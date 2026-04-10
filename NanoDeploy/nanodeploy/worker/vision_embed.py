"""Vision embedding RDMA fetch and injection for EP-separated mode."""

from __future__ import annotations

from collections import defaultdict

import torch

from nanodeploy.context.cache import get_cache_context
from nanodeploy.logging import get_logger

logger = get_logger("NANODEPLOY")


class VisionEmbedManager:
    """Manages vision embedding RDMA fetch and injection."""

    def __init__(self, hf_config):
        self.hf_config = hf_config
        self._vision_embeds: dict[str, torch.Tensor] | None = None

    @property
    def has_embeds(self) -> bool:
        return self._vision_embeds is not None

    def clear(self):
        self._vision_embeds = None

    def inject(
        self, input_ids: torch.Tensor, embed_tokens: torch.nn.Module
    ) -> torch.Tensor | None:
        """Build ``inputs_embeds`` by merging text + vision embeddings.

        Returns ``None`` if no vision embeddings are stored.
        """
        if self._vision_embeds is None:
            return None

        inputs_embeds = embed_tokens(input_ids)
        hf_config = self.hf_config

        # Inject image embeddings
        if "image" in self._vision_embeds:
            image_token_id = getattr(hf_config, "image_token_id", None)
            if image_token_id is not None:
                image_embeds = self._vision_embeds["image"].to(
                    dtype=inputs_embeds.dtype
                )
                mask = input_ids == image_token_id
                n_tokens = mask.sum().item()
                if n_tokens > 0 and n_tokens == image_embeds.shape[0]:
                    mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(
                        mask_expanded, image_embeds
                    )
                    logger.debug(f"Injected {n_tokens} image tokens")
                else:
                    logger.warning(
                        f"Image token count mismatch: input has {n_tokens}, "
                        f"embeds has {image_embeds.shape[0]} — skipping injection"
                    )

        # Inject video embeddings
        if "video" in self._vision_embeds:
            video_token_id = getattr(hf_config, "video_token_id", None)
            if video_token_id is not None:
                video_embeds = self._vision_embeds["video"].to(
                    dtype=inputs_embeds.dtype
                )
                mask = input_ids == video_token_id
                n_tokens = mask.sum().item()
                if n_tokens > 0 and n_tokens == video_embeds.shape[0]:
                    mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds)
                    inputs_embeds = inputs_embeds.masked_scatter(
                        mask_expanded, video_embeds
                    )

        return inputs_embeds

    def fetch_rdma(self, vision_slot_views: list, model_dtype: torch.dtype) -> None:
        """RDMA-fetch vision embeddings from remote encoder(s).

        Reads embeddings from encoder EmbeddingPool into a local receive
        buffer via dlslime, then stores them in ``self._vision_embeds``
        for injection during model forward.

        Args:
            vision_slot_views: List of VisionSlotView from
                ``extract_vision_slots_from_bytes``.
            model_dtype: dtype of the model's embedding layer weights.
        """
        if not vision_slot_views:
            return

        cache_ctx = get_cache_context()
        peer_agent = cache_ctx._peer_agent
        if peer_agent is None:
            logger.warning("PeerAgent not available, cannot RDMA-fetch vision embeds")
            return

        from nanodeploy.context.embedding_pool import _VISION_EMBED_BUFFER_ID

        by_encoder: dict[str, list] = defaultdict(list)
        for v in vision_slot_views:
            by_encoder[v.encoder_engine_id].append(v)

        # Compute total tokens for local receive buffer
        total_tokens = sum(v.num_tokens for v in vision_slot_views)
        hidden_size = vision_slot_views[0].hidden_size
        dtype = model_dtype
        itemsize = torch.tensor([], dtype=dtype).element_size()

        # Allocate local receive buffer on GPU
        recv_buf = torch.zeros(total_tokens, hidden_size, dtype=dtype, device="cuda")
        recv_buf_size = recv_buf.nelement() * recv_buf.element_size()
        recv_mr = peer_agent.register_memory_region(
            "vision_recv",
            recv_buf.data_ptr(),
            int(recv_buf.storage_offset()),
            recv_buf_size,
        )

        # Look up peer_addrs for all encoders via NanoCtrl
        encoder_info_map = cache_ctx._fetch_engine_info_from_nanoctrl(
            set(by_encoder.keys())
        )

        token_offset = 0
        for encoder_id, slots in by_encoder.items():
            encoder_info = encoder_info_map.get(encoder_id, {})
            peer_addrs = encoder_info.get("peer_addrs", [])
            if peer_addrs:
                peer_alias = peer_addrs[0]
            else:
                logger.warning(
                    f"No peer_addrs for encoder {encoder_id} in NanoCtrl, "
                    "falling back to legacy alias"
                )
                peer_alias = f"{encoder_id}:0"

            # Ensure connection
            if peer_alias not in cache_ctx._connected_peers:
                cache_ctx._peer_agent.set_desired_topology(
                    target_peers=list(cache_ctx._connected_peers | {peer_alias}),
                    symmetric=True,
                )
                cache_ctx._peer_agent.wait_for_peers([peer_alias], timeout_sec=30)
                cache_ctx._connected_peers.add(peer_alias)

            # Get remote MR info for vision_embed buffer
            remote_mr_info = peer_agent.get_mr_info(peer_alias, _VISION_EMBED_BUFFER_ID)
            if remote_mr_info is None:
                logger.error(
                    f"Failed to get MR info for vision_embed from {peer_alias}"
                )
                continue

            remote_mr = peer_agent.register_remote_memory_region(
                peer_alias, _VISION_EMBED_BUFFER_ID, remote_mr_info
            )
            endpoint = peer_agent.get_endpoint(peer_alias)
            if endpoint is None:
                logger.error(f"Failed to get endpoint for {peer_alias}")
                continue

            # Build RDMA read ops for all slots from this encoder
            rdma_ops = []
            for v in slots:
                slot_stride = v.max_tokens_per_slot * v.hidden_size * itemsize
                remote_off = v.slot_idx * slot_stride
                read_len = v.num_tokens * v.hidden_size * itemsize
                local_off = token_offset * hidden_size * itemsize

                rdma_ops.append(
                    (
                        recv_mr,  # local MR
                        remote_mr,  # remote MR
                        remote_off,  # remote offset
                        local_off,  # local offset
                        read_len,  # bytes to read
                    )
                )
                token_offset += v.num_tokens

            # Execute batched RDMA reads
            if rdma_ops:
                endpoint.read(rdma_ops)
                logger.debug(
                    f"RDMA-fetched {len(rdma_ops)} vision slots from encoder "
                    f"{encoder_id} ({sum(v.num_tokens for v in slots)} tokens)"
                )

        logger.info(
            f"[VISION_RDMA] Stored vision embeds: shape={recv_buf.shape}, "
            f"dtype={recv_buf.dtype}"
        )
        if logger.isEnabledFor(10):  # DEBUG
            logger.debug(
                f"[VISION_RDMA] norm={recv_buf.norm().item():.4f}, "
                f"nonzero={recv_buf.count_nonzero().item()}/{recv_buf.numel()}"
            )
        self._vision_embeds = {"image": recv_buf}
