import flash_mla
import torch


def prepare_decode_mla_metadata(
    hf_config,
    context_lens: torch.Tensor,
    tile_scheduler_metadata_buffer: torch.Tensor | None = None,
    num_splits_buffer: torch.Tensor | None = None,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Build one set of FlashMLA decode metadata for a model iteration.

    Full CUDA graphs need stable metadata addresses, so their freshly built
    metadata is copied into persistent buffers. Eager and piecewise execution
    can consume the freshly allocated tensors directly.
    """
    if hf_config.num_key_value_heads != 1:
        return None, None

    if (tile_scheduler_metadata_buffer is None) != (num_splits_buffer is None):
        raise ValueError(
            "FlashMLA graph metadata buffers must be provided together"
        )

    tile_scheduler_metadata, num_splits = flash_mla.get_mla_metadata(
        context_lens,
        hf_config.num_attention_heads // hf_config.num_key_value_heads,
        hf_config.num_key_value_heads,
    )
    if tile_scheduler_metadata_buffer is None:
        return tile_scheduler_metadata, num_splits

    if tile_scheduler_metadata.ndim != tile_scheduler_metadata_buffer.ndim:
        raise RuntimeError(
            "FlashMLA tile metadata rank does not match the graph buffer: "
            f"metadata={tuple(tile_scheduler_metadata.shape)} "
            f"buffer={tuple(tile_scheduler_metadata_buffer.shape)}"
        )
    if any(
        metadata_size > buffer_size
        for metadata_size, buffer_size in zip(
            tile_scheduler_metadata.shape,
            tile_scheduler_metadata_buffer.shape,
            strict=True,
        )
    ):
        raise RuntimeError(
            "FlashMLA tile metadata exceeds graph buffer capacity: "
            f"metadata={tuple(tile_scheduler_metadata.shape)} "
            f"buffer={tuple(tile_scheduler_metadata_buffer.shape)}"
        )
    if num_splits.numel() > num_splits_buffer.numel():
        raise RuntimeError(
            "FlashMLA split metadata exceeds graph buffer capacity: "
            f"metadata={num_splits.numel()} buffer={num_splits_buffer.numel()}"
        )

    tile_slices = tuple(slice(0, size) for size in tile_scheduler_metadata.shape)
    tile_scheduler_metadata_view = tile_scheduler_metadata_buffer[tile_slices]
    num_splits_view = num_splits_buffer[: num_splits.numel()]
    tile_scheduler_metadata_view.copy_(tile_scheduler_metadata)
    num_splits_view.copy_(num_splits)
    return tile_scheduler_metadata_view, num_splits_view
