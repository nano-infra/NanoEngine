"""XGrammar-backed JSON Schema and StructuralTag constrained decoding.

The manager is intentionally owned by the sampling rank of a model worker.
Grammar matchers are stateful per sequence, while compiled grammars and token
masks are shared/reused across requests and decode steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch


@dataclass
class _MatcherState:
    grammar_type: str
    grammar: str
    matcher: Any


class StructuredOutputManager:
    """Apply per-request grammar masks and advance matcher state."""

    def __init__(
        self,
        tokenizer: Any,
        vocab_size: int,
        stop_token_ids: Sequence[int],
    ) -> None:
        try:
            import xgrammar as xgr
        except ImportError as exc:  # pragma: no cover - dependency error at startup
            raise RuntimeError(
                "structured-output decoding requires the 'xgrammar' package"
            ) from exc

        self._xgr = xgr
        self._vocab_size = int(vocab_size)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(
            tokenizer,
            vocab_size=self._vocab_size,
            stop_token_ids=[int(token) for token in stop_token_ids],
        )
        self._compiler = xgr.GrammarCompiler(tokenizer_info, cache_enabled=True)
        self._compiled: dict[tuple[str, str], Any] = {}
        self._matchers: dict[int, _MatcherState] = {}
        self._host_bitmask: Optional[torch.Tensor] = None
        self._device_bitmask: Optional[torch.Tensor] = None

    @classmethod
    def from_model(
        cls,
        model_path: str,
        vocab_size: int,
        stop_token_ids: Sequence[int],
    ) -> "StructuredOutputManager":
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        # ``config.eos`` is normally populated by LLMEngine, but RPC entry
        # points such as DLSlime construct ModelRunner directly and can bypass
        # that initialization.  XGrammar requires at least one stop token, so
        # resolve the model metadata again at this ownership boundary.
        from dlengine.models.trait import resolve_eos_token_ids

        resolved_stop_ids = {
            int(token_id) for token_id in stop_token_ids if token_id is not None
        }
        resolved_stop_ids.update(resolve_eos_token_ids(model_path, tokenizer))
        if not resolved_stop_ids:
            raise RuntimeError(
                "structured-output decoding could not resolve any EOS/stop token "
                f"for model {model_path!r}"
            )
        return cls(tokenizer, vocab_size, sorted(resolved_stop_ids))

    @staticmethod
    def has_constraints(
        json_schemas: Sequence[Optional[str]],
        structural_tags: Sequence[Optional[str]],
    ) -> bool:
        return any(json_schemas) or any(structural_tags)

    def _get_matcher(
        self,
        seq_id: int,
        json_schema: Optional[str],
        structural_tag: Optional[str],
    ) -> tuple[Any, bool]:
        if json_schema and structural_tag:
            raise RuntimeError(
                f"seq_id={seq_id} has both JSON Schema and StructuralTag constraints"
            )
        if structural_tag:
            grammar_type, grammar = "structural_tag", structural_tag
        elif json_schema:
            grammar_type, grammar = "json_schema", json_schema
        else:  # guarded by callers
            raise RuntimeError(f"seq_id={seq_id} has no structured-output grammar")

        state = self._matchers.get(seq_id)
        if (
            state is not None
            and state.grammar_type == grammar_type
            and state.grammar == grammar
        ):
            return state.matcher, False

        cache_key = (grammar_type, grammar)
        compiled = self._compiled.get(cache_key)
        if compiled is None:
            if grammar_type == "json_schema":
                compiled = self._compiler.compile_json_schema(grammar, strict_mode=True)
            else:
                compiled = self._compiler.compile_structural_tag(grammar)
            self._compiled[cache_key] = compiled
        matcher = self._xgr.GrammarMatcher(compiled)
        self._matchers[seq_id] = _MatcherState(
            grammar_type=grammar_type, grammar=grammar, matcher=matcher
        )
        return matcher, True

    def _ensure_bitmask(
        self, rows: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        expected_shape = (rows, (self._vocab_size + 31) // 32)
        if (
            self._host_bitmask is None
            or tuple(self._host_bitmask.shape) != expected_shape
        ):
            self._host_bitmask = self._xgr.allocate_token_bitmask(
                rows, self._vocab_size
            )
            self._device_bitmask = None
        if (
            self._device_bitmask is None
            or tuple(self._device_bitmask.shape) != expected_shape
            or self._device_bitmask.device != device
        ):
            self._device_bitmask = torch.empty(
                expected_shape, dtype=torch.int32, device=device
            )
        return self._host_bitmask, self._device_bitmask

    def apply(
        self,
        logits: torch.Tensor,
        seq_ids: Sequence[int],
        json_schemas: Sequence[Optional[str]],
        structural_tags: Sequence[Optional[str]],
        *,
        seed_tokens: Optional[torch.Tensor] = None,
    ) -> None:
        """Mask invalid logits for constrained rows.

        ``seed_tokens`` is supplied on decode.  It is consumed only when a
        matcher is first observed, which restores the grammar state on the
        decode side of a prefill/decode migration without double-accepting
        tokens during ordinary decode.
        """
        rows = logits.shape[0]
        if (
            rows != len(seq_ids)
            or rows != len(json_schemas)
            or rows != len(structural_tags)
        ):
            raise RuntimeError(
                "structured-output batch mismatch: "
                f"logits={rows}, seq_ids={len(seq_ids)}, "
                f"schemas={len(json_schemas)}, tags={len(structural_tags)}"
            )
        if logits.shape[-1] < self._vocab_size:
            raise RuntimeError(
                f"logits vocab {logits.shape[-1]} is smaller than tokenizer vocab "
                f"{self._vocab_size}"
            )

        host_mask, device_mask = self._ensure_bitmask(rows, logits.device)
        pending_seeds: Optional[list[int]] = None
        mask_indices: list[int] = []
        for row, (seq_id, schema, structural_tag) in enumerate(
            zip(seq_ids, json_schemas, structural_tags)
        ):
            if not schema and not structural_tag:
                continue
            matcher, created = self._get_matcher(int(seq_id), schema, structural_tag)
            if created and seed_tokens is not None:
                if pending_seeds is None:
                    pending_seeds = [int(token) for token in seed_tokens.tolist()]
                if not matcher.accept_token(pending_seeds[row]):
                    raise RuntimeError(
                        f"failed to restore grammar state for seq_id={seq_id}"
                    )
            if matcher.fill_next_token_bitmask(host_mask, row):
                mask_indices.append(row)

        if not mask_indices:
            return
        device_mask.copy_(host_mask, non_blocking=True)
        self._xgr.apply_token_bitmask_inplace(
            logits,
            device_mask,
            vocab_size=self._vocab_size,
            indices=mask_indices,
        )

    def accept(
        self,
        seq_ids: Sequence[int],
        json_schemas: Sequence[Optional[str]],
        structural_tags: Sequence[Optional[str]],
        sampled_tokens: torch.Tensor,
    ) -> None:
        """Advance matchers with the sampled tokens and release completed ones."""
        token_ids = [int(token) for token in sampled_tokens.tolist()]
        for seq_id, schema, structural_tag, token_id in zip(
            seq_ids, json_schemas, structural_tags, token_ids
        ):
            if not schema and not structural_tag:
                continue
            state = self._matchers.get(int(seq_id))
            expected_type = "structural_tag" if structural_tag else "json_schema"
            expected_grammar = structural_tag or schema
            if (
                state is None
                or state.grammar_type != expected_type
                or state.grammar != expected_grammar
            ):
                raise RuntimeError(f"missing grammar matcher for seq_id={seq_id}")
            if not state.matcher.accept_token(token_id):
                from dlengine.logging import get_logger
                get_logger("DLENGINE").error(
                    "XGrammar rejected sampled token %d for seq_id=%s. Discarding matcher.",
                    token_id, seq_id
                )
                self._matchers.pop(int(seq_id), None)
                continue
            if state.matcher.is_terminated():
                self._matchers.pop(int(seq_id), None)

    def discard(self, seq_ids: Sequence[int]) -> None:
        """Drop matcher state for aborted/cancelled sequences."""
        for seq_id in seq_ids:
            self._matchers.pop(int(seq_id), None)
