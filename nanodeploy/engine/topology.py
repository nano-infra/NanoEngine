from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class EngineTopology:
    """The ranks owned by one hierarchical LocalEngineCore."""

    engine_id: int
    global_dp_idx: int
    global_ranks: tuple[int, ...]
    attention_sp: int
    attention_tp: int

    @property
    def world_size(self) -> int:
        return self.attention_sp * self.attention_tp

    def global_rank(self, sp_idx: int, tp_idx: int = 0) -> int:
        if not 0 <= sp_idx < self.attention_sp:
            raise ValueError(f"sp_idx must be in [0, {self.attention_sp})")
        if not 0 <= tp_idx < self.attention_tp:
            raise ValueError(f"tp_idx must be in [0, {self.attention_tp})")
        return self.global_ranks[sp_idx * self.attention_tp + tp_idx]

    def engine_local_rank(self, global_rank: int) -> int:
        try:
            return self.global_ranks.index(global_rank)
        except ValueError as exc:
            raise ValueError(
                f"global rank {global_rank} is not owned by engine {self.engine_id}"
            ) from exc


@dataclass(frozen=True, slots=True)
class HierarchicalTopology:
    attention_dp: int
    attention_sp: int
    attention_tp: int
    ffn_dp: int
    ffn_ep: int
    ffn_tp: int
    engines: tuple[EngineTopology, ...]

    @property
    def world_size(self) -> int:
        return self.attention_dp * self.attention_sp * self.attention_tp

    def engine(self, global_dp_idx: int) -> EngineTopology:
        if not 0 <= global_dp_idx < len(self.engines):
            raise ValueError(
                f"global_dp_idx must be in [0, {len(self.engines)})"
            )
        return self.engines[global_dp_idx]


# (attention_dp, attention_sp, attention_tp, ffn_dp, ffn_ep, ffn_tp)
HIERARCHICAL_TOPOLOGY_WHITELIST = frozenset(
    {
        (8, 1, 1, 1, 8, 1),
        (2, 4, 1, 1, 8, 1),
        (1, 8, 1, 1, 8, 1),
        (16, 1, 1, 1, 16, 1),
        (2, 8, 1, 1, 16, 1),
        (32, 1, 1, 1, 32, 1),
        (4, 8, 1, 1, 32, 1),
    }
)


def build_hierarchical_topology(
    *,
    attention_dp: int,
    attention_sp: int,
    attention_tp: int,
    ffn_dp: int,
    ffn_ep: int,
    ffn_tp: int,
) -> HierarchicalTopology:
    signature = (
        attention_dp,
        attention_sp,
        attention_tp,
        ffn_dp,
        ffn_ep,
        ffn_tp,
    )
    if signature not in HIERARCHICAL_TOPOLOGY_WHITELIST:
        supported = ", ".join(
            f"DP{dp} SP{sp} TP{tp} / FFN-DP{fdp} EP{ep} TP{ftp}"
            for dp, sp, tp, fdp, ep, ftp in sorted(
                HIERARCHICAL_TOPOLOGY_WHITELIST
            )
        )
        raise ValueError(
            "hierarchical scheduler topology is not supported: "
            f"DP{attention_dp} SP{attention_sp} TP{attention_tp} / "
            f"FFN-DP{ffn_dp} EP{ffn_ep} TP{ffn_tp}; supported: {supported}"
        )

    attention_world_size = attention_dp * attention_sp * attention_tp
    ffn_world_size = ffn_dp * ffn_ep * ffn_tp
    if attention_world_size != ffn_world_size:
        raise ValueError(
            "hierarchical scheduler requires identical attention and FFN "
            f"world sizes, got {attention_world_size} and {ffn_world_size}"
        )
    if attention_tp != 1 or ffn_tp != 1 or ffn_dp != 1:
        raise ValueError(
            "hierarchical scheduler requires attention_tp=1, ffn_tp=1, "
            "and ffn_dp=1"
        )
    if ffn_ep != attention_dp * attention_sp:
        raise ValueError(
            "hierarchical scheduler requires one deployment-wide FFN EP "
            "group: ffn_ep == attention_dp * attention_sp"
        )

    engine_world_size = attention_sp * attention_tp
    engines = tuple(
        EngineTopology(
            engine_id=global_dp_idx,
            global_dp_idx=global_dp_idx,
            global_ranks=tuple(
                range(
                    global_dp_idx * engine_world_size,
                    (global_dp_idx + 1) * engine_world_size,
                )
            ),
            attention_sp=attention_sp,
            attention_tp=attention_tp,
        )
        for global_dp_idx in range(attention_dp)
    )
    return HierarchicalTopology(
        attention_dp=attention_dp,
        attention_sp=attention_sp,
        attention_tp=attention_tp,
        ffn_dp=ffn_dp,
        ffn_ep=ffn_ep,
        ffn_tp=ffn_tp,
        engines=engines,
    )
