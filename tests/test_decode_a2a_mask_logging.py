import base64
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

import nanodeploy.worker.model_runner as model_runner


def _decode_mask_rows(encoded_rows: list[str], width: int) -> list[list[int]]:
    decoded_rows = []
    for encoded in encoded_rows:
        packed = np.frombuffer(base64.b64decode(encoded), dtype=np.uint8)
        decoded_rows.append(np.unpackbits(packed, count=width).tolist())
    return decoded_rows


def test_decode_a2a_mask_log_uses_reversible_base64_bit_packing() -> None:
    actor_class = model_runner.ModelRunner.__ray_metadata__.modified_class
    runner = actor_class.__new__(actor_class)
    runner.run_count = 17
    runner.rank = 3

    q_mask = torch.tensor(
        [
            [1, 0, 1, 1, 0, 0, 1, 0, 1, 0],
            [0, 1, 0, 0, 1, 1, 0, 1, 0, 1],
        ],
        dtype=torch.int32,
    )
    res_lse_mask = 1 - q_mask
    context = SimpleNamespace(
        use_sp_a2a=True,
        q_mask=q_mask,
        res_lse_mask=res_lse_mask,
        q_offsets=torch.tensor([0, 2, 4], dtype=torch.int32),
        sp_comm_bs=10,
    )
    dist_context = SimpleNamespace(
        attn_dp_rank=0,
        attn_sp_rank=1,
        attn_tp_rank=0,
        attn_sp_world_size=2,
    )

    with (
        patch.object(model_runner, "get_context", return_value=context),
        patch.object(
            model_runner,
            "get_dist_context",
            return_value=dist_context,
        ),
        patch.object(model_runner.logger, "info") as log_info,
    ):
        runner._log_decode_a2a_masks(loop_idx=4, is_dummy=False)

    log_info.assert_called_once()
    payload = log_info.call_args.args[0]
    assert payload["mode"] == "decode_a2a_masks"
    assert payload["global_run_count"] == 17
    assert payload["loop_idx"] == 4
    assert payload["mask_encoding"] == "bit_b64"
    assert payload["max_bs"] == 10
    assert payload["q_offsets"] == [0, 2, 4]
    assert _decode_mask_rows(payload["q_mask"], width=10) == q_mask.tolist()
    assert _decode_mask_rows(
        payload["res_lse_mask"], width=10
    ) == res_lse_mask.tolist()
