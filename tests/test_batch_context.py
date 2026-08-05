import torch

from dlengine.context_v2.batch import reset_batch_context, set_batch_context


def teardown_function():
    reset_batch_context()


def test_gdn_int32_slots_reuse_persistent_graph_buffer():
    slots = torch.tensor([3, 5, 7], dtype=torch.int32)

    context = set_batch_context(is_prefill=False, gdn_state_slots=slots)

    assert context.gdn_state_slots_i32 is slots
    slots[1] = 11
    assert context.gdn_state_slots_i32.tolist() == [3, 11, 7]


def test_gdn_int64_slots_are_normalized_once_per_context():
    slots = torch.tensor([2, 4], dtype=torch.int64)

    context = set_batch_context(is_prefill=True, gdn_state_slots=slots)

    assert context.gdn_state_slots_i32.dtype == torch.int32
    assert context.gdn_state_slots_i32.tolist() == [2, 4]
