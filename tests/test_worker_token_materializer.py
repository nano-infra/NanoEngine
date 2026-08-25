import pytest
import torch

from nanodeploy.worker.token_materializer import AsyncTokenMaterializer


def test_token_materializer_owns_each_slot_until_collection():
    materializer = AsyncTokenMaterializer(num_slots=2)

    first = materializer.submit(torch.tensor([[11], [12]]), slot=0)
    second = materializer.submit(torch.tensor([[21], [22]]), slot=1)

    with pytest.raises(RuntimeError, match="slot 0 is still active"):
        materializer.submit(torch.tensor([[31]]), slot=0)
    assert materializer.collect(first) == [[11], [12]]
    assert materializer.collect(second) == [[21], [22]]

    reused = materializer.submit(torch.tensor([[31]]), slot=0)
    assert materializer.collect(reused) == [[31]]


def test_token_materializer_rejects_foreign_or_duplicate_collection():
    first_materializer = AsyncTokenMaterializer(num_slots=2)
    second_materializer = AsyncTokenMaterializer(num_slots=2)
    pending = first_materializer.submit(torch.tensor([[7]]), slot=0)

    with pytest.raises(ValueError, match="another materializer"):
        second_materializer.collect(pending)
    assert first_materializer.collect(pending) == [[7]]
    with pytest.raises(RuntimeError, match="not active"):
        first_materializer.collect(pending)


@pytest.mark.parametrize("slot", [-1, 2])
def test_token_materializer_rejects_invalid_slot(slot):
    materializer = AsyncTokenMaterializer(num_slots=2)

    with pytest.raises(ValueError, match="invalid token materialization slot"):
        materializer.submit(torch.tensor([[1]]), slot=slot)
