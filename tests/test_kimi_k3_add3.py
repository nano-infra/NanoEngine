import pytest
import torch
from dlengine.runtime.kernel.jit.sgl.add3 import add3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_add3_matches_unfused_rounding_and_is_graph_safe():
    a = torch.randn((8, 7168), device="cuda", dtype=torch.bfloat16)
    b = torch.randn_like(a)
    c = torch.randn_like(a)
    expected = (a + b) + c
    actual = add3(a, b, c)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replayed = add3(a, b, c)
    a.copy_(torch.randn_like(a))
    b.copy_(torch.randn_like(b))
    c.copy_(torch.randn_like(c))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(replayed, (a + b) + c, rtol=0, atol=0)
