"""Backward-compatibility shim for nanodeploy.layers.distributed_routed_experts.

The implementation has moved to:
    nanodeploy.backends.hopper.layers.experts.HopperDistributedRoutedExperts  (Hopper)
    nanodeploy.backends.gpu_generic.layers.experts.GenericDistributedRoutedExperts  (generic)

New code should use:
    from nanodeploy.backends import get_backend
    self.experts = get_backend().get_distributed_routed_experts(...)
"""

from nanodeploy.backends import get_backend


def DistributedRoutedExperts(
    hidden_size,
    intermediate_size,
    num_experts,
    top_k,
    ep_size,
    tp_size,
    ep_group=None,
    tp_group=None,
    n_group=None,
    topk_group=None,
    norm_topk_prob=False,
    routed_scaling_factor=1.0,
    scoring_func="softmax",
    quantization_config=None,
    **kwargs,
):
    return get_backend().get_distributed_routed_experts(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
        ep_size=ep_size,
        tp_size=tp_size,
        ep_group=ep_group,
        tp_group=tp_group,
        n_group=n_group,
        topk_group=topk_group,
        norm_topk_prob=norm_topk_prob,
        routed_scaling_factor=routed_scaling_factor,
        scoring_func=scoring_func,
    )
