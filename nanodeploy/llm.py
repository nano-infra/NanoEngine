import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nanodeploy.engine.llm_engine import LLMEngine
from nanodeploy.engine.ray_utils import get_available_nodes_with_master_first


class LLM(LLMEngine):
    @classmethod
    def as_remote(cls, config):
        ray_address = getattr(config, "ray_address", "127.0.0.1:6379")
        master_address = getattr(config, "master_address", "127.0.0.1:6006")
        ray.init(address=ray_address, ignore_reinit_error=True)

        nodes = get_available_nodes_with_master_first(master_address)
        target_node_id = nodes[0]["NodeID"]

        return (
            ray.remote(num_cpus=1, num_gpus=0)(cls)
            .options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=target_node_id, soft=False
                )
            )
            .remote(config)
        )
