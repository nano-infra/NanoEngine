import os

from nanodeploy.config import Config
from nanodeploy.engine.scheduler import RoutingStrategy, Scheduler


def test_routing_strategy():
    # Mock a model path
    model_path = "/models/qwen3-235B-Instruct-2507-FP8"
    os.makedirs(model_path, exist_ok=True)

    # Test default
    config = Config(model=model_path)
    scheduler = Scheduler(config)
    print(f"Default strategy: {scheduler.routing_strategy}")
    assert scheduler.routing_strategy == RoutingStrategy.RoundRobin

    # Test LeastBatch
    config = Config(model=model_path, routing_strategy="LeastBatch")
    scheduler = Scheduler(config)
    print(f"Set strategy: {scheduler.routing_strategy}")
    assert scheduler.routing_strategy == RoutingStrategy.LeastBatch


if __name__ == "__main__":
    try:
        test_routing_strategy()
        print("Test passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback

        traceback.print_exc()
