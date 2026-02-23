from urllib.parse import urlparse

import ray

from nanodeploy.logging import get_logger

logger = get_logger()


def _clean_and_parse_address(address: str) -> str:
    """
    清理并解析地址，正确处理 'ip:port' 格式。
    """
    if ":" in address and not address.startswith(("http://", "https://")):
        address = f"http://{address}"

    parsed_url = urlparse(address)

    # 如果解析后的 hostname 存在，则返回它
    if parsed_url.hostname:
        return parsed_url.hostname

    # 如果解析失败（例如，输入是一个纯 IP 或主机名），则返回原始地址
    return address


def get_available_nodes_with_master_first(master_address: str):
    """
    Retrieves a list of Ray nodes, sorting them so that the specified master node comes first.
    Excludes nodes that have any ALIVE Placement Groups.

    Args:
        master_address: The address of the master node.

    Returns:
        A list of available Ray node dictionaries, sorted with the master node first.
    """
    all_nodes = ray.nodes()
    if not all_nodes:
        logger.warning("No nodes found in the Ray cluster.")
        return []

    # --------------------------
    # Step 1: Clean and resolve the master address
    # --------------------------
    cleaned_host = _clean_and_parse_address(master_address)
    if cleaned_host in {"localhost", "127.0.0.1"}:
        if not ray.is_initialized():
            raise RuntimeError("Ray must be initialized to resolve 'localhost'")
        resolved_master_ip = ray.util.get_node_ip_address()
    else:
        resolved_master_ip = cleaned_host

    # --------------------------
    # Step 2: Get ALIVE Placement Groups and their nodes
    # --------------------------
    existing_pgs = ray.util.placement_group_table()
    nodes_with_alive_pg = set()

    for _, pg_info in existing_pgs.items():
        pg_state = pg_info.get("state", "")

        # Only consider ALIVE PGs
        if pg_state != "REMOVED":
            # A PG's bundles are spread across nodes. We need all nodes hosting its bundles.
            bundles_to_node_id = pg_info.get("bundles_to_node_id", {})
            for _, node_id in bundles_to_node_id.items():
                if node_id:
                    nodes_with_alive_pg.add(node_id)

    logger.info(f"Node IDs with ALIVE PGs: {nodes_with_alive_pg}")

    # --------------------------
    # Step 3: Filter available nodes
    # --------------------------
    available_nodes = [
        node for node in all_nodes if node["NodeID"] not in nodes_with_alive_pg
    ]

    # --------------------------
    # Step 4: Sort
    # --------------------------
    def sort_key(node):
        node_ip = node.get("NodeManagerAddress")
        logger.info(f"{node_ip=}, {resolved_master_ip=}")
        return 0 if node_ip == resolved_master_ip else 1

    sorted_available_nodes = sorted(available_nodes, key=sort_key)

    logger.info(f"Found {len(sorted_available_nodes)} available nodes.")

    assert sorted_available_nodes, "No available node resources"
    assert (
        sorted_available_nodes[0].get("NodeManagerAddress") == cleaned_host
    ), "master address is occupied or it is not mounted by ray."

    return sorted_available_nodes
