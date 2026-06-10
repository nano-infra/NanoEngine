from urllib.parse import urlparse

import ray

from dlengine.logging import get_logger

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


def _gpus_used_by_alive_pgs() -> dict:
    """Return a mapping of node_id -> number of GPUs reserved by ALIVE PGs.

    A node can host bundles from several placement groups (e.g. a prefill engine
    and a decode engine co-located on the same box). We therefore track GPU
    *usage* per node rather than treating any node that has a PG as fully
    occupied — otherwise a second engine can never schedule on a node that
    already runs one, even when GPUs are free.
    """
    used_gpus_by_node: dict = {}
    existing_pgs = ray.util.placement_group_table()
    for _, pg_info in existing_pgs.items():
        if pg_info.get("state", "") == "REMOVED":
            continue
        bundles = pg_info.get("bundles", {}) or {}
        bundles_to_node_id = pg_info.get("bundles_to_node_id", {}) or {}
        for bundle_idx, node_id in bundles_to_node_id.items():
            if not node_id:
                continue
            # bundle keys may be int or str depending on the Ray version.
            bundle = bundles.get(bundle_idx)
            if bundle is None:
                bundle = bundles.get(str(bundle_idx), {})
            gpu = float((bundle or {}).get("GPU", 0) or 0)
            used_gpus_by_node[node_id] = used_gpus_by_node.get(node_id, 0.0) + gpu
    return used_gpus_by_node


def get_available_nodes_with_master_first(
    master_address: str, required_gpus: float = 1.0
):
    """
    Retrieves a list of Ray nodes, sorting them so that the specified master node comes first.
    Keeps nodes that still have at least ``required_gpus`` free GPUs after
    accounting for GPUs reserved by ALIVE placement groups. This allows several
    engines (e.g. prefill + decode for PD disaggregation) to co-locate on the
    same node when it has spare GPUs.

    Args:
        master_address: The address of the master node.
        required_gpus: Minimum number of free GPUs a node must have to be eligible.

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
    # Step 2: GPUs reserved by ALIVE Placement Groups, per node
    # --------------------------
    used_gpus_by_node = _gpus_used_by_alive_pgs()
    logger.info(f"GPUs reserved by ALIVE PGs (by node): {used_gpus_by_node}")

    def _free_gpus(node) -> float:
        total = float(node.get("Resources", {}).get("GPU", 0) or 0)
        return total - used_gpus_by_node.get(node["NodeID"], 0.0)

    # --------------------------
    # Step 3: Keep nodes with enough free GPUs
    # --------------------------
    available_nodes = [
        node
        for node in all_nodes
        if node.get("Alive", True) and _free_gpus(node) >= required_gpus
    ]

    # --------------------------
    # Step 4: Sort
    # --------------------------
    def sort_key(node):
        node_ip = node.get("NodeManagerAddress")
        logger.info(f"{node_ip=}, {resolved_master_ip=}, free_gpus={_free_gpus(node)}")
        return 0 if node_ip == resolved_master_ip else 1

    sorted_available_nodes = sorted(available_nodes, key=sort_key)

    logger.info(
        f"Found {len(sorted_available_nodes)} available nodes "
        f"(required_gpus={required_gpus})."
    )

    assert sorted_available_nodes, "No available node resources"
    assert (
        sorted_available_nodes[0].get("NodeManagerAddress") == resolved_master_ip
    ), "master address is occupied or it is not mounted by ray."

    return sorted_available_nodes
