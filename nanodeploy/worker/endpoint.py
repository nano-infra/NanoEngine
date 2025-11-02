import dataclasses

import dlslime


@dataclasses.dataclass
class EndpointContext:
    rank: int
    local_rank: int

    world_size: int

    selected_nic: str
    endpoints: dict[str, dict[int, dlslime.RDMAEndpoint]] = None

    def __post_init__(self):
        

    
