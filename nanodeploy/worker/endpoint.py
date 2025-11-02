import dataclasses

import dlslime


@dataclasses.dataclass
class EndpointContext:
    rank: int
    local_rank: int

    world_size: int

    selected_nic: str
    endpoints: dict[str, list[dlslime.RDMAEndpoint]] = None

    def __post_init__(self):
        available_nics = dlslime.available_nic()
        self.selected_nic = available_nics[self.local_rank % len(available_nics)]

        self.endpoints = {}

    def init(
        self,
    ):
        pass
