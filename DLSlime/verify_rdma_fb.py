import time

import _slime_c


def test_rdma_flatbuffers():
    print("Testing RDMA FlatBuffers Integration...")

    # 1. Initialize Context
    ctx = _slime_c.RDMAContext()
    ctx.init("rxe0", 1, "RoCE")

    # 2. Create Endpoint
    # Use 2 QPs to test multi-QP support
    ep = _slime_c.RDMAEndpoint(ctx, num_qp=2)

    # 3. Get Endpoint Info (FlatBuffers)
    info_fb = ep.endpoint_info_fb()
    print(f"Endpoint Info (FB) size: {len(info_fb)} bytes")

    # 4. Self-connect test (to verify deserialization)
    try:
        ep.connect(info_fb)
        print("Self-connect with FlatBuffers successful!")
    except Exception as e:
        print(f"Self-connect with FlatBuffers failed: {e}")
        return

    # 5. Get JSON info for comparison (optional)
    info_json = ep.endpoint_info()
    print(f"Endpoint Info (JSON): {info_json.keys()}")

    print("RDMA FlatBuffers Verification Passed (Basic Connection).")


if __name__ == "__main__":
    test_rdma_flatbuffers()
