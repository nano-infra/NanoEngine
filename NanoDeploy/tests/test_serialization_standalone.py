import os
import sys

# Ensure we can import the shared module directly
# We need to point to where build output put generated .so
# Based on CMakeLists, it installs to ${NANODEPLOY_INSTALL_PATH}, which we set to 'lib'
# But locally build/lib might contain the artifact too.

# Try to import directly from the .so location
# Usually it's in build/lib or build/lib.linux-x86_64-cpython-313
# Let's inspect sys.path
sys.path.append(os.path.join(os.getcwd(), "lib"))

try:
    import _nanodeploy_cpp as cpp
except ImportError:
    # Fallback to build directory if not installed to lib
    sys.path.append(os.path.join(os.getcwd(), "build/lib"))
    import _nanodeploy_cpp as cpp

print("Successfully imported _nanodeploy_cpp")


def test_serialization():
    # Create a Sequence
    token_ids = [101, 202, 303, 404]
    seq = cpp.Sequence(token_ids, 0.7, 128, False)
    seq.seq_id = 999
    seq.status = cpp.SequenceStatus.RUNNING

    # Configure a block context
    ctx = seq.block_ctx(cpp.BlockContextSlot.ACTIVE)
    ctx.reset("test_engine", 1, 1)

    # Add some block data
    # Test mutable proxy behavior before serialization
    ctx.block_location.append((0, 50))
    ctx.group_block_table[0].append(100)

    # Buffer for serialization
    buf_size = 4096

    # We need a writeable buffer. In C++ binding it expects uintptr_t.
    # We can use ctypes.
    import ctypes

    buf = (ctypes.c_ubyte * buf_size)()
    addr = ctypes.addressof(buf)

    print("Serializing...")
    size = cpp.serialize(addr, buf_size, [seq], True)
    print(f"Serialized size: {size} bytes")

    print("Deserializing...")
    seqs_out = cpp.deserialize(addr, size)

    assert len(seqs_out) == 1
    s2 = seqs_out[0]

    # Verification
    print(f"Original token_ids: {seq.token_ids}")
    print(f"Restored token_ids: {s2.token_ids}")
    assert s2.token_ids == token_ids

    print(f"Original ID: {seq.seq_id}")
    print(f"Restored ID: {s2.seq_id}")
    assert s2.seq_id == 999

    assert s2.status == cpp.SequenceStatus.RUNNING

    # Check complex structures
    ctx2 = s2.block_ctx(cpp.BlockContextSlot.ACTIVE)
    print(f"Restored engine_id: {ctx2.engine_id}")
    assert ctx2.engine_id == "test_engine"

    locs = list(ctx2.block_location)
    print(f"Restored block_location: {locs}")
    assert len(locs) == 1
    assert locs[0] == (0, 50)

    # Check group_block_table
    table = ctx2.group_block_table[0]
    print(f"Restored block table: {list(table)}")
    assert list(table) == [100]

    # Test Mutability of deserialized object
    print("Testing mutability of restored object...")
    # s2.token_ids returns a copy (standard pybind11), so we must assign back to update C++
    ids = s2.token_ids
    ids.append(555)
    s2.token_ids = ids
    assert s2.token_ids[-1] == 555
    ctx2.block_location.append((1, 60))
    assert len(ctx2.block_location) == 2

    print("PASS: Serialization round-trip and mutability verified!")


if __name__ == "__main__":
    test_serialization()
