import struct
import unittest

import flatbuffers
from nanodeploy._cpp import deserialize, SamplingParams, Sequence, serialize
from nanodeploy.fbs import (
    SamplingParams as FbSamplingParams,
    Sequence as FbSequence,
    SequenceList,
    SequenceStatus as FbSequenceStatus,
)


class TestSerialization(unittest.TestCase):
    def test_round_trip(self):
        # 1. Create a Python/C++ Sequence object
        # Need to use C++ enum
        from nanodeploy._cpp import SequenceStatus as CppSequenceStatus

        seq = Sequence([101, 102, 103], SamplingParams())
        seq.seq_id = 999
        seq.status = CppSequenceStatus.WAITING

        # 2. Serialize using C++ binding
        import ctypes

        buffer_size = 4096
        buffer = ctypes.create_string_buffer(buffer_size)
        ptr = ctypes.addressof(buffer)

        seqs = [seq]
        size = serialize(ptr, buffer_size, seqs, True)  # is_prefill=True

        print(f"Serialized size: {size}")

        # 3. Deserialize using C++ binding

        restored_seqs = deserialize(ptr, size)

        self.assertEqual(len(restored_seqs), 1)
        r_seq = restored_seqs[0]
        self.assertEqual(r_seq.seq_id, seq.seq_id)
        self.assertEqual(r_seq.token_ids, seq.token_ids)
        print("Round trip successful via C++ serializer.")

    def test_manual_flatbuffer_construction(self):
        """
        Simulate what Rust does: manually build FlatBuffer in Python and pass to C++ deserializer.
        """
        builder = flatbuffers.Builder(1024)

        import nanodeploy.fbs.SamplingParams as SamplingParamsModule

        # Helper to use module functions
        import nanodeploy.fbs.Sequence as SequenceModule
        import nanodeploy.fbs.SequenceList as SequenceListModule

        # Vector of token_ids
        token_ids = [101, 102, 103]
        SequenceModule.StartTokenIdsVector(builder, len(token_ids))
        for t in reversed(token_ids):
            builder.PrependInt32(t)
        token_ids_off = builder.EndVector()

        # SamplingParams
        SamplingParamsModule.Start(builder)
        SamplingParamsModule.AddTemperature(builder, 0.7)
        SamplingParamsModule.AddMaxTokens(builder, 10)
        SamplingParamsModule.AddIgnoreEos(builder, False)
        sp_off = SamplingParamsModule.End(builder)

        # Sequence
        SequenceModule.Start(builder)
        SequenceModule.AddSeqId(builder, 12345)
        SequenceModule.AddStatus(builder, 1)  # WAITING
        SequenceModule.AddTokenIds(builder, token_ids_off)
        SequenceModule.AddSamplingParams(builder, sp_off)
        seq_off = SequenceModule.End(builder)

        # SequenceList
        SequenceListModule.StartSequencesVector(builder, 1)
        builder.PrependUOffsetTRelative(seq_off)
        seqs_vec_off = builder.EndVector()

        SequenceListModule.Start(builder)
        SequenceListModule.AddSequences(builder, seqs_vec_off)
        root = SequenceListModule.End(builder)

    def test_reproduce_payload(self):
        """
        Reproduce failure with the actual payload captured from Rust server.
        Payload Hex: 100000000000000000000600080004000600000004000000010000001c000000000016001800100007000800000000000000000000000c00160000000000000118000000240000007b000000000000000800100008000400080000000a0000009a9999999999b93f03000000010000000200000003000000
        """
        hex_str = "100000000000000000000600080004000600000004000000010000001c000000000016001800100007000800000000000000000000000c00160000000000000118000000240000007b000000000000000800100008000400080000000a0000009a9999999999b93f03000000010000000200000003000000"
        payload = bytes.fromhex(hex_str)
        print(f"Testing payload of size: {len(payload)}")

        import ctypes

        c_buf = ctypes.create_string_buffer(payload, len(payload))
        ptr = ctypes.addressof(c_buf)

        try:
            restored_seqs = deserialize(ptr, len(payload))
            print("Successfully deserialized captured payload!")
            print(f"Num Seqs: {len(restored_seqs)}")
            if len(restored_seqs) > 0:
                s = restored_seqs[0]
                print(f"Seq ID: {s.seq_id}")
                print(f"Token IDs: {s.token_ids}")
        except RuntimeError as e:
            print(f"Deserialization failed as expected: {e}")
            # Raise to fail test if we want, or just pass to confirm reproduction
            raise e


if __name__ == "__main__":
    unittest.main()
