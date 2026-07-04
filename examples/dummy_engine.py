import socket
import struct
import time

from dlengine.server.wire import encode_stepout, SequenceStatus

# Spoke Protocol Constants
MAGIC = 0x504F4B45
HEADER_FMT = "<III"  # Magic(4), MetaSize(4), DataSize(4) -> 12 bytes
META_SIZE = 72  # Fixed NetMeta size


def start_dummy_engine(host="127.0.0.1", port=5000):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(1)

    addr = server.getsockname()
    print(f"DUMMY_ENGINE_LISTENING_ON:{addr[0]}:{addr[1]}", flush=True)

    while True:
        try:
            conn, client_addr = server.accept()
            print(f"Accepted connection from {client_addr}")

            while True:
                # 1. Read Header (12 bytes)
                header_data = conn.recv(12)
                if not header_data:
                    break

                magic, meta_size, data_size = struct.unpack(HEADER_FMT, header_data)

                if magic != MAGIC:
                    print(f"Invalid Magic: {hex(magic)}")
                    break

                # 2. Read Meta (meta_size bytes)
                meta_data = conn.recv(meta_size)

                # 3. Read Data (data_size bytes)
                payload = b""
                if data_size > 0:
                    payload = conn.recv(data_size)
                    while len(payload) < data_size:
                        payload += conn.recv(data_size - len(payload))

                print(f"Received Action. Payload Size: {len(payload)} bytes")

                # 4. Process Request (Assume generic AddRequest for demo)
                # In real scenario, we parse payload as SequenceList.
                # Here we just blindly response with a StepOut.

                resp_payload = encode_stepout(123, [9999], SequenceStatus.FINISHED)

                print(f"Generating StepOut. Size: {len(resp_payload)}")

                # 5. Send Response (Action=2 for StepOut, purely convention here)
                resp_header = struct.pack(
                    HEADER_FMT, MAGIC, META_SIZE, len(resp_payload)
                )
                # Re-use meta for simplicity or zero it out
                resp_meta = meta_data

                time.sleep(0.1)

                conn.sendall(resp_header)
                conn.sendall(resp_meta)
                conn.sendall(resp_payload)
                print("StepOut sent")

        except Exception as e:
            print(f"Connection Error: {e}")
        finally:
            server.close()
            return


if __name__ == "__main__":
    start_dummy_engine(port=5000)
