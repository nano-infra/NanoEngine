import sys

import etcd3

try:
    client = etcd3.client()
    client.get("/")
    print("Successfully connected to etcd!")
except Exception as e:
    print(f"Failed to connect to etcd: {e}")
    sys.exit(1)
