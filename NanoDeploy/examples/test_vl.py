"""Test multimodal (VL) request through NanoRoute."""

import base64
import json
import sys

import requests

IMAGE_PATH = "/tmp/test_image.jpg"
NANOROUTE_URL = "http://127.0.0.1:3001/v1/chat/completions"

with open(IMAGE_PATH, "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "/models/",
    "stream": False,
    "ignore_eos": False,
    "max_tokens": 64,
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ],
}

print(f"Image: {IMAGE_PATH} ({len(b64)} bytes base64)")
print(f"Sending request to {NANOROUTE_URL} ...")

resp = requests.post(NANOROUTE_URL, json=payload, timeout=120)
print(f"Status: {resp.status_code}")
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
