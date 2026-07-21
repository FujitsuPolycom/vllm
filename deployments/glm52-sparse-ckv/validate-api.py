#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import os
import sys
import urllib.request

base_url = os.getenv("BASE_URL", "http://127.0.0.1:5802").rstrip("/")


def request_json(path: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        return json.load(response)


models = request_json("/v1/models").get("data", [])
if not models:
    raise SystemExit("server returned no models")
model = models[0]["id"]
payload = {
    "model": model,
    "messages": [
        {
            "role": "user",
            "content": "Return exactly the text: sparse-ckv-ready",
        }
    ],
    "temperature": 0,
    "max_tokens": 32,
    "reasoning_effort": "none",
}
outputs = [
    request_json("/v1/chat/completions", payload)["choices"][0]["message"]["content"]
    for _ in range(2)
]
if outputs[0] != outputs[1]:
    print(json.dumps(outputs, indent=2), file=sys.stderr)
    raise SystemExit("temperature-zero smoke outputs differ")
expected = "sparse-ckv-ready"
if outputs[0].strip() != expected:
    print(json.dumps(outputs, indent=2), file=sys.stderr)
    raise SystemExit(f"expected {expected!r}")
print(json.dumps({"model": model, "output": outputs[0]}, indent=2))
