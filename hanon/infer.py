"""One place that talks to Prime Inference, so the judge and the sampler agree."""

from __future__ import annotations

import json
import os
import subprocess


def chat(model: str, prompt: str, system: str = "", temperature: float = 0.7,
         max_tokens: int = 4096, timeout: float = 600) -> tuple[str, dict]:
    cmd = ["prime", "inference", "chat", model, prompt,
           "-t", str(temperature), "--max-tokens", str(max_tokens), "-o", "json", "--plain"]
    if system:
        cmd += ["-s", system]
    p = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "PRIME_DISABLE_VERSION_CHECK": "1"},
    )
    if p.returncode != 0:
        raise RuntimeError(f"inference failed: {(p.stderr or p.stdout)[-400:]}")
    i = p.stdout.find("{")
    if i < 0:
        raise RuntimeError(f"no JSON in reply: {p.stdout[:300]}")
    d = json.loads(p.stdout[i:])
    return d["choices"][0]["message"]["content"], d.get("usage", {})
