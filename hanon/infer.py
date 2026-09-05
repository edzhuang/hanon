"""One place that talks to Prime Inference, so the judge and the sampler agree."""

from __future__ import annotations

import json
import os
import random
import subprocess
import time


def chat(model: str, prompt: str, system: str = "", temperature: float = 0.7,
         max_tokens: int = 4096, timeout: float = 600, retries: int = 5) -> tuple[str, dict]:
    cmd = ["prime", "inference", "chat", model, prompt,
           "-t", str(temperature), "--max-tokens", str(max_tokens), "-o", "json", "--plain"]
    if system:
        cmd += ["-s", system]
    env = {**os.environ, "PRIME_DISABLE_VERSION_CHECK": "1"}
    for attempt in range(retries + 1):
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        if p.returncode == 0:
            break
        err = (p.stderr or p.stdout)[-400:]
        transient = any(t in err for t in ("429", "rate_limit", "Server is busy", "Waiting for", "502", "503", "timeout"))
        if not transient or attempt == retries:
            raise RuntimeError(f"inference failed: {err}")
        time.sleep(min(60, 2 ** attempt) + random.random())  # backoff with jitter
    i = p.stdout.find("{")
    if i < 0:
        raise RuntimeError(f"no JSON in reply: {p.stdout[:300]}")
    d = json.loads(p.stdout[i:])
    return d["choices"][0]["message"]["content"], d.get("usage", {})
