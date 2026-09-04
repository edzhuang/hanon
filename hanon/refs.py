"""The rated reference pool the judge compares candidates against.

The blog could not source human p5.brush paintings and had to rate model output. Piano
is the opposite: MAESTRO and GiantMIDI are enormous and world-class -- which is a trap,
not a gift. If every reference is Chopin the candidate loses every comparison, the
reward is a constant, and a constant has no gradient.

So the pool is graded on purpose. `love` is the target the policy is climbing toward,
`ok` is the level it can already beat sometimes, and both are needed: comparisons the
model always loses teach nothing, and comparisons it always wins teach nothing either.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

POOL = Path("refs/pool.json")
TIERS = ("love", "ok", "meh")


@dataclass
class Reference:
    id: str
    tier: str
    source: str      # "human" | "model"
    brief: str       # the prompt it came from, or a description for human pieces
    rendering: str   # the bar-by-bar text the judge reads

    @staticmethod
    def from_midi(path, tier: str, source: str, brief: str) -> "Reference":
        from hanon.render import describe

        if tier not in TIERS:
            raise ValueError(f"tier must be one of {TIERS}, got {tier!r}")
        return Reference(Path(path).stem, tier, source, brief, describe(path))


def load(path: Path = POOL) -> list[Reference]:
    if not Path(path).exists():
        return []
    return [Reference(**r) for r in json.loads(Path(path).read_text())]


def save(refs: list[Reference], path: Path = POOL) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(r) for r in refs], indent=2))


def sample(refs: list[Reference], n: int, rng: random.Random | None = None,
           mix: tuple[float, float] = (0.7, 0.3)) -> list[str]:
    """Draw `n` renderings, mostly from `love` with some `ok` for winnable comparisons."""
    rng = rng or random
    love = [r.rendering for r in refs if r.tier == "love"]
    ok = [r.rendering for r in refs if r.tier == "ok"]
    if not love and not ok:
        return []
    out = []
    for _ in range(n):
        pool = love if (rng.random() < mix[0] and love) or not ok else ok
        out.append(rng.choice(pool))
    return out
