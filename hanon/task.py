"""Prompt selection and the end-to-end aesthetic reward for one piece.

Reward shape is the blog's *post-mortem* design, not the one they started with. Their
first attempt had nine components -- compile, library use, length, human preference,
prompt adherence and four quality judges -- and plateaued at 0.65 with repetitive
output, because the quality judges correlated 0.85-0.95 and were effectively one
component wearing four hats, while length saturated immediately.

Starting from where they finished:

    0.05  compiles                (binary gate)
    0.05  length in range         (binary gate)
    0.30  not degenerate          (local, free, absolute)
    0.60  pairwise vs references  (the only component that scales with taste)

Everything else in metrics.py is logged and rewards nothing. Observability is free;
reward components are not.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from hanon.executor import run_sketch
from hanon.prompt import PROMPTS, SYSTEM
from hanon.refs import Reference, sample
from hanon.rewards import analyze, degeneracy
from hanon.rewards.judge import JUDGE_MODEL, judge_score

W_COMPILE, W_LENGTH, W_QUALITY, W_JUDGE = 0.05, 0.05, 0.30, 0.60
MIN_TOKENS, MAX_TOKENS = 150, 2500   # ~chars/4; the blog's models collapsed to <2k
MIN_SECONDS, MAX_SECONDS = 20, 120


@dataclass
class Result:
    reward: float
    compiled: bool
    length_ok: bool = False
    quality: float = 0.0
    win_rate: float = 0.0
    error: str | None = None
    features: dict = field(default_factory=dict)
    gates: dict = field(default_factory=dict)

    def summary(self) -> str:
        if not self.compiled:
            head = (self.error or "").splitlines()
            return f"reward {self.reward:.3f}  [failed: {head[-1][:70] if head else '?'}]"
        return (f"reward {self.reward:.3f}  quality {self.quality:.2f}  "
                f"wins {self.win_rate:.0%}  {'len ok' if self.length_ok else 'LEN BAD'}")


def evaluate(text: str, brief: str, references: list[Reference] | None = None,
             n_comparisons: int = 2, judge_model: str = JUDGE_MODEL,
             rng: random.Random | None = None, timeout: float = 30.0) -> Result:
    """Score one model response. Gates are cheap and run first; the judge runs last."""
    sk = run_sketch(text, timeout=timeout)
    if not sk.ok:
        return Result(reward=0.0, compiled=False, error=sk.error)

    reward = W_COMPILE
    length_ok = MIN_TOKENS <= len(sk.code) / 4 <= MAX_TOKENS and \
        MIN_SECONDS <= sk.duration_s <= MAX_SECONDS
    reward += W_LENGTH * length_ok

    f = analyze(sk.midi_path)
    quality, gates = degeneracy(f)
    reward += W_QUALITY * quality

    # The judge is the only component that costs money, so it runs on a piece that has
    # already cleared the free gates.
    win = 0.0
    if references:
        win = judge_score(brief, sk.midi_path, sample(references, n_comparisons, rng),
                          n=n_comparisons, model=judge_model, rng=rng)
        reward += W_JUDGE * win

    return Result(reward=reward, compiled=True, length_ok=length_ok, quality=quality,
                  win_rate=win, features=f.as_dict(), gates=gates)


def briefs(n: int | None = None, rng: random.Random | None = None) -> list[str]:
    if n is None:
        return list(PROMPTS)
    return (rng or random).choices(PROMPTS, k=n)
