"""Pairwise aesthetic judging against a pool of rated reference pieces.

The blog's central reward finding was that absolute scoring plateaus and pairwise
comparison does not: nine graders rating a piece 1-10 turned out to be measuring one
thing at 0.85-0.95 correlation, while "which of these two is better" kept producing
signal. So this asks only that question, and never for a number.

Two details that matter more than they look:

- **Position is randomised.** Judges prefer whichever candidate is shown second often
  enough to swamp a 0.6-weighted reward component, and the bias is invisible in
  aggregate scores.
- **The reference pool must be reachable.** If every reference is a masterpiece the
  candidate loses every comparison, the reward is a constant, and a constant has no
  gradient. See refs.py for how the tiers are built.
"""

from __future__ import annotations

import random
import re

from hanon.infer import chat
from hanon.render import describe

JUDGE_MODEL = "anthropic/claude-haiku-4.5"

SYSTEM = """You judge solo piano pieces presented as bar-by-bar summaries.

You will see a brief and two pieces, A and B. Decide which is the better piece of \
music for that brief. Weigh melodic interest, harmonic movement, structure and \
contrast, and use of the instrument. Ignore which is longer or denser -- more notes \
is not better.

Answer with exactly one character: A or B."""

TEMPLATE = """Brief: {brief}

--- PIECE A ---
{a}

--- PIECE B ---
{b}

Which is the better piece for that brief? Answer A or B."""


def compare(brief: str, candidate: str, reference: str, model: str = JUDGE_MODEL,
            rng: random.Random | None = None) -> bool | None:
    """True if the candidate wins. None if the judge gave no usable verdict."""
    rng = rng or random
    flip = rng.random() < 0.5  # candidate is shown second half the time
    a, b = (reference, candidate) if flip else (candidate, reference)
    text, _ = chat(model, TEMPLATE.format(brief=brief, a=a, b=b),
                   system=SYSTEM, temperature=0.0, max_tokens=8)
    m = re.search(r"\b([AB])\b", text.strip().upper())
    if not m:
        return None
    return (m.group(1) == "B") if flip else (m.group(1) == "A")


def judge_score(brief: str, midi_path, references: list[str], n: int = 2,
                model: str = JUDGE_MODEL, rng: random.Random | None = None) -> float:
    """Fraction of comparisons the candidate wins against `n` random references."""
    if not references:
        return 0.0
    rng = rng or random
    cand = describe(midi_path)
    picks = rng.sample(references, min(n, len(references)))
    verdicts = [v for v in (compare(brief, cand, r, model, rng) for r in picks) if v is not None]
    return sum(verdicts) / len(verdicts) if verdicts else 0.0
