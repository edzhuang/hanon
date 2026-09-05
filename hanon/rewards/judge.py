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

You will see two pieces, A and B. Decide which is the better piece of music: melodic \
interest, harmonic movement, structure and contrast, development rather than \
repetition, and idiomatic use of the instrument. Judge the music only, not its style \
or genre; a fine chorale and a fine rag are equals. Ignore which is longer or denser, \
more notes is not better.

Start your answer with the single letter A or B, then one sentence of reasoning."""

TEMPLATE = """Two solo piano pieces.

--- PIECE A ---
{a}

--- PIECE B ---
{b}

Which is the better piece of music? Start with A or B."""

_ANSWER = re.compile(r"^\W*([AB])\b")


def _ask(a: str, b: str, model: str) -> str | None:
    """One call; returns 'A', 'B', or None if the judge gave no usable verdict."""
    text, _ = chat(model, TEMPLATE.format(a=a, b=b), system=SYSTEM, temperature=0.0,
                   max_tokens=120)
    m = _ANSWER.match(text.strip())
    return m.group(1) if m else None


def compare(brief: str, candidate: str, reference: str, model: str = JUDGE_MODEL,
            rng: random.Random | None = None) -> float | None:
    """Candidate's score against one reference: 1 win, 0 loss, 0.5 split, None unusable.

    Both orders are judged. Haiku picks whichever piece is shown second about 68% of
    the time (measured 2026-09-05 over 1,000 verdicts), which randomising the position
    turns into noise but does not remove; asking both orders and requiring agreement
    removes it. `brief` is accepted for interface stability but not shown: references
    are not brief-matched, so judging "fit to the brief" made a Bach chorale lose to
    a two-bar loop whenever the brief said music box.
    """
    first = _ask(candidate, reference, model)    # candidate is A
    second = _ask(reference, candidate, model)   # candidate is B
    if first is None and second is None:
        return None
    wins = [v for v in ((first == "A") if first else None, (second == "B") if second else None) if v is not None]
    return sum(wins) / len(wins)


def judge_score(brief: str, midi_path, references: list[str], n: int = 2,
                model: str = JUDGE_MODEL, rng: random.Random | None = None) -> float:
    """Mean score against `n` random references; each is 1 / 0.5 / 0 over both orders."""
    if not references:
        return 0.0
    rng = rng or random
    cand = describe(midi_path)
    picks = rng.sample(references, min(n, len(references)))
    verdicts = [v for v in (compare(brief, cand, r, model) for r in picks) if v is not None]
    return sum(verdicts) / len(verdicts) if verdicts else 0.0
