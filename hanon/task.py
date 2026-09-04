"""Prompt construction and the end-to-end reward for one counterpoint task."""

from __future__ import annotations

from dataclasses import dataclass, field

from hanon.cantus import Cantus
from hanon.executor import run_sketch
from hanon.rewards.counterpoint import extract_voices, score
from hanon.rewards.spelling import Speller

PROMPT = """\
Write first-species counterpoint above this cantus firmus, in {mode} mode.

Cantus firmus (MIDI pitch numbers, one whole note each):
{pitches}

Write Python using `pretty_midi` that saves a file called `out.mid` containing BOTH
voices: the cantus firmus exactly as given, and your counterpoint above it, aligned
note-against-note with one note of counterpoint per note of cantus firmus.

First species requires: consonant vertical intervals only (no seconds, fourths above
the bass, sevenths or tritones); no parallel fifths or octaves; mostly stepwise motion
with leaps answered by step in the opposite direction; a single melodic high point;
opening on a perfect consonance; and closing on a unison or octave approached by
contrary stepwise motion.

Output only the Python code.
"""


@dataclass
class Result:
    reward: float
    compiled: bool
    cf_preserved: bool
    cp_score: float = 0.0
    violations: list = field(default_factory=list)
    error: str | None = None

    def summary(self) -> str:
        if not self.compiled:
            return f"reward {self.reward:.3f}  [did not compile: {self.error}]"
        if not self.cf_preserved:
            return f"reward {self.reward:.3f}  [cantus firmus altered]"
        return f"reward {self.reward:.3f}  counterpoint {self.cp_score:.3f}, {len(self.violations)} violations"


def build_prompt(c: Cantus) -> str:
    return PROMPT.format(mode=c.mode, pitches=", ".join(str(p) for p in c.pitches))


def evaluate(text: str, c: Cantus, timeout: float = 30.0) -> Result:
    """Grade one model response. Gates first, then the graded counterpoint score."""
    sk = run_sketch(text, timeout=timeout)
    if not sk.ok:
        return Result(reward=0.0, compiled=False, cf_preserved=False, error=sk.error)

    lower, upper = extract_voices(sk.midi_path)

    # The cantus firmus is given, not composed. If the model rewrote it to make its own
    # line fit, the counterpoint is meaningless -- so this gates the whole score rather
    # than costing a few points.
    cf_ok = lower == list(c.pitches)
    if not cf_ok:
        return Result(reward=0.10, compiled=True, cf_preserved=False,
                      error=f"expected cantus {list(c.pitches)}, found {lower}")

    s, vs = score(list(c.pitches), upper, Speller(c.mode, c.final))
    return Result(reward=0.10 + 0.90 * s, compiled=True, cf_preserved=True,
                  cp_score=s, violations=vs)
