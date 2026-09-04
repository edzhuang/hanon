"""Deterministic music features and the anti-degeneracy score.

These metrics cannot tell good music from mediocre music -- that is the judge's job,
and pretending otherwise is how you end up with nine reward components that all
measure the same thing. What they *can* do, cheaply and without a model call, is catch
the pathologies a policy discovers early in training: one pitch hammered 200 times,
every velocity identical, no key at all, four bars looped thirty times, everything
crammed into a single octave.

So they are scored as **gates**, not as quality. Each returns 1.0 for "not pathological"
and falls off toward 0 as the piece degenerates. Passing them earns a little reward;
it does not earn a lot, because passing them is not the same as being good.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")


@dataclass
class Features:
    n_notes: int = 0
    duration_s: float = 0.0
    notes_per_sec: float = 0.0
    n_distinct_pitches: int = 0
    pitch_range: int = 0
    pitch_class_entropy: float = 0.0
    longest_repeat_run: int = 0
    ioi_entropy: float | None = None
    velocity_std: float = 0.0
    mean_polyphony: float = 0.0
    key: str = ""
    key_confidence: float | None = None
    self_similarity: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _entropy(counts) -> float:
    """Shannon entropy normalised to [0, 1] against a uniform distribution."""
    c = np.asarray([x for x in counts if x > 0], dtype=float)
    if c.size < 2:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / math.log(len(p)))


def _self_similarity(pitches, starts, window: float):
    """Mean off-diagonal cosine similarity of per-window pitch-class histograms.

    Near 0 means nothing ever recurs (rambling); near 1 means the same bar over and
    over (a loop). Real music sits in between, which is why the gate rewards the middle.
    """
    if window <= 0 or len(starts) < 8:
        return None  # not enough material to judge; see the None contract in degeneracy()
    idx = (np.asarray(starts) / window).astype(int)
    n = int(idx.max()) + 1
    if n < 3:
        return None
    hist = np.zeros((n, 12))
    for w, p in zip(idx, pitches):
        hist[w, int(p) % 12] += 1
    norm = np.linalg.norm(hist, axis=1, keepdims=True)
    live = (norm.ravel() > 0)
    if live.sum() < 3:
        return None
    h = hist[live] / norm[live]
    sim = h @ h.T
    off = sim[~np.eye(len(h), dtype=bool)]
    return float(off.mean())


def analyze(midi_path: str | Path, run_key_analysis: bool = True) -> Features:
    """Extract features from a MIDI file. Pure pretty_midi + numpy except for the key."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for inst in pm.instruments if not inst.is_drum for n in inst.notes]
    f = Features()
    if not notes:
        return f

    notes.sort(key=lambda n: (n.start, n.pitch))
    pitches = np.array([n.pitch for n in notes])
    starts = np.array([n.start for n in notes])
    ends = np.array([n.end for n in notes])
    vels = np.array([n.velocity for n in notes], dtype=float)

    f.n_notes = len(notes)
    f.duration_s = float(ends.max())
    f.notes_per_sec = f.n_notes / max(f.duration_s, 1e-6)
    f.n_distinct_pitches = int(len(np.unique(pitches)))
    f.pitch_range = int(pitches.max() - pitches.min())
    f.pitch_class_entropy = _entropy(np.bincount(pitches % 12, minlength=12))
    f.velocity_std = float(vels.std())

    run = best = 1
    for a, b in zip(pitches, pitches[1:]):
        run = run + 1 if a == b else 1
        best = max(best, run)
    f.longest_repeat_run = int(best)

    # Inter-onset intervals, quantised so that near-identical rhythms bucket together.
    onsets = np.unique(np.round(starts, 4))
    if len(onsets) > 2:
        iois = np.diff(onsets)
        iois = iois[iois > 1e-4]
        if iois.size:
            f.ioi_entropy = _entropy(np.bincount(np.round(iois / 0.05).astype(int)))

    # Mean sounding voices, sampled on a grid.
    grid = np.arange(0, f.duration_s, 0.05)
    if grid.size:
        f.mean_polyphony = float(
            ((starts[:, None] <= grid) & (ends[:, None] > grid)).sum(0).mean()
        )

    tempo = 100.0
    try:
        _, t = pm.get_tempo_changes()
        if len(t):
            tempo = float(t[0])
    except Exception:
        pass
    f.self_similarity = _self_similarity(pitches, starts, window=4 * 60.0 / max(tempo, 1))

    if run_key_analysis:
        try:
            from music21 import converter

            k = converter.parse(str(midi_path)).analyze("key")
            f.key = str(k)
            f.key_confidence = float(k.correlationCoefficient)
        except Exception:
            pass
    return f


def _band(x, lo, hi, soft=0.25):
    """1.0 inside [lo, hi], tapering linearly to 0 across a margin outside it."""
    if lo <= x <= hi:
        return 1.0
    span = max(hi - lo, 1e-6) * soft
    d = (lo - x) if x < lo else (x - hi)
    return float(max(0.0, 1.0 - d / span))


def degeneracy(f: Features) -> tuple[float, dict]:
    """Score how *un*-pathological a piece is, in [0, 1], with the per-gate breakdown.

    Combined with a soft-min rather than a mean: a piece that is perfect on six axes
    and catastrophic on the seventh is a catastrophe, and averaging would hide it.
    """
    maybe = {
        "density": (f.notes_per_sec, 1.0, 12.0),
        "register": (f.pitch_range, 14, 60),
        "pitch_variety": (f.n_distinct_pitches, 8, 60),
        "tonal": (f.key_confidence, 0.55, 1.0),
        "rhythm": (f.ioi_entropy, 0.25, 0.85),
        "dynamics": (f.velocity_std, 4.0, 40.0),
        "structure": (f.self_similarity, 0.45, 0.85),
    }
    # A None feature means "could not be measured" (too short, analysis failed), which
    # is emphatically not the same as "measured, and terrible". Scoring it 0 would let a
    # ten-second sketch be punished for brevity by every gate that needs length.
    g = {k: _band(v, lo, hi) for k, (v, lo, hi) in maybe.items() if v is not None}
    g["not_stuck"] = (
        1.0 if f.longest_repeat_run <= 4 else max(0.0, 1 - (f.longest_repeat_run - 4) / 12)
    )
    vals = np.array(list(g.values()))
    score = float(vals.min() * 0.5 + vals.mean() * 0.5)
    return score, g
