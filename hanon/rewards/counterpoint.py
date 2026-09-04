"""A first-species counterpoint grader.

This is the whole reason for choosing counterpoint over aesthetics: the rules below
are mechanical, so this file is *provably correct or not*, and can be unit-tested
against textbook examples. No judge, no reference pool, no listening hours, and when
a training run flatlines the reward is not one of the suspects.

Rules follow Fux's first species (note-against-note). Severities are deliberately
graded rather than binary -- a parallel fifth and a repeated note are both "wrong",
but treating them as equally wrong throws away most of the gradient.
"""

from __future__ import annotations

from dataclasses import dataclass

# Simple (octave-reduced) intervals in semitones.
PERFECT = {0, 7}                      # unison/octave, fifth
CONSONANT = {0, 3, 4, 7, 8, 9}        # + m3 M3 m6 M6; P4 counts as dissonant above a bass
MELODIC_OK = {1, 2, 3, 4, 5, 7, 8, 9, 12}  # steps, consonant leaps, octave

# How much each violation costs. These are severities, not probabilities: a dissonance
# on a strong beat is a category error, a repeated note is a stylistic wrinkle.
WEIGHTS = {
    "dissonant_vertical": 1.0,
    "parallel_perfect": 1.0,
    "wrong_length": 1.5,
    "voice_crossing": 0.8,
    "bad_opening": 0.8,
    "bad_final": 0.8,
    "melodic_forbidden": 0.8,
    "cadence": 0.5,
    "direct_perfect": 0.4,
    "unrecovered_leap": 0.3,
    "range_too_wide": 0.3,
    "no_single_climax": 0.25,
    "too_few_steps": 0.25,
    "parallel_imperfect_run": 0.2,
    "repeated_note": 0.1,   # Fux himself repeats notes; discouraged, not an error
}

DECAY = 2.5  # score = exp(-penalty / DECAY); tuned so ~3 real errors lands near 0.3


@dataclass
class Violation:
    rule: str
    index: int
    detail: str

    @property
    def weight(self) -> float:
        return WEIGHTS.get(self.rule, 0.5)

    def __str__(self) -> str:
        return f"{self.rule}@{self.index}: {self.detail}"


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def check(cf: list[int], cp: list[int], cp_above: bool = True) -> list[Violation]:
    """Return every rule violation in a first-species counterpoint line.

    `cf` is the given cantus firmus, `cp` the counterpoint, both as MIDI numbers,
    aligned note-against-note.
    """
    v: list[Violation] = []
    if len(cp) != len(cf):
        v.append(Violation("wrong_length", 0, f"expected {len(cf)} notes, got {len(cp)}"))
        n = min(len(cf), len(cp))
        if n < 2:
            return v
        cf, cp = cf[:n], cp[:n]
    n = len(cf)

    upper, lower = (cp, cf) if cp_above else (cf, cp)
    verts = [abs(u - l) for u, l in zip(upper, lower)]
    simple = [x % 12 for x in verts]

    # --- vertical intervals ---
    for i, (s, raw) in enumerate(zip(simple, verts)):
        # A fourth above the bass is dissonant in two-voice writing; a twelfth is not.
        dissonant = s not in CONSONANT or (s == 5)
        if dissonant:
            v.append(Violation("dissonant_vertical", i, f"{raw} semitones"))

    for i in range(n):
        if cp_above and cp[i] < cf[i]:
            v.append(Violation("voice_crossing", i, "counterpoint below cantus"))
        elif not cp_above and cp[i] > cf[i]:
            v.append(Violation("voice_crossing", i, "counterpoint above cantus"))

    # --- motion between successive verticals ---
    run = 0
    for i in range(n - 1):
        du, dl = upper[i + 1] - upper[i], lower[i + 1] - lower[i]
        a, b = simple[i], simple[i + 1]

        if b in PERFECT and a == b and (du or dl):
            v.append(Violation("parallel_perfect", i, f"consecutive {'unison/octave' if b == 0 else 'fifth'}"))
        elif b in PERFECT and _sign(du) == _sign(dl) and _sign(du) != 0 and abs(du) > 2:
            # Similar motion into a perfect consonance with a leap in the upper voice.
            v.append(Violation("direct_perfect", i, "similar motion into a perfect consonance"))

        if a in (3, 4, 8, 9) and b in (3, 4, 8, 9):
            run += 1
            if run == 4:
                v.append(Violation("parallel_imperfect_run", i, "5+ parallel thirds/sixths"))
        else:
            run = 0

    # --- melodic shape of the counterpoint ---
    steps = 0
    for i in range(n - 1):
        d = cp[i + 1] - cp[i]
        a = abs(d)
        if a == 0:
            v.append(Violation("repeated_note", i, "note repeated"))
            continue
        if a <= 2:
            steps += 1
        if a not in MELODIC_OK:
            v.append(Violation("melodic_forbidden", i, f"melodic interval of {a} semitones"))
        if a >= 5 and i + 2 < n:
            nxt = cp[i + 2] - cp[i + 1]
            if _sign(nxt) == _sign(d) or abs(nxt) > 2:
                v.append(Violation("unrecovered_leap", i, f"leap of {a} not answered by step"))

    if n > 1 and steps / (n - 1) < 0.5:
        v.append(Violation("too_few_steps", 0, f"only {steps}/{n-1} stepwise"))

    span = max(cp) - min(cp)
    if span > 16:
        v.append(Violation("range_too_wide", 0, f"spans {span} semitones"))

    # One climax -- but a final note that happens to equal the peak is idiomatic, not a
    # second climax. Fux's own dorian solution closes on the same D5 it peaked on.
    peak = max(cp)
    occurrences = cp.count(peak) - (1 if cp[-1] == peak else 0)
    if occurrences > 1:
        v.append(Violation("no_single_climax", cp.index(peak), f"peak reached {occurrences}x before the close"))

    # --- opening and cadence ---
    if simple[0] not in PERFECT:
        v.append(Violation("bad_opening", 0, f"opens on {verts[0]} semitones, not a perfect consonance"))
    if simple[-1] != 0:
        v.append(Violation("bad_final", n - 1, f"ends on {verts[-1]} semitones, not unison/octave"))
    elif n >= 2:
        # The classic close: major sixth expanding to the octave by contrary step.
        du, dl = upper[-1] - upper[-2], lower[-1] - lower[-2]
        if simple[-2] not in (9, 3) or _sign(du) == _sign(dl) or abs(du) > 2 or abs(dl) > 2:
            v.append(Violation("cadence", n - 2, "penultimate is not a stepwise contrary approach from a sixth/third"))

    return v


def score(cf: list[int], cp: list[int], cp_above: bool = True) -> tuple[float, list[Violation]]:
    """Grade a counterpoint in [0, 1]. Smooth decay, so partial credit is real."""
    import math

    vs = check(cf, cp, cp_above)
    penalty = sum(x.weight for x in vs)
    return math.exp(-penalty / DECAY), vs


def extract_voices(midi_path) -> tuple[list[int], list[int]]:
    """Pull (lower, upper) note-against-note lines out of a MIDI file.

    Notes are grouped by onset; each onset contributes its lowest and highest pitch.
    """
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = [n for i in pm.instruments if not i.is_drum for n in i.notes]
    if not notes:
        return [], []
    onsets: dict[int, list[int]] = {}
    for nt in notes:
        onsets.setdefault(int(round(nt.start * 32)), []).append(nt.pitch)
    keys = sorted(onsets)
    return [min(onsets[k]) for k in keys], [max(onsets[k]) for k in keys]
