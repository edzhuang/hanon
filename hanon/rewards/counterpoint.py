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

# Intervals are classified by (generic, quality), not semitones -- see spelling.py for
# why. Generic is octave-reduced here, so a twelfth classifies as a fifth.
CONSONANT = {(1, "P"), (5, "P"), (3, "m"), (3, "M"), (6, "m"), (6, "M")}
PERFECT = {(1, "P"), (5, "P")}

# Fux's melodic allowlist: steps, consonant leaps, the ascending minor sixth, the
# octave. Every augmented or diminished interval is excluded by construction, which is
# the whole point of grading on quality.
MELODIC_OK = {(2, "m"), (2, "M"), (3, "m"), (3, "M"), (4, "P"), (5, "P"), (6, "m"), (8, "P")}

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
    "repeated_note": 0.3,          # only charged beyond the second; Fux uses two
    "voice_spacing": 0.3,          # voices drifting past a tenth
    "similar_motion_excess": 0.5,  # Fux: prefer contrary motion
    "consecutive_leaps": 0.4,
    "long_monotonic_run": 0.3,
    "pitch_overuse": 0.4,      # dwelling on one note instead of shaping a line
    "low_variety": 0.5,
}

REPEATS_ALLOWED = 2   # Fux's dorian solution repeats twice; that is idiomatic, not error
STEP_TARGET = 0.75    # Fux sits at 0.90; below this the line stops being a melody
MAX_SPACING = 16      # a tenth
MIN_CONTRARY = 0.4    # fraction of motions that must be contrary or oblique
MAX_SAME_PITCH = 3    # Fux's most-used note appears three times
MIN_VARIETY = 0.5     # distinct pitches as a fraction of length; Fux is 0.55

DECAY = 2.5  # score = exp(-penalty / DECAY); tuned so ~3 real errors lands near 0.3


@dataclass
class Violation:
    rule: str
    index: int
    detail: str
    scale: float = 1.0  # lets a rule charge proportionally to how badly it is missed

    @property
    def weight(self) -> float:
        return WEIGHTS.get(self.rule, 0.5) * self.scale

    def __str__(self) -> str:
        return f"{self.rule}@{self.index}: {self.detail}"


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _reduce(generic: int) -> int:
    """Octave-reduce a generic interval: a twelfth becomes a fifth, an octave a unison."""
    return (generic - 1) % 7 + 1


def check(cf: list[int], cp: list[int], speller, cp_above: bool = True) -> list[Violation]:
    """Return every rule violation in a first-species counterpoint line.

    `cf` is the given cantus firmus, `cp` the counterpoint, both as MIDI numbers,
    aligned note-against-note. `speller` supplies note names for the mode, without
    which augmented seconds are indistinguishable from minor thirds.
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
    quals = [speller.interval(l, u) for u, l in zip(upper, lower)]
    dists = [abs(u - l) for u, l in zip(upper, lower)]
    simple = [(_reduce(g), q) for g, q in quals]

    # --- vertical intervals ---
    for i, ((g, q), (rg, rq)) in enumerate(zip(quals, simple)):
        # A fourth above the bass is dissonant in two-voice writing; so is every
        # augmented and diminished interval, which semitone counting would miss.
        if (rg, rq) not in CONSONANT:
            v.append(Violation("dissonant_vertical", i, f"{q}{g}"))

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
            v.append(Violation("parallel_perfect", i, f"consecutive {'unison/octave' if b[0] == 1 else 'fifth'}"))
        elif b in PERFECT and _sign(du) == _sign(dl) and _sign(du) != 0 and abs(du) > 2:
            # Similar motion into a perfect consonance with a leap in the upper voice.
            v.append(Violation("direct_perfect", i, "similar motion into a perfect consonance"))

        if a[0] in (3, 6) and b[0] in (3, 6):
            run += 1
            if run == 4:
                v.append(Violation("parallel_imperfect_run", i, "5+ parallel thirds/sixths"))
        else:
            run = 0

    # --- melodic shape of the counterpoint ---
    steps = repeats = 0
    prev_leap = 0
    run_len, run_dir = 1, 0
    for i in range(n - 1):
        d = cp[i + 1] - cp[i]
        a = abs(d)

        if _sign(d) == run_dir and d:
            run_len += 1
            if run_len == 5:
                v.append(Violation("long_monotonic_run", i, "5+ notes moving one way"))
        else:
            run_len, run_dir = 1, _sign(d)

        if a >= 3:
            if prev_leap and _sign(d) == _sign(prev_leap):
                v.append(Violation("consecutive_leaps", i, "two leaps in the same direction"))
            prev_leap = d
        else:
            prev_leap = 0

        if a == 0:
            repeats += 1
            continue
        g, q = speller.interval(cp[i], cp[i + 1])
        if a <= 2:
            steps += 1
        if (g, q) not in MELODIC_OK:
            v.append(Violation("melodic_forbidden", i, f"melodic {q}{g} ({speller.name(cp[i])}->{speller.name(cp[i+1])})"))
        if a >= 5 and i + 2 < n:
            nxt = cp[i + 2] - cp[i + 1]
            if _sign(nxt) == _sign(d) or abs(nxt) > 2:
                v.append(Violation("unrecovered_leap", i, f"leap of {a} not answered by step"))

    if repeats > REPEATS_ALLOWED:
        v.append(Violation("repeated_note", 0, f"{repeats} repeated notes",
                           scale=repeats - REPEATS_ALLOWED))

    moves = (n - 1) - repeats
    ratio = steps / moves if moves > 0 else 0.0
    if ratio < STEP_TARGET:
        # Charged in proportion to the shortfall, so the gradient points at melody.
        v.append(Violation("too_few_steps", 0, f"{steps}/{moves} moving notes stepwise ({ratio:.0%})",
                           scale=(STEP_TARGET - ratio) / STEP_TARGET * 3))

    # A line that keeps returning to the same pitch is noodling, not shaping a melody.
    for pitch in set(cp):
        extra = cp.count(pitch) - MAX_SAME_PITCH
        if extra > 0:
            v.append(Violation("pitch_overuse", cp.index(pitch),
                               f"{speller.name(pitch)} used {cp.count(pitch)}x", scale=extra))
    variety = len(set(cp)) / n
    if variety < MIN_VARIETY:
        v.append(Violation("low_variety", 0, f"{len(set(cp))} distinct pitches in {n} notes",
                           scale=(MIN_VARIETY - variety) / MIN_VARIETY * 2))

    # Voices that drift beyond a tenth stop sounding like two parts of one texture.
    for i, dist in enumerate(dists):
        if dist > MAX_SPACING:
            v.append(Violation("voice_spacing", i, f"{dist} semitones apart",
                               scale=min(3.0, (dist - MAX_SPACING) / 4)))

    if n > 2:
        contrary = sum(
            1 for i in range(n - 1)
            if _sign(upper[i + 1] - upper[i]) != _sign(lower[i + 1] - lower[i])
        )
        frac = contrary / (n - 1)
        if frac < MIN_CONTRARY:
            v.append(Violation("similar_motion_excess", 0, f"only {frac:.0%} contrary/oblique",
                               scale=(MIN_CONTRARY - frac) / MIN_CONTRARY * 2))

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
        v.append(Violation("bad_opening", 0, f"opens on {quals[0][1]}{quals[0][0]}, not a perfect consonance"))
    if simple[-1] != (1, "P"):
        v.append(Violation("bad_final", n - 1, f"ends on {quals[-1][1]}{quals[-1][0]}, not unison/octave"))
    elif n >= 2:
        # The classic close: major sixth expanding to the octave by contrary step.
        du, dl = upper[-1] - upper[-2], lower[-1] - lower[-2]
        if simple[-2][0] not in (6, 3) or _sign(du) == _sign(dl) or abs(du) > 2 or abs(dl) > 2:
            v.append(Violation("cadence", n - 2, "penultimate is not a stepwise contrary approach from a sixth/third"))

    return v


def score(cf: list[int], cp: list[int], speller, cp_above: bool = True) -> tuple[float, list[Violation]]:
    """Grade a counterpoint in [0, 1]. Smooth decay, so partial credit is real."""
    import math

    vs = check(cf, cp, speller, cp_above)
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
