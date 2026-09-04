"""Render a MIDI file as compact text for a judge to read.

The judge reads the score, not the audio. Every piece here is solo piano through one
fixed soundfont, so timbre is constant and all the variance lives in the notes --
which makes a symbolic rendering nearly lossless and about fifty times cheaper per
comparison than shipping audio to a multimodal model. Audio judging stays available
for spot-checking; it has no business in the inner loop.
"""

from __future__ import annotations

from pathlib import Path

NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _name(p: int) -> str:
    return f"{NAMES[p % 12]}{p // 12 - 1}"


def _dyn(v: float) -> str:
    return ("ppp", "pp", "p", "mp", "mf", "f", "ff", "fff")[min(7, max(0, int(v) // 16))]


def describe(midi_path: str | Path, max_bars: int = 48) -> str:
    """A bar-by-bar summary: bass, harmony, melodic contour, dynamics, pedal."""
    import pretty_midi

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = sorted(
        (n for i in pm.instruments if not i.is_drum for n in i.notes),
        key=lambda n: (n.start, n.pitch),
    )
    if not notes:
        return "(empty)"

    tempo = 100.0
    try:
        _, t = pm.get_tempo_changes()
        if len(t):
            tempo = float(t[0])
    except Exception:
        pass
    beats = 4.0
    if pm.time_signature_changes:
        ts = pm.time_signature_changes[0]
        beats = ts.numerator * 4.0 / ts.denominator
    bar_s = beats * 60.0 / tempo

    pedals = sorted(
        c.time for i in pm.instruments for c in i.control_changes
        if c.number == 64 and c.value >= 64
    )

    end = notes[-1].end
    n_bars = min(max_bars, int(end / bar_s) + 1)
    lines = [
        f"tempo {tempo:.0f}bpm, {end:.0f}s, {len(notes)} notes, "
        f"{n_bars} bars of {beats:.0f} beats, pedal used {len(pedals)}x",
        "",
    ]

    for b in range(n_bars):
        lo, hi = b * bar_s, (b + 1) * bar_s
        bar = [n for n in notes if lo <= n.start < hi]
        if not bar:
            lines.append(f"{b+1:3} | (rest)")
            continue
        bass = _name(min(n.pitch for n in bar))
        pcs = sorted({n.pitch % 12 for n in bar})
        harmony = " ".join(NAMES[p] for p in pcs)
        top = [n for n in bar if n.pitch >= max(x.pitch for x in bar) - 4]
        melody = " ".join(_name(n.pitch) for n in sorted(top, key=lambda n: n.start)[:8])
        vel = sum(n.velocity for n in bar) / len(bar)
        ped = " ped" if any(lo <= t < hi for t in pedals) else ""
        lines.append(
            f"{b+1:3} | bass {bass:4} | pcs {harmony:22} | top {melody:34} "
            f"| {_dyn(vel)} | {len(bar):2}n{ped}"
        )

    if int(end / bar_s) + 1 > max_bars:
        lines.append(f"... ({int(end / bar_s) + 1 - max_bars} more bars)")
    return "\n".join(lines)
