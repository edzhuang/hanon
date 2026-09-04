"""The hanon DSL: eight verbs, no more.

Deliberately tiny. A small surface the model can hold entirely in its head beats a
large one it has to be told about -- long API docs in a system prompt make models
invent methods that do not exist. Everything here is dependency-free so the sandbox
starts instantly.
"""

from __future__ import annotations

import struct
from contextlib import contextmanager

__all__ = ["note", "chord", "phrase", "arp", "pedal", "voice", "repeat", "render"]

_STEPS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _pitch(p):
    """Accept 60, 'C4', 'F#3', 'Bb5'. Middle C is C4 = 60."""
    if isinstance(p, int):
        if not 0 <= p <= 127:
            raise ValueError(f"pitch {p} outside MIDI range 0-127")
        return p
    s = str(p).strip()
    if not s or s[0].upper() not in _STEPS:
        raise ValueError(f"bad pitch {p!r}: expected like 'C4', 'F#3', 'Bb5' or 0-127")
    v, i = _STEPS[s[0].upper()], 1
    while i < len(s) and s[i] in "#b":
        v += 1 if s[i] == "#" else -1
        i += 1
    try:
        v += (int(s[i:]) + 1) * 12
    except ValueError:
        raise ValueError(f"bad pitch {p!r}: missing or non-numeric octave") from None
    if not 0 <= v <= 127:
        raise ValueError(f"pitch {p!r} resolves to {v}, outside MIDI range 0-127")
    return v


def _seq(x):
    return list(x) if isinstance(x, (list, tuple)) else str(x).split()


class _Score:
    def __init__(self):
        self.notes = []   # (track, start_beat, pitch, dur_beats, velocity)
        self.pedals = []  # (start_beat, dur_beats)
        self.tracks = ["main"]
        self.cur = "main"


_score = _Score()


def _reset():
    global _score
    _score = _Score()
    return _score


def note(pitch, at, dur=1.0, vel=80):
    """One note. `at` and `dur` are in beats."""
    at, dur = float(at), float(dur)
    if dur <= 0:
        raise ValueError(f"dur must be positive, got {dur}")
    if at < 0:
        raise ValueError(f"at must be non-negative, got {at}")
    _score.notes.append((_score.cur, at, _pitch(pitch), dur, max(1, min(127, int(vel)))))


def chord(pitches, at, dur=1.0, vel=80):
    """Simultaneous notes: chord("C4 E4 G4", at=0, dur=2)."""
    for p in _seq(pitches):
        note(p, at, dur, vel)


def phrase(pitches, at=0.0, dur=0.5, vel=80, gap=None):
    """A melodic line. `dur` and `vel` may be scalars or per-note lists.

    Returns the beat where the phrase ends, so phrases chain:
        t = phrase("C4 D4 E4", at=0)
        t = phrase("F4 G4", at=t)
    """
    ps = _seq(pitches)
    durs = dur if isinstance(dur, (list, tuple)) else [dur] * len(ps)
    vels = vel if isinstance(vel, (list, tuple)) else [vel] * len(ps)
    if len(durs) != len(ps) or len(vels) != len(ps):
        raise ValueError(
            f"phrase got {len(ps)} pitches but {len(durs)} durs and {len(vels)} vels"
        )
    t = float(at)
    for p, d, v in zip(ps, durs, vels):
        note(p, t, d, v)
        t += float(gap) if gap is not None else float(d)
    return t


def arp(pitches, at, dur, step=0.25, vel=70, updown=False):
    """Arpeggiate `pitches` across `dur` beats, one attack every `step` beats."""
    ps = [_pitch(p) for p in _seq(pitches)]
    if not ps:
        raise ValueError("arp needs at least one pitch")
    if updown and len(ps) > 2:
        ps = ps + ps[-2:0:-1]
    step, t, end, i = float(step), float(at), float(at) + float(dur), 0
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    while t < end - 1e-9:
        note(ps[i % len(ps)], t, min(step, end - t), vel)
        t += step
        i += 1
    return end


def pedal(at, dur):
    """Sustain pedal down for `dur` beats."""
    if float(dur) <= 0:
        raise ValueError(f"pedal dur must be positive, got {dur}")
    _score.pedals.append((float(at), float(dur)))


@contextmanager
def voice(name):
    """Write into a named track: with voice("left_hand"): ..."""
    prev = _score.cur
    if name not in _score.tracks:
        _score.tracks.append(name)
    _score.cur = name
    try:
        yield
    finally:
        _score.cur = prev


def repeat(fn, times, every, transpose=0, at=0.0):
    """Call fn(t, i) `times` times, `every` beats apart, shifting pitch by
    `transpose` semitones each iteration. This is how structure gets built."""
    if times < 1:
        raise ValueError(f"times must be >= 1, got {times}")
    for i in range(int(times)):
        mark = len(_score.notes)
        fn(float(at) + i * float(every), i)
        if transpose:
            shift = int(transpose) * i
            for j in range(mark, len(_score.notes)):
                tr, s, p, d, v = _score.notes[j]
                _score.notes[j] = (tr, s, max(0, min(127, p + shift)), d, v)
    return float(at) + times * float(every)


# --- MIDI serialisation ------------------------------------------------------

_TPB = 480  # ticks per beat


def _vlq(n):
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.append((n & 0x7F) | 0x80)
        n >>= 7
    return bytes(reversed(out))


def _track(events):
    """events: list of (tick, priority, status, d1, d2)."""
    body, prev = bytearray(), 0
    for tick, _, status, d1, d2 in sorted(events, key=lambda e: (e[0], e[1])):
        body += _vlq(tick - prev) + bytes([status, d1, d2])
        prev = tick
    body += b"\x00\xff\x2f\x00"
    return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


def render(path="out.mid", tempo=100):
    """Write the score to a MIDI file. Call this once, at the end."""
    if not _score.notes:
        raise ValueError("nothing to render: the score has no notes")
    if not 20 <= float(tempo) <= 300:
        raise ValueError(f"tempo {tempo} outside sane range 20-300 bpm")

    upb = int(60_000_000 / float(tempo))
    head = bytearray(b"\x00\xff\x51\x03" + struct.pack(">I", upb)[1:])
    head += b"\x00\xff\x2f\x00"
    chunks = [
        b"MThd" + struct.pack(">IHHH", 6, 1, len(_score.tracks) + 1, _TPB),
        b"MTrk" + struct.pack(">I", len(head)) + bytes(head),
    ]

    for ti, tname in enumerate(_score.tracks):
        ch = min(ti, 15)
        ev = [(0, 0, 0xC0 | ch, 0, 0)]  # program 0: acoustic grand
        for tr, start, p, d, v in _score.notes:
            if tr != tname:
                continue
            on = int(round(start * _TPB))
            off = max(on + 1, int(round((start + d) * _TPB)))
            ev.append((on, 1, 0x90 | ch, p, v))
            ev.append((off, 0, 0x80 | ch, p, 0))
        if ti == 0:
            for start, d in _score.pedals:
                ev.append((int(round(start * _TPB)), 0, 0xB0 | ch, 64, 127))
                ev.append((int(round((start + d) * _TPB)), 0, 0xB0 | ch, 64, 0))
        chunks.append(_track(ev))

    data = b"".join(chunks)
    with open(path, "wb") as f:
        f.write(data)
    return path
