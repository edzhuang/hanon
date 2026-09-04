"""Cantus firmi to write counterpoint against.

Each is a modal melody in the traditional shape: stepwise, one climax, ending
scale-degree 2 -> 1. Pitches are MIDI numbers; `final` is the modal final.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cantus:
    name: str
    mode: str
    final: int
    pitches: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.pitches)


CANTUS_FIRMI: list[Cantus] = [
    Cantus("fux_dorian", "dorian", 62, (62, 65, 64, 62, 67, 65, 69, 67, 65, 64, 62)),
    Cantus("fux_ionian", "ionian", 60, (60, 62, 64, 65, 64, 62, 67, 65, 64, 62, 60)),
    Cantus("fux_aeolian", "aeolian", 57, (57, 60, 59, 57, 62, 60, 64, 62, 60, 59, 57)),
    Cantus("fux_phrygian", "phrygian", 64, (64, 65, 67, 65, 64, 69, 67, 65, 64)),
    Cantus("fux_lydian", "lydian", 65, (65, 69, 67, 65, 72, 71, 69, 67, 65)),
    Cantus("fux_mixolydian", "mixolydian", 67, (67, 69, 71, 69, 67, 72, 71, 69, 67)),
    Cantus("short_dorian", "dorian", 62, (62, 64, 65, 67, 65, 64, 62)),
    Cantus("short_ionian", "ionian", 60, (60, 64, 62, 65, 64, 62, 60)),
]

BY_NAME = {c.name: c for c in CANTUS_FIRMI}
