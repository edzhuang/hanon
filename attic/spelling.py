"""Recover note spelling from MIDI, so intervals have real quality.

MIDI records key numbers, not note names: pitch 68 is G# or Ab and the file cannot
say which. But interval *quality* is what counterpoint rules are actually about --
F->G# (augmented second, forbidden) and F->Ab (minor third, ordinary) are both three
semitones. Grading on semitones alone silently permits the single most famous
forbidden melodic interval in modal counterpoint.

The mode and final are given, so the diatonic collection is known. Anything outside it
is musica ficta -- in practice a raised leading tone -- which is enough to assign every
note a letter and therefore every interval a quality.
"""

from __future__ import annotations

NATURAL_PC = (0, 2, 4, 5, 7, 9, 11)  # C D E F G A B
LETTERS = "CDEFGAB"

MODES = {
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
}

# Semitones spanned by each generic interval when perfect or major.
_BASE = {1: 0, 2: 2, 3: 4, 4: 5, 5: 7, 6: 9, 7: 11, 8: 12}
_PERFECT = {1, 4, 5, 8}


class Speller:
    """Maps MIDI pitches to (letter index, accidental) for one mode."""

    def __init__(self, mode: str, final: int):
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {sorted(MODES)}")
        try:
            root = NATURAL_PC.index(final % 12)
        except ValueError:
            raise ValueError(f"final {final} is not a natural note; modes here use naturals") from None

        self.table: dict[int, tuple[int, int]] = {}
        degrees = []
        for i, step in enumerate(MODES[mode]):
            letter = (root + i) % 7
            pc = (final + step) % 12
            acc = (pc - NATURAL_PC[letter] + 6) % 12 - 6  # nearest signed accidental
            self.table[pc] = (letter, acc)
            degrees.append((letter, acc, pc))

        # Chromatic notes are ficta. Prefer raising the diatonic note below (the leading
        # tone case) over flattening the one above -- G# in A aeolian, never Ab.
        for pc in range(12):
            if pc in self.table:
                continue
            for letter, acc, dpc in degrees:
                if (dpc + 1) % 12 == pc:
                    self.table[pc] = (letter, acc + 1)
                    break
            else:
                for letter, acc, dpc in degrees:
                    if (dpc - 1) % 12 == pc:
                        self.table[pc] = (letter, acc - 1)
                        break

    def position(self, pitch: int) -> int:
        """Absolute diatonic staff position, so letter distance is subtraction."""
        letter, acc = self.table[pitch % 12]
        octave = round((pitch - acc - NATURAL_PC[letter]) / 12)
        return 7 * octave + letter

    def name(self, pitch: int) -> str:
        letter, acc = self.table[pitch % 12]
        mark = "#" * acc if acc > 0 else "b" * -acc
        return f"{LETTERS[letter]}{mark}"

    def interval(self, a: int, b: int) -> tuple[int, str]:
        """(generic, quality) between two pitches, e.g. (2, 'A') for an augmented second.

        Quality is one of P, M, m, A, d. Generic counts inclusively: 1 unison, 2 second.
        """
        generic = abs(self.position(b) - self.position(a)) + 1
        semis = abs(b - a)
        octaves, g = divmod(generic - 1, 7)
        g += 1
        if g == 1 and octaves:  # an octave and its compounds
            g, octaves = 8, octaves - 1
        diff = semis - (_BASE[g] + 12 * octaves)
        if g in _PERFECT:
            q = {0: "P", 1: "A", -1: "d"}.get(diff, "A" if diff > 0 else "d")
        else:
            q = {0: "M", -1: "m", 1: "A", -2: "d"}.get(diff, "A" if diff > 0 else "d")
        return g if not octaves else g + 7 * octaves, q
