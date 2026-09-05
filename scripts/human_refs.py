"""Build the `love` tier of the reference pool from public-domain scores in music21's corpus.

Scores, not performances: no human timing or velocity curves, so the judge can't tell
human from model by format alone. Each piece is cut to the first ~45s at a bar
boundary, written as MIDI through pretty_midi, and rendered with the same describe()
the candidates get.

Usage: uv run scripts/human_refs.py   -> refs/human/*.mid and refs/pool.json (human entries)
"""
import sys
from pathlib import Path

sys.path.insert(0, ".")
from music21 import corpus, tempo as m21tempo, note as m21note, chord as m21chord
import pretty_midi
from hanon.refs import Reference, load, save

OUT = Path("refs/human")
MAX_S = 45.0

PIECES = [  # (corpus path, brief-like description)
    ("bach/bwv846.mxl", "a flowing prelude in C major, arpeggiated throughout"),
    ("chopin/mazurka06-2.krn", "a wistful mazurka in a minor key"),
    ("joplin/maple_leaf_rag.mxl", "a bright, syncopated ragtime piece"),
    ("schumann_clara/polonaise_op1n1.mxl", "a stately polonaise for piano"),
    ("schumann_clara/polonaise_op1n2.mxl", "a stately polonaise for piano"),
    ("schumann_clara/polonaise_op1n3.mxl", "a stately polonaise for piano"),
    ("schumann_clara/polonaise_op1n4.mxl", "a stately polonaise for piano"),
    ("bach/bwv80.8.mxl", "a warm, hymn-like chorale"),
    ("bach/bwv84.5.mxl", "a warm, hymn-like chorale"),
    ("bach/bwv8.6.mxl", "a warm, hymn-like chorale"),
    ("bach/bwv81.7.mxl", "a warm, hymn-like chorale"),
    ("bach/bwv83.5.mxl", "a warm, hymn-like chorale"),
]


def export(path, name):
    s = corpus.parse(path)
    # seconds map needs a tempo; default to a sensible one when the score has none
    if not list(s.recurse().getElementsByClass(m21tempo.MetronomeMark)):
        s.insert(0, m21tempo.MetronomeMark(number=100))
    flat = s.flatten()
    sm = flat.secondsMap
    pm = pretty_midi.PrettyMIDI(initial_tempo=100)
    piano = pretty_midi.Instrument(program=0)
    # cut at the first bar boundary at or past MAX_S: earliest note onset per measure
    starts = {}
    for e in sm:
        el = e["element"]
        if isinstance(el, (m21note.Note, m21chord.Chord)) and el.measureNumber is not None:
            starts[el.measureNumber] = min(starts.get(el.measureNumber, 1e9), e["offsetSeconds"])
    cut = next((t for _, t in sorted(starts.items()) if t >= MAX_S), None)
    for e in sm:
        el = e["element"]
        if isinstance(el, (m21note.Note, m21chord.Chord)):
            st, en = e["offsetSeconds"], e["endTimeSeconds"]
            if cut is not None and st >= cut:
                continue
            en = min(en, cut) if cut is not None else en
            if en - st <= 0.01:
                continue
            pitches = [p.midi for p in el.pitches]
            for p in pitches:
                piano.notes.append(pretty_midi.Note(velocity=80, pitch=p, start=st, end=en))
    pm.instruments.append(piano)
    out = OUT / f"{name}.mid"
    pm.write(str(out))
    return out, (max(n.end for n in piano.notes) if piano.notes else 0), len(piano.notes)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    keep = [r for r in load() if r.source != "human"]
    for path, brief in PIECES:
        name = "human_" + path.replace("/", "_").rsplit(".", 1)[0]
        out, dur, n = export(path, name)
        keep.append(Reference.from_midi(out, "love", "human", brief))
        print(f"{name:<40} {dur:5.1f}s {n:4d} notes")
    save(keep)
    print(f"pool: {len(keep)} refs -> refs/pool.json")
