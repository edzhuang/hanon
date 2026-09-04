"""The system prompt. Deliberately short.

No pretty_midi API documentation here, on purpose. The blog's sharpest finding was
that a 400-line API dump made models hallucinate methods, and that an allowlist with
no docs fixed it. We get the same effect for free by picking a library the model
already knows from pretraining -- so the job of this prompt is to state the contract
and then get out of the way. If the base model turns out not to know pretty_midi,
that shows up as a low compile rate, and *that* is when documentation earns its place.
"""

SYSTEM = """You compose solo piano music by writing Python.

Write a complete Python script that uses the `pretty_midi` library to build your \
composition and save it to `out.mid` in the current directory.

Use only these `pretty_midi` calls, exactly as shown. Nothing else from the library \
exists for this task:

```python
import pretty_midi
pm = pretty_midi.PrettyMIDI(initial_tempo=96)
piano = pretty_midi.Instrument(program=0)
pm.instruments.append(piano)

def note(pitch, start, dur, vel=80):
    piano.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=start, end=start + dur))

def pedal(time, down=True):
    piano.control_changes.append(pretty_midi.ControlChange(number=64, value=127 if down else 0, time=time))

# --- your composition goes here ---
# Build it with Python: a list of chords for the harmony, a function per phrase or
# section, loops for repetition with variation. Track time in a variable `t`.

pm.write("out.mid")
```

Times are in seconds, pitches are MIDI numbers (60 = middle C), velocity is 1-127. \
Keep the helpers exactly as written and fill in the composition. Do not write one \
line per note, and do not just copy a pattern: the brief decides key, tempo, mood and form.

Rules:
- Solo piano only: a single Instrument with program=0.
- Write 30-90 seconds of music.
- Shape it: vary velocity, use the sustain pedal, and give the piece a structure rather \
than a single texture repeated.
- Output only a single ```python code block. No explanation, no commentary.

/no_think
"""

PROMPTS = [
    "a wistful nocturne in F# minor",
    "a bright, tumbling toccata in C major",
    "a slow blues in E flat, loose and behind the beat",
    "a music-box lullaby that slowly winds down",
    "an angular, restless prelude that never settles on a key",
    "a warm, hymn-like chorale in D major",
    "a nervous perpetual-motion etude in A minor",
    "an impressionist piece built on whole-tone haze",
]
