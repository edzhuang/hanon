"""Run a model-written pretty_midi sketch and get a MIDI file back.

The subprocess isolation here is hygiene, not a security boundary -- it stops runaway
loops and memory hogs from taking down a rollout worker. Untrusted-code containment is
the container's job under training (Prime sandbox / Docker), and deliberately not
attempted in-process: blocking imports via sys.modules is both trivially reversible and
liable to break honest libraries (mido imports socket).
"""

from __future__ import annotations

import json
import re
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

OUT_NAME = "out.mid"

@dataclass
class Sketch:
    """The outcome of running one model-written sketch."""

    ok: bool
    code: str
    midi_path: Path | None = None
    error: str | None = None
    stdout: str = ""
    duration_s: float = 0.0
    info: dict = field(default_factory=dict)


def extract_code(text: str) -> str:
    """Pull Python out of a model reply. Models fence, prose, and apologise."""
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    if fences:
        return max(fences, key=len).strip()
    return text.strip()


def _limits():
    """Best-effort resource caps. macOS silently rejects some of these, and a missing
    cap locally is fine -- training runs in a container where they actually bind."""
    caps = [(resource.RLIMIT_CPU, (60, 60)), (resource.RLIMIT_NOFILE, (512, 512))]
    if sys.platform != "darwin":  # RLIMIT_AS is unreliable on macOS
        caps.append((resource.RLIMIT_AS, (4 << 30, 4 << 30)))
    for what, vals in caps:
        try:
            resource.setrlimit(what, vals)
        except (ValueError, OSError):
            pass


def run_sketch(text: str, timeout: float = 30.0, keep_dir: Path | None = None) -> Sketch:
    """Execute a sketch and return the MIDI it wrote, or why it didn't."""
    code = extract_code(text)
    if not code:
        return Sketch(ok=False, code="", error="empty response")

    workdir = Path(tempfile.mkdtemp(prefix="hanon_", dir=keep_dir))
    script = workdir / "sketch.py"
    script.write_text(code)  # no prelude: keeps tracebacks line-accurate

    try:
        proc = subprocess.run(
            [sys.executable, "-I", str(script)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_limits,
        )
    except subprocess.TimeoutExpired:
        return Sketch(ok=False, code=code, error=f"timeout after {timeout}s")
    except Exception as e:  # pragma: no cover - platform oddities
        return Sketch(ok=False, code=code, error=f"launch failed: {e}")

    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()
        return Sketch(
            ok=False,
            code=code,
            stdout=proc.stdout,
            error="\n".join(tail[-4:]) or f"exit {proc.returncode}",
        )

    midi = workdir / OUT_NAME
    if not midi.exists():
        found = sorted(workdir.glob("*.mid")) + sorted(workdir.glob("*.midi"))
        if not found:
            return Sketch(ok=False, code=code, stdout=proc.stdout,
                          error=f"ran cleanly but wrote no MIDI (expected {OUT_NAME})")
        midi = found[0]

    try:
        import pretty_midi

        pm = pretty_midi.PrettyMIDI(str(midi))
        n = sum(len(i.notes) for i in pm.instruments)
        dur = float(pm.get_end_time())
    except Exception as e:
        return Sketch(ok=False, code=code, midi_path=midi, stdout=proc.stdout,
                      error=f"unparseable MIDI: {e}")

    if n == 0:
        return Sketch(ok=False, code=code, midi_path=midi, stdout=proc.stdout,
                      error="MIDI contains no notes")

    return Sketch(ok=True, code=code, midi_path=midi, stdout=proc.stdout,
                  duration_s=dur, info={"n_notes": n})


if __name__ == "__main__":
    r = run_sketch(sys.stdin.read())
    print(json.dumps({"ok": r.ok, "error": r.error, "dur": r.duration_s, **r.info}, indent=2))
