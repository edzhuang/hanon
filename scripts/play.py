"""Generate piano pieces with a base model, execute them, render audio.

Usage: uv run scripts/play.py "a wistful nocturne in F# minor" -n 4 -m qwen/qwen3-8b
"""
import argparse, json, os, shutil, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, ".")
from hanon.executor import run_sketch
from hanon.prompt import SYSTEM
from hanon.rewards import analyze, degeneracy

SF2 = Path("assets/piano.sf2")
OUT = Path("out")


def generate(model, prompt, temperature, max_tokens):
    env = {**os.environ, "PRIME_DISABLE_VERSION_CHECK": "1"}
    p = subprocess.run(
        ["prime", "inference", "chat", model, prompt, "-s", SYSTEM,
         "-t", str(temperature), "--max-tokens", str(max_tokens), "-o", "json", "--plain"],
        capture_output=True, text=True, env=env, timeout=600,
    )
    if p.returncode != 0:
        raise RuntimeError(f"inference failed: {(p.stderr or p.stdout)[-500:]}")
    start = p.stdout.find("{")
    if start < 0:
        raise RuntimeError(f"no JSON in reply: {p.stdout[:300]}")
    d = json.loads(p.stdout[start:])
    return d["choices"][0]["message"]["content"], d.get("usage", {})


def render_wav(midi: Path, wav: Path):
    if not SF2.exists() or not shutil.which("fluidsynth"):
        return None
    subprocess.run(["fluidsynth", "-a", "file", "-F", str(wav), "-r", "44100",
                    "-q", str(SF2), str(midi)], capture_output=True, timeout=180)
    return wav if wav.exists() and wav.stat().st_size > 1000 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("-n", type=int, default=4)
    ap.add_argument("-m", "--model", default="qwen/qwen3-8b")
    ap.add_argument("-t", "--temperature", type=float, default=0.9)
    ap.add_argument("--max-tokens", type=int, default=4096)
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    tag = "".join(c if c.isalnum() else "_" for c in a.prompt)[:32]
    rows, toks = [], 0

    for i in range(a.n):
        t0 = time.time()
        try:
            text, usage = generate(a.model, a.prompt, a.temperature, a.max_tokens)
        except Exception as e:
            print(f"[{i}] generation failed: {e}"); continue
        toks += usage.get("completion_tokens", 0)
        r = run_sketch(text)
        stem = f"{tag}_{i}"
        (OUT / f"{stem}.py").write_text(r.code)
        if not r.ok:
            print(f"[{i}] FAILED  {r.error.splitlines()[-1][:90] if r.error else ''}")
            rows.append({"i": i, "ok": False, "error": r.error}); continue
        midi = OUT / f"{stem}.mid"
        shutil.copy(r.midi_path, midi)
        f = analyze(midi); s, g = degeneracy(f)
        wav = render_wav(midi, OUT / f"{stem}.wav")
        print(f"[{i}] ok  {f.duration_s:5.1f}s  {f.n_notes:4d} notes  "
              f"key={f.key or '?':<12} degeneracy={s:.2f}  "
              f"({time.time()-t0:.0f}s, {usage.get('completion_tokens',0)} tok)")
        rows.append({"i": i, "ok": True, "score": s, "gates": g,
                     "features": f.as_dict(), "wav": str(wav) if wav else None})

    ok = [r for r in rows if r["ok"]]
    print(f"\ncompile rate {len(ok)}/{len(rows)}"
          + (f"   degeneracy mean {sum(r['score'] for r in ok)/len(ok):.2f}"
             f"  best {max(r['score'] for r in ok):.2f}" if ok else ""))
    print(f"completion tokens: {toks}")
    (OUT / f"{tag}_results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
