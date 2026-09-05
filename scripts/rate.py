"""Rate sampled pieces into love / ok / meh for the reference pool.

Usage:
  uv run scripts/rate.py            # listen and rate; resumable, ctrl-c or q to stop
  uv run scripts/rate.py --stats    # tallies so far
  uv run scripts/rate.py --build    # write refs/pool.json from the ratings

Keys while a piece plays:  l = love   o = ok   m = meh   r = replay   s = skip   q = quit

Ratings go to refs/ratings.jsonl, one line per piece, append-only. The brief is shown
(the judge sees it too) but the degeneracy score is hidden so it can't bias you; the
ratings are what those thresholds get calibrated against.
"""
import argparse, json, random, subprocess, sys, termios, tty
from pathlib import Path

sys.path.insert(0, ".")

OUT = Path("out")
RATINGS = Path("refs/ratings.jsonl")
KEYS = {"l": "love", "o": "ok", "m": "meh"}


def pieces():
    """Every compiled piece in out/ with its brief, from the *_results.json files."""
    rows = []
    for rj in sorted(OUT.glob("*_results.json")):
        tag = rj.name[: -len("_results.json")]
        for r in json.loads(rj.read_text()):
            if not r.get("ok") or not r.get("wav"):
                continue
            wav = Path(r["wav"])
            if not wav.exists():
                continue
            rows.append({"id": f"{tag}_{r['i']}", "wav": str(wav),
                         "mid": str(wav.with_suffix(".mid")),
                         "brief": brief_for(tag), "duration_s": r["features"]["duration_s"]})
    return rows


def brief_for(tag):
    from hanon.prompt import PROMPTS
    for p in PROMPTS:
        if "".join(c if c.isalnum() else "_" for c in p)[:32] == tag:
            return p
    return tag


def load_ratings():
    if not RATINGS.exists():
        return {}
    return {json.loads(l)["id"]: json.loads(l) for l in RATINGS.read_text().splitlines() if l.strip()}


def getch():
    fd = sys.stdin.fileno(); old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd); return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def play(wav):
    return subprocess.Popen(["afplay", wav])


def rate(seed):
    done = load_ratings()
    todo = [p for p in pieces() if p["id"] not in done]
    random.Random(seed).shuffle(todo)   # fixed seed: same order each session, interleaves briefs
    print(f"{len(done)} rated, {len(todo)} to go.  l=love o=ok m=meh r=replay s=skip q=quit\n")
    RATINGS.parent.mkdir(exist_ok=True)
    with RATINGS.open("a") as f:
        for k, p in enumerate(todo, 1):
            print(f"[{k}/{len(todo)}] {p['duration_s']:.0f}s  \"{p['brief']}\"", end="  ", flush=True)
            proc = play(p["wav"])
            while True:
                c = getch().lower()
                if c == "r":
                    proc.terminate(); proc = play(p["wav"]); continue
                if c in ("q", "\x03"):
                    proc.terminate(); print("\nbye"); return
                if c == "s":
                    proc.terminate(); print("skip"); break
                if c in KEYS:
                    proc.terminate(); print(KEYS[c])
                    f.write(json.dumps({**p, "tier": KEYS[c]}) + "\n"); f.flush(); break
    print("\nall rated")


def stats():
    r = load_ratings()
    by = {t: sum(1 for x in r.values() if x["tier"] == t) for t in ("love", "ok", "meh")}
    print(f"{len(r)} rated: {by}")
    for b in sorted({x["brief"] for x in r.values()}):
        xs = [x for x in r.values() if x["brief"] == b]
        print(f"  {sum(x['tier']=='love' for x in xs):2d} love {sum(x['tier']=='ok' for x in xs):2d} ok "
              f"{sum(x['tier']=='meh' for x in xs):2d} meh   {b}")


def build():
    from hanon.refs import Reference, load, save, POOL
    keep = {r.id: r for r in load() if r.source == "human"}   # human tier is built elsewhere
    n = 0
    for x in load_ratings().values():
        if x["tier"] in ("love", "ok"):
            keep[x["id"]] = Reference.from_midi(x["mid"], x["tier"], "model", x["brief"]); n += 1
    save(list(keep.values()))
    print(f"wrote {POOL}: {n} model refs + {len(keep) - n} human refs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    if a.stats: stats()
    elif a.build: build()
    else: rate(a.seed)
