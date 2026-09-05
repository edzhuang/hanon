"""Headroom probe: is there anything for RL to climb?

For every compiled piece in out/:
  - vs_human: win rate against `n_human` random human `love` references
  - vs_model: win rate against `n_model` random other pieces from the same brief
Then: does the judge order the pieces at all, and how big is the best-of-8 vs
mean-of-8 gap per brief? That gap is what GRPO climbs; no gap, no point renting a GPU.

Verdicts are cached in out/headroom.json so re-running is free.
Usage: uv run scripts/headroom.py [--n-human 2] [--n-model 3] [--workers 8]
"""
import argparse, json, random, statistics as st, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, ".")
from hanon.refs import load
from hanon.render import describe
from hanon.rewards.judge import compare

CACHE = Path("out/headroom.json")


def pieces():
    from hanon.prompt import PROMPTS
    tag = lambda p: "".join(c if c.isalnum() else "_" for c in p)[:32]
    rows = []
    for rj in sorted(Path("out").glob("*_results.json")):
        t = rj.name[:-len("_results.json")]
        brief = next((p for p in PROMPTS if tag(p) == t), t)
        for r in json.loads(rj.read_text()):
            if r.get("ok"):
                rows.append({"id": f"{t}_{r['i']}", "brief": brief, "mid": f"out/{t}_{r['i']}.mid",
                             "degeneracy": r["score"]})
    return rows


def main(a):
    rng = random.Random(0)
    rows = pieces()
    human = [r for r in load() if r.source == "human" and r.tier == "love"]
    print(f"{len(rows)} pieces, {len(human)} human refs")
    text = {r["id"]: describe(r["mid"]) for r in rows}
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    jobs = []  # (key, brief, cand_text, ref_text)
    by_brief = {}
    for r in rows: by_brief.setdefault(r["brief"], []).append(r)
    for r in rows:
        for h in rng.sample(human, min(a.n_human, len(human))):
            jobs.append((f"{r['id']}|H|{h.id}", r["brief"], text[r["id"]], h.rendering))
        others = [o for o in by_brief[r["brief"]] if o["id"] != r["id"]]
        for o in rng.sample(others, min(a.n_model, len(others))):
            jobs.append((f"{r['id']}|M|{o['id']}", r["brief"], text[r["id"]], text[o["id"]]))
    todo = [j for j in jobs if j[0] not in cache]
    print(f"{len(jobs)} comparisons, {len(todo)} to run")

    def run(j):
        key, brief, c, ref = j
        return key, compare(brief, c, ref, rng=random.Random(key))
    with ThreadPoolExecutor(a.workers) as ex:
        for i, (key, v) in enumerate(ex.map(run, todo), 1):
            cache[key] = v
            if i % 100 == 0:
                CACHE.write_text(json.dumps(cache)); print(f"  {i}/{len(todo)}", flush=True)
    CACHE.write_text(json.dumps(cache))

    # ---- analysis
    def rate(rid, kind):
        vs = [v for k, v in cache.items() if k.startswith(f"{rid}|{kind}|") and v is not None]
        return sum(vs) / len(vs) if vs else None
    for r in rows:
        r["vs_human"] = rate(r["id"], "H"); r["vs_model"] = rate(r["id"], "M")
    none = sum(1 for k, v in cache.items() if v is None)
    print(f"\nunusable verdicts: {none}/{len(cache)}")
    vh = [r["vs_human"] for r in rows if r["vs_human"] is not None]
    vm = [r["vs_model"] for r in rows if r["vs_model"] is not None]
    print(f"vs human: mean win {st.mean(vh):.3f}   pieces with any win: {sum(x > 0 for x in vh)}/{len(vh)}")
    print(f"vs model: mean win {st.mean(vm):.3f} (0.5 = no ordering)   std {st.pstdev(vm):.3f}   "
          f"win-rate histogram: " + " ".join(f"{x:.2f}:{sum(abs(y - x) < 0.01 for y in vm)}" for x in sorted(set(round(y, 2) for y in vm))))
    # best-of-8 vs mean-of-8 on the model-vs-model win rate, per brief
    print("\nbest-of-8 vs mean-of-8 (model-vs-model win rate), per brief:")
    gaps = []
    for b, rs in by_brief.items():
        xs = [r["vs_model"] for r in rs if r["vs_model"] is not None]; rng.shuffle(xs)
        groups = [xs[i:i + 8] for i in range(0, len(xs) - 7, 8)]
        gm = st.mean(st.mean(g) for g in groups); gb = st.mean(max(g) for g in groups); gaps.append(gb - gm)
        print(f"  mean {gm:.2f}  best {gb:.2f}  gap {gb - gm:+.2f}   {b[:45]}")
    print(f"  average gap: {st.mean(gaps):+.2f}")
    # does the judge agree with the local metrics?
    import math
    def corr(a, b):
        ma, mb = st.mean(a), st.mean(b); d = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
        return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / d if d else float("nan")
    loop = {r["id"]: r for r in json.loads(Path("out/loopiness.json").read_text())} if Path("out/loopiness.json").exists() else {}
    ok = [r for r in rows if r["vs_model"] is not None and r["id"] in loop]
    print(f"\ncorr(vs_model, degeneracy metric) = {corr([r['vs_model'] for r in ok], [r['degeneracy'] for r in ok]):+.2f}")
    print(f"corr(vs_model, loopiness)         = {corr([r['vs_model'] for r in ok], [loop[r['id']]['loopy'] for r in ok]):+.2f}")
    print(f"corr(vs_human, vs_model)          = {corr([r['vs_human'] for r in ok], [r['vs_model'] for r in ok]):+.2f}")
    print("\ntop 5 by judge (model-vs-model), for listening:")
    for r in sorted(ok, key=lambda r: -r["vs_model"])[:5]:
        print(f"  {r['vs_model']:.2f}  vs_human={r['vs_human']:.2f}  loopy={loop[r['id']]['loopy']:.2f}  out/{r['id']}.wav")
    Path("out/headroom_rows.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-human", type=int, default=2)
    ap.add_argument("--n-model", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    main(ap.parse_args())
