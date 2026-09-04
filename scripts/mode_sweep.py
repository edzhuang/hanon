"""Every cantus must admit good solutions, and not too many.

Too few and the reward is unsatisfiable -- RL never sees a success to reinforce.
Too many and it is trivially satisfiable. Both are failure modes, so measure both.
"""
import sys; sys.path.insert(0, ".")
from hanon.cantus import CANTUS_FIRMI
from hanon.rewards.counterpoint import score
from hanon.rewards.spelling import Speller

for c in CANTUS_FIRMI:
    sp, cf = Speller(c.mode, c.final), list(c.pitches)
    lo, hi = c.final, c.final + 22
    cands = [[p] for p in range(lo, hi + 1)]
    for i in range(1, len(cf)):
        nxt = [x + [p] for x in cands for p in range(lo, hi + 1)]
        scored = sorted(((score(cf[: len(x)], x, sp)[0], x) for x in nxt), key=lambda t: -t[0])
        cands = [x for _, x in scored[:1500]]
    finals = sorted(((score(cf, x, sp)[0], x) for x in cands), key=lambda t: -t[0])
    perfect = [x for s, x in finals if s >= 0.999]
    best_s, best = finals[0]
    steps = sum(1 for a, b in zip(best, best[1:]) if 0 < abs(b - a) <= 2)
    print(f"{c.name:16} n={len(cf):2}  best={best_s:.3f}  perfect_found={len(perfect):4}  "
          f"range={max(best)-min(best):2}  steps={steps}/{len(best)-1}")
