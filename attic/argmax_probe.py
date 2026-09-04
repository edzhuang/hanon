"""Search for the highest-scoring counterpoint the grader admits, then look at it.

RL converges to the argmax of the reward. If the argmax is musically dull, the reward
is wrong -- and finding that out costs nothing here, versus a GPU-day to discover it
from a training curve.
"""
import sys; sys.path.insert(0, ".")
from hanon.cantus import BY_NAME
from hanon.rewards.counterpoint import score
from hanon.rewards.spelling import Speller

C = BY_NAME["fux_dorian"]; SP = Speller(C.mode, C.final); CF = list(C.pitches)
FUX = [69, 69, 67, 69, 71, 72, 72, 71, 74, 73, 74]
LO, HI = 62, 84

def beam(width=4000):
    """Grow lines left to right, keeping the best partials by their own partial score."""
    cands = [[p] for p in range(LO, HI + 1)]
    for i in range(1, len(CF)):
        nxt = []
        for c in cands:
            for p in range(LO, HI + 1):
                nxt.append(c + [p])
        # Score each partial against the cantus prefix; cheap and prunes hard.
        scored = []
        for c in nxt:
            s, _ = score(CF[: len(c)], c, SP)
            scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        cands = [c for _, c in scored[:width]]
    return max(((score(CF, c, SP)[0], c) for c in cands), key=lambda x: x[0])

best_s, best = beam()
fux_s, fux_v = score(CF, FUX, SP)
print(f"argmax found : {best_s:.3f}  {best}")
print(f"fux          : {fux_s:.3f}  {FUX}")
print(f"              fux penalised for: {[x.rule for x in fux_v]}")
print()
steps = sum(1 for a, b in zip(best, best[1:]) if abs(b - a) <= 2)
print(f"argmax shape : range {max(best)-min(best)} semitones, "
      f"{steps}/{len(best)-1} stepwise, {len(set(best))} distinct pitches")
print(f"fux shape    : range {max(FUX)-min(FUX)} semitones, "
      f"{sum(1 for a,b in zip(FUX,FUX[1:]) if abs(b-a)<=2)}/{len(FUX)-1} stepwise, "
      f"{len(set(FUX))} distinct pitches")
