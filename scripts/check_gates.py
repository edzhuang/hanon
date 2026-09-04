"""Validate the gates by feeding them pieces that are broken in known ways.

If a gate cannot catch the pathology it was written for, it is decoration.
"""
import sys; sys.path.insert(0, ".")
import pretty_midi as pm
from hanon.rewards import analyze, degeneracy

def build(name, fn):
    m = pm.PrettyMIDI(initial_tempo=100); ins = pm.Instrument(program=0)
    fn(ins); m.instruments.append(ins)
    path = f"/tmp/gate_{name}.mid"; m.write(path); return path

def stuck(i):        # one pitch, 120 times
    for k in range(120): i.notes.append(pm.Note(80, 60, k*0.25, k*0.25+0.24))
def loop(i):         # two bars, looped 16 times, nothing else
    for r in range(16):
        for k, p in enumerate([60,64,67,72]):
            t = r*2 + k*0.5; i.notes.append(pm.Note(80, p, t, t+0.5))
def flat(i):         # musical shape, zero dynamics, rigid grid
    for k, p in enumerate([60,62,64,65,67,69,71,72]*6):
        i.notes.append(pm.Note(80, p, k*0.25, k*0.25+0.25))
def cramped(i):      # everything inside a fifth
    import random; random.seed(0)
    for k in range(200):
        i.notes.append(pm.Note(random.randint(60,100), random.choice([60,61,62,63,64]), k*0.15, k*0.15+0.2))
def noise(i):        # random pitches, random rhythm, no key
    import random; random.seed(1); t=0.0
    for _ in range(220):
        t += random.choice([0.07,0.13,0.21,0.34,0.55])
        i.notes.append(pm.Note(random.randint(40,110), random.randint(21,108), t, t+0.3))

for name, fn in [("stuck",stuck),("loop",loop),("flat",flat),("cramped",cramped),("noise",noise)]:
    f = analyze(build(name, fn)); s, g = degeneracy(f)
    worst = sorted(g.items(), key=lambda kv: kv[1])[:3]
    print(f"{name:9} score={s:.2f}  worst: " + ", ".join(f"{k}={v:.2f}" for k, v in worst))
