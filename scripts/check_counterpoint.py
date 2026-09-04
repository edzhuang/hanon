"""Validate the grader against Fux's own textbook example and deliberate errors.

A grader that flags Gradus ad Parnassum is a broken grader, so the authentic example
is the positive control the aesthetic metrics never had.
"""
import sys; sys.path.insert(0, ".")
from hanon.cantus import BY_NAME
from hanon.rewards.counterpoint import score

CF = list(BY_NAME["fux_dorian"].pitches)          # D F E D G F A G F E D
FUX = [69, 69, 67, 69, 71, 72, 72, 71, 74, 73, 74]  # Fux's counterpoint above it

CASES = {
    "fux (authentic)": FUX,
    "parallel 5ths":   [69, 72, 74, 69, 71, 72, 72, 71, 74, 73, 74],
    "dissonances":     [68, 66, 70, 68, 73, 71, 70, 68, 74, 73, 74],
    "static":          [69] * 11,
    "wrong length":    [69, 69, 67, 69, 71],
    "bad final":       [69, 69, 67, 69, 71, 72, 72, 71, 74, 73, 72],
}

for name, cp in CASES.items():
    s, vs = score(CF, cp)
    print(f"{name:17} {s:.3f}  ({len(vs)} violations)")
    for x in vs[:5]:
        print(f"{'':19}- {x}")
