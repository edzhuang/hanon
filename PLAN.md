# hanon — RL a small LLM to compose piano music by writing code

Following surya.website/rling-qwen-to-paint-with-code, which RL'd Qwen to paint via
p5.brush sketches. Same loop: **prompt → model writes code → sandbox renders artifact
→ judge scores → GRPO.**

| Paint-with-code | hanon |
| --- | --- |
| p5.brush JS sketch | Python using `pretty_midi` |
| Puppeteer → PNG | exec in sandbox → `.mid` (→ wav via fluidsynth) |
| HPSv3 aesthetic score | local anti-degeneracy metrics |
| pairwise judge vs. rated reference paintings | pairwise judge vs. rated reference pieces |
| GEPA on the system prompt | same |
| GRPO on a 35B model | GRPO on Qwen3-8B + LoRA, prime-rl |

## Reward

Their *post-mortem* shape, not the one they started with. The first attempt had nine
components and plateaued at 0.65 with repetitive output, because four quality judges
correlated 0.85–0.95 — one component wearing four hats — while length saturated
immediately. Starting where they finished:

```
0.05  compiles                (binary gate)
0.05  length in range         (binary gate)
0.30  not degenerate          (local, free, absolute)
0.60  pairwise vs references  (the only component that scales with taste)
```

Everything in `metrics.py` is logged and rewards nothing.

## Two departures from the blog

**The judge reads the score, not the audio.** Solo piano through one fixed soundfont
means timbre is constant and all the variance is in the notes, so a bar-by-bar text
rendering is nearly lossless. A 30s piece is ~220 tokens, so a comparison is ~1.8k —
about **$0.002 per judgment on Haiku**, or ~$8 per training run. Audio judging stays
for spot-checks, not the inner loop.

**The reference pool is graded on purpose.** They *couldn't* source human p5.brush
paintings. Piano is the opposite — MAESTRO and GiantMIDI are world-class and enormous —
and that is a trap: if every reference is Chopin, the candidate loses every comparison,
the reward is constant, and a constant has no gradient. The pool mixes `love` (the
target) with `ok` (beatable), sampled 70/30.

## Status

| Piece | File | State |
| --- | --- | --- |
| System prompt + briefs | `hanon/prompt.py` | done |
| Sketch executor | `hanon/executor.py` | done |
| Anti-degeneracy metrics | `hanon/rewards/metrics.py` | done, **thresholds still guesses** |
| Bar-by-bar renderer | `hanon/render.py` | done |
| Pairwise judge | `hanon/rewards/judge.py` | done, discriminates correctly live |
| Reference pool | `hanon/refs.py` | schema done, **pool empty** |
| End-to-end reward | `hanon/task.py` | done |
| Sampling harness | `scripts/play.py` | done |
| verifiers v1 taskset | — | next |
| GEPA prompt optimisation | — | after baseline |
| GRPO run | — | after headroom probe |

## Next, in order

1. **Sample a few hundred pieces** with `scripts/play.py` on qwen3-8b ($0.117/$0.455 per
   1M via Prime Inference — a few dollars).
2. **Rate ~200 of them** into `love` / `ok` / `meh`. About an hour of listening. This is
   the real bottleneck and nothing downstream works without it.
3. **Calibrate the metric thresholds** against the rated pool instead of my guesses.
4. **Headroom probe**: is best-of-8 meaningfully better than mean-of-8? That gap is what
   RL climbs. No gap means no amount of GPU time helps.
5. **GEPA** on the system prompt — where the blog got its cheapest wins.
6. **GRPO**, Qwen3-8B + LoRA, 1×H100 (~$1.75/hr).

## Budget

**$150 cap**, ~$60 to a first result. Judge inference is ~$8/run, not a barrier.
Discipline that keeps it there: debug reward offline against cached rollouts, spend $2
on the headroom probe before $60 on a run, smoke-test at 20 steps, and terminate the
instance — a forgotten H100 over a weekend costs more than a successful run.

## attic/

First-species counterpoint: a fully verifiable grader, validated to rank Fux's own
dorian solution at the top with nothing able to outscore it. Retired when the judge
turned out to be affordable. Worth keeping — a rule filter in front of the judge would
mean the judge never spends a call on malformed music, and `argmax_probe.py` (search
the reward's maximiser offline and look at it) applies to any reward, including this one.
