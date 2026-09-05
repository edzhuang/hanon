# hanon

RL a small LLM to compose piano music by writing `pretty_midi` code. Following
surya.website/rling-qwen-to-paint-with-code, which RL'd Qwen to paint via p5.brush
sketches. Same loop: **prompt → model writes code → sandbox renders artifact → judge
scores → GRPO.**

| Paint-with-code | hanon |
| --- | --- |
| p5.brush JS sketch | Python using `pretty_midi` |
| Puppeteer → PNG | exec in sandbox → `.mid` (→ wav via fluidsynth) |
| HPSv3 aesthetic score | local anti-degeneracy metrics |
| pairwise judge vs. rated reference paintings | pairwise judge vs. rated reference pieces |
| GEPA on the system prompt | same |
| GRPO on a 35B model | GRPO on Qwen3-8B + LoRA, prime-rl |

## Decisions already made, and why

Reopening these without new evidence wastes time — each was argued out and several
reversed an earlier wrong call.

**`pretty_midi` with a six-call allowlist, not a custom DSL.** The blog's
allowlist-of-eight-methods finding was about p5.brush being *niche*: the model had no
pretraining knowledge, had to be told, and hallucinated. `pretty_midi` is in
pretraining, but only partly: the first sampling run (2026-09-04) showed qwen3-8b knows
`Note` and `Instrument` cold and hallucinates everything rarer — 12 of the first 14
sketches failed on invented `ControlChange` keyword names, because the prompt told it to
use the pedal without showing the signature. So the system prompt now carries the six
calls the task needs, with exact signatures, and says nothing else exists. Same shape as
the blog's fix, scaled to the actual gap; the library still does the work and ordinary
Python still supplies the structure. Do not document more of the library than this —
that is how tokens end up on time signatures instead of music.

**Thinking mode on, always.** qwen3-8b with `/no_think` collapses to a handful of outputs
per brief: 39 unique pieces in a 252-piece run, and 5 identical outputs from 5 calls at
temperature 1.2. Not caching (response ids differ, a nonce in the prompt doesn't help);
the non-thinking distribution is just a point mass on this task. With thinking on, 5
calls gave 5 unique pieces at ~10k tokens each. A collapsed policy gives GRPO zero
advantage and zero gradient, so diversity is non-negotiable. The cost is ~10k-token
rollouts, so `max_tokens` is 12000 and training steps are slower. The no-think runs are
kept in `out/v3` and `out/v4` as the record. The retired DSL is in git
history (`git show 7431b49:attic/dsl/__init__.py`).

**Qwen3-8B, not 4B.** RL sharpens what a model can already sometimes do; it does not
install knowledge. Composition is knowledge-bound and 4B is where that falls off. 4B and
8B also cost the same per GPU-hour, so 4B saves nothing and risks everything.

**The judge reads the score, not audio.** One fixed soundfont means timbre is constant
and all variance is in the notes, so `render.py`'s bar-by-bar text is near-lossless.
~220 tokens for 30s, so a comparison is ~1.8k — about $0.002 on Haiku, ~$8 per training
run. Audio judging is for spot-checks only, never the inner loop.

**Four reward components, not nine.** Their nine-component rubric plateaued at 0.65 with
repetitive output because four quality judges correlated 0.85–0.95 — one component
wearing four hats — while length saturated immediately. Start where they finished:

```
0.05  compiles                (binary gate)
0.05  length in range         (binary gate)
0.30  not degenerate          (local, free, absolute)
0.60  pairwise vs references  (the only component that scales with taste)
```

Everything in `metrics.py` is logged and rewards nothing. Observability is free; reward
components are not.

**The reference pool is graded on purpose.** They *couldn't* source human p5.brush
paintings. Piano is the opposite — MAESTRO and GiantMIDI are world-class and enormous —
and that is a trap: if every reference is Chopin the candidate loses every comparison,
the reward is constant, and a constant has no gradient. Mix `love` (target) with `ok`
(beatable), 70/30.

So we do *not* share the blog's shortage, and the two tiers come from different places:

- **`love` comes from human data, not from rating.** 30s excerpts from MAESTRO /
  GiantMIDI / public-domain MIDI, pushed through the same bar-by-bar renderer. Better
  than a hand-rated tier because it doesn't share the model's tics, so the judge can't
  be fooled by "sounds like the other model samples." Two things must be handled or the
  judge learns the wrong lesson: (1) *format is a tell* — performed MIDI has human
  timing, velocity curves and pedal, model output is quantized code, so human excerpts
  get quantized and velocity-flattened to look like something `pretty_midi` code could
  have produced; (2) *excerpts need edges* — 30s from the middle of a sonata has no
  beginning or cadence and loses to a model piece that has both, so cut at phrase
  boundaries or use short complete pieces.
- **`ok` comes from the model's own better samples**, hand-rated. This is what keeps a
  gradient alive. ~100 rated samples covers it, and the same labels calibrate the
  `metrics.py` thresholds, which human music can't do because it isn't degenerate.

**GH200 96GB @ $1.99/hr, not H100 @ $3.29.** More memory than an H100 for well under the
price — room for 8B + LoRA + vLLM + a colocated judge, which makes judge inference free.
Check `prime availability list` before assuming; prices move and I quoted them wrong once.

## Methods worth reusing

**Search the reward's argmax offline before spending anything on GPU.** RL converges
there, so look at it first. On the counterpoint grader this caught two exploits that
would have appeared in training as a clean 1.0 reward curve next to bad music — the
worst kind of bug, because the curve looks like success. The probe that found them is
`git show 7431b49:attic/argmax_probe.py`.

**Validate a grader against a known-good example.** The counterpoint grader was tested
against Fux's own dorian solution; when it flagged him, the rule was wrong, not Fux.
Find the equivalent positive control for any new reward component.

**Iterate on reward offline against cached rollouts.** Generate once, re-score for free.
Never rent a GPU to tune reward weights.

## Headroom probe result (2026-09-05)

64 pieces (8 per brief, thinking-mode qwen3-8b), judged both ways against 2 human
`love` refs and 3 other model pieces. $1.65. There is a slope:

| Signal | Value | Reading |
| --- | --- | --- |
| model vs human, mean win | 0.20 | humans win 80%, not 100%: the 0.60 component is not constant |
| model vs model, std of win rate | 0.32 | wider than the noise floor; the judge orders the pieces |
| corr(vs human, vs model) | +0.47 | two independent comparison sets agree on which pieces are better |
| corr(judge, bar-level loopiness) | −0.33 | the judge dislikes loops, which is what the ear disliked |
| corr(judge, `metrics.py` degeneracy) | +0.07 | the local metric measures nothing the judge cares about |
| judge disagrees with itself across orders | 24% | the per-comparison noise floor; ask both orders, always |

Caveats: best-of-8 minus mean-of-8 came out +0.47 but is inflated by max-of-noisy-values;
the correlations are the honest evidence. The ceiling is low: every piece in the pool
loops and the user rated all 216 `meh`. RL can climb from "loops 90%" toward "loops
less"; it cannot reach "good" from this base model.

**Judge findings that changed the design.** Probe v1 (1,080 verdicts, $2.50) measured
the judge rather than the model: Haiku picks whichever piece is shown *second* 68% of
the time, and asked "which is better *for this brief*" it preferred model loops to
Bach 62% of the time, because human refs are not brief-matched and a chorale is a poor
music-box lullaby. The judge now scores musical quality only, asks both orders (1 /
0.5 / 0), and gets 120 tokens so "I need to evaluate…" openers don't truncate to
nothing. Sanity: the loopiest piece loses to every human ref both ways; Bach vs Chopin
splits; the least-loopy model piece beats the loopiest both ways.

**The `structure` gate rewards loops.** It bands pitch-class-histogram self-similarity
to 0.45–0.85 and a two-bar loop with a drifting melody note lands inside the band. Its
correlation with bar-level repetition is 0.01; `not_stuck` returned one value for all
216 pieces. Replace with a transposition-aware bar-repetition metric, calibrated on the
two classes we have: human refs vs the v5 pool.

## Status

| Piece | File | State |
| --- | --- | --- |
| System prompt + briefs | `hanon/prompt.py` | done |
| Sketch executor | `hanon/executor.py` | done |
| Anti-degeneracy metrics | `hanon/rewards/metrics.py` | done, **`structure` gate rewards loops, needs replacing** |
| Bar-by-bar renderer | `hanon/render.py` | done |
| Pairwise judge | `hanon/rewards/judge.py` | done; quality-only, both orders, validated on known pairs |
| Reference pool | `hanon/refs.py` | 12 human `love` refs from music21 corpus (`scripts/human_refs.py`); `ok` = model's own samples |
| End-to-end reward | `hanon/task.py` | done |
| Sampling harness | `scripts/play.py`, `scripts/sample_all.sh` | done; v5 pool in `out/` (216 compiled, 212 unique) |
| Headroom probe | `scripts/headroom.py` | done, see result above |
| verifiers v1 taskset | — | next |
| GEPA prompt optimisation | — | after baseline |
| GRPO run | — | after headroom probe |

## Next, in order

1. **Sample a few hundred pieces** with `scripts/play.py` on qwen3-8b ($0.117/$0.455 per
   1M via Prime Inference — a few dollars).
2. **Build the reference pool.** Two halves: (a) a script that excerpts, quantizes and
   velocity-flattens human MIDI into `love`; (b) rate ~100 model samples into `ok` /
   `meh`, about half an hour of listening. Nothing downstream works without this.
3. **Calibrate the metric thresholds** against the rated pool instead of my guesses.
4. **Headroom probe**: is best-of-8 meaningfully better than mean-of-8? That gap is what
   RL climbs. No gap means no amount of GPU time helps.
5. **GEPA** on the system prompt — where the blog got its cheapest wins.
6. **GRPO**, Qwen3-8B + LoRA, 1×GH200 ($1.99/hr, verified 2026-09-04).

## Budget

Hard cap **$75**, prepaid into the Prime wallet in two tranches. Everything this
project spends (pods, qwen3-8b sampling, the Haiku judge via Prime Inference) draws
from that one wallet, so `prime wallet` is the single source of truth for spend.

| Tranche | Covers | Amount |
| --- | --- | --- |
| Deposited 2026-09-04 | Steps 1–5: sampling, judge passes, calibration, headroom probe, GEPA. API only. | $25 |
| After the headroom probe passes | Step 6: ~25 GH200-hours. One careful GRPO run plus a retry, not a comfortable four. | $50 |

The GRPO tranche is deliberately lean, so the offline preparation is not optional:
reward tuned on cached rollouts, argmax probe run, config smoke-tested on a tiny model
locally, before the first GPU hour.

Rules:

- The second tranche is gated on the headroom probe, not on the balance. No gap
  between best-of-8 and mean-of-8 means the project ends with $0 of GPU spend.
- When the wallet is empty the project stops. Write up, don't top up.
- Auto top-up stays off. Confirm in billing settings that a pod is terminated at zero
  balance rather than the account going negative.
- `prime pods list` at the end of every session. An idle pod costs the same as a
  training one, and a forgotten weekend is $140.

## Working agreements

- Verify prices, model availability, and API shapes with the CLI rather than from
  memory. Several confident numbers here were wrong on first quote.
- `metrics.py` thresholds are still guesses and hold 0.30 of the reward. Recalibrate
  against the rated pool before trusting them.
- The user owns the reward design; explain it rather than only implementing it.

## Environment

`uv` + `.venv` (Python 3.12). Inference via `prime inference chat` (already
authenticated) — set `PRIME_DISABLE_VERSION_CHECK=1`. Soundfont at `assets/piano.sf2`,
rendering via `fluidsynth`. `hanon/infer.py` is the single place that calls Prime.

## Removed, but recoverable

First-species counterpoint — a fully verifiable grader, validated to rank Fux's own
dorian solution at the top with nothing able to outscore it — and the original custom
DSL both lived in `attic/` until they were deleted. Everything is in git history at
`7431b49`; `git show 7431b49:attic/counterpoint.py` and friends. Two pieces are worth
resurrecting if the need comes up: a rule filter in front of the judge, so the judge
never spends a call on malformed music, and `argmax_probe.py`, which searches a
reward's maximiser offline and applies to any reward, including the current one.
