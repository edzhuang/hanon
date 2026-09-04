# hanon

RL a small LLM to compose piano music by writing `pretty_midi` code. Following
surya.website/rling-qwen-to-paint-with-code: **prompt → model writes code → sandbox
renders → judge scores → GRPO.** See `PLAN.md` for the current state and next steps.

## Decisions already made, and why

Reopening these without new evidence wastes time — each was argued out and several
reversed an earlier wrong call.

**`pretty_midi`, not a custom DSL.** The blog's allowlist-of-eight-methods finding was
about p5.brush being *niche*: the model had no pretraining knowledge, had to be told,
and hallucinated. `pretty_midi` is already in pretraining, so it needs no docs in the
system prompt — same endpoint, reached for free. The retired DSL is in `attic/dsl/`.

**Qwen3-8B, not 4B.** RL sharpens what a model can already sometimes do; it does not
install knowledge. Composition is knowledge-bound and 4B is where that falls off. 4B and
8B also cost the same per GPU-hour, so 4B saves nothing and risks everything.

**The judge reads the score, not audio.** One fixed soundfont means timbre is constant
and all variance is in the notes, so `render.py`'s bar-by-bar text is near-lossless.
~220 tokens for 30s, ~$0.002 per comparison. Audio judging is for spot-checks only.

**Four reward components, not nine.** Their nine-component rubric plateaued because four
quality judges correlated 0.85–0.95 — one component wearing four hats. Start where they
finished: `0.05 compile + 0.05 length + 0.30 degeneracy + 0.60 pairwise judge`.
Everything in `metrics.py` is logged and rewards nothing.

**The reference pool is graded on purpose.** MAESTRO and GiantMIDI are world-class,
which is a trap: if every reference is Chopin the candidate loses every comparison, the
reward is constant, and a constant has no gradient. Mix `love` (target) with `ok`
(beatable), 70/30.

**GH200 96GB @ $1.99/hr, not H100 @ $4.29.** More memory than an H100 for under half the
price — room for 8B + LoRA + vLLM + a colocated judge, which makes judge inference free.
Check `prime availability list` before assuming; prices move and I quoted them wrong once.

## Methods worth reusing

**Search the reward's argmax offline before spending anything on GPU.** RL converges
there, so look at it first. On the counterpoint grader this caught two exploits that
would have appeared in training as a clean 1.0 reward curve next to bad music — the
worst kind of bug, because the curve looks like success. See `attic/argmax_probe.py`.

**Validate a grader against a known-good example.** The counterpoint grader was tested
against Fux's own dorian solution; when it flagged him, the rule was wrong, not Fux.
Find the equivalent positive control for any new reward component.

**Iterate on reward offline against cached rollouts.** Generate once, re-score for free.
Never rent a GPU to tune reward weights.

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
