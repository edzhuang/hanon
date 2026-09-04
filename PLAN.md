# hanon — RL a small LLM to write counterpoint by writing code

Inspired by surya.website/rling-qwen-to-paint-with-code, which RL'd a model to paint
via p5.brush sketches. Same skeleton — **prompt → model writes code → sandbox renders
artifact → score → GRPO** — but with the reward deliberately swapped from aesthetic
judgement to mechanical verification.

## Why counterpoint instead of "compose something beautiful"

The blog's hard part was the reward: subjective, judge-based, and it took them a
rebuild to get right. With a judge-based reward, a flatlined run has three suspects —
broken reward, miscalibrated reference pool, or a model too weak — and no ground truth
to tell them apart. Each hypothesis costs another GPU-day to test.

First-species counterpoint has strict, mechanical, centuries-old rules. So the grader
is ordinary code that can be **unit-tested against textbook examples**, and when a run
flatlines the reward is not one of the suspects. It is also still real music, which a
math or code benchmark would not be.

Validation that matters: the grader scores **Fux's own dorian solution at 0.92**, and
deliberately broken lines at 0.02–0.45. It earned that number — the first version
flagged Fux for a second climax on his final note, which was the grader being wrong,
not Fux.

## Shape

| Piece | File | Status |
| --- | --- | --- |
| Cantus firmus bank (8 modal CFs) | `hanon/cantus.py` | done |
| Sketch executor (subprocess, timeout, MIDI validation) | `hanon/executor.py` | done |
| First-species grader (15 rules, graded severities) | `hanon/rewards/counterpoint.py` | done |
| Prompt + end-to-end reward | `hanon/task.py` | done |
| Anti-degeneracy metrics (now observability, not reward) | `hanon/rewards/metrics.py` | done |
| verifiers v1 taskset | `hanon/taskset.py` | next |
| Baseline eval + headroom probe | `scripts/` | needs an API key |
| GRPO run (Qwen3-8B + LoRA, prime-rl) | — | after the probe |

## Reward

Collapsed from the start, learning from their nine-component rubric that plateaued
because the judges correlated 0.85–0.95:

```
reward = 0.10 * compiles + 0.90 * counterpoint_score   (× cantus-firmus-preserved gate)
counterpoint_score = exp(-Σ violation_weights / 2.5)
```

Severities are graded, not binary: a parallel fifth and a repeated note are both
"wrong", but scoring them equally throws away most of the gradient. The cantus firmus
check is a **gate**, not a deduction — a model that rewrites the cantus to fit its own
line has not written counterpoint.

Everything in `metrics.py` is logged and rewards nothing. Observability is free;
reward components are not.

## Model and budget

**Qwen3-8B + LoRA on 1×H100** (~$1.75/hr on the Prime Intellect marketplace). Not 4B:
RL sharpens what a model can already sometimes do rather than installing knowledge, and
4B is where music-theory knowledge falls off. 4B and 8B cost the same per hour anyway —
the cliff is at 32B, where you start needing multiple GPUs.

Cap: **$150**, of which ~$60 reaches a first real result and the rest is reserve for the
runs that don't work. Discipline that keeps it there:

- **Debug the reward offline.** Cache a few hundred rollouts once, re-score them for
  free. Never rent a GPU to iterate on reward weights.
- **Spend $2 before $60.** Check best-of-8 vs mean-of-8 on the base model first. No gap
  means no headroom, and no amount of GPU time fixes that.
- **Smoke-test at 20 steps** (~$1) before committing to a 12-hour run.
- **Terminate the instance.** A forgotten H100 over a weekend costs more than a
  successful run.

## Later

Aesthetic judgement is stage 3, layered on a model that already writes correct music —
a better curriculum than starting with taste. Second through fifth species, then free
counterpoint, are the natural difficulty ladder before that.
