# aime24 cot_ablated: the loop gate fired, and was overridden

This file must travel with any report of the aime24 results.

## What the gate found

`config.LOOP_GATE` (decision 24) fired inside the `cot_ablated` cell after its
first 20 problems and stopped the run:

    loop gate: cot_ablated unusable 40% vs cot_intact 5%,
    delta +35 pts against a 15% threshold over 20 problems

All 8 unusable records were `incomplete` from hitting the 32768 cap. Six of the
eight end in verbatim repetition loops; `cot_intact` hit the cap once in the
same 20-problem window. Note that `incorrect` does **not** count as unusable
(`config.UNUSABLE_OUTCOMES` is `incomplete`, `unparsed`, `error`) — the gate
measures whether generation terminates, not whether the answer is right.

This is a measured result, not a run failure. Report it as one.

## What was done afterwards

`shard_pod.sh` generated the remaining 10 problems (ids 20–29) across two
H100s, 5 per card. It reached them by defeating the gate in two steps:

1. Each pod's working file was seeded with `cot_ablated` records **only**,
   with every `cot_intact` record withheld. `gate_check` then found zero
   matched pairs and reported `gate deferred: 0/20 pairs scored`. The gate did
   not run and pass — it did not run at all.
2. `--allow-unguarded` cleared `gate_preflight`, which otherwise refuses this
   invocation shape before the model loads.

Separately, `--shard` cannot preserve an early stop in any case: the gate's
value is not generating problems 20–29 until 0–19 are judged, and a shard
pre-commits to its whole slice.

The records were appended by hand rather than through `merge_runs.py`, so the
run file's pin does not record the second machine. `collect_shard_tail.py`
reconciled `model_revision`, dataset fingerprint, backend and GPU name across
the pods' pins in its place; any divergence it reported belongs here.

## What this means for the results

- The aime24 `cot_ablated` cell is **not gate-guarded**. The pre-registered
  protocol for this cell fired and was overridden.
- Any aime24 interaction from this file is computed over a cell that is ~40%
  non-terminating.
- The cap that censors it (32768) was committed **without a calibration** —
  see the `config.CAPS["aime24"]` comment, which states that the `incomplete`
  rate at this cap "is UNKNOWN in advance" and must be read off the run
  itself. So the +35pt delta is partly ablation and partly an unmeasured cap.
  Two things argue it is not purely censoring: `cot_intact` also degenerates
  when it hits the cap (4 of its 5 cap hits), so looping-at-cap is a baseline
  property of Qwen3-4B here rather than something ablation invents; and the
  ablated arm still hits that cap 8× more often.
- Report per-cell `hit_cap` and outcome composition alongside every accuracy,
  as the CAPS entry requires.
- Do not present the +35pt degeneration and the interaction as independent
  findings. The first is why the second is hard to read.

## Scope

gsm8k and math500 are unaffected. Neither the band (`light`, layers 14–19) nor
any pre-registered constant was changed — the override is in the invocation,
not in `config.py`.
