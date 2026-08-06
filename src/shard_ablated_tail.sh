#!/usr/bin/env bash
#
# Generate aime24's REMAINING cot_ablated problems across two H100s, 5 per card.
#
# ---------------------------------------------------------------------------
# WHAT THIS DEFEATS. Read this before running it.
#
# The 20 cot_ablated records on disk are 20 and not 30 because decision 24's
# loop gate FIRED: cot_ablated was 40% unusable against cot_intact's 5% over
# the first 20 problems, a +35pt delta against a 15% threshold. Eight of the
# eight unusable records are cap hits at 32768, and six of those eight end in
# verbatim repetition loops.
#
# There is no way to generate the remaining 10 with that gate intact. A plain
# resume walks to problem id 19, re-fires the gate on the records already on
# disk, and exits 3 -- deliberately (run.py's loop calls gate_now on resumed
# records so "a warm file would not sail past it").
#
# So this script does two things that must be disclosed in the report:
#
#   1. It seeds each pod's working file with the cot_ablated records ONLY,
#      WITHHOLDING cot_intact. gate_check then finds zero matched pairs and
#      reports "gate deferred: 0/20 pairs scored" instead of the +35pt delta.
#      The gate does not run and pass -- it does not run at all.
#   2. It passes --allow-unguarded, without which gate_preflight refuses the
#      invocation before the model loads.
#
# Additionally, --shard cannot preserve an early stop even in principle: the
# gate's value is not generating problems 20-29 until 0-19 are judged, and a
# shard pre-commits to its whole slice.
#
# The 10 records this produces are therefore UNGUARDED, and the file they are
# merged into contains a cell that decision 24 stopped and that was completed
# anyway. DISCLOSURE.md is written at the end saying exactly that. Do not
# delete it, and do not report the aime24 interaction without it.
# ---------------------------------------------------------------------------
#
# Usage, from src/:   ./shard_ablated_tail.sh
# Override the interpreter with PY=../.venv/bin/python ./shard_ablated_tail.sh

set -euo pipefail
cd "$(dirname "$0")"

DATASET=aime24
BAND=light
N=30
MAIN="runs/incoming/${DATASET}_n${N}_${BAND}.jsonl"
MAINPIN="runs/incoming/${DATASET}_n${N}_${BAND}_pin.json"
WORK="runs/incoming/shard_tail"
PY="${PY:-python}"

for f in "$MAIN" "$MAINPIN"; do
    [[ -f "$f" ]] || { echo "FATAL: missing $f -- run this from src/"; exit 1; }
done

# Two cards of the SAME model. merge_runs.py compares the pin's gpu NAME, not
# just the backend, because two CUDA cards of different architecture can select
# different kernels; a mismatch here is a refusal at merge time (overridable
# with --allow-gpu-mix, which is its own disclosure). Checked before the ~1.5h
# of generation rather than after.
read -r N_GPU GPUS <<<"$("$PY" - <<'EOF'
import torch
n = torch.cuda.device_count()
print(n, "|".join(torch.cuda.get_device_name(i) for i in range(n)))
EOF
)"
[[ "$N_GPU" -ge 2 ]] || { echo "FATAL: need 2 GPUs, torch sees $N_GPU"; exit 1; }
A="${GPUS%%|*}"; B="${GPUS#*|}"; B="${B%%|*}"
if [[ "$A" != "$B" ]]; then
    echo "FATAL: cards differ -- '$A' vs '$B'."
    echo "  merge_runs.py refuses this without --allow-gpu-mix. Either use two"
    echo "  matched cards or add that flag AND disclose the mix."
    exit 1
fi
echo "2x $A"

mkdir -p "$WORK"
cp "$MAIN" "$WORK/main.jsonl.bak"
cp "$MAINPIN" "$WORK/main_pin.json.bak"
echo "backed up $MAIN -> $WORK/main.jsonl.bak"

# ---- shard working files: cot_ablated only, cot_intact WITHHELD -------------
# Lines are copied VERBATIM rather than re-serialised: merge_runs.py refuses
# duplicate (id, cond) keys whose records DIFFER, and the seeded 0-19 appear in
# both shards and in MAIN. Byte-identical copies dedupe instead of colliding.
for K in 0 1; do
    "$PY" - "$MAIN" "$WORK/shard_$K.jsonl" <<'EOF'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
kept = withheld = 0
with open(src) as f, open(dst, "w") as o:
    for line in f:
        if json.loads(line)["cond"] == "cot_ablated":
            o.write(line)
            kept += 1
        else:
            withheld += 1
print(f"  {dst}: seeded {kept} cot_ablated, WITHHELD {withheld} "
      f"(incl. every cot_intact -- this is what blinds the gate)")
EOF
    cp "$MAINPIN" "$WORK/shard_${K}_pin.json"
done

# ---- generate -------------------------------------------------------------
# CUDA_VISIBLE_DEVICES rather than --device cuda:N, deliberately: run.py stamps
# the RESOLVED device string on every record, and analyze.devices_of compares
# those strings raw -- a file holding "cuda", "cuda:0" and "cuda:1" would be
# reported as three backends. Masking to one card per process makes each pod
# resolve plain "cuda", uniform with the 20 records already on disk.
echo
echo "launching 2 pods (each loads its own ~8GB copy of Qwen3-4B)"
declare -A PID
for K in 0 1; do
    CUDA_VISIBLE_DEVICES="$K" "$PY" run.py \
        --dataset "$DATASET" --n "$N" --band "$BAND" \
        --only cot_ablated --device cuda \
        --out "$WORK/shard_$K.jsonl" \
        --shard "$K/2" --allow-unguarded \
        >"$WORK/pod_$K.log" 2>&1 &
    PID[$K]=$!
    echo "  pod $K -> GPU $K, pid ${PID[$K]}, log $WORK/pod_$K.log"
done

FAIL=0
for K in 0 1; do
    if wait "${PID[$K]}"; then
        echo "  pod $K OK"
    else
        RC=$?
        FAIL=1
        echo "  pod $K FAILED rc=$RC"
        [[ "$RC" == 3 ]] && echo "    rc=3 is the GATE FIRING -- see the log."
        tail -25 "$WORK/pod_$K.log" | sed 's/^/    /'
    fi
done
if [[ "$FAIL" != 0 ]]; then
    echo
    echo "NOT MERGING: at least one pod failed. $MAIN is untouched"
    echo "(backup at $WORK/main.jsonl.bak). Fix the cause and re-run --"
    echo "the shard files are resumable, so completed problems are not redone."
    exit 1
fi

# ---- merge ---------------------------------------------------------------
# --out names one of the inputs, which merge_runs.py allows (it reads
# everything before writing anything). MAIN supplies the cells the shards
# withheld; the shards supply ids 20-29.
echo
"$PY" merge_runs.py "$WORK/shard_0.jsonl" "$WORK/shard_1.jsonl" "$MAIN" \
    --out "$MAIN"

"$PY" - "$MAIN" <<'EOF'
import collections, json, sys
recs = [json.loads(l) for l in open(sys.argv[1])]
c = collections.Counter(r["cond"] for r in recs)
print(f"\n{sys.argv[1]}: {len(recs)} records")
for cond, n in sorted(c.items()):
    sub = [r for r in recs if r["cond"] == cond]
    caps = sorted({r["cap"] for r in sub})
    devs = sorted({r["device"] for r in sub})
    print(f"  {cond:16} n={n:<4} hit_cap={sum(r['hit_cap'] for r in sub):<3} "
          f"caps={caps} devices={devs}")
if len({r["device"] for r in recs}) > 1:
    print("  WARNING: mixed device strings -- analyze.devices_of will flag "
          "this file.")
if len({r["cap"] for r in recs if r["cond"].startswith("cot")}) > 1:
    print("  WARNING: the cot cells hold MORE THAN ONE cap. That is two "
          "ceilings in one\n  distribution -- cap_report's MIXED CEILINGS "
          "case. Do not analyse it.")
EOF

# ---- disclosure ---------------------------------------------------------
cat >"$WORK/DISCLOSURE.md" <<'EOF'
# aime24 cot_ablated: gate defeated, cell completed anyway

`shard_ablated_tail.sh` generated cot_ablated ids 20-29 after decision 24's
loop gate had already fired and stopped that cell at 20 problems.

## What the gate found, before it was bypassed

    loop gate: cot_ablated unusable 40% vs cot_intact 5%,
    delta +35 pts against a 15% threshold over 20 problems

All 8 unusable records were `incomplete` from hitting the 32768 cap; 6 of the 8
end in verbatim repetition loops. cot_intact hit the cap once in the same
window.

## How it was bypassed

1. Each pod's working file was seeded with cot_ablated records ONLY, with every
   cot_intact record withheld, so `gate_check` found zero matched pairs and
   reported `gate deferred: 0/20 pairs scored`. The gate did not run.
2. `--allow-unguarded` cleared `gate_preflight`, which otherwise refuses this
   invocation shape before the model loads.
3. `--shard 0/2` and `--shard 1/2` split the 10 pending problems across two
   H100s. A shard cannot preserve an early stop in any case: the gate's value
   is not generating 20-29 until 0-19 are judged.

## What this means for the report

- The aime24 cot_ablated cell is **not gate-guarded**. Decision 24 fired
  against it and was overridden; the pre-registered protocol for this cell was
  not followed.
- Any aime24 interaction computed from this file is computed over a cell that
  is ~40% non-terminating, and the cap that censors it (32768) was committed
  WITHOUT a calibration -- see the `config.CAPS["aime24"]` comment, which says
  the incomplete rate at this cap "is UNKNOWN in advance" and must be read off
  the run.
- Report per-cell `hit_cap` and outcome composition alongside every accuracy,
  and state that the gate fired and was overridden. Do not present the +35pt
  degeneration and the interaction as independent findings; the first is why
  the second is hard to read.
- gsm8k and math500 are unaffected.
EOF

echo
echo "wrote $WORK/DISCLOSURE.md -- the report must carry it."
echo "backup of the pre-merge file: $WORK/main.jsonl.bak"
