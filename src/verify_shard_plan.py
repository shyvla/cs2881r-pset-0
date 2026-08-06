"""Pre-flight for shard_ablated_tail.sh: prove out the plan with no GPU.

    cd src && python verify_shard_plan.py

Loads no model and writes nothing outside a scratch dir. It answers the four
questions worth answering BEFORE renting two H100s for ~1.5h:

  1. Does gate_preflight refuse the invocation without --allow-unguarded?
     (It should. That refusal is the thing the script overrides.)
  2. Does it pass WITH --allow-unguarded?
  3. Does the in-loop gate go quiet against a cot_intact-withheld shard file?
     (It should report `gate deferred: 0/20 pairs scored` -- NOT fire, and
     NOT pass. It does not run at all.)
  4. Do the two shards split the 10 pending problems 5/5?

Expected output is printed inline below each check. Anything else means the
shell script will not do what its comments claim, and you should not run it.
"""
import json
import os
import sys
import tempfile

import config
import run as R

MAIN = "runs/incoming/aime24_n30_light.jsonl"
GATE = config.LOOP_GATE
N = 30

if not os.path.exists(MAIN):
    sys.exit(f"run this from src/ -- {MAIN} not found")

work = tempfile.mkdtemp(prefix="shard_plan_")
blinded = os.path.join(work, "shard_0.jsonl")

# Exactly the seeding shard_ablated_tail.sh does: cot_ablated only.
withheld = 0
with open(MAIN) as f, open(blinded, "w") as o:
    for line in f:
        if json.loads(line)["cond"] == "cot_ablated":
            o.write(line)
        else:
            withheld += 1

ids_ = config.run_ids("aime24", N, N)
done = set()
for line in open(blinded):
    r = json.loads(line)
    done.add((r["id"], r["cond"]))

print(f"scratch: {work}")
print(f"seeded {len(done)} cot_ablated records, withheld {withheld} others")
print(f"  ablated ids on disk: {sorted(i for i, _ in done)}")

ok = True

print("\n[1] gate_preflight WITHOUT --allow-unguarded")
print("    expect: SystemExit refusing an unguarded cot_ablated")
try:
    R.gate_preflight(["cot_ablated"], done, ids_, GATE, N)
    print("    !! NOT REFUSED -- unexpected")
    ok = False
except SystemExit as e:
    print(f"    refused: {str(e).splitlines()[0]}")

print("\n[2] gate_preflight WITH --allow-unguarded")
print("    expect: a WARNING, then return")
try:
    R.gate_preflight(["cot_ablated"], done, ids_, GATE, N,
                     allow_unguarded=True)
except SystemExit as e:
    print(f"    !! REFUSED anyway: {e}")
    ok = False

print("\n[3] in-loop gate at id 19, against the BLINDED shard file")
print("    expect: fired=False, 'gate deferred: 0/20 pairs scored'")
fired, msg = R.gate_check(blinded, ("cot_ablated", "cot_intact"), GATE,
                          n_run=N)
print(f"    fired={fired} | {msg}")
if fired or "deferred" not in msg:
    print("    !! unexpected -- the shard file is not blinding the gate")
    ok = False

print("\n[3b] the same gate against the REAL file, for contrast")
print("    expect: fired=True, the +35pt delta")
fired2, msg2 = R.gate_check(MAIN, ("cot_ablated", "cot_intact"), GATE,
                            n_run=N)
print(f"    fired={fired2} | {msg2}")
if not fired2:
    print("    !! the gate no longer fires on the real file -- investigate "
          "before doing anything else")
    ok = False

print("\n[4] shard slices over the pending problems")
print("    expect: [20,22,24,26,28] and [21,23,25,27,29]")
pending = [i for i in ids_ if (i, "cot_ablated") not in done]
print(f"    pending: {pending}")
slices = [sorted(set(pending[k::2])) for k in (0, 1)]
for k, s in enumerate(slices):
    print(f"    shard {k}/2 -> {s}  ({len(s)} problems)")
if sorted(slices[0] + slices[1]) != pending or len(slices[0]) != 5:
    print("    !! slices do not partition the pending list 5/5")
    ok = False

print("\n" + ("ALL CHECKS AS EXPECTED" if ok else "SOMETHING IS OFF -- read "
                                                  "the !! lines above"))
sys.exit(0 if ok else 1)
