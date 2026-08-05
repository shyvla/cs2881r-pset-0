"""Merge staged run files from parallel machines into ONE resumable file.

    python merge_runs.py podA.jsonl podB.jsonl --out runs/math500_n150_light.jsonl

WHY THIS EXISTS. `--only` stages a run's cells across invocations, and two of
them (cot_intact, cot_random) are independent, so the economical schedule
generates them on two pods at once. Each pod then holds a partial .jsonl and
its own _pin.json, and cot_ablated needs ONE file holding both cells: its
pre-registered gate (run.ablated_gate) reads the matched intact records from
the file it appends to, and a missing pair does not stop the run -- it
degrades the gate to an UNGUARDED warning. Concatenation is the easy half.
The half nothing checked is whether two files from two machines describe the
same pinned world: run.pin_guard reconciles one invocation against one file
on disk and never sees the second pin at all.

WHAT IS COMPARED across pins, and why each is a refusal:

  model_revision       different weights answering the same prompt (as in
                       pin_guard).
  dataset rows +       different problems under the same ids (as in
  content_sha256 + ids pin_guard). Deliberately NO n-extension handling:
                       extension is a resume workflow with its own guard,
                       and a merge whose inputs disagree about the sample is
                       a coordination error, not an extension.
  backend              bf16 kernels differ across backends, and the cells in
                       one file are differenced against each other (as in
                       pin_guard). No override here: --allow-device-change
                       exists on resume for a machine that died mid-run,
                       and a deliberate merge is not that.
  gpu name             STRICTER than pin_guard, which compares backends
                       only. Two CUDA cards of different architecture (a
                       4090 and a 5090) can select different kernels, so
                       "cuda" == "cuda" does not make two pods one machine.
                       --allow-gpu-mix overrides; the report must then say
                       so, exactly like a device change on resume.

Per record: duplicate (id, cond) keys are deduplicated when the records are
identical and refused when they differ -- two pods generated the same cell
differently, and choosing one silently is the confound-shaped mistake this
repo exists to refuse. Calibration records are refused outright (they live
in scoring.calib_path's separate namespace and are never run data). Every id
must sit inside the pinned sample, and dataset/seed/band must be uniform,
because a file that mixes them is not one experiment.

The merged pin keeps the FIRST input's `hardware` (pin_guard's convention:
"where this file began"), unions the hardware histories in input order, and
records the merge in `merged_from` so a later reader can see the file was
assembled rather than generated in place.

--out may name one of the inputs (everything is read before anything is
written); any OTHER existing path is refused rather than overwritten,
including a stray pin sitting beside it.

Stdlib only, on purpose: no torch, no model, so it runs on the laptop
coordinating the pods. The backend is read from the pin's recorded
`hardware.backend` (falling back to the device string's prefix for pins
written before the field existed), not re-derived from local hardware.
"""
import argparse
import json
import os


def pin_path_of(path: str) -> str:
    return path.replace(".jsonl", "_pin.json")


def load_pin(path: str) -> dict:
    pp = pin_path_of(path)
    if not os.path.exists(pp):
        raise SystemExit(
            f"refusing to merge {path}: no pin file at {pp}. A run file "
            f"without its pin cannot be attributed to weights, a sample or "
            f"a backend -- copy the _pin.json that run.write_pin wrote "
            f"beside it.")
    with open(pp) as f:
        pin = json.load(f)
    if not pin.get("hardware"):
        raise SystemExit(
            f"refusing to merge {path}: its pin records no hardware, so the "
            f"backend that generated it cannot be verified. (Pins written "
            f"before hardware recording -- the committed GSM8K file -- are "
            f"resumable in place, but a merge asserts two files share a "
            f"backend and this one cannot.)")
    return pin


def backend_of_pin(pin: dict) -> str:
    hw = pin["hardware"]
    return hw.get("backend") or hw["device"].split(":")[0]


def check_pins(paths: list[str], pins: list[dict], allow_gpu_mix: bool):
    """Every cross-pin refusal, against the first pin as the reference."""
    base, base_path = pins[0], paths[0]
    for path, pin in zip(paths[1:], pins[1:]):
        old, new = base.get("model_revision"), pin.get("model_revision")
        if old != new:
            raise SystemExit(
                f"refusing to merge:\n  {base_path}: revision {old}\n"
                f"  {path}: revision {new}\n"
                f"Different weights answering the same prompts. Pooling them "
                f"makes the interaction a difference between checkpoints.")
        a, b = base.get("dataset", {}), pin.get("dataset", {})
        for field in ("rows", "content_sha256", "ids"):
            if a.get(field) != b.get(field):
                raise SystemExit(
                    f"refusing to merge: dataset {field} differs\n"
                    f"  {base_path}: {str(a.get(field))[:60]}\n"
                    f"  {path}: {str(b.get(field))[:60]}\n"
                    f"The records denote different problems (or a different "
                    f"sample) under the same ids. If one file is an "
                    f"n-extension of the other, that is a RESUME workflow -- "
                    f"see run.pin_guard -- not a merge.")
        if backend_of_pin(base) != backend_of_pin(pin):
            raise SystemExit(
                f"refusing to merge:\n"
                f"  {base_path}: {base['hardware']['device']}\n"
                f"  {path}: {pin['hardware']['device']}\n"
                f"bf16 kernels differ across backends, so these cells were "
                f"not generated by one deterministic function. There is no "
                f"override here; --allow-device-change on resume is for a "
                f"machine that died, and a merge is a plan.")
        g1, g2 = base["hardware"].get("gpu"), pin["hardware"].get("gpu")
        if g1 != g2 and not allow_gpu_mix:
            raise SystemExit(
                f"refusing to merge:\n  {base_path}: gpu {g1}\n"
                f"  {path}: gpu {g2}\n"
                f"Same backend, different card. Different architectures can "
                f"select different kernels, and the cells in one file are "
                f"differenced against each other -- run.pin_guard does not "
                f"check this, so this script does. Pass --allow-gpu-mix if "
                f"the mixing is deliberate, and disclose it in the report.")
        t1, t2 = base["hardware"].get("torch"), pin["hardware"].get("torch")
        if t1 != t2:
            print(f"WARNING: torch differs ({base_path}: {t1}, {path}: {t2})."
                  f" Same class of risk as a card mix, one rung down; the "
                  f"merged pin records both.")


def merge_records(paths: list[str], pinned_ids) -> tuple[dict, dict]:
    """All records keyed by (id, cond), deduplicated, refused on conflict.

    Returns ({key: record} in first-seen order, {path: records contributed},
    identical-duplicate count).
    """
    ids = set(pinned_ids or [])
    records, contributed, dupes = {}, {}, 0
    uniform = {"dataset": set(), "seed": set(), "band": set()}
    for path in paths:
        contributed[path] = 0
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("calibration"):
                    raise SystemExit(
                        f"refusing to merge {path}: it holds calibration "
                        f"records. Calibration files live in a separate "
                        f"namespace (scoring.calib_path) because they are "
                        f"generated at a measurement ceiling, not at CAPS; "
                        f"they are never run data and never merged into it.")
                key = (r["id"], r["cond"])
                if key in records:
                    if records[key] != r:
                        raise SystemExit(
                            f"refusing to merge: {path} holds a DIFFERENT "
                            f"record for id={key[0]} cond={key[1]} than an "
                            f"earlier input. Two machines generated the same "
                            f"cell and disagreed; that is the coordination "
                            f"error the plan was supposed to prevent, and "
                            f"choosing one silently would bury it. Decide "
                            f"which file is the run and drop the other.")
                    dupes += 1
                    continue
                if ids and r["id"] not in ids:
                    raise SystemExit(
                        f"refusing to merge {path}: id {r['id']} is outside "
                        f"the pinned sample. Same refusal as run.py's stale-id "
                        f"scan -- this file holds problems the pin never "
                        f"sampled.")
                records[key] = r
                contributed[path] += 1
                for field in uniform:
                    uniform[field].add(r.get(field))
    for field, vals in uniform.items():
        if len(vals) > 1:
            raise SystemExit(
                f"refusing to merge: records disagree on {field}: "
                f"{sorted(map(str, vals))}. One run file is one experiment; "
                f"a mixed {field} means these inputs are not halves of the "
                f"same one.")
    return records, contributed, dupes


def merged_pin(paths: list[str], pins: list[dict], contributed: dict) -> dict:
    keep = dict(pins[0])
    hist = []
    for pin in pins:
        for hw in pin.get("hardware_history") or [pin["hardware"]]:
            if hw not in hist:
                hist.append(hw)
    keep["hardware_history"] = hist
    keep["merged_from"] = [
        {"file": os.path.basename(p),
         "device": pin["hardware"]["device"],
         "gpu": pin["hardware"].get("gpu"),
         "records": contributed[p]}
        for p, pin in zip(paths, pins)]
    return keep


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+",
                    help="two or more staged run .jsonl files, each with its "
                         "_pin.json beside it. Order matters only for ties: "
                         "the first input's pin is the merged file's origin.")
    ap.add_argument("--out", required=True,
                    help="destination .jsonl. May name one of the inputs "
                         "(everything is read before anything is written); "
                         "any other existing path is refused.")
    ap.add_argument("--allow-gpu-mix", action="store_true",
                    help="permit merging files generated on different GPU "
                         "models within the same backend. Refused by default "
                         "because different architectures can select "
                         "different kernels; when passed, the report must "
                         "disclose the mix.")
    a = ap.parse_args(argv)
    if len(a.inputs) < 2:
        ap.error("need at least two files to merge")

    out = os.path.abspath(a.out)
    ins = [os.path.abspath(p) for p in a.inputs]
    if len(set(ins)) != len(ins):
        ap.error("the same file is listed twice")
    if os.path.exists(out) and out not in ins:
        raise SystemExit(
            f"refusing to overwrite {a.out}: it already exists and is not "
            f"one of the inputs. If it is a partial run of its own, merge it "
            f"IN as an input; if it is stale, delete it yourself.")

    pins = [load_pin(p) for p in a.inputs]
    check_pins(a.inputs, pins, a.allow_gpu_mix)
    records, contributed, dupes = merge_records(
        a.inputs, pins[0].get("dataset", {}).get("ids"))
    pin = merged_pin(a.inputs, pins, contributed)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f:
        for r in records.values():
            f.write(json.dumps(r) + "\n")
    with open(pin_path_of(out), "w") as f:
        json.dump(pin, f, indent=1)

    by_cond = {}
    for (_, cond) in records:
        by_cond[cond] = by_cond.get(cond, 0) + 1
    print(f"merged {len(records)} records from {len(a.inputs)} files "
          f"into {a.out}")
    for cond in sorted(by_cond):
        print(f"  {cond:16} {by_cond[cond]}")
    if dupes:
        print(f"  ({dupes} identical duplicate(s) deduplicated)")
    print(f"pin: revision {pin.get('model_revision')}, "
          f"{len(pin['hardware_history'])} hardware entr"
          f"{'y' if len(pin['hardware_history']) == 1 else 'ies'}, "
          f"origin {pin['hardware']['device']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
