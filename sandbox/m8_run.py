"""Milestone 8 -- the 2x2 plus the random-direction control. Produces DATA.

    python m8_run.py --n 1                     # time one problem, six cells
    python m8_run.py --n 150 --layers 26-31    # the run
    python m8_run.py --tiny                    # weightless wiring check

Resumable: re-running appends only what is missing, keyed by (id, cond).

SIX CELLS, from the scoring.cond_name grid rather than typed out (decision 17):

    direct_intact   direct_ablated   direct_random
    cot_intact      cot_ablated      cot_random

    interaction = (direct_intact - direct_ablated) - (cot_intact - cot_ablated)

and the same arithmetic with the `random` cells substituted for the `ablated`
ones is the control -- analysis.CELLS_RANDOM_CONTROL expresses it as a remap so
it runs through the identical pipeline instead of a second copy of it. If the
control interaction is not ~0, what we measured is broad degradation.

WHAT THIS DOES NOT DECIDE. Every pre-registered parameter comes from config.py
via require(), so nothing here can be chosen at the keyboard: the band, k, the
caps, the projection mode, the gain, the exclusion rule and the loop gate are
all fixed before this script runs and are recorded in the manifest afterwards.
`--layers` is the one exception and it prints a warning, because an exploratory
window is not a pre-registered band.

THE LOOP GATE (decision 24, settled in config.LOOP_GATE) fires between cells,
not after the whole run: the ablated CoT cell is the expensive one and the
gate's whole purpose is to stop before paying for it. It compares the
`unusable` rate -- outcome in {incomplete, unparsed, error} -- against the
matched intact cell, because cot_intact already sits near 5% and an absolute
threshold would fire on noise. distinct10 is NOT the signal; see config.py for
the measurement that disqualified it.

COST. With config.USE_EXCLUSION True every ablated generation runs a second
clean forward pass per token, so the two ablated CoT cells cost roughly 2x the
intact one plus the per-position projection. That number was never measured, so
`--n 1` exists: it prints per-cell seconds and the extrapolation to the full
run, which is the input power.py --secs wants.
"""
import argparse
import json
import os
import sys
import time

import torch

import config
from hooks import make_ablation, n_layers, resolve_band
from m7_calibration import load_real, load_tiny

DIRECT_SUFFIX = ("\n\nRespond with only the final numeric answer and nothing "
                 "else. Do not show any reasoning.")
DIRECT_PREFILL = "\\boxed{"


def conditions():
    """The six cells, as {name: (thinking, suffix, prefill, kind)}.

    `kind` is None for intact, else the hooks.make_ablation kind. Built from
    the grid so a cell cannot be omitted or misspelled.
    """
    from scoring import cond_name
    out = {}
    for level, (think, suffix, prefill) in (
            ("direct", (False, DIRECT_SUFFIX, DIRECT_PREFILL)),
            ("cot", (True, "", ""))):
        for state, kind in (("intact", None), ("ablated", "ablate"),
                            ("random", "rand_tok")):
            out[cond_name(level, state)] = (think, suffix, prefill, kind)
    return out


def gate_check(path, cells, gate):
    """Decision 24. Returns (fired, message). Reads the scored outcomes of the
    ablated cell against its matched intact cell over the first gate['n'] ids
    they share."""
    from scoring import score
    bad = set(config.UNUSABLE_OUTCOMES)
    recs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            recs.setdefault(r["cond"], {})[r["id"]] = r
    abl, base = recs.get(cells[0], {}), recs.get(cells[1], {})
    shared = sorted(set(abl) & set(base))[:gate["n"]]
    if len(shared) < gate["n"]:
        return False, f"gate deferred: {len(shared)}/{gate['n']} pairs scored"

    def rate(store):
        n = 0
        for i in shared:
            r = store[i]
            o = score(r["raw"], r["gold"], hit_cap=r["hit_cap"],
                      thinking=r["cond"].startswith("cot"))[0]
            n += o in bad
        return n / len(shared)

    ra, rb = rate(abl), rate(base)
    delta = ra - rb
    msg = (f"loop gate: {cells[0]} unusable {ra:.0%} vs {cells[1]} {rb:.0%}, "
           f"delta {delta * 100:+.0f} pts against a {gate['threshold']:.0%} "
           f"threshold over {len(shared)} problems")
    return delta > gate["threshold"], msg


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default=None)
    ap.add_argument("--band", default=config.PRIMARY_BAND,
                    choices=sorted(config.BANDS))
    ap.add_argument("--layers", default=None, metavar="LO-HI")
    ap.add_argument("--dataset", default="gsm8k")
    ap.add_argument("--only", default=None,
                    help="restrict to a level ('direct'|'cot') or a "
                         "comma-separated list of cell names. The file is "
                         "resumable and keyed by (id, cond), so the remaining "
                         "cells can be appended later without redoing these. "
                         "Exists because the three direct cells cost ~9 s a "
                         "problem against ~980 for the three cot cells, and "
                         "the direct arm is what sizes the decision to pay "
                         "for the cot arm at all.")
    ap.add_argument("--smoke-cap", type=int, default=None,
                    help="--tiny only: shrink max_new_tokens so the wiring "
                         "check is fast. Refused on the real model, because a "
                         "cap is a pre-registered parameter (config.CAPS) and "
                         "not something a flag gets to change mid-experiment.")
    a = ap.parse_args(argv)
    device = a.device or ("cpu" if a.tiny else "mps")
    if a.smoke_cap and not a.tiny:
        raise SystemExit("--smoke-cap is --tiny only; the real run's caps come "
                         "from config.CAPS")

    # Every one of these raises rather than defaulting. Reading them BEFORE
    # the model loads means a missing pre-registration choice costs zero
    # seconds instead of one cell.
    MODE, GAIN = config.projection()
    GATE = config.require("LOOP_GATE")
    USE_EX = config.require("USE_EXCLUSION")
    K, KX = config.K_ABLATE, config.EXCLUDE_TOPK

    model, tok = (load_tiny if a.tiny else load_real)(device)
    NL = n_layers(model)
    try:
        band, nm = resolve_band(NL, config.BANDS[a.band], a.layers)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    band_name = a.band if nm == "band" else nm

    out = a.out or f"runs/m8_{a.dataset}_n{a.n}_{band_name}.jsonl".replace(
        "(", "").replace(")", "")
    os.makedirs("runs", exist_ok=True)

    print(f"band={band_name} {band.start}-{band.stop - 1} (width {len(band)})"
          f"   k={K}   projection={MODE}/gain={GAIN}")
    print(f"exclusion={'ON top-' + str(KX) if USE_EX else 'OFF'}"
          f"   gate={GATE}   out={out}")
    if a.layers:
        print("WARNING: --layers is an EXPLORATORY window, not a "
              "pre-registered band. Say so in the report.")

    CONDS = conditions()
    if a.tiny:
        ds, ids_, golds = None, list(range(min(a.n, 2))), {}
    else:
        from datasets import load_dataset
        from scoring import GOLD_FIELD, dataset_fingerprint, model_revision
        ds = load_dataset("openai/gsm8k", "main", split="test")
        ids_ = config.problem_ids(a.n, len(ds))
        golds = {i: GOLD_FIELD["gsm8k"](ds[i]) for i in ids_}
        # Pinned BEFORE the first generation, not after the last: this run may
        # span an instance restart, and a checkpoint that resolves differently
        # between the first cell and the sixth is a confound with no signature
        # anywhere in the data.
        prov = {"model_revision": model_revision(model),
                "dataset": dataset_fingerprint(ds, ids_, path="openai/gsm8k",
                                               name="main", split="test")}
        with open(out.replace(".jsonl", "_pin.json"), "w") as f:
            json.dump(prov, f, indent=1)
        print(f"pinned revision {prov['model_revision']}   "
              f"content {prov['dataset']['content_sha256']}")

    done = set()
    if os.path.exists(out):
        with open(out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["id"], r["cond"]))
        stale = {i for i, _ in done} - set(ids_)
        if stale:
            raise SystemExit(f"{out} holds ids outside this sample: "
                             f"{sorted(stale)[:5]}... refusing to mix samples")
        print(f"resuming: {len(done)} records on disk")

    # Cheap cells first, and the ablated CoT cell LAST so the loop gate can
    # stop before it. Intact before intervened within a level, because the
    # gate compares against the intact rate.
    order = ["direct_intact", "direct_ablated", "direct_random",
             "cot_intact", "cot_random", "cot_ablated"]
    assert sorted(order) == sorted(CONDS), (order, sorted(CONDS))
    if a.only:
        want = set(a.only.split(","))
        order = [c for c in order if c in want or c.split("_")[0] in want]
        if not order:
            raise SystemExit(f"--only {a.only!r} matched no cells of "
                             f"{sorted(CONDS)}")
        print(f"restricted to {order} -- the interaction needs all six, so "
              f"this is a\nstaged run, not a smaller experiment. Resume "
              f"appends the rest.")

    secs = {}
    for cond in order:
        think, suffix, prefill, kind = CONDS[cond]
        cap = a.smoke_cap or config.cap_for(cond)
        t_cell = time.time()
        nrec = 0
        for i in ids_:
            if (i, cond) in done:
                continue
            if a.tiny:
                enc = {"input_ids": torch.randint(0, 256, (1, 12)).to(device)}
                gold = "0"
            else:
                from scoring import render_prompt
                enc = tok(render_prompt(tok, ds[i]["question"], think, suffix,
                                        prefill), return_tensors="pt").to(device)
                gold = golds[i]
            t0 = time.time()
            n_mod = 0
            if kind is None:
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=cap,
                                       do_sample=False)
                n_new = int(g.shape[1] - enc["input_ids"].shape[1])
                body = g[0, enc["input_ids"].shape[1]:]
            else:
                from hooks import generate_ablated
                fn = make_ablation(model, K, mode=MODE, gain_scaled=GAIN,
                                   kind=kind, seed=config.SEED)
                g, n_new, iv = generate_ablated(
                    model, enc["input_ids"], list(band), fn, cap,
                    exclude_topk=KX if USE_EX else None,
                    eos_token_id=(None if a.tiny
                                  else tok.eos_token_id))
                body = g[0, enc["input_ids"].shape[1]:]
                n_mod = iv.n_modified
            raw = prefill + ("" if a.tiny else
                             tok.decode(body, skip_special_tokens=False))
            rec = dict(id=i, cond=cond, seed=config.SEED, raw=raw, gold=gold,
                       n_tok=n_new, hit_cap=bool(n_new >= cap),
                       secs=round(time.time() - t0, 1),
                       # exposure, for the difficulty-length confound: harder
                       # problems write longer CoT and so get more ablated
                       # positions. The effect must be reported against this.
                       n_modified=n_mod, band=f"{band.start}-{band.stop - 1}")
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            nrec += 1
            print(f"{cond:16} id={i:<5} {n_new:5d} tok "
                  f"{rec['secs']:7.1f}s  mod={n_mod}", flush=True)
        secs[cond] = time.time() - t_cell
        if nrec:
            print(f"  {cond}: {nrec} problems in {secs[cond] / 60:.1f} min")

        # Gate between cells, before the most expensive one is paid for.
        if cond == "cot_random" and not a.tiny:
            fired, msg = gate_check(out, ("cot_random", "cot_intact"), GATE)
            print(f"  {msg}")
            if fired:
                print("\nGATE FIRED. Stopping before cot_ablated, as decision "
                      "24 pre-registered.\n  Revise the band or the cap; do "
                      "NOT relax the gate having seen this.")
                return 3

    print("\nper-cell seconds" + (" (this is the power.py --secs input)"
                                  if a.n == 1 else ""))
    tot = 0.0
    for cond in [c for c in order if c in secs]:
        per = secs[cond] / max(1, a.n)
        tot += per
        print(f"   {cond:16}{per:9.1f} s/problem")
    print(f"   {'TOTAL':16}{tot:9.1f} s/problem"
          f"   -> {tot * 150 / 3600:.1f} h at n=150")
    print(f"\nnext:  python power.py <accuracies> --rho <measured> "
          f"--secs {tot:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
