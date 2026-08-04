"""Milestone 8 -- the 2x2 plus the random-direction control. Produces DATA.

    python m8_run.py --n 1                     # time one problem, six cells
    python m8_run.py --n 150 --layers 26-31    # the run
    python m8_run.py --tiny                    # weightless wiring check
    python m8_run.py --check-data --dataset math500   # no model, no GPU

Resumable: re-running appends only what is missing, keyed by (id, cond).

--dataset SELECTS THE DATASET, and did not always. It was accepted as a flag
and then ignored: the loader, the gold field and the fingerprint all named
GSM8K literally, so `--dataset math500` produced GSM8K problems, scored against
GSM8K golds, in a file called m8_math500_*.jsonl. The pin file was the only
thing that would have told you. Everything per-dataset now comes from
scoring.DATASETS (how to read it) and config (caps and the prompt), and
`--check-data` verifies the reading half before a GPU is involved.

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
from loaders import load_real, load_tiny

def conditions(dataset: str):
    """The six cells, as {name: (thinking, suffix, prefill, kind)}.

    `kind` is None for intact, else the hooks.make_ablation kind. Built from
    the grid so a cell cannot be omitted or misspelled.

    The direct prompt comes from config.direct_prompt(dataset) -- it used to be
    a module constant here, and three copies of the same string in the probes.
    Calling it here also means an unset instruction stops the run at argument
    time, before the model loads.
    """
    from scoring import cond_name
    suffix, prefill = config.direct_prompt(dataset)
    out = {}
    for level, (think, sfx, pre) in (
            ("direct", (False, suffix, prefill)),
            ("cot", (True, "", ""))):
        for state, kind in (("intact", None), ("ablated", "ablate"),
                            ("random", "rand_tok")):
            out[cond_name(level, state)] = (think, sfx, pre, kind)
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


def check_n(dataset, n):
    """`n` against the RECORDED split length, before any network or model.

    config.problem_ids already refuses n > dataset_size, but only once the split
    is loaded -- which on aime24 means a download and a model load to discover
    that the default n=150 was never possible against 30 problems. Checking the
    recorded row count costs nothing and fails in the right place.
    """
    import scoring
    rows = scoring.DATASETS[dataset]["rows"]
    if n > rows:
        raise SystemExit(
            f"--n {n} exceeds {dataset}'s {rows} problems. AIME has 30, so "
            f"n=150 is not meaningful there at all -- use --n {rows} and "
            f"report it as the whole dataset rather than a sample.")


def check_data(dataset, n):
    """`--check-data`. Everything about a dataset that can be wrong for free.

    Exists because the expensive failures here are all cheap to detect: a
    renamed hub repo, a problem field called `problem` instead of `question`, a
    gold format the scorer cannot parse, a cap or a prompt still unset. Every
    one of them otherwise surfaces minutes into a rented GPU.
    """
    import scoring
    check_n(dataset, n)
    ds, path, name = scoring.load_problems(dataset)
    ids_ = config.problem_ids(n, len(ds))
    field = scoring.question_field(dataset, ds[ids_[0]])
    bad = scoring.validate_gold(dataset, [ds[i] for i in ids_])
    fp = scoring.dataset_fingerprint(ds, ids_)
    print(f"{dataset}: {ds.num_rows} rows "
          f"(scoring.DATASETS says {scoring.DATASETS[dataset]['rows']})")
    print(f"  resolved mirror {path}" + (f"/{name}" if name else ""))
    print(f"  problem field   {field!r}   of {sorted(ds[ids_[0]])}")
    print(f"  sample          n={n}  ids {ids_[:5]}{'...' if n > 5 else ''}")
    print(f"  content sha256  {fp['content_sha256']}")
    gold = scoring.GOLD_FIELD[dataset](ds[ids_[0]])
    print(f"  first gold      {gold!r}")
    if bad:
        print(f"  GOLD UNPARSEABLE on {len(bad)}/{n} problems: "
              f"{[ids_[i] for i in bad[:5]]}")
    else:
        print(f"  gold parses     all {n}")
    open_items = config.dataset_ready(dataset)
    if open_items:
        print(f"  NOT RUNNABLE -- unset pre-registration choices:")
        for k in open_items:
            print(f"      {k}")
        print("  Set them in config.py and commit before generating.")
    else:
        print("  pre-registration complete for this dataset")
    if ds.num_rows != scoring.DATASETS[dataset]["rows"]:
        print(f"  WARNING: row count differs from the recorded "
              f"{scoring.DATASETS[dataset]['rows']} -- a different split or "
              f"release. Every sample id means a different problem than it did.")
    return 1 if (bad or open_items) else 0


def main(argv=None):
    import scoring

    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default=None)
    ap.add_argument("--band", default=config.PRIMARY_BAND,
                    choices=sorted(config.BANDS))
    ap.add_argument("--layers", default=None, metavar="LO-HI")
    ap.add_argument("--dataset", default="gsm8k",
                    choices=sorted(scoring.DATASETS),
                    help="selects the dataset for real. `choices` rather than "
                         "a free string because the failure mode of a typo "
                         "here used to be a silently-GSM8K run under another "
                         "dataset's filename.")
    ap.add_argument("--check-data", action="store_true",
                    help="pre-flight the dataset and exit: load it, resolve "
                         "which field holds the problem, parse every gold, "
                         "report the pre-registration items still unset. No "
                         "model, no GPU, no generation. Run this before "
                         "renting anything -- scoring.DATASETS' entries for "
                         "math500 and aime24 are published conventions this "
                         "repo has never actually observed.")
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
    if a.check_data:
        return check_data(a.dataset, a.n)
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
    # Per-dataset choices, read here for the same reason: an unset cap or
    # instruction should cost zero seconds, not one model load. conditions()
    # resolves the prompt and cap_for every cell's cap, both raising on None.
    if config.dataset_ready(a.dataset):
        raise SystemExit(
            f"{a.dataset} is not runnable -- unset pre-registration choices:\n"
            + "\n".join(f"  {k}" for k in config.dataset_ready(a.dataset))
            + f"\nSet them in config.py and commit first. "
              f"`--check-data --dataset {a.dataset}` reports this without "
              f"loading a model.")
    CONDS = conditions(a.dataset)
    for cond in CONDS:
        config.cap_for(cond, a.dataset)
    if not a.tiny:
        check_n(a.dataset, a.n)

    model, tok = (load_tiny if a.tiny else load_real)(device)
    NL = n_layers(model)
    try:
        band, nm = resolve_band(NL, config.BANDS[a.band], a.layers)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    band_name = a.band if nm == "band" else nm

    out = a.out or scoring.run_path(a.dataset, a.n, band_name)
    os.makedirs("runs", exist_ok=True)

    print(f"band={band_name} {band.start}-{band.stop - 1} (width {len(band)})"
          f"   k={K}   projection={MODE}/gain={GAIN}")
    print(f"exclusion={'ON top-' + str(KX) if USE_EX else 'OFF'}"
          f"   gate={GATE}   out={out}")
    if a.layers:
        print("WARNING: --layers is an EXPLORATORY window, not a "
              "pre-registered band. Say so in the report.")

    if a.tiny:
        ds, ids_, golds, qfield = None, list(range(min(a.n, 2))), {}, None
    else:
        spec = scoring.DATASETS[a.dataset]
        ds, path, name = scoring.load_problems(a.dataset)
        ids_ = config.problem_ids(a.n, len(ds))
        # Resolved ONCE, and before the loop: a wrong field name is then a
        # pre-flight error naming the keys that do exist, not a KeyError on the
        # first problem of the first cell with the model already resident.
        qfield = scoring.question_field(a.dataset, ds[ids_[0]])
        golds = {i: scoring.GOLD_FIELD[a.dataset](ds[i]) for i in ids_}
        bad = scoring.validate_gold(a.dataset, [ds[i] for i in ids_])
        if bad:
            raise SystemExit(
                f"{len(bad)}/{len(ids_)} gold answers do not parse "
                f"({[ids_[i] for i in bad[:5]]}...). Every one of those "
                f"problems is unscoreable in every cell, so generating them "
                f"buys nothing. Fix GOLD_FIELD[{a.dataset!r}] in scoring.py, "
                f"or check the split.")
        # Pinned BEFORE the first generation, not after the last: this run may
        # span an instance restart, and a checkpoint that resolves differently
        # between the first cell and the sixth is a confound with no signature
        # anywhere in the data.
        prov = {"model_revision": scoring.model_revision(model),
                "dataset": scoring.dataset_fingerprint(
                    ds, ids_, dataset=a.dataset, path=path, name=name,
                    split=spec["split"], question_field=qfield)}
        with open(out.replace(".jsonl", "_pin.json"), "w") as f:
            json.dump(prov, f, indent=1)
        print(f"dataset={a.dataset} rows={ds.num_rows} question={qfield!r}")
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
        cap = a.smoke_cap or config.cap_for(cond, a.dataset)
        t_cell = time.time()
        nrec = 0
        for i in ids_:
            if (i, cond) in done:
                continue
            if a.tiny:
                enc = {"input_ids": torch.randint(0, 256, (1, 12)).to(device)}
                gold = "0"
            else:
                enc = tok(scoring.render_prompt(tok, ds[i][qfield], think,
                                                suffix, prefill),
                          return_tensors="pt").to(device)
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
            rec = dict(id=i, cond=cond, dataset=a.dataset, seed=config.SEED,
                       raw=raw, gold=gold,
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
