"""The experiment -- the 2x2 plus the random-direction control. Produces DATA.

    python run.py --n 1                     # time one problem, six cells
    python run.py --layers 26-31            # the run, at the pre-registered n
    python run.py --tiny                    # weightless wiring check
    python run.py --check-data --dataset math500   # no model, no GPU
    python run.py --calibrate-caps --dataset math500       # size its caps

Resumable: re-running appends only what is missing, keyed by (id, cond).

WHERE IT RUNS. The device is resolved once by loaders.pick_device (CUDA, then
MPS, then CPU; --device pins it), recorded in the pin file, and stamped on
every record. Greedy decoding is deterministic on a backend but NOT across
them -- bf16 kernels differ -- so resuming a file onto a different backend is
refused unless --allow-device-change says the mixing is deliberate. That is the
one confound this file's own manifest used to absorb: the pin was rewritten on
every invocation, so an arm generated elsewhere left no trace at all.

--n DEFAULTS FROM config.N_DEFAULT[dataset], and is not an argparse constant.
It used to default to 150, which is a GSM8K number derived from GSM8K
accuracies: impossible on aime24 (30 problems) and unjustified on math500. n is
a power-analysis output, so it is pre-registered per dataset like everything
else here, and a dataset whose n has not been derived yet refuses to run.

--calibrate-caps EXISTS BECAUSE THE PRE-REGISTRATION REQUIRED A MEASUREMENT
THAT NOTHING COULD TAKE. config.CAPS prescribes "run the intact cells first,
read the hit_cap rate, set these above it with headroom" -- but every route to
generating anything went through cap_for(), which raises on an unset cap, so
the prescribed step was not expressible and the realistic outcome was a guessed
cap. This mode runs the INTACT cells only, at the deliberately generous
config.MEASURE_CAP ceiling, and reports the length distribution plus a
suggested cap per level. It writes to scoring.calib_path (a separate namespace)
and stamps every record `calibration: true`, because generations made at a
measurement ceiling are not run data and analyze.py refuses to read them as if
they were.

ITS SAMPLE IS THE HARD END, from config.CALIB_SAMPLE, and --n is refused
because which problems are measured sets the cap as surely as how many. A cap
that binds is a differential bias and a cap that is loose costs only seconds on
generations that were running away regardless, so the quantity to observe is
the UPPER TAIL -- and a random draw from the run's own sample underestimates
the maximum of a larger run. On math500 the sample is level-5 problems taken
from OUTSIDE the run sample, which the split is large enough to afford; on
aime24 it is problems 11-15 of each exam, a proxy, with no holdout possible
because n=30 is the whole dataset. A contrast group of easier problems rides
along on math500 purely to check that difficulty predicts length at all, and
never enters the cap arithmetic.

--dataset SELECTS THE DATASET, and did not always. It was accepted as a flag
and then ignored: the loader, the gold field and the fingerprint all named
GSM8K literally, so `--dataset math500` produced GSM8K problems, scored against
GSM8K golds, in a file named as if it were MATH-500. The pin file was the only
thing that would have told you. Everything per-dataset now comes from
scoring.DATASETS (how to read it) and config (caps and the prompt), and
`--check-data` verifies the reading half before a GPU is involved.

SIX CELLS, from the scoring.cond_name grid rather than typed out (a name
written by hand can drift from the grid, and a drifted name flips the sign of
the headline number without erroring):

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

THE LOOP GATE (pre-registered in config.LOOP_GATE) fires INSIDE the ablated
CoT cell, after its first gate-n problems: config's decision 24 says to run
ablated CoT on a few problems and stop before paying for the rest if
generation has degenerated. It used to fire after the cot_random cell instead
-- a proxy that never observed ablated generation at all, when "targeted
ablation degenerates uniquely" is the hypothesis itself -- and a staged
`--only cot_ablated` resume skipped it entirely, because it was attached to a
cell that invocation never ran. Attached to the cell it protects, it reads
records whether they were generated this invocation or resumed from disk. It
compares the `unusable` rate -- outcome in {incomplete, unparsed, error} --
against the matched intact cell, because cot_intact already sits near 5% and
an absolute threshold would fire on noise. distinct10 is NOT the signal; see
config.py for the measurement that disqualified it. The old between-cells
check survives as a free secondary tripwire after cot_random: both cells it
reads are on disk by then, and broad degradation under random directions is
worth stopping for even though it is not what decision 24 gates.

A staged `--only cot_ablated` invocation whose gate pairs are not on disk is
refused at pre-flight (gate_preflight) rather than warned about at problem
gate-n, because the warning arrives only after the generations the gate
exists to stop are paid for. Merge cot_intact in first (merge_runs.py);
--allow-unguarded overrides and must be disclosed.

COST. With config.USE_EXCLUSION True every J-space-ablated generation runs a
second clean forward pass per token, so cot_ablated costs roughly 2x the
intact cell plus the per-position projection. The random cells skip the clean
pass -- hooks.make_ablation consumes the exclusion set for kind="ablate" only,
so paying for it there computes a set nothing reads (see exclusion_for). That
number was never measured, so `--n 1` exists: it prints per-cell seconds and
the extrapolation to the full run, which is the input power.py --secs wants.
"""
import argparse
import json
import os
import sys
import time

import torch

import config
import loaders
from hooks import make_ablation, n_layers, resolve_band
from loaders import backend_of, load_real, load_tiny, pick_device

# Cheap cells first, and the ablated CoT cell LAST: its own gate stops it
# after the first gate-n problems, and the cot_random tripwire before it needs
# that cell complete. Intact before intervened within a level, because the
# gate compares against the intact rate.
#
# NAMES ONLY, and separate from conditions() on purpose: which cells a run will
# generate has to be known BEFORE the prompts are resolved, because the
# readiness check is now scoped to the levels actually requested and
# conditions() raises on an unset direct instruction. Deriving the level set
# from conditions() would mean a cot-only staged run still had to satisfy the
# direct arm's pre-registration to discover it did not need it.
ORDER = ["direct_intact", "direct_ablated", "direct_random",
         "cot_intact", "cot_random", "cot_ablated"]


def cells_for(only: str | None, intact_only: bool = False) -> list[str]:
    """The cells this invocation will generate, in generation order."""
    available = [c for c in ORDER
                 if not intact_only or c.endswith("_intact")]
    if not only:
        return available
    want = set(only.split(","))
    order = [c for c in available if c in want or c.split("_")[0] in want]
    if not order:
        raise SystemExit(f"--only {only!r} matched no cells of {available}")
    return order


def conditions(dataset: str, levels=None):
    """The six cells, as {name: (thinking, suffix, prefill, kind)}.

    `kind` is None for intact, else the hooks.make_ablation kind. Built from
    the grid so a cell cannot be omitted or misspelled.

    The direct prompt comes from config.direct_prompt(dataset) -- it used to be
    a module constant here, and three copies of the same string in the probes.
    Calling it here also means an unset instruction stops the run at argument
    time, before the model loads.

    `levels` restricts which levels are built, and defaults to
    config.RUN_LEVELS. It exists so a cot-only run does not resolve the direct
    prompt at all: the prompt for an arm this invocation will not generate is
    not a precondition for generating the other one.
    """
    from scoring import cond_name
    levels = config.RUN_LEVELS if levels is None else levels
    specs = {"direct": lambda: (False, *config.direct_prompt(dataset)),
             "cot": lambda: (True, "", ""),
             "nothink": lambda: (False, "", "")}
    out = {}
    for level in levels:
        if level not in specs:
            raise KeyError(f"no prompt spec for level {level!r}; "
                           f"known: {sorted(specs)}")
        think, sfx, pre = specs[level]()
        for state, kind in (("intact", None), ("ablated", "ablate"),
                            ("random", "rand_tok")):
            out[cond_name(level, state)] = (think, sfx, pre, kind)
    return out


def exclusion_for(kind: str | None, use_ex: bool, kx: int) -> int | None:
    """generate_ablated's exclude_topk for one cell, or None to skip the
    paired clean pass.

    ONE place, because the decision has two parts that drifted apart once
    already: whether the pre-registration wants the exclusion rule at all
    (config.USE_EXCLUSION), and whether this cell's selection can consume it.
    hooks.make_ablation reads the exclusion set for kind="ablate" only -- the
    random kinds draw uniformly from all 151,936 unembedding rows, where a
    draw hitting a clean top-10 token is a ~0.07% event per position -- so the
    clean pass under "rand_tok" doubled one of the two most expensive cells to
    compute a set nothing looked at. Skipping it leaves the generated tokens
    bit-identical: the clean pass runs paused, on its own cache.
    """
    return kx if (use_ex and kind == "ablate") else None


def gate_index(cond: str, ids, gate, tiny: bool = False,
               calib: bool = False) -> int | None:
    """The problem id after which the pre-registered gate must run, or None.

    The id, not the loop position, so the check works identically whether the
    record was generated this invocation or resumed from disk -- which is what
    makes the gate `--only`-proof: a staged `--only cot_ablated` resume hits
    it exactly where a full run does. Attached to cot_ablated because that is
    the cell decision 24 gates; the ids are generated in sorted order, so the
    first min(gate_n, n) of them are exactly the pairs gate_check will read.
    """
    if cond != "cot_ablated" or tiny or calib or not ids:
        return None
    return ids[min(gate["n"], len(ids)) - 1]


def resolve_n(dataset: str, n: int | None) -> int:
    """`--n` when given, else the pre-registered config.N_DEFAULT[dataset].

    An explicit --n is still allowed -- --n 1 for timing and a small
    --calibrate-caps sample both need it -- but the DEFAULT is a committed
    number rather than an argparse literal, so the ordinary invocation cannot
    quietly run a GSM8K sample size against another dataset.
    """
    if n is not None:
        return n
    try:
        return config.n_default(dataset)
    except ValueError as e:
        raise SystemExit(
            f"{e}\nPass --n explicitly to override for a pilot or a timing "
            f"run; commit N_DEFAULT[{dataset!r}] before the real one.") from None


def _sample_extension(prev_ds: dict, new_ds: dict, rehash) -> bool:
    """True when the new sample EXTENDS the pinned one: the same problems,
    plus more of them.

    config.problem_ids is a shuffle prefix, so problem_ids(100) nests inside
    problem_ids(150) by construction and config promises the extension is
    free -- but the pin's content_sha256 is hashed over the sampled ids, so a
    superset hashes differently and the guard used to refuse the very
    workflow config prescribes, with no override. Nesting alone is not
    enough, though: the ids could nest while the split changed underneath
    them. `rehash` recomputes the content hash over the PINNED ids from the
    dataset as loaded now, and only a match proves the records on disk denote
    the same problems they did when written.
    """
    old_ids, new_ids = prev_ds.get("ids"), new_ds.get("ids")
    if not (rehash and old_ids and new_ids):
        return False
    if not set(old_ids) < set(new_ids):
        return False
    return rehash(old_ids) == prev_ds.get("content_sha256")


def pin_guard(prev, prov, allow_device_change=False, n_done=0, out="the file",
              rehash=None):
    """Reconcile the pin already on disk with this invocation's, or refuse.

    Returns the pin dict to write.

    THE BUG THIS CLOSES. The pin was rewritten unconditionally at the top of
    every invocation, and the resume scan ran after it. So generating the
    direct arm on a Mac and resuming the cot arm on a rented L4 -- exactly the
    workflow `--only` exists to support -- left a file whose manifest said
    "cuda" and whose MPS half had no signature anywhere. The confound the
    manifest was written to catch was the one it silently absorbed.

    WHAT IS COMPARED, and why each is a refusal rather than a warning:

      model_revision  Different weights answering the same prompt. Pinned in a
                      commit now (loaders.MODEL_REVISION), so a mismatch here
                      means the pin moved mid-file.
      dataset         rows and content_sha256. Different problems under the
                      same ids. `fingerprint` is deliberately NOT compared: it
                      is the datasets library's private hash and can differ
                      across library versions and machines, which is precisely
                      the situation a resume on a new box is in.

                      ONE content mismatch is legitimate and accepted: an
                      n-EXTENSION, where the pinned ids nest inside the new
                      sample and their rows still hash identically (see
                      _sample_extension). The workflow is to copy the .jsonl
                      AND its _pin.json to the larger n's run_path, then
                      resume there -- the filename keeps telling the truth
                      about n, and nothing already generated is regenerated.
                      The pin records each extension in `sample_history`.
      hardware        The BACKEND, per loaders.backend_of. bf16 kernels differ
                      between MPS and CUDA, so greedy decoding is deterministic
                      on a backend and not across them.

    Only the backend check takes an override, because it is the only one with a
    legitimate use: a laptop that died mid-run, and a deliberate decision to
    finish elsewhere and say so in the report. The other two have no reading
    under which pooling is correct.

    A pin with no `hardware` key predates hardware recording. That is not a
    mismatch and must not refuse -- the committed GSM8K run is such a file --
    but it cannot be verified either, so it says so and does not pretend the
    existing rows were generated here.
    """
    keep = dict(prov)
    now = prov["hardware"]
    # Carried forward on every resume, extension or not: dict(prov) starts
    # from this invocation's pin, and the history of past extensions lives
    # only in the file being replaced.
    if prev.get("sample_history"):
        keep["sample_history"] = list(prev["sample_history"])

    old_rev, new_rev = prev.get("model_revision"), prov.get("model_revision")
    if old_rev and new_rev and old_rev != new_rev:
        raise SystemExit(
            f"refusing to resume {out}:\n"
            f"  pinned checkpoint: {old_rev}\n"
            f"  this invocation:   {new_rev}\n"
            f"The {n_done} records already on disk were generated by different "
            f"weights than these. Pooling them makes the interaction a "
            f"difference between checkpoints as much as between conditions. "
            f"Write to a new --out, or check loaders.MODEL_REVISION against "
            f"the commit that produced this file.")

    old_ds, new_ds = prev.get("dataset", {}), prov.get("dataset", {})
    for field in ("rows", "content_sha256"):
        old, new = old_ds.get(field), new_ds.get(field)
        if old is not None and new is not None and old != new:
            if field == "content_sha256" and _sample_extension(old_ds, new_ds,
                                                               rehash):
                print(
                    f"NOTE: extending {out} from n={len(old_ds['ids'])} to "
                    f"n={len(new_ds['ids'])}. The pinned ids nest inside this "
                    f"sample and their rows hash identically, so the {n_done} "
                    f"records on disk denote the same problems. Recorded in "
                    f"the pin's sample_history.\n"
                    f"  If a calibration sample was drawn DISJOINT from the "
                    f"run sample (config.CALIB_SAMPLE), that claim was "
                    f"verified at the old n -- recompute the overlap and "
                    f"report it.")
                keep["sample_history"] = (keep.get("sample_history") or []) + \
                    [{"n": len(old_ds["ids"]), "content_sha256": old}]
                continue
            hint = ""
            if field == "content_sha256":
                hint = ("\nAn n-EXTENSION -- the same problems, more of them "
                        "-- is accepted automatically when the pinned ids "
                        "nest inside the new sample and still hash the same: "
                        "copy the .jsonl and its _pin.json to the larger n's "
                        "run_path and resume there.")
            raise SystemExit(
                f"refusing to resume {out}:\n"
                f"  pinned dataset {field}: {old}\n"
                f"  this invocation:        {new}\n"
                f"The sample ids in the {n_done} records on disk denote "
                f"different problems than they would now. Write to a new "
                f"--out." + hint)

    was = prev.get("hardware")
    if not was:
        # Pins written before hardware was recorded at all.
        print(f"NOTE: {out} was pinned without a hardware record, so the "
              f"{n_done} records already on disk cannot be attributed to a "
              f"backend. Recording {now['device']} for this invocation; new "
              f"records carry it individually.")
        keep["hardware"] = now
        keep["hardware_history"] = [now]
        return keep

    # `hardware` stays as FIRST pinned, never overwritten -- that is the field
    # a later reader treats as "where this file began". Everything since is in
    # the history.
    keep["hardware"] = was
    hist = list(prev.get("hardware_history") or [was])
    if backend_of(was["device"]) != backend_of(now["device"]):
        if not allow_device_change:
            raise SystemExit(
                f"refusing to resume {out}:\n"
                f"  pinned hardware: {was['device']}"
                f" (torch {was.get('torch')}"
                + (f", {was['gpu']}" if was.get("gpu") else "") + ")\n"
                f"  this invocation: {now['device']}"
                f" (torch {now.get('torch')}"
                + (f", {now['gpu']}" if now.get("gpu") else "") + ")\n"
                f"Greedy decoding is deterministic on a backend, not across "
                f"them: bf16 kernels differ, so the same prompt can decode to "
                f"different text. The {n_done} records already on disk were "
                f"generated on {backend_of(was['device'])}, and this run's "
                f"headline number is a difference of differences across those "
                f"cells.\n"
                f"Pass --allow-device-change to proceed anyway -- the pin will "
                f"record both and every record carries its own device -- or "
                f"write to a new --out and keep the arms on one machine.")
        print(f"WARNING: continuing {out} on {now['device']}; the "
              f"{n_done} records on disk were generated on {was['device']}. "
              f"This file now spans two backends and the report must say so.")
    if hist[-1] != now:
        hist.append(now)
    keep["hardware_history"] = hist
    return keep


def write_pin(out, prov, allow_device_change=False, n_done=0, rehash=None):
    """Reconcile this invocation's pin with the one on disk, and write it.

    Returns what was written. Split out of main so the branch that matters can
    be tested without a model: `n_done` and not merely "a pin file exists" is
    what makes this a resume. A pin beside an empty or absent generations file
    describes nothing yet -- an aborted start, or a run whose output was
    deleted -- and re-pinning it is correct. A pin beside 300 records is a
    claim about how those records were made.
    """
    pin_path = out.replace(".jsonl", "_pin.json")
    if n_done and os.path.exists(pin_path):
        with open(pin_path) as f:
            prov = pin_guard(json.load(f), prov, allow_device_change,
                             n_done=n_done, out=out, rehash=rehash)
    else:
        prov = {**prov, "hardware_history": [prov["hardware"]]}
    with open(pin_path, "w") as f:
        json.dump(prov, f, indent=1)
    return prov


def resume_scan(out, calib: bool, measure_cap=None,
                check_ceiling: bool = True) -> set:
    """The records already on disk, as {(id, cond)}, or refuse to resume.

    Split out of main for the same reason write_pin was: the refusals here are
    the cheap protection against expensive mistakes, and a branch that can only
    be reached with a model resident is a branch that gets tested never.

    Two refusals:

      * STAMP MISMATCH. A calibration file and a run file are different
        experiments -- different caps, and one of them has no intervention at
        all. Resuming across the boundary would silently mix them, which is
        the confound calib_path's separate namespace exists to prevent, so the
        stamp is checked too and not just the name.

      * STALE CEILING, calibration only. Raising config.MEASURE_CAP is the
        prescribed response to a censored measurement -- but resume is keyed
        by (id, cond) and will never regenerate what is already on disk. So a
        re-run against records measured at the old ceiling either reports the
        old censored distribution forever (nothing missing, nothing generated,
        cap_report's advice loops) or generates only the missing records and
        pools two ceilings into one "distribution". cap_report's MIXED
        CEILINGS branch catches the second case, but only after the GPU
        seconds are spent; this refuses both cases before any are.

    `measure_cap` defaults to config.MEASURE_CAP and is a parameter only so
    the refusal can be tested without editing config. `check_ceiling=False`
    disables the second refusal ALONE -- `--smoke-cap` shrinks a tiny
    calibration's ceiling on purpose, and the guard would misread that
    throwaway file as a censored measurement. The stamp check is not
    switchable: a tiny file that mixes calibration and run records is exactly
    as mixed as a real one.
    """
    caps = config.MEASURE_CAP if measure_cap is None else measure_cap
    done, old = set(), {}
    with open(out) as f:
        for line in f:
            r = json.loads(line)
            done.add((r["id"], r["cond"]))
            if bool(r.get("calibration")) != calib:
                raise SystemExit(
                    f"{out} holds "
                    f"{'calibration' if r.get('calibration') else 'run'} "
                    f"records but this is a "
                    f"{'calibration' if calib else 'run'} invocation. "
                    f"They are generated at different caps and must not "
                    f"be pooled -- write elsewhere or delete the file.")
            if calib and check_ceiling:
                level = r["cond"].split("_")[0]
                ceiling = caps.get(level)
                if ceiling and (r.get("cap") or 0) < ceiling:
                    old.setdefault(level, set()).add(r.get("cap"))
    if old:
        raise SystemExit(
            f"refusing to resume {out}: it holds calibration records "
            f"measured at a LOWER\nceiling than the current "
            f"config.MEASURE_CAP -- "
            + "; ".join(f"{lvl} at {sorted(cs)} vs {caps[lvl]} now"
                        for lvl, cs in sorted(old.items()))
            + f".\nResume is keyed by (id, cond) and will never regenerate "
              f"them, so re-running\nhere would report the old censored "
              f"distribution again -- or, if any records\nare missing, pool "
              f"two ceilings into one distribution. Delete the stale\n"
              f"records (or the file and its _pin.json) and re-measure at "
              f"one ceiling.")
    return done


GATE_MIN_N = 5


def gate_check(path, cells, gate, n_run=None):
    """The loop gate. Returns (fired, message). Reads the scored outcomes of the
    ablated cell against its matched intact cell over the first gate['n'] ids
    they share.

    The gate is FREE: both cells it reads are already generated by the time it
    runs, so it costs no extra generation however large gate['n'] is. What it
    cannot do is read more pairs than the run has.

    SCALED TO THE RUN. gate['n'] is 20, and on a run smaller than that the gate
    used to defer -- i.e. report nothing and let the most expensive cell proceed
    unguarded, which is the one situation it exists for. It now runs at
    min(gate['n'], n_run) and says so, because a coarse tripwire is worth more
    than an absent one. Below GATE_MIN_N even that is noise, so it declines
    explicitly rather than firing on one problem.
    """
    from scoring import score, thinking_of
    bad = set(config.UNUSABLE_OUTCOMES)
    recs = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            recs.setdefault(r["cond"], {})[r["id"]] = r
    abl, base = recs.get(cells[0], {}), recs.get(cells[1], {})
    want = gate["n"] if n_run is None else min(gate["n"], n_run)
    if want < GATE_MIN_N:
        return False, (f"gate DECLINED: n={n_run} is below GATE_MIN_N="
                       f"{GATE_MIN_N}. cot_ablated runs UNGUARDED -- read its "
                       f"outcome composition yourself before trusting it.")
    shared = sorted(set(abl) & set(base))[:want]
    if len(shared) < want:
        return False, f"gate deferred: {len(shared)}/{want} pairs scored"
    if want < gate["n"]:
        print(f"  NOTE: gate running at n={want}, not its pre-registered "
              f"{gate['n']} -- the run is smaller. One problem is "
              f"{100 / want:.0f} pts here, so the {gate['threshold']:.0%} "
              f"threshold is {gate['threshold'] * want:.1f} problems.")

    def rate(store):
        n = 0
        for i in shared:
            r = store[i]
            o = score(r["raw"], r["gold"], hit_cap=r["hit_cap"],
                      thinking=thinking_of(r["cond"]))[0]
            n += o in bad
        return n / len(shared)

    ra, rb = rate(abl), rate(base)
    delta = ra - rb
    msg = (f"loop gate: {cells[0]} unusable {ra:.0%} vs {cells[1]} {rb:.0%}, "
           f"delta {delta * 100:+.0f} pts against a {gate['threshold']:.0%} "
           f"threshold over {len(shared)} problems")
    return delta > gate["threshold"], msg


def ablated_gate(path, gate, n_run, n_left):
    """Decision 24, executed: the gate over the first gate-n cot_ablated
    problems, against their matched cot_intact records. Returns True when
    the run must stop before generating the rest of the cell.

    `n_left` is how many cot_ablated problems are not yet on disk -- what
    firing actually saves. It is printed rather than assumed because the gate
    also runs on resume: over a completed cell it can no longer prevent
    anything, and the message must not claim it did.
    """
    fired, msg = gate_check(path, ("cot_ablated", "cot_intact"), gate,
                            n_run=n_run)
    print(f"  {msg}")
    if fired:
        print(f"\nGATE FIRED, as decision 24 pre-registered: ablated CoT has "
              f"degenerated against\n  its intact baseline. Stopping with "
              f"{n_left} cot_ablated problem(s) ungenerated;\n  the "
              f"gate-sample records stay on disk. Revise the band or the "
              f"cap; do NOT\n  relax the gate having seen this.")
    elif "deferred" in msg:
        print("  WARNING: the gate could not run -- cot_intact is not on "
              "disk for these ids,\n  so cot_ablated proceeds UNGUARDED. "
              "Generate cot_intact first for a guarded run.")
    return fired


def gate_preflight(order, done, ids, gate, n, allow_unguarded=False,
                   tiny=False, calib=False):
    """Refuse the staged invocation decision 24 cannot guard, BEFORE it
    generates anything.

    ablated_gate already notices the missing-pairs case -- but at problem
    gate-n, AFTER the gate-window generations are paid for, and only as a
    warning: a staged `--only cot_ablated` against a file with no cot_intact
    printed UNGUARDED and then generated the most expensive cell anyway. On
    one machine ORDER makes that sequence impossible; on the staged multi-pod
    workflow (arms generated in parallel, merged, then cot_ablated -- see
    merge_runs.py) it is one forgotten merge away, and the price of the
    mistake is the 2x-per-token cell the gate exists to stop. A refusal here
    costs a model load; the warning it replaces costs the gate window.

    Refuses iff this invocation will generate cot_ablated records while the
    gate's intact pairs are neither on disk nor scheduled earlier in this
    same invocation. It never second-guesses the gate's own pre-registered
    judgements: below GATE_MIN_N the gate DECLINES by design and this stays
    out of it; --tiny and calibration have no gate at all; and a cell already
    complete on disk has nothing left to protect (the in-loop gate still
    reads it and can still return 3). --allow-unguarded proceeds with the
    same UNGUARDED language the loop prints, and the report must say so.
    """
    if tiny or calib or "cot_ablated" not in order:
        return
    if "cot_intact" in order:
        return  # ORDER puts intact first, so the pairs exist by gate time
    want = min(gate["n"], n)
    if want < GATE_MIN_N:
        return
    if all((i, "cot_ablated") in done for i in ids):
        return
    missing = [i for i in ids[:want] if (i, "cot_intact") not in done]
    if not missing:
        return
    if allow_unguarded:
        print(f"WARNING: --allow-unguarded -- cot_ablated generates "
              f"UNGUARDED, with {len(missing)}/{want} of its gate pairs "
              f"absent, so decision 24 cannot run. The report must disclose "
              f"this.")
        return
    raise SystemExit(
        f"refusing to generate cot_ablated UNGUARDED: cot_intact is missing "
        f"for {len(missing)} of the first {want} problems "
        f"({missing[:5]}{'...' if len(missing) > 5 else ''}), so the "
        f"pre-registered gate (decision 24) could not run until after the "
        f"generations it exists to stop were already paid for.\n"
        f"Generate cot_intact into this file first -- on the staged "
        f"multi-pod workflow that is\n"
        f"    python merge_runs.py <pod files...> --out <this file>\n"
        f"before the cot_ablated invocation -- or pass --allow-unguarded to "
        f"proceed anyway and say so in the report.")


def cap_report(path, dataset):
    """`--calibrate-caps`' actual output: what CAPS entries to commit.

    Reads the calibration file back rather than accumulating in memory, so a
    resumed calibration reports over everything on disk and not just this
    invocation's new records.

    Reports, per level:
      n_tok quantiles   the distribution the cap has to clear.
      hit-rate at the ceiling. THE LOAD-BEARING NUMBER. Anything above zero
                        means the distribution is CENSORED -- the true tail is
                        longer than observed, so the suggestion is a lower
                        bound and MEASURE_CAP itself must be raised first.
      unusable rate     intact generations that did not produce a parseable
                        answer. A cap sized on generations that never finished
                        answering is sized on the wrong thing.
    """
    import scoring
    by = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            by.setdefault(r["cond"].split("_")[0], []).append(r)

    bad = set(config.UNUSABLE_OUTCOMES)

    def median(xs):
        s = sorted(xs)
        return s[len(s) // 2] if s else 0

    def unusable_rate(recs):
        return sum(
            scoring.score(r["raw"], r["gold"], hit_cap=r["hit_cap"],
                          thinking=scoring.thinking_of(r["cond"]))[0] in bad
            for r in recs) / len(recs)

    print(f"\n{'=' * 72}\nCAP CALIBRATION -- {dataset}\n{'=' * 72}")
    print("CAP SAMPLE ONLY -- the contrast group is excluded from every number "
          "here.\nIt is easier by construction and would drag the percentile "
          "down.\n")
    print(f"{'level':9}{'n':>4}{'p50':>7}{'p90':>7}{'p99':>7}{'max':>7}"
          f"{'ceiling':>9}{'hit':>6}{'unusable':>10}   suggest")
    censored, suggestions, empty, mixed = [], {}, [], []
    for level in sorted(by):
        recs = [r for r in by[level] if r.get("calib_role") != "contrast"]
        if not recs:                      # contrast-only level: nothing to size
            empty.append(level)
            continue
        toks = sorted(r["n_tok"] for r in recs)
        # EVERY record's ceiling, not the first one's. Raising MEASURE_CAP and
        # re-running is the prescribed response to a censored measurement, and
        # resume is keyed by (id, cond) -- so a file can end up holding records
        # generated at two different ceilings, whose n_tok values are not one
        # distribution. Reading recs[0] reported whichever came first and
        # pooled them silently, which is the same failure as pooling two
        # backends.
        ceilings = sorted({r.get("cap", config.MEASURE_CAP.get(level))
                           for r in recs})
        if len(ceilings) > 1:
            mixed.append((level, ceilings))
        ceiling = max(ceilings)
        q = lambda p: toks[min(len(toks) - 1, int(p * len(toks)))]
        hit = sum(r["hit_cap"] for r in recs) / len(recs)
        sug = config.suggest_cap(toks)
        suggestions[level] = sug
        if hit:
            censored.append((level, hit))
        print(f"{level:9}{len(toks):>4}{q(0.5):>7}{q(0.9):>7}{q(0.99):>7}"
              f"{toks[-1]:>7}{ceiling:>9}{hit:>6.0%}"
              f"{unusable_rate(recs):>10.0%}   {sug}")

    # ------------------------------------------- does difficulty predict length
    #
    # The sampling rule in config.CALIB_SAMPLE rests entirely on the premise
    # that harder problems write longer traces. That premise is cheap to check
    # from data already in hand, and if it is false the cap was sized on the
    # wrong tail -- so it is checked here rather than assumed in a comment.
    broken = []
    print(f"\n{'level':9}{'difficulty':>11}{'role':>10}{'n':>4}"
          f"{'median':>9}{'max':>8}")
    for level in sorted(by):
        strata = {}
        for r in by[level]:
            strata.setdefault((r.get("difficulty"), r.get("calib_role")),
                              []).append(r["n_tok"])
        for (d, rl), toks in sorted(strata.items(),
                                    key=lambda kv: (kv[0][0] is None,
                                                    kv[0][0] or 0)):
            print(f"{level:9}{str(d):>11}{str(rl):>10}{len(toks):>4}"
                  f"{median(toks):>9}{max(toks):>8}")
        # Two ways the premise can fail, both reported the same way.
        capd = [(r.get("difficulty"), r["n_tok"]) for r in by[level]
                if r.get("calib_role") != "contrast"
                and r.get("difficulty") is not None]
        contrast = [r["n_tok"] for r in by[level]
                    if r.get("calib_role") == "contrast"]
        if contrast and capd:
            if median(contrast) >= median(t for _, t in capd):
                broken.append(f"{level}: the EASIER contrast group writes as "
                              f"much as the hard cap sample "
                              f"({median(contrast)} vs "
                              f"{median(t for _, t in capd)} tokens)")
        levels_present = sorted({d for d, _ in capd})
        if len(levels_present) >= 2:
            lo = median([t for d, t in capd if d == levels_present[0]])
            hi = median([t for d, t in capd if d == levels_present[-1]])
            if hi < lo:
                broken.append(
                    f"{level}: within the cap sample, difficulty "
                    f"{levels_present[-1]} is SHORTER than difficulty "
                    f"{levels_present[0]} ({hi} vs {lo} tokens)")

    if broken:
        print("\nPREMISE CHECK FAILED -- difficulty does not predict length "
              "here:")
        for b in broken:
            print(f"  {b}")
        print("  config.CALIB_SAMPLE samples the hard end precisely because "
              "hard problems\n  are assumed to write longest. Where that is "
              "false, this sample is not the\n  upper tail and the suggestion "
              "below is not conservative. Re-draw the\n  calibration on a "
              "sample that IS the tail before committing anything.")
    else:
        print("\npremise check: difficulty predicts length as assumed "
              "(or there is only\none stratum, in which case nothing is "
              "claimed either way).")
    if empty:
        print(f"\nNOTE: {empty} had contrast records only -- no cap sized.")

    print(f"\nsuggestion = {config.CAP_HEADROOM}x p99 of the CAP SAMPLE, "
          f"rounded up to a multiple\nof {config.CAP_ROUNDING} "
          f"(config.suggest_cap)")
    if mixed:
        print("\nMIXED CEILINGS -- this file holds records generated at more "
              "than one\nmax_new_tokens, so its n_tok values are not one "
              "distribution:")
        for level, cs in mixed:
            print(f"  {level}: {cs}")
        print("  Raising MEASURE_CAP and re-running is the right response to "
              "censoring, but\n  resume is keyed by (id, cond) and will not "
              "regenerate what is already on\n  disk. Delete the file and "
              "re-measure at one ceiling.")
        return 1
    if broken:
        return 1
    if censored:
        # WHICH instruction gets printed is decided by the pre-registered
        # stopping rule, not by this function: config.CEILING_RETRY_MAX_HIT
        # was committed before the re-measurement precisely so that the
        # response to censoring is not a judgement made on the outcome. This
        # is the message the operator follows, so printing "raise it once
        # more" above the threshold would re-open the decision the rule
        # closed.
        rule = config.CEILING_RETRY_MAX_HIT
        raise_first = sorted(l for l, h in censored if h <= rule)
        stop = sorted((l, h) for l, h in censored if h > rule)
        if raise_first:
            print(f"\nDO NOT COMMIT THESE. {raise_first} hit the ceiling, so "
                  f"the length\ndistribution is censored and every number "
                  f"above is a lower bound. Raise\nconfig.MEASURE_CAP for "
                  f"those levels and re-run this calibration first.")
        for level, h in stop:
            print(f"\nSTOPPING RULE (config.CEILING_RETRY_MAX_HIT): {level} "
                  f"hit the ceiling on {h:.0%}\nof the cap sample, over the "
                  f"{rule:.0%} limit. DO NOT RAISE MEASURE_CAP AGAIN.\n"
                  f"The finding is that the model does not reliably terminate "
                  f"on this dataset --\na property of the model and the "
                  f"benchmark, not a measurement a bigger\nceiling will fix. "
                  f"Set the {level} cap from what the run can afford, commit "
                  f"it as\na budget decision rather than a measurement, and "
                  f"report the per-cell\n`incomplete` rate as a stated "
                  f"limitation of the design. The suggestion\nabove is a "
                  f"lower bound and is NOT the number to commit.")
        return 1
    print("\nTo commit, edit config.CAPS in a reviewable diff:")
    print(f"    CAPS[{dataset!r}] = {{"
          + ", ".join(f"{k!r}: {v}" for k, v in sorted(suggestions.items()))
          + ", ...}")
    print("Nothing here writes config.py. A pre-registration a script can edit "
          "is not\na pre-registration -- the commit is what makes it one.")
    return 0


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


def check_data(dataset, n, levels=None):
    """`--check-data`. Everything about a dataset that can be wrong for free.

    Exists because the expensive failures here are all cheap to detect: a
    renamed hub repo, a problem field called `problem` instead of `question`, a
    gold format the scorer cannot parse, a cap or a prompt still unset. Every
    one of them otherwise surfaces minutes into a rented GPU.

    NEVER RAISES on an unset pre-registration choice -- it reports them. That
    is the whole point of a pre-flight, and it is why an unset N_DEFAULT falls
    back to validating the WHOLE split here rather than exiting: a dataset whose
    n has not been derived yet is exactly the one whose fields and golds you
    want checked before spending time on the power analysis.
    """
    import scoring
    levels = config.RUN_LEVELS if levels is None else levels
    explicit_n = n is not None
    if explicit_n:
        check_n(dataset, n)
    ds, path, name = scoring.load_problems(dataset)
    if not explicit_n:
        try:
            n = config.n_default(dataset)
            check_n(dataset, n)
        except ValueError:
            n = ds.num_rows
            print(f"note: N_DEFAULT[{dataset!r}] is unset and --n was not "
                  f"given, so this validates the WHOLE split ({n} rows).")
    diff = ([scoring.difficulty_of(dataset, ds[i]) for i in range(ds.num_rows)]
            if config.RUN_SAMPLE.get(dataset) else None)
    ids_ = config.run_ids(dataset, n, len(ds), difficulty=diff)
    field = scoring.question_field(dataset, ds[ids_[0]])
    bad = scoring.validate_gold(dataset, [ds[i] for i in ids_])
    fp = scoring.dataset_fingerprint(ds, ids_)
    print(f"{dataset}: {ds.num_rows} rows "
          f"(scoring.DATASETS says {scoring.DATASETS[dataset]['rows']})")
    print(f"  resolved mirror {path}" + (f"/{name}" if name else ""))
    print(f"  problem field   {field!r}   of {sorted(ds[ids_[0]])}")
    print(f"  sample          n={n}  ids {ids_[:5]}{'...' if n > 5 else ''}")
    if diff is not None:
        by_level = {}
        for i in ids_:
            by_level[diff[i]] = by_level.get(diff[i], 0) + 1
        print(f"  stratified      {dict(sorted(by_level.items()))} per level "
              f"(config.RUN_SAMPLE)")
    print(f"  content sha256  {fp['content_sha256']}")
    gold = scoring.GOLD_FIELD[dataset](ds[ids_[0]])
    print(f"  first gold      {gold!r}")
    if bad:
        print(f"  GOLD UNPARSEABLE on {len(bad)}/{n} problems: "
              f"{[ids_[i] for i in bad[:5]]}")
    else:
        print(f"  gold parses     all {n}")
    open_items = config.dataset_ready(dataset, levels=levels)
    if open_items:
        print(f"  NOT RUNNABLE ({'+'.join(sorted(levels))}) -- unset "
              f"pre-registration choices:")
        for k in open_items:
            print(f"      {k}")
        print("  Set them in config.py and commit before generating.")
        if any(k.startswith("CAPS") for k in open_items):
            print(f"  For the caps, MEASURE rather than guess:\n"
                  f"      python run.py --calibrate-caps --dataset {dataset}")
        if any(k.startswith("N_DEFAULT") for k in open_items):
            print(f"  For n, run power.py at this dataset's intact accuracies "
                  f"and measured rho.")
    else:
        print(f"  pre-registration complete for {'+'.join(sorted(levels))}")
    if ds.num_rows != scoring.DATASETS[dataset]["rows"]:
        print(f"  WARNING: row count differs from the recorded "
              f"{scoring.DATASETS[dataset]['rows']} -- a different split or "
              f"release. Every sample id means a different problem than it did.")
    return 1 if (bad or open_items) else 0


def main(argv=None):
    import scoring

    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default=None,
                    help="cpu, mps, cuda or cuda:N. Defaults to the best "
                         "available (loaders.pick_device) and is validated "
                         "before the model loads, so a typo costs no download.")
    ap.add_argument("--allow-device-change", action="store_true",
                    help="permit resuming a file onto a different backend "
                         "than it was pinned on. Refused by default: bf16 "
                         "kernels differ between MPS and CUDA, so cells "
                         "differenced against each other must come from one "
                         "backend. When passed, the pin records both devices "
                         "and the report has to disclose it.")
    ap.add_argument("--allow-unguarded", action="store_true",
                    help="permit a staged `--only cot_ablated` invocation to "
                         "generate without the gate's cot_intact pairs on "
                         "disk. Refused by default: decision 24's gate would "
                         "otherwise first run AFTER the generations it "
                         "exists to stop were paid for, and only as a "
                         "warning. Merge cot_intact in first (merge_runs.py) "
                         "for a guarded run; when passed, the report must "
                         "disclose the unguarded cell.")
    ap.add_argument("--n", type=int, default=None,
                    help="problems to run. Defaults to the PRE-REGISTERED "
                         "config.N_DEFAULT[dataset] -- 150 was an argparse "
                         "literal, i.e. a GSM8K number applied to every "
                         "dataset. Pass it explicitly for a pilot, a timing "
                         "run or a calibration.")
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
    ap.add_argument("--calibrate-caps", action="store_true",
                    help="measure the caps instead of running the experiment: "
                         "INTACT cells only, at the config.MEASURE_CAP "
                         "ceiling, reporting the length distribution and a "
                         "suggested config.CAPS entry per level. Writes to "
                         "scoring.calib_path and stamps every record "
                         "`calibration: true`, so the output can never be "
                         "analysed as run data. This is the step config.CAPS "
                         "prescribes and that nothing could previously take. "
                         "The sample comes from config.CALIB_SAMPLE (the hard "
                         "end of the difficulty range), so --n is refused.")
    ap.add_argument("--smoke-cap", type=int, default=None,
                    help="--tiny only: shrink max_new_tokens so the wiring"
                         "check is fast. Refused on the real model, because a "
                         "cap is a pre-registered parameter (config.CAPS) and "
                         "not something a flag gets to change mid-experiment.")
    a = ap.parse_args(argv)

    # WHICH CELLS, resolved first: the readiness check below is scoped to the
    # levels this invocation will actually generate, so a cot-only staged run
    # is not blocked on the direct arm's prompt. Names only -- no prompt is
    # resolved yet, which is what makes that scoping possible.
    CALIB = a.calibrate_caps
    order = cells_for(a.only, intact_only=CALIB)
    levels = sorted({c.split("_")[0] for c in order})

    if a.check_data:
        return check_data(a.dataset, a.n, levels)
    device = pick_device(a.device, a.tiny)
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
    #
    # CALIBRATION EXEMPTS THE CAPS, and only the caps: it exists to measure
    # them, so requiring them would be the circularity this mode was added to
    # break. Everything else -- the prompt for any direct cell, the dataset's
    # fields and golds -- still has to be settled, because a calibration run
    # against the wrong prompt sizes a cap for a condition that will not run.
    open_items = config.dataset_ready(a.dataset, levels=levels,
                                      need_n=(a.n is None))
    if CALIB:
        open_items = [k for k in open_items if not k.startswith("CAPS")]
    if open_items and not a.tiny:
        raise SystemExit(
            f"{a.dataset} is not runnable for {'+'.join(levels)} -- unset "
            f"pre-registration choices:\n"
            + "\n".join(f"  {k}" for k in open_items)
            + f"\nSet them in config.py and commit first. "
              f"`--check-data --dataset {a.dataset}` reports this without "
              f"loading a model.")
    # AFTER the readiness report, not before: an unset N_DEFAULT is one of the
    # things dataset_ready lists, and resolving n first would exit naming only
    # that -- sending you back for a second round trip to discover the caps
    # were unset too. A pre-flight that reports one problem at a time is a
    # pre-flight you run five times.
    #
    # --tiny generates random token ids against a weightless model, so no
    # dataset and no pre-registered n are involved: it must stay runnable
    # against a dataset whose n is still undecided, since checking the wiring
    # is a prerequisite for doing anything else.
    n = (a.n if a.n is not None else 2) if a.tiny else resolve_n(a.dataset, a.n)

    # THE CALIBRATION SAMPLE IS PRE-REGISTERED, so --n has nothing to say about
    # it: which problems get measured changes the cap, exactly as much as how
    # many do. config.CALIB_SAMPLE fixes both, and n stays resolved from
    # N_DEFAULT because the math500 rule draws from OUTSIDE the run sample and
    # cannot verify that claim without knowing what the run sample is.
    n_calib = None
    if CALIB and not a.tiny:
        if a.n is not None:
            raise SystemExit(
                f"--n is not accepted with --calibrate-caps. The calibration "
                f"sample is pre-registered in config.CALIB_SAMPLE"
                f"[{a.dataset!r}] -- which problems are measured determines the "
                f"cap just as much as how many, so it is a committed choice "
                f"and not a flag.")
        if config.CALIB_SAMPLE.get(a.dataset) is None:
            raise SystemExit(
                f"no calibration sampling rule for {a.dataset!r} "
                f"(config.CALIB_SAMPLE is None). gsm8k's caps are already "
                f"committed; for anything else, add a rule and commit it "
                f"before measuring.")
        n_calib = config.calib_n(a.dataset)

    CONDS = conditions(a.dataset, levels=levels)
    assert set(order) <= set(CONDS), (order, sorted(CONDS))
    if not CALIB:
        for cond in order:
            config.cap_for(cond, a.dataset)
    if not a.tiny and not CALIB:
        check_n(a.dataset, n)

    # BEFORE the load, and flushed. Every print this script makes used to come
    # after from_pretrained, so on a fresh instance the first ~9 GB and several
    # minutes produced no output of its own and the run was indistinguishable
    # from a hang -- the one moment a user is most likely to kill it. Naming the
    # device and the cache location here also makes the usual cause of a genuine
    # stall (HF_HOME unset, so the download is going to a small root disk)
    # visible without a second terminal.
    if not a.tiny:
        print(f"loading {loaders.MODEL} @ {loaders.MODEL_REVISION[:12]} "
              f"onto {device}\n"
              f"  HF_HOME={os.environ.get('HF_HOME') or '(unset)'}"
              f"   HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE') or '0'}\n"
              f"  {loaders.cache_status()}\n"
              f"  No further output until the weights land.", flush=True)
    model, tok = (load_tiny if a.tiny else load_real)(device)
    NL = n_layers(model)
    try:
        band, nm = resolve_band(NL, config.BANDS[a.band], a.layers)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    band_name = a.band if nm == "band" else nm

    # Anchored to src/ (scoring.resolve), not to the caller's cwd. Launched
    # from the repo root, "runs/..." used to name a directory that did not
    # exist yet: the run was silently unresumable and analyze.py looked
    # somewhere else for the output. An explicit --out is honoured verbatim.
    out = scoring.resolve(
        a.out or (scoring.calib_path(a.dataset, n_calib or n) if CALIB
                  else scoring.run_path(a.dataset, n, band_name)))
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if CALIB:
        spec = config.CALIB_SAMPLE.get(a.dataset)
        print(f"CALIBRATION -- intact cells only, at the config.MEASURE_CAP "
              f"ceiling.\nNo ablation is applied and NO RUN DATA IS PRODUCED. "
              f"ceilings={ {l: config.MEASURE_CAP[l] for l in levels} }")
        if spec:
            print(f"sample: {spec['n']} problems at difficulty "
                  f"{spec['difficulty']}"
                  + (f" drawn from OUTSIDE the n={n} run sample"
                     if spec["disjoint_from_run"] else "")
                  + (f", plus {spec['contrast_n']} at {spec['contrast']} as a "
                     f"monotonicity check (NOT cap input)"
                     if spec["contrast_n"] else ""))
    else:
        print(f"band={band_name} {band.start}-{band.stop - 1} "
              f"(width {len(band)})   k={K}   projection={MODE}/gain={GAIN}")
        print(f"exclusion={'ON top-' + str(KX) if USE_EX else 'OFF'}"
              f"   gate={GATE}")
    print(f"n={n}"
          + ("" if a.n is not None or a.tiny else " (config.N_DEFAULT)")
          + f"   cells={order}   out={out}")
    if a.layers:
        print("WARNING: --layers is an EXPLORATORY window, not a "
              "pre-registered band. Say so in the report.")

    role, diffic = {}, {}
    if a.tiny:
        ds, ids_, golds, qfield = None, list(range(min(n, 2))), {}, None
    else:
        spec = scoring.DATASETS[a.dataset]
        ds, path, name = scoring.load_problems(a.dataset)
        if CALIB:
            # Difficulty read via scoring (how to read a record), the SELECTION
            # applied by config (which records). Resolved here, once, so an
            # unreadable difficulty field is an error about the sample rather
            # than a KeyError inside a lambda on problem 1.
            diff = [scoring.difficulty_of(a.dataset, ds[i])
                    for i in range(ds.num_rows)]
            try:
                sel = config.calib_ids(a.dataset, diff, run_n=n)
            except ValueError as e:
                raise SystemExit(str(e)) from None
            role = {i: "cap" for i in sel["cap"]}
            role.update({i: "contrast" for i in sel["contrast"]})
            ids_ = sorted(role)
            diffic = {i: diff[i] for i in ids_}
            print(f"cap sample {sel['cap']}")
            if sel["contrast"]:
                print(f"contrast   {sel['contrast']}")
        else:
            # config.run_ids, not problem_ids: math500's committed sample is
            # STRATIFIED (config.RUN_SAMPLE, 30 per difficulty level), so the
            # difficulty column is part of drawing the sample there, read the
            # same way the calibration branch reads it.
            diff = ([scoring.difficulty_of(a.dataset, ds[i])
                     for i in range(ds.num_rows)]
                    if config.RUN_SAMPLE.get(a.dataset) else None)
            try:
                ids_ = config.run_ids(a.dataset, n, len(ds), difficulty=diff)
            except ValueError as e:
                raise SystemExit(str(e)) from None
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
        #
        # BUILT here, WRITTEN after the resume scan below -- pin_guard needs to
        # know how many records the file already holds, and refusing to resume
        # is worth more than refusing a millisecond earlier.
        prov = {"model_revision": scoring.model_revision(model),
                # WHERE it ran. Greedy decoding is deterministic on a given
                # backend, not across them: bf16 kernels differ between MPS and
                # CUDA, so the same prompt can decode to different text. The
                # deliverable is a difference of differences, and the arms are
                # generated hours or days apart -- if one arm moves machine,
                # that lands in the headline number with no other signature in
                # the data. Same class of confound as an unpinned checkpoint,
                # and recorded for the same reason.
                "hardware": loaders.hardware(device),
                "dataset": scoring.dataset_fingerprint(
                    ds, ids_, dataset=a.dataset, path=path, name=name,
                    split=spec["split"], question_field=qfield)}
        print(f"dataset={a.dataset} rows={ds.num_rows} question={qfield!r}")

    done = set()
    if os.path.exists(out):
        done = resume_scan(out, CALIB, check_ceiling=not a.tiny)
        stale = {i for i, _ in done} - set(ids_)
        if stale:
            raise SystemExit(f"{out} holds ids outside this sample: "
                             f"{sorted(stale)[:5]}... refusing to mix samples")
        print(f"resuming: {len(done)} records on disk")

    # BEFORE the pin is written and anything generates: a staged cot_ablated
    # invocation whose gate pairs are absent is refused here, while the
    # mistake still costs nothing but a model load. See gate_preflight.
    gate_preflight(order, done, ids_, GATE, n,
                   allow_unguarded=a.allow_unguarded, tiny=a.tiny,
                   calib=CALIB)

    if not a.tiny:
        # AFTER the resume scan, so a mismatch is reported against the number
        # of records it would have contaminated -- and so that the pin on disk
        # is only replaced once this invocation is known to be a legitimate
        # continuation of the file it is appending to.
        # `rehash` re-fingerprints an arbitrary id list against the dataset
        # as loaded NOW -- what lets pin_guard verify that an n-extension's
        # nested ids still denote the rows they did when pinned.
        prov = write_pin(out, prov, a.allow_device_change, len(done),
                         rehash=lambda pinned: scoring.dataset_fingerprint(
                             ds, pinned)["content_sha256"])
        print(f"pinned revision {prov['model_revision']}   "
              f"content {prov['dataset']['content_sha256']}   "
              f"device {device}")

    if a.only and not CALIB:
        print(f"restricted to {order} -- the interaction needs all six, so "
              f"this is a\nstaged run, not a smaller experiment. Resume "
              f"appends the rest.")

    secs, counts = {}, {}
    for cond in order:
        think, suffix, prefill, kind = CONDS[cond]
        level = cond.split("_")[0]
        cap = (a.smoke_cap or (config.MEASURE_CAP[level] if CALIB
                               else config.cap_for(cond, a.dataset)))
        t_cell = time.time()
        nrec = 0
        # WHERE decision 24 runs: inside this cell, keyed by problem id
        # (gate_index), so a staged `--only cot_ablated` resume hits it
        # exactly where a full run does.
        gate_at = gate_index(cond, ids_, GATE, tiny=a.tiny, calib=CALIB)

        def gate_now(i, cond=cond, gate_at=gate_at):
            if i != gate_at:
                return False
            left = sum(1 for j in ids_ if j > i and (j, cond) not in done)
            return ablated_gate(out, GATE, n, left)

        for i in ids_:
            if (i, cond) in done:
                # The gate must see resumed records too, or a warm file
                # would sail past it.
                if gate_now(i):
                    return 3
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
                # `problem=i`: the random kinds key their draws by problem,
                # so the control averages over selections instead of
                # conditioning every problem on one shared pattern.
                fn = make_ablation(model, K, mode=MODE, gain_scaled=GAIN,
                                   kind=kind, seed=config.SEED, problem=i)
                # eos deliberately unspecified: generate_ablated falls back
                # to model.generation_config.eos_token_id, the LIST
                # [<|im_end|>, <|endoftext|>] that model.generate stops on.
                # Passing tok.eos_token_id here stopped on <|im_end|> alone,
                # so a degenerate intervened generation emitting the other
                # token ran to the cap while its intact partner terminated --
                # hit_cap and `incomplete` inflated in the intervened cells
                # only, straight into the gate and the outcome table.
                g, n_new, iv = generate_ablated(
                    model, enc["input_ids"], list(band), fn, cap,
                    exclude_topk=exclusion_for(kind, USE_EX, KX))
                body = g[0, enc["input_ids"].shape[1]:]
                n_mod = iv.n_modified
            raw = prefill + ("" if a.tiny else
                             tok.decode(body, skip_special_tokens=False))
            rec = dict(id=i, cond=cond, dataset=a.dataset, seed=config.SEED,
                       raw=raw, gold=gold,
                       # `cap` alongside hit_cap because hit_cap alone does not
                       # say WHAT was hit, and calibration records are written
                       # at a ceiling rather than at CAPS. A file whose records
                       # were made under two different caps is then visible
                       # instead of silently pooled.
                       n_tok=n_new, cap=cap, hit_cap=bool(n_new >= cap),
                       # WHERE THIS RECORD was generated, not where the file
                       # was pinned. The pin describes a file; a resumed run
                       # can span machines (--allow-device-change), and then
                       # only a per-record stamp can say which cells shared a
                       # backend. Costs ~12 bytes a row and is the only thing
                       # that makes a mixed file diagnosable after the fact.
                       device=device,
                       secs=round(time.time() - t0, 1),
                       # exposure, for the difficulty-length confound: harder
                       # problems write longer CoT and so get more ablated
                       # positions. The effect must be reported against this.
                       n_modified=n_mod, band=f"{band.start}-{band.stop - 1}",
                       # Never analysable as run data. analyze.py refuses it.
                       calibration=CALIB,
                       # Calibration only. `difficulty` is what the sample was
                       # drawn on, so the report can show length against it and
                       # the premise of the sampling rule checks itself;
                       # `calib_role` keeps the contrast group out of the cap
                       # arithmetic it must never influence.
                       difficulty=diffic.get(i), calib_role=role.get(i))
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            nrec += 1
            print(f"{cond:16} id={i:<5} {n_new:5d} tok "
                  f"{rec['secs']:7.1f}s  mod={n_mod}", flush=True)
            if gate_now(i):
                return 3
        secs[cond] = time.time() - t_cell
        counts[cond] = nrec
        if nrec:
            print(f"  {cond}: {nrec} problems in {secs[cond] / 60:.1f} min")

        # SECONDARY tripwire, not decision 24: random-direction generation
        # degenerating against intact is broad breakage upstream of the
        # hypothesis, worth stopping for before cot_ablated starts. The
        # pre-registered gate is the one inside cot_ablated (ablated_gate),
        # which observes the cell it protects rather than a proxy for it.
        # Never during calibration: no intervened cell exists.
        if cond == "cot_random" and not a.tiny and not CALIB:
            fired, msg = gate_check(out, ("cot_random", "cot_intact"), GATE,
                                    n_run=n)
            print(f"  {msg}")
            if fired:
                print("\nTRIPWIRE FIRED: generation degenerated under RANDOM "
                      "directions, so the\n  breakage is upstream of the "
                      "hypothesis. Stopping before cot_ablated. Revise\n  "
                      "the band or the cap; do NOT relax the check having "
                      "seen this.")
                return 3

    # Divided by what this invocation ACTUALLY GENERATED, not by n. Dividing by
    # n was wrong twice over: a calibration generates its own sample size (25
    # on math500, not the run's 100) and so under-reported 4x, and any RESUMED
    # run charges only the new records against the full n. Cells that generated
    # nothing are skipped rather than reported as free.
    generated = [c for c in order if counts.get(c)]
    tot = 0.0
    if not generated:
        print("\nnothing generated this invocation; no timings to report")
    else:
        print("\nper-generation seconds"
              + (" (this is the power.py --secs input)" if n == 1 else ""))
        for cond in generated:
            per = secs[cond] / counts[cond]
            tot += per
            print(f"   {cond:16}{per:9.1f} s  x{counts[cond]:<4} generations")
        skipped = [c for c in order if not counts.get(c)]
        if skipped:
            print(f"   (already on disk, not timed: {skipped})")
        # Extrapolation only for a real run: the calibration sample is a
        # different size AND a deliberately harder draw, so scaling it to the
        # run's n would forecast the run from its worst problems.
        if not CALIB and len(generated) == len(order):
            try:
                full = config.n_default(a.dataset)
            except (ValueError, KeyError):
                full = n
            print(f"   {'TOTAL':16}{tot:9.1f} s/problem"
                  f"   -> {tot * full / 3600:.1f} h at n={full}")

    if CALIB:
        return cap_report(out, a.dataset)
    print(f"\nnext:  python power.py <accuracies> --rho <measured> "
          f"--secs {tot:.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
