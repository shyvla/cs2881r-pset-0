"""Analyse a run.py generations file. Scoring is separate from generation.

    python analyze.py                                  # default run file
    python analyze.py --file runs/archive/m8_gsm8k_n150_light.jsonl
    python analyze.py --dataset math500 --n 500        # default file for it

Generation costs hours; scoring costs milliseconds. Keeping them apart means a
change to the scorer is a diff and a re-run of THIS, not of the GPU job -- and
it means the numbers in the report can be regenerated from the repo instead of
from a shell heredoc that only ever existed in a terminal.

WHAT IT REPORTS, and why in this order:

  1. Per-cell accuracy and outcome composition. `incomplete` / `unparsed` /
     `error` counts matter as much as `correct`: the pre-registration fixes a
     sensitivity analysis where the headline counts them wrong and a secondary
     restricts to clean terminations, and a differential rate across cells
     lands straight in the interaction. The `norm` column counts answers the
     markdown fallback rescued, reported per cell for the same reason: a
     uniform rescue rate is harmless and an asymmetric one is a bias, and
     scoring.py's comment demanding that comparison was, until here, a
     comparison nothing computed.

  2. SELECTIVITY, for whichever arms are present. This is the contrast the
     paper's own control defines: does removing the top-10 J-lens directions
     cost more accuracy than removing ten RANDOM unembedding rows? Expressed
     as a difference of differences so analysis.paired_bootstrap -- already
     tested, already used for the interaction -- computes it unchanged:

         (intact - ablated) - (intact - random)  ==  random - ablated

  3. The INTERACTION, only if the cot arm exists. Reported with the random
     control interaction beside it, per analysis.CELLS_RANDOM_CONTROL: a
     control interaction that is not ~0 means broad degradation.

  4. The exploratory subset, separately. config.problem_ids(EXPLORED_N) is the
     set the band was chosen on, and it is NOT representative -- on the n=150
     GSM8K run it scores direct_intact 50% against the full sample's 26%, so
     effect sizes taken from it were inflated about tenfold. Pooling it with
     the holdout would import that inflation, so it is always printed apart.

WHICH DATASET. `problem_ids` is a shuffle prefix over the SPLIT LENGTH, so
recovering which problems were the exploratory ones needs the row count -- and
that used to be the literal 1319, i.e. GSM8K, whatever file was passed. On a
MATH-500 run (500 rows) that silently named a different set of 12 problems and
mislabelled the holdout. The row count now comes from the run's own _pin.json,
falling back to scoring.DATASETS, and a mismatch is reported rather than
absorbed.

And section 4 runs ONLY for config.EXPLORED_ON. Getting the row count right
makes problem_ids(12) name the right 12 problems for the dataset -- but on a
dataset the band was never swept on, no problems were explored at all, so
there is no exploratory subset to hold out. It is now skipped with that stated,
rather than splitting a run into two arbitrary pieces.
"""
import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

import config
import scoring
from analysis import mcnemar, paired_bootstrap
from scoring import score_detail

# The grid, from scoring -- a hand-typed copy here was ("direct", "cot"),
# which silently dropped any nothink cell from every table: the file loaded,
# the header said the cell was present, and no number for it appeared
# anywhere. The INTERACTION is a different matter: it is pre-registered over
# the outer two rungs of the externalisation axis, so that pair stays
# explicit. nothink, if present, is reported per-cell and per-arm only.
LEVELS = scoring.LEVELS
STATES = scoring.STATES
INTERACTION_ARMS = ("direct", "cot")


def load(path):
    """Score every record -> (by, errors). `thinking` follows the cell,
    because strip_think must know whether to expect a <think> block -- getting
    that wrong silently scores the trace instead of the answer. It is resolved
    by scoring.thinking_of, from the condition grid: the old startswith("cot")
    was right for the current grid and silently wrong for the archived letter
    naming ("A_cot" IS a cot cell), so an off-grid name now refuses loudly
    rather than guessing a stripping rule.

    `by` maps cond -> id -> (correct, outcome, record, detail), where `detail`
    is scoring.score_detail's dict -- carried through because `normalized` (the
    markdown rescue) has to be reported PER CELL: scoring.py's own design
    principle is that only an asymmetric rescue rate is a bias, and a check
    nothing computes is a check that does not happen.

    A record that raises during scoring becomes outcome="error" and an entry in
    `errors`, exactly as scoring.score_file does it: one malformed record must
    not take down the analysis of an hour of generation, and "error" is a
    pre-registered outcome, not an excuse to drop the row.

    REFUSES CALIBRATION FILES. `run.py --calibrate-caps` writes intact cells
    generated at the config.MEASURE_CAP ceiling rather than at the
    pre-registered cap. Those are a different condition -- max_new_tokens is
    part of the design, not an implementation detail -- so an intact cell
    borrowed from a calibration would not be comparable to the ablated cells it
    was differenced against. The separate filename makes that unlikely; the
    stamp makes it impossible.
    """
    by, errors = {}, []
    for line in open(path):
        r = json.loads(line)
        if r.get("calibration"):
            raise SystemExit(
                f"{path} holds CALIBRATION records (id={r.get('id')} "
                f"{r.get('cond')}), generated at the config.MEASURE_CAP "
                f"ceiling and not at a pre-registered cap. They are not run "
                f"data and nothing here will analyse them.")
        try:
            thinking = scoring.thinking_of(r["cond"])
        except ValueError as e:
            raise SystemExit(
                f"{path}: {e}. Files under the archived letter naming carry "
                f"conditions this grid cannot interpret; score them with the "
                f"mapping they were generated under, not this tool.") from None
        try:
            d = score_detail(r["raw"], r["gold"], hit_cap=r["hit_cap"],
                             thinking=thinking)
        except Exception as e:                       # noqa: BLE001 - reported
            d = dict(outcome="error", extracted="", normalized=False)
            errors.append((r.get("id"), r["cond"],
                           f"{type(e).__name__}: {e}"))
        by.setdefault(r["cond"], {})[r["id"]] = (
            int(d["outcome"] == "correct"), d["outcome"], r, d)
    return by, errors


def dataset_of(by, fallback):
    """Which dataset the file says it is, or `fallback`.

    run.py now stamps every record. Records written before it did carry no
    field, so absence falls back to the flag instead of guessing -- and a file
    holding two datasets is refused rather than averaged.
    """
    seen = {r.get("dataset") for v in by.values() for _, _, r, *_ in v.values()}
    seen.discard(None)
    if len(seen) > 1:
        raise SystemExit(f"this file mixes datasets {sorted(seen)}; the cells "
                         f"are not comparable and nothing here should pool "
                         f"them")
    return (seen.pop() if seen else fallback), bool(seen)


def devices_of(by):
    """Which backend generated each cell, as (report_lines, crosses_arms).

    run.py refuses to resume a file onto another backend unless
    --allow-device-change says the mixing is deliberate, and stamps every
    record with its device. Nothing read that stamp: the guard could be
    overridden at generation time and the resulting file would then analyse
    exactly like a clean one, which makes the override a decision with no
    consequences and the stamp write-only.

    WARNS, never refuses. The refusal already happened, at the point where it
    could still have been avoided for free; by the time a file is being
    analysed the GPU hours are spent, and a tool that declines to report data
    the user deliberately generated just gets worked around.

    `crosses_arms` is the case worth shouting about. Two cells of one arm on
    different backends is bad; the DIRECT arm on one backend and the COT arm on
    another puts the backend change directly into
    (direct_intact - direct_ablated) - (cot_intact - cot_ablated), because
    every term on one side of that subtraction was decoded by different
    kernels than every term on the other. That is the headline number.
    """
    per = {}
    for cond, v in by.items():
        seen = {r.get("device") for _, _, r, *_ in v.values()}
        seen.discard(None)
        per[cond] = seen
    devices = set().union(*per.values()) if per else set()
    if len(devices) <= 1:
        return [], False
    lines = [f"WARNING: this file was generated on {len(devices)} backends: "
             f"{sorted(devices)}. Greedy decoding is deterministic on a "
             f"backend, not across them."]
    for cond in sorted(per):
        if per[cond]:
            lines.append(f"   {cond:16} {', '.join(sorted(per[cond]))}"
                         + ("   <- MIXED WITHIN THE CELL"
                            if len(per[cond]) > 1 else ""))
    # Crossing is about the INTERACTION's two arms specifically: those are the
    # cells the headline subtraction spans. LEVELS would drag nothink in and
    # make `all(...)` false whenever the middle rung was (correctly) not run.
    by_arm = {lvl: set().union(*[per[c] for c in per
                                 if c.startswith(lvl + "_")] or [set()])
              for lvl in INTERACTION_ARMS}
    crosses = all(by_arm.get(l) for l in INTERACTION_ARMS) and \
        by_arm["direct"] != by_arm["cot"]
    if crosses:
        lines.append("   The two ARMS differ, so the interaction below is a "
                     "difference of\n   differences across a backend change. "
                     "Report it as such or regenerate.")
    return lines, crosses


def rows_for(path, dataset):
    """The split length, needed to reconstruct config.problem_ids.

    Preference order: the run's own _pin.json, which is authoritative because
    it records what the run actually loaded; then scoring.DATASETS, the expected
    value. A disagreement is printed rather than absorbed -- it means the sample
    ids denote different problems than this analysis assumes, which silently
    mislabels the exploratory/holdout split.
    """
    expect = scoring.DATASETS.get(dataset, {}).get("rows")
    pin = path.replace(".jsonl", "_pin.json")
    if os.path.exists(pin):
        with open(pin) as f:
            rows = json.load(f).get("dataset", {}).get("rows")
        if rows:
            if expect and rows != expect:
                print(f"WARNING: {pin} records {rows} rows but "
                      f"scoring.DATASETS[{dataset!r}] expects {expect} -- "
                      f"different split or release")
            return int(rows), pin
    if not expect:
        raise SystemExit(f"cannot determine the split length for {dataset!r}: "
                         f"no {pin}, and no 'rows' in scoring.DATASETS. "
                         f"problem_ids is a shuffle prefix over that length, "
                         f"so the exploratory subset cannot be identified "
                         f"without it.")
    return int(expect), "scoring.DATASETS"


def arm(by, level):
    """Complete cases for one level, as (ids, {state: array}). Paired by
    construction: a problem missing from any cell of the arm is dropped from
    all of them, and the count is reported."""
    cells = [f"{level}_{s}" for s in STATES if f"{level}_{s}" in by]
    if len(cells) < 2:
        return None, None, cells
    ids = sorted(set.intersection(*(set(by[c]) for c in cells)))
    return ids, {c.split("_")[1]: np.array([by[c][i][0] for i in ids], float)
                 for c in cells}, cells


def selectivity(a, b, c, label=""):
    """random - ablated, via the tested difference-of-differences estimator."""
    r = paired_bootstrap({"direct_intact": a, "direct_ablated": b,
                          "cot_intact": a, "cot_ablated": c})
    verdict = ("CI EXCLUDES zero -- the SELECTION matters"
               if (r["lo"] > 0 or r["hi"] < 0) else
               "CI includes zero -- no selectivity detected")
    wide = r["width"] > 0.25
    print(f"   {label}point {100 * r['point']:+.1f} pts   95% CI "
          f"[{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]   p={r['p']:.3f}   "
          f"width {100 * r['width']:.1f} pts")
    print(f"   {label}{verdict}")
    if wide:
        print(f"   {label}CI wider than 25 pts: this cannot distinguish 'no "
              f"effect' from\n   {label}'not enough problems'. Do not read it "
              f"as a null.")
    return r


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default=None,
                    help="explicit generations file; otherwise built from "
                         "--dataset/--n/--band by scoring.run_path, the same "
                         "function run.py names its output with")
    ap.add_argument("--dataset", default="gsm8k",
                    choices=sorted(scoring.DATASETS),
                    help="only used to locate the default file and as the "
                         "fallback for records that predate the dataset stamp; "
                         "the file's own records win")
    ap.add_argument("--n", type=int, default=None,
                    help="only used to locate the default file; defaults to "
                         "the pre-registered config.N_DEFAULT[dataset]. It was "
                         "a literal 150, so --dataset aime24 looked for a file "
                         "that can never exist (30 problems).")
    ap.add_argument("--band", default=config.PRIMARY_BAND)
    a = ap.parse_args(argv)
    if a.file:
        path = a.file
    else:
        try:
            n = a.n if a.n is not None else config.n_default(a.dataset)
        except ValueError as e:
            raise SystemExit(f"{e}\nOr pass --file / --n to name the run "
                             f"directly.") from None
        # Anchored to src/, exactly as run.py anchors the file it writes. The
        # two constructions have to agree from ANY working directory, or the
        # analysis silently reads a different run than the one just produced --
        # which is the failure run_path was centralised to prevent, one level up.
        path = scoring.resolve(scoring.run_path(a.dataset, n, a.band))
    by, errors = load(path)
    dataset, stamped = dataset_of(by, a.dataset)
    rows, rows_src = rows_for(path, dataset)
    print(f"{path}\ncells present: {sorted(by)}")
    print(f"dataset: {dataset}"
          + ("" if stamped else f" (assumed from --dataset; records carry no "
                               f"stamp)")
          + f"   split rows {rows} (from {rows_src})\n")
    dev_lines, dev_crosses = devices_of(by)
    if dev_lines:
        print("\n".join(dev_lines) + "\n")
    if errors:
        print(f"WARNING: {len(errors)} record(s) raised during scoring, "
              f"counted as outcome='error'\n(not-correct in every number "
              f"below):")
        for i, c, m in errors[:5]:
            print(f"   id={i} cond={c} {m}")
        print()
    if scoring.SLOW_PARSES:
        # A parse that times out returns empty and the cache makes that
        # permanent for the process -- and timing is machine-dependent. The
        # warning lives here because THIS is the tool the headline numbers
        # come from; score_file printing it was not enough.
        print(f"WARNING: {len(scoring.SLOW_PARSES)} parse(s) ran within 10% "
              f"of the {scoring.PARSE_KW['parsing_timeout']}s timeout. These "
              f"records may score differently on another machine.\n")

    print(f"{'cell':17}{'n':>5}{'acc':>7}{'norm':>6}   composition")
    for level in LEVELS:
        for s in STATES:
            c = f"{level}_{s}"
            if c not in by:
                continue
            v = by[c]
            comp = Counter(o for _, o, _, _ in v.values())
            acc = np.mean([x[0] for x in v.values()])
            n_norm = sum(x[3]["normalized"] for x in v.values())
            print(f"{c:17}{len(v):>5}{acc:>7.1%}{n_norm:>6}   "
                  + "  ".join(f"{k}={n}" for k, n in comp.most_common()))

    # The rescue rate, compared across cells -- the comparison scoring.py's
    # design principle demands and nothing previously computed. No threshold:
    # what counts as "similar" is a judgment call, so the spread is surfaced
    # and the judging left to the reader.
    n_norm_total = sum(x[3]["normalized"]
                       for v in by.values() for x in v.values())
    if n_norm_total:
        rates = {c: sum(x[3]["normalized"] for x in v.values()) / len(v)
                 for c, v in by.items()}
        hi = max(rates, key=rates.get)
        lo = min(rates, key=rates.get)
        print(f"\nmarkdown rescue (the norm column) fired on {n_norm_total} "
              f"record(s). The rescue is safe\nonly if its rate is similar "
              f"across cells -- an asymmetric rate is a differential\nbias "
              f"(see scoring.unwrap_markdown). Spread: {rates[hi]:.1%} ({hi}) "
              f"vs {rates[lo]:.1%} ({lo}).")

    arms = {}
    for level in LEVELS:
        ids, m, cells = arm(by, level)
        if m is None:
            # One cell present is worth saying (a staged run mid-way); zero
            # cells is not a finding -- with nothink on the grid, every real
            # run would otherwise print a line about an arm it never planned.
            if cells:
                print(f"\n{level}: only {cells} present, nothing paired to "
                      f"compare")
            continue
        arms[level] = (ids, m)
        print(f"\n{level.upper()} ARM   {len(ids)} paired problems")
        base = m["intact"]
        for s in ("ablated", "random"):
            if s in m:
                print(f"   {s:9} {m[s].mean():>6.1%}   "
                      f"drop {100 * (base.mean() - m[s].mean()):+.1f} pts   "
                      f"McNemar vs intact "
                      + "  ".join(f"{k}={v}" for k, v in
                                  mcnemar(base, m[s]).items()))
        if "ablated" in m and "random" in m:
            print(f"   SELECTIVITY = (intact-ablated) - (intact-random) "
                  f"= random - ablated")
            selectivity(base, m["ablated"], m["random"])

    # ------------------------------------------------------ the interaction
    if "direct" in arms and "cot" in arms:
        di, dm = arms["direct"]
        ci, cm = arms["cot"]
        ids = sorted(set(di) & set(ci))
        if not ids:
            print("\nno problem appears in both arms; interaction undefined")
            return 1
        pick = lambda m, src, s: np.array(
            [m[s][src.index(i)] for i in ids], float)
        print(f"\nINTERACTION   {len(ids)} problems in both arms")
        for lbl, s in (("ablation", "ablated"), ("random control", "random")):
            if s not in dm or s not in cm:
                continue
            r = paired_bootstrap({
                "direct_intact": pick(dm, di, "intact"),
                "direct_ablated": pick(dm, di, s),
                "cot_intact": pick(cm, ci, "intact"),
                "cot_ablated": pick(cm, ci, s)})
            print(f"   {lbl:16}{100 * r['point']:+.1f} pts   95% CI "
                  f"[{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}]   "
                  f"p={r['p']:.3f}   width {100 * r['width']:.1f}")
        print("   A control interaction that is not ~0 means broad "
              "degradation, not\n   a J-space effect. Read the two lines "
              "together or neither.")
        if dev_crosses:
            # Repeated here, next to the number it damages, and not only in
            # the header: the header scrolls off, and this is the line that
            # gets pasted into the report.
            print("   NOTE: the two arms were generated on different "
                  "backends (see above).\n   Part of this number is a kernel "
                  "difference, not an ablation effect.")
    else:
        print("\nINTERACTION: needs both arms. The direct arm alone cannot "
              "test it --\n  the interaction asks whether WRITTEN reasoning "
              "protects against removing\n  INTERNAL reasoning, which "
              "presupposes the direct condition is damaged.")

    # -------------------------------------------- exploratory vs holdout
    #
    # ONLY ON THE DATASET THE BAND WAS SELECTED ON. The sweep ran on GSM8K, so
    # problem_ids(12) names the explored set for GSM8K alone. Applied to
    # another dataset those 12 indices are 12 arbitrary problems -- nothing was
    # selected on them, so splitting them out protects against nothing and the
    # "holdout" label is false. On aime24 it would also discard 40% of the run
    # to report a 12-problem subset of nothing.
    if dataset != config.EXPLORED_ON:
        print(f"\nEXPLORATORY SUBSET: not applicable. The band was selected on "
              f"{config.EXPLORED_ON}\n   (config.EXPLORED_ON), so every "
              f"{dataset} problem here is confirmatory and the\n   whole "
              f"sample above is the holdout. Note in the report that the band "
              f"was\n   nonetheless CHOSEN on {config.EXPLORED_ON} and "
              f"transferred, which is an assumption,\n   not a measurement on "
              f"this dataset.")
        return 0

    explored = set(config.problem_ids(config.EXPLORED_N, rows))
    print(f"\nEXPLORATORY SUBSET (config.problem_ids({config.EXPLORED_N}), "
          f"the set the band was chosen on)")
    print("   Reported apart, never pooled: selection on outcome data inflates "
          "whatever\n   it selected for, and this subset is measurably not "
          "representative.")
    for level in LEVELS:
        if level not in arms:
            continue
        ids, m = arms[level]
        sub = [k for k, i in enumerate(ids) if i in explored]
        hold = [k for k, i in enumerate(ids) if i not in explored]
        if not sub or not hold:
            continue
        for s in STATES:
            if s not in m:
                continue
            print(f"   {level}_{s:9} explored {m[s][sub].mean():>6.0%} "
                  f"({len(sub)})    holdout {m[s][hold].mean():>6.0%} "
                  f"({len(hold)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
