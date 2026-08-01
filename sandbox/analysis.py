"""
Analysis for the 2x2 J-space ablation experiment.

The deliverable is ONE number:

    interaction = (direct_intact - direct_ablated)
                - (cot_intact    - cot_ablated)

It measures how much MORE ablation hurts direct answering than CoT. That is
the fingerprint of internal/external substitution. Reporting the two drops
separately invites "the ablation just broke everything"; the interaction is
what distinguishes substitution from broad degradation.

A point estimate is not a result. Everything here is built to produce the
interval alongside it, because an interaction is a difference of two
differences of four noisy proportions -- noise compounds four times while
signal does not. Simulation: at n=20 per cell, a REAL 20-point interaction is
detected only ~26% of the time. See power.py.

NAMING (FIX 6)
--------------
There are no A/B/C/D letters in this codebase. They previously meant one
thing in the design table (A = direct intact) and another in the Milestone-4
run script (A_cot = CoT intact); mapping them backwards flips the sign of the
headline number and nothing errors. Roles and conditions now share one
`{level}_{state}` grid defined in scoring.LEVELS / scoring.STATES.
"""
import json
import math
from collections import Counter
from statistics import NormalDist

import numpy as np

from scoring import DEGENERATE_BELOW, OUTCOMES, parse_cond

_NORM = NormalDist()
_TRAPZ = getattr(np, "trapezoid", None) or np.trapz   # numpy 2 renamed trapz

# Order matters: the interaction is defined as (ROLES[0]-ROLES[1]) -
# (ROLES[2]-ROLES[3]).
ROLES = ("direct_intact", "direct_ablated", "cot_intact", "cot_ablated")

# role -> condition name in your scores file.
CELLS = {r: r for r in ROLES}

# The random-direction matched control, expressed as a remap rather than a
# second copy of the analysis: substitute the random-direction cells for the
# ablated ones and re-run the identical pipeline. This interaction should be
# ~0. If it is not, the effect is broad degradation, not a J-space effect.
CELLS_RANDOM_CONTROL = {
    "direct_intact":  "direct_intact",
    "direct_ablated": "direct_random",
    "cot_intact":     "cot_intact",
    "cot_ablated":    "cot_random",
}


def load_scores(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def to_matrix(records, cells=CELLS):
    """Pivot to complete cases: problems present in EVERY cell.

    Completeness is what makes the design paired, which is what makes the
    bootstrap and McNemar valid. Dropping incomplete problems is reported,
    never silent -- a cell that systematically fails to produce records
    (e.g. crashes under ablation) would otherwise vanish from the analysis.

    FIX 8: an empty cell now raises. Previously it returned ids=[] and the
    bootstrap returned nan for point/lo/hi with p=0.0; the verdict test
    (`lo > 0 or hi < 0`) is False under NaN, so `report` printed

        "CI includes zero -> NOT resolved at this n"

    A total pipeline failure presented itself as a clean honest negative.
    That is the single worst way this code could fail, so it is now loud.
    """
    by = {role: {} for role in cells}
    inv = {v: k for k, v in cells.items()}
    dups = set()
    for r in records:
        role = inv.get(r["cond"])
        if role is None:
            continue
        if r["id"] in by[role]:
            dups.add((cells[role], r["id"]))
        by[role][r["id"]] = int(r["correct"])

    # A duplicate silently overwrote (last wins) here while cell_table
    # double-counted it, so the table and the interaction disagreed for a
    # reason neither one showed. Most likely cause: a resumed generation run
    # that appended the same problems twice.
    if dups:
        raise ValueError(
            f"{len(dups)} duplicate (cond, id) record(s), e.g. "
            f"{sorted(dups)[:5]}. Which generation was scored is ambiguous; "
            f"de-duplicate the generations file before analysing.")

    empty = [f"{role} (expects cond={cells[role]!r})"
             for role, d in by.items() if not d]
    if empty:
        seen = sorted({r["cond"] for r in records})
        raise ValueError(
            "no records found for cell(s): " + "; ".join(empty)
            + f"\n  conditions present in the data: {seen}"
            + "\n  either the run did not produce these cells, or CELLS is "
              "mapped to the wrong condition names.")

    ids = sorted(set.intersection(*(set(d) for d in by.values())))
    if not ids:
        raise ValueError(
            "no problem id appears in all cells, so nothing is paired; "
            "check that every cell ran on the SAME problem ids.")

    dropped = {role: sorted(set(d) - set(ids)) for role, d in by.items()}
    mat = {role: np.array([by[role][i] for i in ids], dtype=float)
           for role in cells}
    return ids, mat, dropped


def interaction(mat) -> float:
    a, b, c, d = (mat[r] for r in ROLES)
    return (a.mean() - b.mean()) - (c.mean() - d.mean())


def paired_bootstrap(mat, n_boot=10000, seed=0, alpha=0.05):
    """Percentile bootstrap resampling PROBLEMS, not observations.

    Resampling problems preserves the pairing (a problem is drawn into all
    four cells together), so the correlation induced by problem difficulty is
    handled automatically and no distributional assumption is needed.

    The returned `p` is an inverted-CI p-value, not a test statistic from a
    null resampling. It is floored at 1/(n_boot+1) because a bootstrap over
    n_boot draws cannot resolve anything smaller, and printing "p=0.000" for
    what is really "p < 1e-4" overstates the evidence.
    """
    rng = np.random.default_rng(seed)
    a, b, c, d_ = (mat[r] for r in ROLES)
    n = len(a)
    point = (a.mean() - b.mean()) - (c.mean() - d_.mean())
    idx = rng.integers(0, n, size=(n_boot, n))
    d = ((a[idx].mean(1) - b[idx].mean(1))
         - (c[idx].mean(1) - d_[idx].mean(1)))
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_raw = 2 * min((d <= 0).mean(), (d >= 0).mean())
    floor = 1.0 / (n_boot + 1)
    return dict(point=point, lo=lo, hi=hi, p=min(1.0, max(p_raw, floor)),
                p_at_floor=bool(p_raw < floor), width=hi - lo, n=n)


def mcnemar(x, y):
    """Exact paired test for two conditions on the same problems.

    b = x right / y wrong, c = x wrong / y right. Problems both cells got
    right (or both wrong) carry no information about the difference and are
    correctly ignored -- that is the power gain from pairing.
    """
    b = int(((x == 1) & (y == 0)).sum())
    c = int(((x == 0) & (y == 1)).sum())
    n = b + c
    if n == 0:
        return dict(b=b, c=c, p=1.0)
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return dict(b=b, c=c, p=min(1.0, p))


def _degenerate(r) -> bool:
    """Looping anywhere -- in the trace OR in the answer body (FIX 4).

    Falls back to the pre-fix `distinct10` field so old score files still
    load, but those files cannot see trace looping at all.
    """
    vals = [r.get("distinct10_trace"), r.get("distinct10_body"),
            r.get("distinct10")]
    vals = [v for v in vals if v is not None]
    return bool(vals) and min(vals) < DEGENERATE_BELOW


def cell_table(records, cells=CELLS, ids=None):
    """Accuracy AND outcome composition per cell.

    Composition matters: an accuracy collapse made of `incomplete` is a
    termination failure, not a reasoning failure, and the two support very
    different claims.

    FIX 7: pass `ids` to restrict to the paired complete cases. Without it
    this table is computed over ALL records while the interaction is computed
    over complete cases only, so the two disagree. Measured on a synthetic
    run where one cell lost its ten hardest problems:

        direct_intact   cell_table acc=75% (n=20)   complete-case acc=50%

    `report` printed both a few lines apart, so anyone recomputing
    (A-B)-(C-D) from the table got a different number than the reported
    interaction.
    """
    ids = None if ids is None else set(ids)
    rows = {}
    for role, cond in cells.items():
        rs = [r for r in records
              if r["cond"] == cond and (ids is None or r["id"] in ids)]
        if not rs:
            continue
        c = Counter(r["outcome"] for r in rs)
        rows[role] = dict(
            cond=cond,
            n=len(rs),
            acc=c["correct"] / len(rs),
            **{o: c[o] for o in OUTCOMES},
            hit_cap=sum(bool(r.get("hit_cap")) for r in rs) / len(rs),
            degenerate=sum(_degenerate(r) for r in rs) / len(rs),
            normalized=sum(bool(r.get("normalized")) for r in rs) / len(rs),
            mean_tok=(sum(r.get("n_tok") or 0 for r in rs) / len(rs)),
        )
    return rows


def cap_warnings(rows, abs_thresh=0.05, diff_thresh=0.05) -> list[str]:
    """FIX 5: max_new_tokens asymmetry is a confound, not a detail.

    Caps of 2048 (cot) / 512 (nothink) / 32 (direct) mean that under ablation
    -- which makes every condition longer and loopier -- only the direct
    condition has a cap that binds immediately. The extra verbosity becomes
    `incomplete`, which counts as not-correct, which inflates the direct drop
    and therefore the interaction, IN THE DIRECTION WE PREDICTED.

    So: set the direct cap high enough that it does not bind (128-256 costs
    seconds, not minutes), and check here that it did not. A materially
    higher cap rate in an ablated cell than its intact partner contaminates
    the estimate regardless of what the interval says.
    """
    warns = []
    for role, r in sorted(rows.items()):
        if r["hit_cap"] > abs_thresh:
            warns.append(f"{role} hit max_new_tokens on {r['hit_cap']:.0%} of "
                         f"problems -- cap is binding, raise it")
    for level in ("direct", "nothink", "cot"):
        a, b = f"{level}_intact", f"{level}_ablated"
        if a in rows and b in rows:
            diff = rows[b]["hit_cap"] - rows[a]["hit_cap"]
            if abs(diff) > diff_thresh:
                warns.append(
                    f"{level}: cap rate differs by {diff:+.0%} between intact "
                    f"({rows[a]['hit_cap']:.0%}) and ablated "
                    f"({rows[b]['hit_cap']:.0%}) -- truncation is doing part "
                    f"of the work attributed to ablation")
    return warns


def report(records, cells=CELLS, n_boot=10000, seed=0, label=""):
    ids, mat, dropped = to_matrix(records, cells)
    rows = cell_table(records, cells, ids=ids)        # FIX 7: paired only
    rows_all = cell_table(records, cells)

    if label:
        print(f"### {label}")
    if any(role != cond for role, cond in cells.items()):
        print("cell mapping: " + ",  ".join(f"{r} <- {c}"
                                            for r, c in cells.items()
                                            if r != c))

    print(f"{'cell':<17}{'n':>4}{'acc':>7}{'corr':>6}{'inc':>5}{'trunc':>6}"
          f"{'unp':>5}{'err':>5}{'cap%':>7}{'degen%':>8}{'norm%':>7}{'tok':>7}")
    for role in ROLES:
        if role not in rows:
            continue
        r = rows[role]
        print(f"{role:<17}{r['n']:>4}{r['acc']:>7.1%}{r['correct']:>6}"
              f"{r['incorrect']:>5}{r['incomplete']:>6}{r['unparsed']:>5}"
              f"{r['error']:>5}{r['hit_cap']:>7.0%}{r['degenerate']:>8.0%}"
              f"{r['normalized']:>7.0%}{r['mean_tok']:>7.0f}")
    print("(table is over the PAIRED complete cases, same rows the "
          "interaction uses)")

    n_drop = sum(len(v) for v in dropped.values())
    print(f"\npaired on {len(ids)} complete problems"
          + (f"  ({n_drop} cell-records dropped as incomplete cases)"
             if n_drop else ""))
    if n_drop:
        shifted = [f"{role} {rows[role]['acc']:.0%} paired vs "
                   f"{rows_all[role]['acc']:.0%} over all records"
                   for role in ROLES
                   if role in rows and role in rows_all
                   and abs(rows[role]['acc'] - rows_all[role]['acc']) > 0.01]
        if shifted:
            print("  NOTE: dropping is not neutral -- " + "; ".join(shifted))

    for w in cap_warnings(rows):
        print(f"  CAP WARNING: {w}")
    for role in ROLES:
        if role in rows and rows[role]["degenerate"] > 0.05:
            print(f"  DEGENERACY WARNING: {role} is looping on "
                  f"{rows[role]['degenerate']:.0%} of problems -- an "
                  f"interaction can be halved by this alone")

    a, b, c, d_ = (mat[r] for r in ROLES)
    dd, dc = a.mean() - b.mean(), c.mean() - d_.mean()
    m_direct, m_cot = mcnemar(a, b), mcnemar(c, d_)
    print(f"\nablation drop, DIRECT: {dd:+.1%}   "
          f"McNemar b={m_direct['b']} c={m_direct['c']} p={m_direct['p']:.3f}")
    print(f"ablation drop, CoT:    {dc:+.1%}   "
          f"McNemar b={m_cot['b']} c={m_cot['c']} p={m_cot['p']:.3f}")

    bs = paired_bootstrap(mat, n_boot=n_boot, seed=seed)
    pstr = (f"< {1 / (n_boot + 1):.1e}" if bs["p_at_floor"]
            else f"= {bs['p']:.3f}")
    print(f"\nINTERACTION (direct drop) - (CoT drop) = {bs['point']:+.1%}")
    print(f"  95% CI [{bs['lo']:+.1%}, {bs['hi']:+.1%}]  "
          f"width {bs['width']:.1%}  bootstrap p {pstr}")
    verdict = ("CI excludes zero -> substitution supported"
               if bs["lo"] > 0 or bs["hi"] < 0 else
               "CI includes zero -> NOT resolved at this n")
    print(f"  {verdict}")
    if bs["lo"] <= 0 <= bs["hi"] and bs["width"] > 0.25:
        print("  NOTE: CI width >25pts. This is an UNDERPOWERED null, which is"
              "\n        not the same as evidence of no effect. Report the"
              "\n        interval, or raise n, before claiming a negative.")
    return dict(cells=rows, cells_all=rows_all, bootstrap=bs,
                mcnemar_direct=m_direct, mcnemar_cot=m_cot,
                n_paired=len(ids), dropped=dropped,
                cap_warnings=cap_warnings(rows))


def report_with_control(records, n_boot=10000, seed=0):
    """Main interaction, then the same analysis on the random-direction cells.

    This pairing is the answer to "distinguish a J-space effect from broad
    degradation", and it is the paper's own control. Run them together so the
    comparison is never optional.
    """
    main = report(records, CELLS, n_boot, seed, label="J-SPACE ABLATION")
    print()
    try:
        ctrl = report(records, CELLS_RANDOM_CONTROL, n_boot, seed,
                      label="RANDOM-DIRECTION CONTROL (expect ~0)")
    except ValueError as e:
        print(f"### RANDOM-DIRECTION CONTROL\n  not available: {e}")
        return dict(main=main, control=None)
    print(f"\ncontrol interaction {ctrl['bootstrap']['point']:+.1%} "
          f"vs main {main['bootstrap']['point']:+.1%}")
    return dict(main=main, control=ctrl)


def _bvn_cdf(h, k, rho, steps=257):
    """P(X<h, Y<k) for a standard bivariate normal, by 1-D quadrature.

    Uses the identity Phi2(h,k;rho) = Phi(h)Phi(k) + integral_0^rho phi2 dr,
    so no scipy dependency. Accurate to ~1e-6 at this step count.
    """
    base = _NORM.cdf(h) * _NORM.cdf(k)
    if rho == 0:
        return base
    r = np.linspace(0.0, rho, steps)
    dens = (np.exp(-(h * h - 2 * r * h * k + k * k) / (2 * (1 - r * r)))
            / (2 * np.pi * np.sqrt(1 - r * r)))
    return base + float(_TRAPZ(dens, r))


def tetrachoric(x, y, iters=60) -> float:
    """Latent-scale correlation between two 0/1 vectors.

    NOT the Pearson correlation of the 0/1 values. Dichotomising two
    correlated normals attenuates the correlation -- for balanced cells the
    Pearson phi is (2/pi)*arcsin(rho), so a latent rho of 0.50 shows up as
    0.33. power.py's `rho` is the LATENT correlation, so handing it a phi
    would understate the pairing and overstate the n you need.

    Solved by bisection on rho until the model's joint P(both correct)
    matches the observed one. Returns nan for a cell at 0% or 100%, where
    the latent threshold is infinite and rho is unidentified.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    px, py = x.mean(), y.mean()
    if not (0 < px < 1 and 0 < py < 1):
        return float("nan")
    h, k = _NORM.inv_cdf(px), _NORM.inv_cdf(py)
    target = float(((x == 1) & (y == 1)).mean())
    lo, hi = -0.999, 0.999
    for _ in range(iters):                    # Phi2 is increasing in rho
        mid = (lo + hi) / 2
        if _bvn_cdf(h, k, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def observed_rho(mat) -> float:
    """Mean pairwise tetrachoric correlation -- the `rho` power.py assumes.

    Power at n=150 ranges from 78% (rho=0) to 96% (rho=0.75), so this is a
    load-bearing assumption and it was previously hard-coded at 0.5. After
    the baseline run you can measure it instead of guessing, then feed it
    back in as `power.py ... --rho X` before pre-registering n.
    """
    keys = [k for k in mat if len(mat[k]) > 1]
    cors = []
    for i, ki in enumerate(keys):
        for kj in keys[i + 1:]:
            r = tetrachoric(mat[ki], mat[kj])
            if not math.isnan(r):
                cors.append(r)
    return float(np.mean(cors)) if cors else float("nan")


def baseline_table(records, conditions):
    """Milestone-4 helper: cell table for a baseline-only run.

    `report` needs all four 2x2 cells and will raise on baseline-only data.
    Pass the condition names you generated, e.g.
    ("cot_intact", "nothink_intact", "direct_intact").
    """
    cells = {c: c for c in conditions}
    rows = cell_table(records, cells)
    print(f"{'cell':<17}{'n':>4}{'acc':>7}{'corr':>6}{'inc':>5}{'trunc':>6}"
          f"{'unp':>5}{'err':>5}{'cap%':>7}{'degen%':>8}{'norm%':>7}{'tok':>7}")
    for cond in conditions:
        if cond not in rows:
            print(f"{cond:<17}  -- no records --")
            continue
        r = rows[cond]
        print(f"{cond:<17}{r['n']:>4}{r['acc']:>7.1%}{r['correct']:>6}"
              f"{r['incorrect']:>5}{r['incomplete']:>6}{r['unparsed']:>5}"
              f"{r['error']:>5}{r['hit_cap']:>7.0%}{r['degenerate']:>8.0%}"
              f"{r['normalized']:>7.0%}{r['mean_tok']:>7.0f}")
    for w in cap_warnings(rows):
        print(f"  CAP WARNING: {w}")
    for cond, r in rows.items():
        if r["normalized"] > 0.05:
            print(f"  EXTRACTION NOTE: {cond} needed markdown normalisation "
                  f"on {r['normalized']:.0%} of answers")
        if r["unparsed"] / max(r["n"], 1) > 0.05:
            print(f"  EXTRACTION WARNING: {cond} is {r['unparsed']}/{r['n']} "
                  f"unparsed -- hand-read these before trusting the accuracy")
    # sanity: every condition must be on the grid, so a typo cannot survive
    for cond in conditions:
        parse_cond(cond)
    return rows
