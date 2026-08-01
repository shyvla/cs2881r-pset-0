"""
How many problems per cell do you need? Run BEFORE spending GPU hours.

Usage:
    python power.py                                  # defaults below
    python power.py 0.50 0.20 0.90 0.80              # direct/cot accuracies
    python power.py 0.50 0.20 0.90 0.80 --rho 0.62   # measured correlation
    python power.py 0.50 0.20 0.90 0.80 --loop 0.15  # looping contamination

Positional args are, in order:
    direct_intact  direct_ablated  cot_intact  cot_ablated

Take direct_intact and cot_intact from the baseline run (Milestone 4 gives
them for free) plus a guess at the ablation drops, then read off the n you
need. This is a pre-registration input, not a post-hoc excuse.

TWO THINGS THIS VERSION MODELS THAT THE FIRST DID NOT
-----------------------------------------------------
--rho   the problem-difficulty correlation across cells. It was hard-coded at
        0.5, and it is load-bearing: at n=150 with the default accuracies,
        power runs 78% (rho=0) / 88% (rho=0.5) / 96% (rho=0.75). You can
        MEASURE it after the baseline run -- analysis.observed_rho on the
        intact cells -- instead of guessing.

--loop  the fraction of ablated-CoT generations that degenerate into looping
        and therefore fail. The integration test showed 15% looping cutting a
        true +20 interaction to a measured +9 with a CI spanning zero. A power
        table that ignores it is optimistic about the case you already know
        happens, so the n it recommends is too small.
"""
import argparse
from statistics import NormalDist

import numpy as np

from analysis import CELLS, ROLES, paired_bootstrap, to_matrix

_NORM = NormalDist()


def simulate(n, P, rho=0.5, loop_rate=0.0, trials=200, n_boot=800, seed=0):
    """Gaussian-copula synthetic runs through the REAL analysis pipeline.

    One latent difficulty per problem, shared across cells at correlation
    `rho`, thresholded at each cell's target accuracy. Because it calls the
    actual to_matrix/paired_bootstrap, it measures the power of the estimator
    you will really use, not of an idealised one.

    `loop_rate` flips that fraction of correct answers in cot_ablated to
    incorrect, standing in for degenerate looping.
    """
    rng = np.random.default_rng(seed)
    thresh = {role: _NORM.inv_cdf(p) for role, p in P.items()}
    widths, detect = [], 0
    for _ in range(trials):
        z = rng.normal(size=n)
        recs = []
        for role in P:
            lat = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.normal(size=n)
            ok = (lat < thresh[role]).astype(int)
            if role == "cot_ablated" and loop_rate > 0:
                ok = ok * (rng.random(n) >= loop_rate)
            recs += [{"id": i, "cond": CELLS[role], "correct": int(c)}
                     for i, c in enumerate(ok)]
        _, mat, _ = to_matrix(recs)
        bs = paired_bootstrap(mat, n_boot=n_boot,
                              seed=int(rng.integers(1_000_000)))
        widths.append(bs["width"])
        detect += (bs["lo"] > 0) or (bs["hi"] < 0)
    return float(np.mean(widths)), detect / trials


def main(P, ns=(20, 30, 50, 100, 150, 250), rho=0.5, loop_rate=0.0,
         secs_per_problem=70.0):
    nominal = (P["direct_intact"] - P["direct_ablated"]) \
            - (P["cot_intact"] - P["cot_ablated"])
    # looping depresses cot_ablated, which widens the CoT drop and shrinks
    # the interaction you actually have to detect.
    effective = nominal - loop_rate * P["cot_ablated"]

    print("assumed cell accuracies: "
          + "  ".join(f"{k}={v:.0%}" for k, v in P.items()))
    print(f"rho = {rho:.2f}   looping in cot_ablated = {loop_rate:.0%}")
    print(f"nominal interaction     = {nominal:+.1%}")
    if loop_rate:
        print(f"effective interaction   = {effective:+.1%}  "
              f"<- what you must detect once looping is included")
    print()
    print(f"{'n/cell':>7}{'95% CI width':>15}{'power':>9}   "
          f"{'~GPU-hrs (6 cells, L4)':>24}")
    for n in ns:
        w, pw = simulate(n, P, rho=rho, loop_rate=loop_rate)
        hrs = n * secs_per_problem / 3600
        print(f"{n:>7}{w:>14.0%}{pw:>9.0%}{hrs:>22.1f}")
    print("\nRule of thumb: aim for power >= 0.80 on your MVP dataset.")
    print("A CI wider than ~25 points cannot distinguish 'no effect' from")
    print("'we did not look hard enough'.")
    print("\nGPU-hours assume ~%.0fs/problem summed across 6 cells on an L4"
          % secs_per_problem)
    print("(the M4's measured 7.7 tok/s scaled by ~6x). Re-measure on the")
    print("actual instance before trusting the last column.")


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("acc", nargs="*", type=float,
                    help="four accuracies: direct_intact direct_ablated "
                         "cot_intact cot_ablated")
    ap.add_argument("--rho", type=float, default=0.5,
                    help="problem-difficulty correlation across cells "
                         "(measure with analysis.observed_rho)")
    ap.add_argument("--loop", type=float, default=0.0,
                    help="fraction of cot_ablated generations lost to looping")
    ap.add_argument("--ns", type=int, nargs="+",
                    default=[20, 30, 50, 100, 150, 250])
    ap.add_argument("--secs", type=float, default=70.0,
                    help="seconds per problem summed across 6 cells")
    a = ap.parse_args(argv)
    if a.acc and len(a.acc) != 4:
        # used to accept 2 args and die with a bare KeyError deep inside main
        ap.error(f"need exactly 4 accuracies (got {len(a.acc)}), in the order "
                 f"{' '.join(ROLES)}")
    if not all(0.0 <= x <= 1.0 for x in a.acc):
        ap.error("accuracies must be in [0, 1]")
    if not 0.0 <= a.rho < 1.0:
        ap.error("--rho must be in [0, 1)")
    if not 0.0 <= a.loop <= 1.0:
        ap.error("--loop must be in [0, 1]")
    return a


if __name__ == "__main__":
    args = _parse_args()
    accs = args.acc or [0.50, 0.20, 0.90, 0.80]
    main(dict(zip(ROLES, accs)), ns=tuple(args.ns), rho=args.rho,
         loop_rate=args.loop, secs_per_problem=args.secs)
