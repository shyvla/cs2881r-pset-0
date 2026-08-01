"""Validate analysis.py by parameter recovery and CI coverage."""
import numpy as np

from analysis import (CELLS, CELLS_RANDOM_CONTROL, ROLES, cap_warnings,
                      cell_table, mcnemar, observed_rho, paired_bootstrap,
                      report, report_with_control, to_matrix)

RNG = np.random.default_rng(7)
_REF = np.random.default_rng(1).normal(size=200000)


def synth(n, P, rho=0.5, rng=RNG, cells=CELLS, hit_cap=0.0, degen=0.0):
    """Paired synthetic data: one latent difficulty per problem, shared
    across all four cells, so the cells are correlated like real ones."""
    z = rng.normal(size=n)
    recs = []
    for role, p in P.items():
        lat = np.sqrt(rho) * z + np.sqrt(1 - rho) * rng.normal(size=n)
        correct = (lat < np.quantile(_REF, p)).astype(int)
        caps = rng.random(n) < (hit_cap.get(role, 0.0)
                                if isinstance(hit_cap, dict) else hit_cap)
        degens = rng.random(n) < (degen.get(role, 0.0)
                                  if isinstance(degen, dict) else degen)
        for i, c in enumerate(correct):
            recs.append({"id": i, "cond": cells[role], "correct": int(c),
                         "outcome": "correct" if c else "incorrect",
                         "n_tok": 500, "hit_cap": bool(caps[i]),
                         "normalized": False,
                         "distinct10_body": 0.9,
                         "distinct10_trace": 0.2 if degens[i] else 0.9})
    return recs


P_EFFECT = {"direct_intact": .50, "direct_ablated": .20,
            "cot_intact": .90, "cot_ablated": .80}
P_NULL = {"direct_intact": .50, "direct_ablated": .30,      # both drop 20
          "cot_intact": .90, "cot_ablated": .70}


def test_mcnemar_hand():
    x = np.array([1, 1, 1, 0, 0, 0, 1, 0])
    y = np.array([0, 0, 0, 1, 1, 0, 1, 0])
    r = mcnemar(x, y)            # b=3 (x right y wrong), c=2
    assert (r["b"], r["c"]) == (3, 2), r
    # exact two-sided binomial, n=5, k=min=2: 2*(C(5,0)+C(5,1)+C(5,2))/32
    assert abs(r["p"] - 2 * (1 + 5 + 10) / 32) < 1e-12, r
    print(f"ok   mcnemar exact  b=3 c=2 p={r['p']:.4f}")
    return True


def test_recovery():
    truth = (P_EFFECT["direct_intact"] - P_EFFECT["direct_ablated"]) \
          - (P_EFFECT["cot_intact"] - P_EFFECT["cot_ablated"])
    _, mat, _ = to_matrix(synth(4000, P_EFFECT))
    bs = paired_bootstrap(mat, n_boot=2000)
    err = abs(bs["point"] - truth)
    assert err < 0.03, (bs["point"], truth)
    print(f"ok   recovery       truth={truth:+.0%} est={bs['point']:+.1%} "
          f"err={err:.3f} (n=4000)")
    return True


def test_coverage(n=100, trials=300):
    """The critical test: does the 95% CI actually cover 95% of the time?"""
    truth = 0.20
    rng = np.random.default_rng(11)
    hits = widths = 0
    for _ in range(trials):
        _, mat, _ = to_matrix(synth(n, P_EFFECT, rng=rng))
        bs = paired_bootstrap(mat, n_boot=800, seed=int(rng.integers(1e6)))
        hits += bs["lo"] <= truth <= bs["hi"]
        widths += bs["width"]
    cov = hits / trials
    assert 0.90 <= cov <= 0.99, cov
    print(f"ok   coverage       {cov:.0%} of 95% CIs contain truth "
          f"(n={n}, mean width {widths/trials:.0%})")
    return True


def test_false_positive(n=100, trials=300):
    """True interaction = 0. CI should exclude zero ~5% of the time."""
    rng = np.random.default_rng(23)
    fp = 0
    for _ in range(trials):
        _, mat, _ = to_matrix(synth(n, P_NULL, rng=rng))
        bs = paired_bootstrap(mat, n_boot=800, seed=int(rng.integers(1e6)))
        fp += (bs["lo"] > 0) or (bs["hi"] < 0)
    rate = fp / trials
    assert rate < 0.12, rate
    print(f"ok   false positive  {rate:.0%} when true interaction is 0")
    return True


def test_complete_cases():
    recs = synth(10, {r: .5 for r in ROLES})
    recs = [r for r in recs
            if not (r["cond"] == CELLS["cot_ablated"] and r["id"] in (3, 7))]
    ids, mat, dropped = to_matrix(recs)
    assert ids == [0, 1, 2, 4, 5, 6, 8, 9], ids
    assert all(len(v) == 8 for v in mat.values())
    assert dropped["direct_intact"] == [3, 7]
    print(f"ok   complete cases  paired on {len(ids)}/10, dropped {[3, 7]}")
    return True


# ==================================================== FIX 8: loud failure
def test_empty_cell_raises():
    """An ablated cell that produced nothing must NOT read as a clean null.

    Before the fix: to_matrix returned ids=[], the bootstrap returned nan for
    point/lo/hi with p=0.0, and because `lo > 0 or hi < 0` is False under NaN
    the report printed "CI includes zero -> NOT resolved at this n". A total
    pipeline failure looked exactly like an honest negative.
    """
    recs = [r for r in synth(10, {r: .5 for r in ROLES})
            if r["cond"] != CELLS["cot_ablated"]]
    try:
        to_matrix(recs)
    except ValueError as e:
        assert "cot_ablated" in str(e), e
        print("ok   empty cell     raises instead of returning a NaN null")
        return True
    raise AssertionError("to_matrix silently accepted an empty cell")


def test_no_overlap_raises():
    """Cells that ran on disjoint problem ids are not paired at all."""
    recs = synth(6, {r: .5 for r in ROLES})
    recs = [r for r in recs
            if not (r["cond"] == CELLS["cot_ablated"] and r["id"] < 6)]
    recs += [{"id": 99, "cond": CELLS["cot_ablated"], "correct": 1,
              "outcome": "correct", "n_tok": 1, "hit_cap": False}]
    try:
        to_matrix(recs)
    except ValueError as e:
        assert "paired" in str(e), e
        print("ok   no overlap     raises instead of pairing on nothing")
        return True
    raise AssertionError("to_matrix accepted non-overlapping ids")


# ============================== FIX 7: table must match the interaction
def test_cell_table_complete_cases():
    """cell_table over all records disagrees with the paired analysis."""
    recs = []
    for i in range(20):
        for role in ROLES:
            if role == "cot_ablated" and i >= 10:
                continue                       # cell lost its hardest half
            correct = 1 if (role != "cot_ablated" and i >= 10) else i % 2
            recs.append({"id": i, "cond": CELLS[role], "correct": correct,
                         "outcome": "correct" if correct else "incorrect",
                         "n_tok": 10, "hit_cap": False, "normalized": False,
                         "distinct10_body": 0.9, "distinct10_trace": 0.9})
    ids, mat, _ = to_matrix(recs)
    all_rows = cell_table(recs)
    paired_rows = cell_table(recs, ids=ids)
    assert all_rows["direct_intact"]["acc"] == 0.75
    assert paired_rows["direct_intact"]["acc"] == 0.50
    for role in ROLES:
        assert abs(paired_rows[role]["acc"] - mat[role].mean()) < 1e-12
    print(f"ok   paired table    all-records acc="
          f"{all_rows['direct_intact']['acc']:.0%} vs paired "
          f"{paired_rows['direct_intact']['acc']:.0%}; report now prints the "
          f"paired one")
    return True


# ==================================================== minor fixes
def test_p_floor():
    """A bootstrap over n_boot draws cannot resolve p below 1/(n_boot+1)."""
    mat = {"direct_intact": np.ones(60), "direct_ablated": np.zeros(60),
           "cot_intact": np.ones(60), "cot_ablated": np.ones(60)}
    bs = paired_bootstrap(mat, n_boot=10000)
    assert bs["p_at_floor"] and bs["p"] == 1 / 10001, bs
    print(f"ok   p floor        p={bs['p']:.2e} flagged, not printed as 0.000")
    return True


def test_cap_warnings():
    """FIX 5: a cap that binds only under ablation inflates the interaction."""
    rows = cell_table(synth(200, P_EFFECT,
                            hit_cap={"direct_intact": 0.0,
                                     "direct_ablated": 0.30,
                                     "cot_intact": 0.0, "cot_ablated": 0.0}))
    warns = cap_warnings(rows)
    assert any("direct" in w and "cap rate differs" in w for w in warns), warns
    assert not cap_warnings(cell_table(synth(200, P_EFFECT))), "false alarm"
    print(f"ok   cap warnings   {len(warns)} raised on a 30% direct-only cap")
    return True


def test_duplicate_records_raise():
    """A resumed run that appends twice must not analyse silently.

    The duplicate overwrote (last wins) in to_matrix while cell_table counted
    it twice, so the printed table and the interaction disagreed for a reason
    neither one displayed.
    """
    recs = synth(10, {r: .5 for r in ROLES})
    recs = recs + [dict(recs[0])]
    try:
        to_matrix(recs)
    except ValueError as e:
        assert "duplicate" in str(e), e
        print("ok   duplicates    raise instead of last-write-wins")
        return True
    raise AssertionError("to_matrix accepted a duplicate (cond, id)")


def test_observed_rho():
    for target in (0.0, 0.3, 0.5, 0.75):
        _, mat, _ = to_matrix(synth(5000, {r: .5 for r in ROLES}, rho=target))
        got = observed_rho(mat)
        assert abs(got - target) < 0.06, (target, got)
        print(f"       rho: target={target:.2f} measured={got:.3f}")
    print("ok   observed_rho    recovers the correlation power.py assumes")
    return True


if __name__ == "__main__":
    import sys
    res = [test_mcnemar_hand(), test_recovery(), test_coverage(),
           test_false_positive(), test_complete_cases(),
           test_empty_cell_raises(), test_no_overlap_raises(),
           test_cell_table_complete_cases(), test_p_floor(),
           test_cap_warnings(), test_duplicate_records_raise(),
           test_observed_rho()]
    print(f"\n{'ALL PASS' if all(res) else 'FAILURES PRESENT'}\n")

    print("=" * 78)
    print("DEMO: realistic n=100 run, true interaction +20%")
    print("=" * 78)
    report(synth(100, P_EFFECT))

    print("\n" + "=" * 78)
    print("DEMO: the underpowered case, n=20, same truth")
    print("=" * 78)
    report(synth(20, P_EFFECT))

    print("\n" + "=" * 78)
    print("DEMO: 15% looping in cot_ablated + a direct-only cap that binds")
    print("      -- the interaction shrinks and BOTH warnings fire")
    print("=" * 78)
    report(synth(150, P_EFFECT,
                 hit_cap={"direct_intact": 0.0, "direct_ablated": 0.25,
                          "cot_intact": 0.0, "cot_ablated": 0.0},
                 degen={"direct_intact": 0.0, "direct_ablated": 0.0,
                        "cot_intact": 0.0, "cot_ablated": 0.15}))

    print("\n" + "=" * 78)
    print("DEMO: main analysis + random-direction control together")
    print("=" * 78)
    rng = np.random.default_rng(5)
    main_cells = synth(150, P_EFFECT, rng=rng)
    # the two random-direction cells only; the intact cells are shared
    rnd = [r for r in synth(150, {"direct_intact": .50, "direct_ablated": .48,
                                  "cot_intact": .90, "cot_ablated": .88},
                            rng=rng, cells=CELLS_RANDOM_CONTROL)
           if r["cond"].endswith("_random")]
    report_with_control(main_cells + rnd)
    sys.exit(0 if all(res) else 1)
