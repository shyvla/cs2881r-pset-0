"""Tests for run.py's orchestration -- the parts that decide WHAT runs.

No model and no dataset: everything here is about cell selection, sample-size
resolution, the loop gate's arithmetic and the calibration/run boundary. Those
are exactly the decisions that are expensive to get wrong (a staged run blocked
on the other arm's prompt, a gate that silently declines to guard the most
expensive cell, calibration generations pooled into a run) and free to test.

Run: python -m tests.test_run
"""
import json
import os
import sys
import tempfile

import config
import run
import scoring


# ============================================== which cells this run generates

def test_cells_for_covers_the_grid_in_gate_order():
    """The order is load-bearing: the ablated CoT cell must be LAST so the loop
    gate can stop before the run pays for it, and intact must precede
    intervened within a level because the gate compares against it."""
    order = run.cells_for(None)
    assert order[-1] == "cot_ablated", order
    assert order.index("cot_intact") < order.index("cot_random") < \
           order.index("cot_ablated"), order
    assert order.index("direct_intact") < order.index("direct_ablated"), order
    assert len(order) == 6, order


def test_order_and_the_condition_grid_cannot_drift():
    """ORDER is a hand-written list and conditions() builds from
    scoring.cond_name. main() used to assert they matched; splitting them apart
    so the level set is knowable before the prompts are resolved would
    otherwise have dropped that guard. A cell present in one and not the other
    is a cell silently never generated."""
    assert set(run.ORDER) == set(run.conditions("gsm8k")), (
        sorted(run.ORDER), sorted(run.conditions("gsm8k")))
    assert len(run.ORDER) == len(set(run.ORDER)), run.ORDER
    assert {c.split("_")[0] for c in run.ORDER} == set(config.RUN_LEVELS)


def test_cells_for_is_resolvable_without_a_prompt():
    """The whole point of splitting ORDER out of conditions(): the readiness
    check is scoped to the levels being run, so the level set has to be
    knowable BEFORE direct_prompt() is consulted. Otherwise a cot-only stage
    has to satisfy the direct arm's pre-registration to discover it does not
    need it."""
    saved = config.DIRECT_INSTRUCTION["gsm8k"]
    try:
        config.DIRECT_INSTRUCTION["gsm8k"] = None
        assert run.cells_for("cot") == ["cot_intact", "cot_random",
                                        "cot_ablated"]
    finally:
        config.DIRECT_INSTRUCTION["gsm8k"] = saved


def test_cells_for_only_accepts_levels_and_cell_names():
    assert run.cells_for("direct_intact") == ["direct_intact"]
    assert set(run.cells_for("direct")) == {"direct_intact", "direct_ablated",
                                            "direct_random"}
    assert set(run.cells_for("cot_intact,direct_intact")) == {
        "cot_intact", "direct_intact"}
    try:
        run.cells_for("nonsense")
    except SystemExit:
        return
    raise AssertionError("--only matching nothing must not silently run all")


def test_calibration_restricts_to_intact_cells():
    """Ablated generations run longer and loop, so a cap calibrated on them is
    sized to degenerate behaviour -- and it would mean seeing ablated outcomes
    before the pre-registration is closed."""
    order = run.cells_for(None, intact_only=True)
    assert order == ["direct_intact", "cot_intact"], order
    assert run.cells_for("cot", intact_only=True) == ["cot_intact"]


def test_conditions_only_builds_the_levels_asked_for():
    """A cot-only run must not resolve the direct prompt at all."""
    saved = config.DIRECT_INSTRUCTION["gsm8k"]
    try:
        config.DIRECT_INSTRUCTION["gsm8k"] = None
        conds = run.conditions("gsm8k", levels=("cot",))
        assert set(conds) == {"cot_intact", "cot_ablated", "cot_random"}
    finally:
        config.DIRECT_INSTRUCTION["gsm8k"] = saved
    # ...and the default still builds the full MVP grid.
    assert len(run.conditions("gsm8k")) == 6


def test_conditions_prefill_reaches_only_the_direct_cells():
    conds = run.conditions("math500")
    for state in ("intact", "ablated", "random"):
        think, suffix, prefill, _ = conds[f"direct_{state}"]
        assert think is False and prefill == "\\boxed{"
        assert "numeric" not in suffix, suffix
        think, suffix, prefill, _ = conds[f"cot_{state}"]
        assert think is True and suffix == "" and prefill == ""


# ==================================================== the sample size

def test_resolve_n_prefers_the_flag_and_falls_back_to_the_committed_n():
    assert run.resolve_n("gsm8k", 20) == 20
    assert run.resolve_n("gsm8k", None) == config.N_DEFAULT["gsm8k"]
    assert run.resolve_n("aime24", None) == 30
    assert run.resolve_n("math500", None) == 100


def test_resolve_n_exits_rather_than_borrowing_gsm8ks_n():
    """The failure this closes: --n defaulting to 150 meant `run.py --dataset
    math500` silently ran a GSM8K sample size. An undecided n must stop the
    run, not pick one.

    Against a temporarily-unset entry, so settling a dataset's n does not turn
    this red -- and an explicit --n must still override, since a pilot or a
    calibration on an undecided dataset is exactly how it gets decided."""
    saved = config.N_DEFAULT["gsm8k"]
    try:
        config.N_DEFAULT["gsm8k"] = None
        assert run.resolve_n("gsm8k", 20) == 20
        try:
            run.resolve_n("gsm8k", None)
        except SystemExit as e:
            assert "N_DEFAULT" in str(e) and "power.py" in str(e), e
            return
        raise AssertionError("an unset N_DEFAULT must stop the run")
    finally:
        config.N_DEFAULT["gsm8k"] = saved


# ==================================================== the loop gate

def _gate_file(path, n_pairs, bad_abl, bad_base):
    """A generations file with `n_pairs` matched cot_random/cot_intact records,
    the first `bad_*` of each unusable (an unclosed <think>, which score()
    grades `incomplete`)."""
    with open(path, "w") as f:
        for cond, n_bad in (("cot_random", bad_abl), ("cot_intact", bad_base)):
            for i in range(n_pairs):
                raw = ("<think>looping and looping" if i < n_bad
                       else "<think>ok</think>\\boxed{7}")
                f.write(json.dumps({"id": i, "cond": cond, "raw": raw,
                                    "gold": "7", "hit_cap": False}) + "\n")


def test_gate_fires_only_above_its_threshold():
    gate = dict(config.LOOP_GATE)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.jsonl")
        # 20 pairs, 5 unusable vs 0 = +25 pts against a 15 pt threshold
        _gate_file(p, 20, 5, 0)
        fired, msg = run.gate_check(p, ("cot_random", "cot_intact"), gate, 20)
        assert fired, msg
        # 2 vs 0 = +10 pts, under it
        _gate_file(p, 20, 2, 0)
        fired, msg = run.gate_check(p, ("cot_random", "cot_intact"), gate, 20)
        assert not fired, msg


def test_gate_scales_to_a_run_smaller_than_its_own_n():
    """It used to defer on any run under gate['n']=20 -- reporting nothing and
    letting the most expensive cell proceed unguarded, which is the one
    situation it exists for. The gate reads already-generated cells, so running
    it at the run's n costs nothing."""
    gate = dict(config.LOOP_GATE)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.jsonl")
        _gate_file(p, 10, 4, 0)           # +40 pts over 10 problems
        fired, msg = run.gate_check(p, ("cot_random", "cot_intact"), gate, 10)
        assert fired, msg
        assert "deferred" not in msg, msg


def test_gate_declines_rather_than_firing_on_noise():
    """Below GATE_MIN_N one problem swings the rate by more than the threshold,
    so the gate must say it cannot run -- not fire, and not stay silent."""
    gate = dict(config.LOOP_GATE)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.jsonl")
        _gate_file(p, 2, 2, 0)            # 100% vs 0%, on two problems
        fired, msg = run.gate_check(p, ("cot_random", "cot_intact"), gate, 2)
        assert not fired, msg
        assert "DECLINED" in msg and "UNGUARDED" in msg, msg


def test_gate_defers_when_the_cells_are_not_both_generated():
    gate = dict(config.LOOP_GATE)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.jsonl")
        _gate_file(p, 20, 0, 0)
        fired, msg = run.gate_check(p, ("cot_ablated", "cot_intact"), gate, 20)
        assert not fired and "deferred" in msg, msg


# ==================================== the calibration / run-data boundary

def test_calib_path_is_a_separate_namespace_from_run_path():
    """Calibration generations are made at the MEASURE_CAP ceiling, so they are
    a different condition and must not be mistakable for run data in a
    directory listing."""
    assert scoring.calib_path("math500", 20) == "runs/calib_math500_n20.jsonl"
    for d in ("gsm8k", "math500", "aime24"):
        assert scoring.calib_path(d, 20) != scoring.run_path(d, 20, "light")
    # No band: no intervention was applied, so there is none to record.
    assert "light" not in scoring.calib_path("gsm8k", 20)


def test_analyze_refuses_calibration_records():
    """The filename makes pooling unlikely; the stamp makes it impossible. An
    intact cell borrowed from a calibration would have been generated at a
    different max_new_tokens than the ablated cells it is differenced against."""
    import analyze
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "calib.jsonl")
        with open(p, "w") as f:
            f.write(json.dumps({"id": 0, "cond": "cot_intact", "raw": "x",
                                "gold": "7", "hit_cap": False,
                                "calibration": True}) + "\n")
        try:
            analyze.load(p)
        except SystemExit as e:
            assert "CALIBRATION" in str(e), e
            return
    raise AssertionError("analyze.load must refuse calibration records")


# ================================== analyze.py scores by the same policies

def _write_run(d, recs, name="run.jsonl"):
    p = os.path.join(d, name)
    with open(p, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    return p


def test_analyze_load_scores_errors_and_carries_normalized():
    """load() must apply scoring.py's own policies: a record that raises
    becomes outcome='error' (never a crash, never silently dropped), the
    markdown-rescue flag is carried through so it can be reported PER CELL,
    and `thinking` comes from the grid -- an unclosed trace in a cot cell is
    incomplete, not graded."""
    import analyze
    with tempfile.TemporaryDirectory() as d:
        p = _write_run(d, [
            {"id": 0, "cond": "direct_intact", "raw": "**72**", "gold": "72",
             "hit_cap": False},
            {"id": 1, "cond": "direct_intact", "raw": "72", "gold": "72",
             "hit_cap": False},
            # empty gold raises inside score_detail -> outcome='error'
            {"id": 2, "cond": "direct_intact", "raw": "72", "gold": "",
             "hit_cap": False},
            {"id": 0, "cond": "cot_intact", "raw": "<think>I think 72",
             "gold": "72", "hit_cap": False},
        ])
        by, errors = analyze.load(p)
    di = by["direct_intact"]
    assert di[0][1] == "correct" and di[0][3]["normalized"] is True
    assert di[1][1] == "correct" and di[1][3]["normalized"] is False
    assert di[2][1] == "error" and di[2][0] == 0
    assert len(errors) == 1 and errors[0][0] == 2, errors
    assert by["cot_intact"][0][1] == "incomplete"
    print("  ok    analyze.load: error outcome + normalized carried per record")


def test_analyze_load_refuses_an_off_grid_cond():
    """startswith('cot') silently mis-scored archived letter-named files --
    'A_cot' IS a cot cell and the prefix test said it was not, so its traces
    would have been graded as answers. Off-grid names refuse loudly now."""
    import analyze
    with tempfile.TemporaryDirectory() as d:
        p = _write_run(d, [{"id": 0, "cond": "A_cot", "raw": "72",
                            "gold": "72", "hit_cap": False}])
        try:
            analyze.load(p)
        except SystemExit as e:
            assert "A_cot" in str(e), e
            print("  ok    analyze.load refuses off-grid condition names")
            return
    raise AssertionError("analyze.load must refuse off-grid condition names")


def test_conditions_thinking_flags_agree_with_the_grid():
    """run.conditions() builds (thinking, ...) specs from its own table;
    scoring.thinking_of resolves the same flag for consumers that only have
    the cond name. If the two tables drift, analyze.py scores a cell under
    the wrong stripping rule -- silently."""
    conds = run.conditions("gsm8k", levels=list(scoring.LEVELS))
    assert len(conds) == len(scoring.LEVELS) * len(scoring.STATES)
    for name, (think, *_rest) in conds.items():
        assert think == scoring.thinking_of(name), name
    print("  ok    run.conditions' thinking flags match scoring.thinking_of")


def test_analyze_main_reports_norm_and_errors_per_cell():
    """The report must surface the markdown-rescue rate per cell (the check
    scoring.py's design principle demands and nothing previously computed)
    and scoring errors -- not swallow either."""
    import contextlib
    import io

    import analyze
    with tempfile.TemporaryDirectory() as d:
        p = _write_run(d, [
            {"id": 0, "cond": "direct_intact", "raw": "**72**", "gold": "72",
             "hit_cap": False, "dataset": "gsm8k"},
            {"id": 1, "cond": "direct_intact", "raw": "71", "gold": "72",
             "hit_cap": False, "dataset": "gsm8k"},
            {"id": 2, "cond": "direct_intact", "raw": "70", "gold": "",
             "hit_cap": False, "dataset": "gsm8k"},
        ])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = analyze.main(["--file", p])
    out = buf.getvalue()
    assert rc == 0, out
    assert "norm" in out, out                       # the per-cell column
    assert "markdown rescue" in out, out            # the cross-cell spread
    assert "outcome='error'" in out, out            # the scoring failure
    assert "error=1" in out, out                    # ...in the composition
    print("  ok    analyze.main reports norm column, rescue spread, errors")


def test_analyze_reports_nothink_cells():
    """analyze.py's LEVELS used to be a hand-typed ('direct', 'cot'), so a
    nothink cell loaded but appeared in no table: the header said it was
    present and no number for it existed anywhere. The middle rung of the
    externalisation axis must be reported wherever it was run."""
    import contextlib
    import io

    import analyze
    with tempfile.TemporaryDirectory() as d:
        recs = []
        for i in range(5):
            recs.append({"id": i, "cond": "nothink_intact", "raw": "72",
                         "gold": "72", "hit_cap": False, "dataset": "gsm8k"})
            recs.append({"id": i, "cond": "nothink_ablated",
                         "raw": "72" if i < 2 else "71", "gold": "72",
                         "hit_cap": False, "dataset": "gsm8k"})
        p = _write_run(d, recs)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = analyze.main(["--file", p])
    out = buf.getvalue()
    assert rc == 0, out
    assert "nothink_intact" in out and "nothink_ablated" in out, out
    assert "NOTHINK ARM" in out, out                # paired comparison ran
    print("  ok    analyze.main reports nothink cells and their arm")


def test_cap_report_flags_a_censored_distribution():
    """The load-bearing number. Any hit at the ceiling means the observed tail
    is not the real tail, so every quantile -- and the suggestion built on it
    -- is a lower bound and must not be committed."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "calib.jsonl")
        with open(p, "w") as f:
            for i in range(10):
                hit = i == 0
                f.write(json.dumps({
                    "id": i, "cond": "cot_intact", "dataset": "math500",
                    "raw": "<think>ok</think>\\boxed{7}", "gold": "7",
                    "n_tok": 8192 if hit else 300, "cap": 8192,
                    "hit_cap": hit, "calibration": True}) + "\n")
        assert run.cap_report(p, "math500") == 1, "censored report must fail"

        # ...and a clean distribution reports a committable suggestion.
        with open(p, "w") as f:
            for i in range(10):
                f.write(json.dumps({
                    "id": i, "cond": "cot_intact", "dataset": "math500",
                    "raw": "<think>ok</think>\\boxed{7}", "gold": "7",
                    "n_tok": 300 + i, "cap": 8192,
                    "hit_cap": False, "calibration": True}) + "\n")
        assert run.cap_report(p, "math500") == 0


# ============================== which problems the calibration measures on

# Stand-in for a split's difficulty column, so none of this needs a download:
# 100 problems cycling through levels 1-5, i.e. 20 of each.
_DIFF = [1 + (i % 5) for i in range(100)]


def test_calib_sample_draws_only_from_the_hard_end():
    """The cap's loss is asymmetric -- too low is a differential bias, too high
    costs seconds -- so the quantity to observe is the upper tail, not the
    middle. A random draw from the run's own sample underestimates the maximum
    of a larger run, which is what CAP_HEADROOM was silently covering for."""
    saved = config.CALIB_SAMPLE["math500"]
    try:
        config.CALIB_SAMPLE["math500"] = {
            "difficulty": (5, 5), "n": 10, "contrast": (3, 3),
            "contrast_n": 4, "disjoint_from_run": False}
        sel = config.calib_ids("math500", _DIFF)
        assert len(sel["cap"]) == 10 and len(sel["contrast"]) == 4
        assert {_DIFF[i] for i in sel["cap"]} == {5}, "cap sample must be hard"
        assert {_DIFF[i] for i in sel["contrast"]} == {3}
        assert not set(sel["cap"]) & set(sel["contrast"])
    finally:
        config.CALIB_SAMPLE["math500"] = saved


def test_calib_sample_can_avoid_the_run_sample_entirely():
    """math500 has 500 rows and the run uses 100, so the calibration can be
    drawn from problems the analysis will never see -- worst-case coverage AND
    zero overlap, which a prefix of the run sample cannot give."""
    saved = config.CALIB_SAMPLE["math500"]
    try:
        config.CALIB_SAMPLE["math500"] = {
            "difficulty": (5, 5), "n": 5, "contrast": None, "contrast_n": 0,
            "disjoint_from_run": True}
        run_sample = set(config.problem_ids(40, len(_DIFF)))
        sel = config.calib_ids("math500", _DIFF, run_n=40)
        assert not set(sel["cap"]) & run_sample, (sel["cap"], run_sample)
        # ...and it refuses to claim disjointness it cannot verify.
        try:
            config.calib_ids("math500", _DIFF, run_n=None)
        except ValueError as e:
            assert "run n" in str(e), e
        else:
            raise AssertionError("disjointness needs a known run sample")
    finally:
        config.CALIB_SAMPLE["math500"] = saved


def test_calib_sample_nests_so_a_bigger_calibration_is_a_superset():
    """Same shuffle-prefix property as problem_ids: re-running a calibration at
    a larger n must extend the previous one, not replace it, or the two are not
    comparable and the earlier generations are wasted."""
    saved = config.CALIB_SAMPLE["math500"]
    try:
        base = {"difficulty": (5, 5), "contrast": None, "contrast_n": 0,
                "disjoint_from_run": False}
        config.CALIB_SAMPLE["math500"] = {**base, "n": 5}
        small = set(config.calib_ids("math500", _DIFF)["cap"])
        config.CALIB_SAMPLE["math500"] = {**base, "n": 12}
        big = set(config.calib_ids("math500", _DIFF)["cap"])
        assert small <= big, (sorted(small), sorted(big))
    finally:
        config.CALIB_SAMPLE["math500"] = saved


def test_calib_sample_refuses_a_pool_it_cannot_fill():
    """Silently returning 3 problems when the rule asked for 15 would size a
    cap on a sample nobody chose. The likeliest cause is a mirror without the
    difficulty field, so the message has to name that."""
    saved = config.CALIB_SAMPLE["math500"]
    try:
        config.CALIB_SAMPLE["math500"] = {
            "difficulty": (5, 5), "n": 999, "contrast": None, "contrast_n": 0,
            "disjoint_from_run": False}
        config.calib_ids("math500", _DIFF)
    except ValueError as e:
        assert "only 20 exist" in str(e) and "DIFFICULTY" in str(e), e
        return
    finally:
        config.CALIB_SAMPLE["math500"] = saved
    raise AssertionError("an unfillable calibration pool must raise")


def test_difficulty_reads_the_real_fields_and_survives_a_mirror_without_them():
    """math500 carries MATH's own `level`; aime24 has no difficulty field at
    all, so problem number is parsed out of the AoPS url as a PROXY. A mirror
    lacking either must give None, not a KeyError from inside a lambda."""
    assert scoring.difficulty_of("math500", {"level": 5}) == 5
    assert scoring.difficulty_of("aime24", {
        "url": "https://artofproblemsolving.com/wiki/index.php/"
               "2024_AIME_II_Problems/Problem_13"}) == 13
    assert scoring.difficulty_of("gsm8k", {"question": "q"}) is None
    for d, rec in (("math500", {"problem": "p"}),
                   ("aime24", {"problem": "p"}),
                   ("aime24", {"url": "nonsense"})):
        assert scoring.difficulty_of(d, rec) is None, (d, rec)


def test_committed_calib_rules_are_coherent():
    """The shipped rules, checked against the splits' real shapes: math500 has
    134 level-5 problems and aime24 has 30 problems numbered 1-15 twice."""
    m = config.CALIB_SAMPLE["math500"]
    assert m["difficulty"] == (5, 5), "the cap sample must be the hard end"
    assert m["disjoint_from_run"], "500 rows can afford a holdout"
    assert m["contrast"][1] < m["difficulty"][0], "contrast must be EASIER"
    a = config.CALIB_SAMPLE["aime24"]
    assert a["difficulty"][1] <= 15, "AIME exams have 15 problems"
    assert not a["disjoint_from_run"], "n=30 IS the dataset; nothing to hold out"
    # Smaller than a random 20 would have been: less of the dataset AND the
    # tail rather than the middle.
    assert config.calib_n("aime24") < 20, config.calib_n("aime24")
    assert config.CALIB_SAMPLE["gsm8k"] is None, "gsm8k's caps are committed"


# ================================= the premise the sampling rule rests on

def _calib_file(path, rows):
    with open(path, "w") as f:
        for i, (d, role, ntok) in enumerate(rows):
            f.write(json.dumps({
                "id": i, "cond": "cot_intact", "dataset": "math500",
                "raw": "<think>ok</think>\\boxed{7}", "gold": "7",
                "n_tok": ntok, "cap": 8192, "hit_cap": False,
                "calibration": True, "difficulty": d,
                "calib_role": role}) + "\n")


def test_contrast_group_never_reaches_the_cap_suggestion():
    """It is easier by construction, so letting it into the percentile would
    drag the cap down -- the exact failure the hard-end sample exists to
    avoid."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.jsonl")
        hard = [(5, "cap", 3000 + i) for i in range(10)]
        _calib_file(p, hard)
        alone = config.suggest_cap([r[2] for r in hard])
        _calib_file(p, hard + [(3, "contrast", 100) for _ in range(10)])
        assert run.cap_report(p, "math500") == 0
        # The suggestion is a pure function of the cap sample, so adding ten
        # very short contrast records must not move it.
        assert config.suggest_cap([r[2] for r in hard]) == alone


def test_cap_report_fails_when_difficulty_does_not_predict_length():
    """The whole sampling rule assumes hard problems write longest. If the
    easier contrast group writes as much, this sample is not the upper tail and
    the suggestion is not conservative -- so it must refuse, not print a
    number."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "c.jsonl")
        _calib_file(p, [(5, "cap", 1000 + i) for i in range(10)]
                    + [(3, "contrast", 3000 + i) for i in range(5)])
        assert run.cap_report(p, "math500") == 1

        # Same check inside a cap sample that spans levels, which is aime24's
        # shape -- there is no contrast group there to catch it.
        _calib_file(p, [(11, "cap", 3000), (11, "cap", 3100),
                        (15, "cap", 900), (15, "cap", 950)])
        assert run.cap_report(p, "aime24") == 1


# ============================================ --n has no say in calibration

def test_n_is_refused_with_calibrate_caps():
    """Which problems are measured sets the cap as surely as how many, so the
    sample is a committed choice in config.CALIB_SAMPLE and not a flag."""
    try:
        run.main(["--calibrate-caps", "--dataset", "math500", "--n", "20"])
    except SystemExit as e:
        assert "CALIB_SAMPLE" in str(e), e
        return
    raise AssertionError("--n must be refused with --calibrate-caps")


def test_calibrating_a_dataset_with_no_rule_is_refused():
    """gsm8k's caps are already committed and its data generated. Calibrating
    it would produce a sample nobody pre-registered."""
    try:
        run.main(["--calibrate-caps", "--dataset", "gsm8k"])
    except SystemExit as e:
        assert "no calibration sampling rule" in str(e), e
        return
    raise AssertionError("a dataset without a CALIB_SAMPLE rule must refuse")


# ================================= the pin guard: one file, one machine

def _pin(device="mps", rev="abc", sha="deadbeef", gpu=None, **extra):
    return {"model_revision": rev,
            "hardware": {"device": device, "backend": device.split(":")[0],
                         "torch": "2.13.0", "gpu": gpu},
            "dataset": {"rows": 1319, "content_sha256": sha},
            **extra}


def test_resuming_onto_another_backend_is_refused():
    """The confound the manifest existed to catch and used to absorb. The pin
    was rewritten on every invocation, so generating the direct arm on a Mac
    and resuming the cot arm on a rented GPU left a file claiming to be all
    CUDA -- and the interaction is a difference of differences across exactly
    those cells."""
    try:
        run.pin_guard(_pin("mps"), _pin("cuda", gpu="NVIDIA L4"),
                      allow_device_change=False, n_done=300, out="r.jsonl")
    except SystemExit as e:
        assert "mps" in str(e) and "cuda" in str(e), e
        assert "300" in str(e), "the refusal must say what is at stake"
        assert "--allow-device-change" in str(e), "and how to proceed"
        return
    raise AssertionError("a backend change on resume must refuse")


def test_the_override_records_both_devices_and_keeps_the_first():
    """--allow-device-change makes the mixing deliberate, not invisible.
    `hardware` stays as pinned so the file's origin is never overwritten; the
    history is what says the run moved."""
    out = run.pin_guard(_pin("mps"), _pin("cuda", gpu="NVIDIA L4"),
                        allow_device_change=True, n_done=300)
    assert out["hardware"]["device"] == "mps", out["hardware"]
    assert [h["device"] for h in out["hardware_history"]] == ["mps", "cuda"]


def test_the_same_backend_on_another_card_is_not_a_drift():
    """backend_of, not the device string: bf16 kernels differ between MPS and
    CUDA, not between two cards of the same architecture. Refusing here would
    make the guard cost more than it buys."""
    out = run.pin_guard(_pin("cuda:0"), _pin("cuda:1"), n_done=10)
    assert [h["device"] for h in out["hardware_history"]] == ["cuda:0",
                                                              "cuda:1"]


def test_repeated_resumes_on_one_machine_do_not_grow_the_history():
    out = run.pin_guard(_pin("mps"), _pin("mps"), n_done=10)
    assert len(out["hardware_history"]) == 1, out["hardware_history"]


def test_a_pin_predating_hardware_recording_still_resumes():
    """The committed GSM8K n=150 pin has no hardware key at all. Refusing
    would strand real data behind a guard added after it was generated -- the
    honest handling is to say the existing rows are unattributable and record
    this invocation."""
    old = {"model_revision": "abc",
           "dataset": {"rows": 1319, "content_sha256": "deadbeef"}}
    out = run.pin_guard(old, _pin("cuda"), n_done=900)
    assert out["hardware"]["device"] == "cuda"
    assert len(out["hardware_history"]) == 1


def test_a_checkpoint_change_on_resume_is_refused_with_no_override():
    """Different weights answering the same prompts. --allow-device-change is
    deliberately not a general 'mix anything' flag: there is no reading under
    which pooling two checkpoints into one interaction is correct."""
    try:
        run.pin_guard(_pin(rev="aaa"), _pin(rev="bbb"),
                      allow_device_change=True, n_done=50)
    except SystemExit as e:
        assert "aaa" in str(e) and "bbb" in str(e), e
        return
    raise AssertionError("a checkpoint change on resume must refuse")


def test_a_dataset_change_on_resume_is_refused():
    """Same ids, different problems."""
    try:
        run.pin_guard(_pin(sha="1111"), _pin(sha="2222"), n_done=50)
    except SystemExit as e:
        assert "content_sha256" in str(e), e
        return
    raise AssertionError("a dataset content change on resume must refuse")


def test_the_datasets_private_fingerprint_is_not_compared():
    """`fingerprint` is the datasets library's own hash and can differ across
    library versions and machines -- which is precisely the situation a resume
    on a rented box is in. content_sha256 is ours and is what carries the
    claim, so a differing fingerprint must not block a legitimate resume."""
    a = _pin()
    a["dataset"]["fingerprint"] = "local"
    b = _pin()
    b["dataset"]["fingerprint"] = "cloud"
    run.pin_guard(a, b, n_done=50)


def test_a_pin_beside_no_records_is_simply_replaced():
    """What makes an invocation a resume is records on disk, not a pin file. A
    pin left by an aborted start, or beside a generations file that was
    deleted, is a claim about nothing -- and refusing on it would send you
    hunting for a stale json before you could run at all."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.jsonl")
        run.write_pin(out, _pin("mps"))
        got = run.write_pin(out, _pin("cuda"), n_done=0)
        assert got["hardware"]["device"] == "cuda"
        with open(out.replace(".jsonl", "_pin.json")) as f:
            assert json.load(f)["hardware"]["device"] == "cuda"


def test_the_first_write_records_a_history_of_one():
    with tempfile.TemporaryDirectory() as d:
        got = run.write_pin(os.path.join(d, "r.jsonl"), _pin("cuda:0"))
        assert [h["device"] for h in got["hardware_history"]] == ["cuda:0"]


def test_write_pin_refuses_a_real_resume_across_backends():
    """The wiring, not just the arithmetic: the guard has to be reached from
    the path that actually writes the file."""
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "r.jsonl")
        run.write_pin(out, _pin("mps"))
        try:
            run.write_pin(out, _pin("cuda"), n_done=300)
        except SystemExit as e:
            assert "--allow-device-change" in str(e), e
            # And the pin on disk is untouched: the refusal must not leave the
            # file describing a run that was never allowed to happen.
            with open(out.replace(".jsonl", "_pin.json")) as f:
                assert json.load(f)["hardware"]["device"] == "mps"
            return
    raise AssertionError("write_pin must refuse a cross-backend resume")


# ============================ the device stamp, read back at analysis time

def _by(**cells):
    """{cond: [device, ...]} -> analyze.load's shape."""
    return {cond: {i: (1, "correct", {"id": i, "cond": cond, "device": d})
                   for i, d in enumerate(devs)}
            for cond, devs in cells.items()}


def test_one_backend_is_reported_silently():
    import analyze
    lines, crosses = analyze.devices_of(
        _by(direct_intact=["mps"] * 3, cot_intact=["mps"] * 3))
    assert lines == [] and not crosses, lines


def test_records_with_no_device_stamp_are_not_a_mix():
    """Everything generated before the stamp existed. Absence is not
    disagreement, and warning about it on every committed file would train the
    reader to skip the warning that matters."""
    import analyze
    by = {"cot_intact": {0: (1, "correct", {"id": 0, "cond": "cot_intact"})}}
    lines, crosses = analyze.devices_of(by)
    assert lines == [] and not crosses, lines


def test_a_mixed_file_is_named_cell_by_cell():
    """--allow-device-change makes the mixing possible, so something has to
    make it visible. Otherwise the override is a decision with no consequence
    and the per-record stamp is written but never read."""
    import analyze
    lines, _ = analyze.devices_of(
        _by(direct_intact=["mps", "cuda"], cot_intact=["mps"]))
    joined = "\n".join(lines)
    assert "2 backends" in joined, joined
    assert "MIXED WITHIN THE CELL" in joined, joined


def test_arms_on_different_backends_is_the_loud_case():
    """The interaction subtracts one arm from the other. If the arms disagree
    about the backend, the kernel difference is inside the headline number."""
    import analyze
    lines, crosses = analyze.devices_of(
        _by(direct_intact=["mps"], direct_ablated=["mps"],
            cot_intact=["cuda"], cot_ablated=["cuda"]))
    assert crosses, lines
    assert "ARMS differ" in "\n".join(lines)


def test_the_same_two_backends_in_both_arms_does_not_cross():
    """Still bad, still warned about -- but not the specific failure where the
    subtraction itself spans the change, so it must not claim to be."""
    import analyze
    lines, crosses = analyze.devices_of(
        _by(direct_intact=["mps", "cuda"], cot_intact=["cuda", "mps"]))
    assert lines and not crosses, (lines, crosses)


def _run_all():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"  ok    {name}")
        except Exception as e:
            failed.append(name)
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
