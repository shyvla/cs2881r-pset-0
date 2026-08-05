"""Pre-registered experimental parameters. The pre-registration, executable.

WHY THIS FILE EXISTS
--------------------
`CAPS` previously lived in an unexecuted cell of experiments.ipynb. That means
no test could reach it, a change to it produced no reviewable diff, and the
value recorded in the run manifest came from a notebook cell rather than from
anything under version control. The same was true of the k values and the
layer bands, which existed only as prose in planning notes outside the repo.

Everything here is a number that must be fixed BEFORE ablated data is seen,
and recorded in the manifest afterwards. Keeping them in one importable module
is the same discipline as condition naming (names come from a grid in
scoring.py, never hand-typed): a constant that is written twice can drift,
and drift here is silent.

UNDECIDED values are None, and the accessor for each raises. A pre-registration
choice cannot then be made accidentally at the keyboard -- the code stops.
"""

import random

# ------------------------------------------------------------ problem sample

SEED = 0


def problem_ids(n: int, dataset_size: int, seed: int = SEED) -> list[int]:
    """The ONE place problem ids are drawn. Same rule as render_prompt being
    the one place a prompt is built: a selection expressed twice can drift,
    and drift here means the pilot and the real run describe different
    problems with nothing to warn you.

    Prefix of a shuffle, so the samples NEST BY CONSTRUCTION:
        problem_ids(20) subset problem_ids(50) subset problem_ids(150)
    for every dataset size. That is what lets an n=20 pilot, an n=50
    calibration and an n=150 run be compared at all.

    WHY NOT random.sample, which the notebook used. It nests on GSM8K, but by
    accident: CPython switches between a pool shuffle and rejection sampling
    on a `setsize` heuristic that at k=150 sits at 1045. GSM8K test is 1319,
    so it stays on the rejection path and the first 20 accepted draws are the
    same whether you asked for 20 or 150. MATH-500 is 500 -- below the
    threshold, different algorithm, and the nesting breaks: measured, only 18
    of the 20 pilot problems appear in the 150. AIME has 30 problems, so
    n=150 is not meaningful there at all.

    A 1.26x margin on a CPython implementation detail is not a foundation for
    the comparability of every number in the report.

    NOTE: this returns DIFFERENT ids than the n=20 baseline run, which used
    random.sample. That is free now, because the baseline is pilot data being
    re-run at n=150, and expensive later.
    """
    if n > dataset_size:
        raise ValueError(f"asked for {n} problems from a dataset of "
                         f"{dataset_size}")
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    rng = random.Random(seed)
    perm = list(range(dataset_size))
    rng.shuffle(perm)
    return sorted(perm[:n])


# Problems already used to make design decisions. The band was selected by
# comparing flip rates across five layer windows on problem_ids(12) -- that is
# outcome data, and choosing a band on it is a garden of forking paths. It is a
# legitimate exploratory step; what is not legitimate is then testing the
# chosen band on the same problems and calling it confirmation.
EXPLORED_N = 12

# ...ON THIS DATASET. The band sweep ran on GSM8K problems, so problem_ids(12)
# names the explored set for GSM8K ALONE. The same 12 indices on another
# dataset are 12 arbitrary problems: nothing was selected on them, so removing
# them buys no protection and the "holdout" label is simply false. analyze.py
# reads this and reports the exploratory/holdout split only for this dataset --
# it used to apply the split unconditionally, which on aime24 would have
# discarded 40% of the run for nothing.
EXPLORED_ON = "gsm8k"


# n IS A PRE-REGISTRATION CHOICE, not a convenience default. It comes out of
# power.py -- the accuracies, the measured rho and the looping rate give the n
# at which the interaction is detectable -- and choosing it at the keyboard is
# choosing how many problems to look at after seeing how the first ones went.
#
# It lives here for the same reason CAPS does: `--n 150` was an argparse
# default, i.e. a GSM8K number silently applied to every dataset. On aime24 it
# is not merely wrong but impossible (30 problems), and on math500 it is 30% of
# the split with no stated justification.
#
# PER-DATASET because the power calculation is per-dataset: it depends on the
# intact accuracies, and Qwen3-4B's accuracy on MATH-500 is not its accuracy on
# GSM8K.
N_DEFAULT = {
    # The committed run. 150 was the power.py recommendation at the pilot's
    # accuracies with rho=0.5.
    "gsm8k": 150,

    # DECIDED: 150, AMENDED from 100 -- before any MATH-500 run data exists,
    # which is what makes it an amendment rather than a rescue.
    #
    # 100 was a committed budget decision, not a power output. The amendment
    # is a power decision made the same pre-data way: power.py at a
    # GSM8K-sized interaction (0.40/0.10/0.90/0.80, rho=0.5) gives n=100 a
    # power of 0.82 with a 27-point CI -- at the edge of its own >=0.80 rule
    # and past the ~25-point width its output calls uninterpretable -- against
    # 0.97 and 22 points at n=150. Under a +10 interaction both are hopeless
    # (0.30 vs 0.40 power; 0.80 needs n~400, which is 80% of the split and
    # starves CALIB_SAMPLE's disjoint pool). So the extra 50 problems buy
    # margin exactly and only in the world where the effect is detectable at
    # all. Deliberately still not GSM8K's 150-by-coincidence: the number was
    # chosen on MATH-500 scenario accuracies, and the reasoning above is what
    # a later reader should disagree with.
    #
    # ONE CONSEQUENCE, priced in: calib_ids draws DISJOINT from the run
    # sample, so moving n re-draws the calibration sample -- measured, only 4
    # of the 20 cap problems survive from the n=100 draw, and 91 of the 134
    # level-5 problems remain outside the n=150 sample (4.5x the draw).
    # Nothing measured is lost: the only calibration taken at n=100 came back
    # censored at the 8192 ceiling and has to be deleted and re-measured
    # anyway (see MEASURE_CAP). But this amendment must be committed BEFORE
    # that re-measurement runs, or the caps would be sized on problems the
    # run sample now contains.
    #
    # WHAT TO DO WITH IT LATER: once the intact cells exist, run power.py at
    # the observed accuracies with analysis.observed_rho, and REPORT the power
    # this n actually had. If it is low, that is a stated limitation of a
    # pre-registered design; it is not a licence to raise n and re-read the
    # result. problem_ids is a shuffle prefix, so a later extension is a
    # strict superset and nothing already generated is wasted -- but an
    # extension decided after seeing the interaction is a different experiment
    # and has to be labelled as one.
    #
    # HOW TO EXTEND, mechanically: copy the .jsonl AND its _pin.json to the
    # larger n's run_path, then resume there. run.pin_guard accepts the new
    # sample when the pinned ids nest inside it and their rows still hash
    # identically, and records the old n in the pin's sample_history. Note
    # that the calibration sample's disjoint-from-run claim was verified at
    # the OLD n -- recompute the overlap and report it.
    "math500": 150,

    # The WHOLE dataset, not a sample. 30 problems is what AIME 2024 has, so
    # there is nothing to choose and nothing to power-analyse -- report it as
    # the whole dataset. Note that at n=30 the CIs will be wide; that is a
    # property of the benchmark, and the report should say so rather than
    # implying a null.
    "aime24": 30,
}


def n_default(dataset: str) -> int:
    """The pre-registered problem count for `dataset`.

    Raises rather than falling back to another dataset's n, for the same
    reason cap_for does: the failure mode of a default is a GSM8K number
    running under another dataset's filename.
    """
    if dataset not in N_DEFAULT:
        raise KeyError(f"no N_DEFAULT for dataset {dataset!r}; "
                       f"known: {sorted(N_DEFAULT)}")
    n = N_DEFAULT[dataset]
    if n is None:
        raise ValueError(
            f"N_DEFAULT[{dataset!r}] is unset -- an UNDECIDED "
            f"pre-registration choice. Derive it from power.py and commit it "
            f"before generating; a sample size chosen after seeing the first "
            f"problems is the post-hoc rescue this file exists to prevent.")
    return n


def holdout_ids(n: int, dataset_size: int, seed: int = SEED) -> list[int]:
    """The confirmatory sample: problem_ids(n) minus the explored ones.

    Free only because problem_ids is a shuffle prefix and therefore nests, so
    the first EXPLORED_N problems are a subset of any larger draw and removing
    them leaves the rest untouched. With random.sample this would not have been
    expressible at all.

    The report gets both: the exploratory 12 labelled as the band-selection
    set, and this as the test. Pooling them would inflate the result by exactly
    the amount the selection was worth.
    """
    explored = set(problem_ids(EXPLORED_N, dataset_size, seed))
    return [i for i in problem_ids(n, dataset_size, seed) if i not in explored]


# ---------------------------------------------------------------- generation

# The levels the MVP run actually generates. `nothink` is in LEVELS and in CAPS
# as the middle rung of the externalisation axis, but run.conditions() builds
# only these two, so a readiness check that demanded CAPS[d]['nothink'] would
# block a dataset on a cap for a cell that is never generated. It went unnoticed
# on GSM8K only because that entry happens to be set.
#
# run.conditions() iterates this, so the grid and the readiness check cannot
# drift apart: adding `nothink` to the run means adding it here, which
# immediately makes its cap a blocking requirement, which is correct.
RUN_LEVELS = ("cot", "direct")

# Keyed by DATASET, then by LEVEL. The cap is a property of how much the
# condition writes, not of whether it is ablated, so the Half B states
# (ablated, random) need no new entries. This is also what stops the asymmetry
# `cap_warnings` was written to detect: a cap that binds in one state and not
# its partner converts verbosity into wrongness.
#
# PER-DATASET because every number below was calibrated on GSM8K and on nothing
# else. MATH-500 and AIME problems are harder and their traces are longer, so
# reusing GSM8K's caps would let the cap bind on one dataset and not another --
# and `cap_warnings` cannot see that, because it compares cells WITHIN a run.
# One constant with two meanings is the same silent-drift failure as CAPS
# living in an unexecuted notebook cell.
CAPS = {
    "gsm8k": {
        "cot": 3072,
        "nothink": 512,
        # 32 -> 128. The cap was first set to 32 on the basis of "max observed
        # was 7 tokens", which was measured on INTACT runs only. The
        # intervention-placement probe (probes/intervention.py) then showed
        # the cap binding on 100% of intervened direct generations.
        # cap_warnings' own docstring already prescribed 128-256 for exactly
        # this reason.
        #
        # The score is probably unaffected either way: the \boxed{ prefill
        # closed at token 2 in both observed cases and
        # extraction_mode="first_match" always takes that first box however
        # much junk follows. But that is two observations of NOISE, not of the
        # real ablation, and the case where the brace never closes inside the
        # cap is reachable -- and there truncation really would be doing part
        # of the work. 128 costs ~4 extra seconds per direct generation and
        # removes the ambiguity.
        "direct": 128,
    },
    # UNDECIDED, and deliberately NOT copied from GSM8K. Both need a measured
    # cap before any ablated data exists: run the intact cells first, read the
    # hit_cap rate, set these above it with headroom. A guessed cap is cheap to
    # write and expensive to discover, because a binding cap looks exactly like
    # the ablation making the model fail to answer.
    "math500": {"cot": None, "nothink": None, "direct": None},
    "aime24": {"cot": None, "nothink": None, "direct": None},
}


# ------------------------------------------------- the calibration ceiling
#
# NOT a cap, and never used by a run that produces analysable data. This is the
# ceiling for `run.py --calibrate-caps`, whose only job is to make the CAPS
# entries above measurable instead of guessed.
#
# The chicken-and-egg it resolves: CAPS' own comment prescribes "run the intact
# cells first, read the hit_cap rate, set these above it with headroom", but
# every path to running anything went through cap_for(), which raises on an
# unset cap. The prescribed measurement was not expressible. A guessed cap is
# cheap to write and expensive to discover, because a binding cap looks exactly
# like the ablation making the model fail to answer -- so the measurement has
# to be reachable or the guess happens anyway.
#
# Chosen to be GENEROUS, not tight: the point is a distribution of n_tok that
# is NOT censored, so that a percentile means what it says. Roughly 2.7x GSM8K's
# cot cap and 4x its direct cap, which is the right order for datasets whose
# traces are longer. --calibrate-caps reports the hit-rate AT this ceiling, so a
# ceiling that still binds is visible rather than silently truncating the very
# distribution being measured.
#
# INTACT CELLS ONLY. Ablated generations run longer and loop, so calibrating a
# cap on them would size it to degenerate behaviour -- and it would also mean
# seeing ablated outcomes before the pre-registration is closed, which is the
# thing this file exists to prevent.
# cot RAISED 8192 -> 16384 after the first math500 calibration came back
# censored: 35% of the 20 level-5 problems ran to 8192, with a median of 6684,
# so a real part of the mass sits just above the old ceiling. The direct
# ceiling is untouched -- it measured a max of 10 tokens against 512.
MEASURE_CAP = {"cot": 16384, "nothink": 2048, "direct": 512}

# THE STOPPING RULE, committed BEFORE the 16384 calibration is read, because a
# rule written afterwards is a judgement made on the outcome.
#
#     If the re-measurement still hits the ceiling on more than 15% of the cap
#     sample, DO NOT RAISE AGAIN. The finding is then that Qwen3-4B does not
#     reliably terminate on this dataset, which is a property of the model and
#     the benchmark, not a measurement that a bigger number will fix. Set the
#     cot cap from what the run can afford, and report the per-cell
#     `incomplete` rate as a stated limitation of the design.
#
# WHY A RULE AND NOT A JUDGEMENT. Each doubling costs a full re-calibration and
# then roughly +50% on a ~21h run, and the temptation at 20% censoring will be
# to raise "just once more". The first calibration also showed the runaway
# behaviour is NOT confined to the hard tail -- one of the five level-3
# contrast problems hit 8192 too -- so the tail may not be bounded at all, and
# an unbounded tail cannot be chased to a cap.
CEILING_RETRY_MAX_HIT = 0.15

# Headroom over the measured distribution when suggesting a cap. 1.5x the p99,
# rounded up to a multiple of 128. Stated here rather than buried in run.py
# because it is the rule that turns a measurement into a pre-registered number,
# and a later reader should be able to disagree with it.
CAP_HEADROOM = 1.5
CAP_ROUNDING = 128


def suggest_cap(n_tok: list[int]) -> int:
    """A cap suggestion from observed intact generation lengths.

    Deliberately a SUGGESTION printed by --calibrate-caps, never written to
    CAPS automatically: a pre-registration choice that a script can edit is not
    a pre-registration. The caller commits it by hand, which is what makes it
    reviewable as a diff.
    """
    if not n_tok:
        raise ValueError("no generations to size a cap from")
    ordered = sorted(n_tok)
    # p99 by nearest-rank, which on a small sample is simply the max -- the
    # honest behaviour, since at n=20 there is no 99th percentile to estimate.
    p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))]
    want = CAP_HEADROOM * max(p99, 1)
    return int(CAP_ROUNDING * -(-want // CAP_ROUNDING))


# ------------------------------------- WHICH problems the calibration reads
#
# NOT a random sample, and the reason is that the cap's loss is ASYMMETRIC. A
# cap set too low converts verbosity into wrongness, and does it preferentially
# in the degraded cells -- straight into the interaction term. A cap set too
# high costs GPU seconds, and only on the generations that would have run away
# anyway, since a short answer terminates early whatever the ceiling is.
#
# So the quantity the cap has to clear is the UPPER TAIL of the length
# distribution, and a random 20 drawn from a run of 100 systematically
# underestimates the maximum of 100. CAP_HEADROOM was silently doing that job;
# sampling the hard end replaces a fudge factor with a measurement.
#
# Both datasets carry a difficulty signal (scoring.DIFFICULTY), so this is
# selection on METADATA, never on outcome data -- no result is looked at, and
# the garden of forking paths that EXPLORED_N records does not apply. It is
# nonetheless a pre-registered choice, committed here rather than typed at the
# keyboard, because "which problems did you measure on" changes the number.
#
#   difficulty    inclusive range on scoring.DIFFICULTY. The CAP SAMPLE.
#   n             how many to draw from it.
#   contrast      an EASIER range, drawn only to check that difficulty really
#                 does predict length. NEVER feeds the cap suggestion -- easier
#                 problems are shorter and would drag the percentile down.
#   disjoint_from_run  exclude the run's own sample, where the split is big
#                 enough to afford it.
CALIB_SAMPLE = {
    # Caps already committed and the data already generated. Nothing to draw.
    "gsm8k": None,

    # 134 of the 500 problems are level 5, and 43 of those fall inside the
    # n=150 run sample -- so 20 can be drawn from the 91 the run will never
    # touch, with the pool still 4.5x the draw. Worst-case coverage AND
    # zero overlap with the analysed sample, which is the one combination a
    # prefix of the run sample cannot give. (The numbers were 31 inside/103
    # outside at the original n=100; see N_DEFAULT for why the amendment
    # re-drew this sample and why no measurement was lost.)
    #
    # 20 rather than 15 because the number this sample exists to estimate is a
    # TAIL, and the tail is what a small sample estimates worst: suggest_cap
    # reads the p99 by nearest rank, which at n=15 or n=20 is simply the
    # longest trace observed. More draws is a longer observed maximum and a
    # cap less likely to bind on the run. The cost is 5 more intact
    # generations, paid once.
    #
    # The 5 level-3 problems are the monotonicity check, and stay at 5 -- they
    # never enter the cap arithmetic, so more of them would buy nothing. If
    # level 3 traces come out LONGER than level 5, the premise of this whole
    # sampling rule is wrong and cap_report says so instead of quietly sizing
    # on the wrong tail.
    "math500": {"difficulty": (5, 5), "n": 20,
                "contrast": (3, 3), "contrast_n": 5,
                "disjoint_from_run": True},

    # 30 problems, all of them in the run, so there is nothing to hold out and
    # `disjoint_from_run` would leave an empty pool. Problems 11-15 of each
    # exam give 10 at the hard end -- BETTER than a random 20 on both counts:
    # a third of the dataset instead of two thirds, and the tail instead of the
    # middle.
    #
    # No contrast group: 11-15 already spans five difficulty values, so the
    # slope is visible inside the cap sample itself. Adding easier problems
    # here would only push the overlap with the run back up, which is the thing
    # the smaller n was chosen to avoid.
    "aime24": {"difficulty": (11, 15), "n": 10,
               "contrast": None, "contrast_n": 0,
               "disjoint_from_run": False},
}


def calib_n(dataset: str) -> int:
    """How many problems the calibration will generate, cap + contrast.

    Derivable without loading the dataset, so the output filename is known
    before anything is downloaded.
    """
    spec = CALIB_SAMPLE.get(dataset)
    if spec is None:
        raise ValueError(f"no CALIB_SAMPLE rule for {dataset!r}")
    return spec["n"] + spec["contrast_n"]


def calib_ids(dataset: str, difficulty, run_n: int | None = None,
              seed: int = SEED) -> dict:
    """The calibration sample, as {"cap": [ids], "contrast": [ids]}.

    `difficulty` is a sequence of per-problem difficulty values indexed by
    problem id -- built by the caller from scoring.difficulty_of, because
    reading a record is scoring's job and choosing which records is this
    file's.

    Shuffled with SEED and taken as a prefix, exactly like problem_ids, so a
    calibration re-run at a larger n is a superset of the smaller one and the
    two are comparable.
    """
    spec = CALIB_SAMPLE.get(dataset)
    if spec is None:
        raise ValueError(
            f"CALIB_SAMPLE[{dataset!r}] is None -- no calibration sampling "
            f"rule is pre-registered for this dataset. gsm8k's caps are "
            f"already committed; for anything else, add a rule to config.py "
            f"and commit it before measuring.")

    exclude = set()
    if spec["disjoint_from_run"]:
        if run_n is None:
            raise ValueError(
                f"CALIB_SAMPLE[{dataset!r}] asks to exclude the run sample but "
                f"no run n is known -- set N_DEFAULT[{dataset!r}] first, or the "
                f"'disjoint' claim is unverifiable.")
        exclude = set(problem_ids(run_n, len(difficulty), seed))

    def draw(rng_range, k, label):
        if not k or rng_range is None:
            return []
        lo, hi = rng_range
        pool = [i for i, d in enumerate(difficulty)
                if d is not None and lo <= d <= hi and i not in exclude]
        if len(pool) < k:
            raise ValueError(
                f"{dataset} {label} sample wants {k} problems with difficulty "
                f"in [{lo}, {hi}]"
                + (f" outside the n={run_n} run sample" if exclude else "")
                + f", but only {len(pool)} exist. Either the difficulty field "
                  f"did not read (scoring.DIFFICULTY[{dataset!r}]) or "
                  f"CALIB_SAMPLE[{dataset!r}] asks for more than the split "
                  f"holds.")
        rng = random.Random(seed)
        rng.shuffle(pool)
        return sorted(pool[:k])

    return {"cap": draw(spec["difficulty"], spec["n"], "cap"),
            "contrast": draw(spec["contrast"], spec["contrast_n"],
                             "contrast")}


def cap_for(cond: str, dataset: str) -> int:
    """max_new_tokens for a condition name like 'direct_ablated' on `dataset`.

    `dataset` is REQUIRED and has no default. A default of "gsm8k" is precisely
    the bug this signature exists to close: run.py once accepted a --dataset flag
    and then loaded GSM8K regardless, so a MATH-500 run would have generated
    GSM8K problems under a MATH-500 filename. A parameter that CAN be forgotten
    eventually is.

    Raises on an unset cap rather than returning None, because None reaches
    model.generate as max_new_tokens=None -- which does not fail, it generates
    to the context limit.
    """
    if dataset not in CAPS:
        raise KeyError(f"no caps for dataset {dataset!r}; "
                       f"known: {sorted(CAPS)}")
    level = cond.split("_")[0]
    if level not in CAPS[dataset]:
        raise KeyError(f"no cap for level {level!r} (from cond {cond!r}) on "
                       f"dataset {dataset!r}; known levels: "
                       f"{sorted(CAPS[dataset])}")
    cap = CAPS[dataset][level]
    if cap is None:
        raise ValueError(
            f"CAPS[{dataset!r}][{level!r}] is unset -- an UNDECIDED "
            f"pre-registration choice. Set it in config.py and commit before "
            f"running ablated data; choosing a cap after seeing results is the "
            f"post-hoc rescue this file exists to prevent.")
    return cap


# --------------------------------------------------- direct-condition prompt

# THE one definition of the direct condition's instruction and prefill. This
# string was copy-pasted into three probes and the run script -- the same "a
# constant written twice can drift" failure as CAPS in a notebook cell, and
# worse here, because probes/capture.py asserts a prompt FINGERPRINT
# (683d8ea5f9e42c80) against the committed run manifest: a one-word edit to any
# single copy makes that probe describe a condition that never ran.
#
# The prefill is shared across datasets. All three want the answer in \boxed{},
# and scoring's extraction_mode="first_match" is built on that.
DIRECT_PREFILL = "\\boxed{"

DIRECT_INSTRUCTION = {
    # Byte-for-byte what the n=20 baseline and n=150 GSM8K runs used. Do NOT
    # reword it: the archived run manifest and the fingerprint assertion in
    # probes/capture.py both pin this exact string, and the committed n=150
    # data stays comparable to later runs only while it holds.
    "gsm8k": ("\n\nRespond with only the final numeric answer and nothing "
              "else. Do not show any reasoning."),

    # DECIDED: GSM8K's wording MINUS the word "numeric", and nothing else.
    #
    # GSM8K's string is wrong here and must not be reused: it asks for "the
    # final numeric answer", and MATH-500 answers frequently are not numeric --
    # \frac{3}{2}, 2\sqrt{2}, \frac{\pi}{2}, (2,5), intervals. test_scoring.py
    # already carries all of those shapes, so the SCORER handles them; it is
    # the PROMPT that would be instructing the model away from the format its
    # own gold uses. That inflates `unparsed`, and it does so preferentially in
    # the degraded (ablated) cells, which puts the artifact straight into the
    # interaction term.
    #
    # WHY A ONE-WORD EDIT rather than a better prompt. The direct condition's
    # job is to be the SAME instruction across datasets, differing only where a
    # dataset forces it to: any other wording change would confound "MATH-500
    # is harder" with "MATH-500 was asked differently". The \boxed{} prefill --
    # which is what actually makes the condition direct -- is identical across
    # all three, and no format instruction is needed because the prefill has
    # already committed the model to the format the scorer reads.
    "math500": ("\n\nRespond with only the final answer and nothing else. "
                "Do not show any reasoning."),

    # DECIDED: GSM8K's wording VERBATIM. AIME answers ARE integers 0-999 (which
    # scoring.validate_gold enforces), so "the final numeric answer" is
    # semantically correct as written, and the byte-identical string keeps the
    # aime24 and gsm8k direct cells as close to the same condition as two
    # datasets allow.
    "aime24": ("\n\nRespond with only the final numeric answer and nothing "
               "else. Do not show any reasoning."),
}


def direct_prompt(dataset: str) -> tuple[str, str]:
    """(suffix, prefill) for the direct condition on `dataset`.

    Resolved in ONE place for the same reason `projection()` is: the probes and
    the run loop must build an identical prompt, or a measurement describes a
    different condition than the run it was taken to calibrate.
    """
    if dataset not in DIRECT_INSTRUCTION:
        raise KeyError(f"no direct instruction for dataset {dataset!r}; "
                       f"known: {sorted(DIRECT_INSTRUCTION)}")
    suffix = DIRECT_INSTRUCTION[dataset]
    if suffix is None:
        raise ValueError(
            f"DIRECT_INSTRUCTION[{dataset!r}] is unset -- an UNDECIDED "
            f"pre-registration choice. Write the instruction in config.py and "
            f"commit it before generating. A prompt chosen after seeing which "
            f"wording scored better is the post-hoc rescue this file exists "
            f"to prevent, with extra steps.")
    return suffix, DIRECT_PREFILL


def dataset_ready(dataset: str, levels=RUN_LEVELS,
                  need_n: bool = True) -> list[str]:
    """Which per-dataset pre-registration choices are still unset.

    Empty means the dataset can be run. The per-dataset counterpart to
    `undecided()`, which covers top-level scalars: _UNDECIDED and require()
    operate on module globals, and a nested table entry is not one. The
    accessors above raise on their own behalf; this reports the same thing
    without raising, for a pre-flight or a report appendix.

    SCOPED TO `levels`, which is what the caller is actually going to generate.
    Two things were previously demanded that no run needed:

      * CAPS[d]['nothink'], for a cell run.conditions() does not build. See
        RUN_LEVELS.
      * DIRECT_INSTRUCTION[d] for a cot-only staged run, which never renders a
        direct prompt. The file is resumable and `--only` exists precisely so
        the arms can be paid for separately, so a staged cot run must not be
        blocked on a choice belonging to the other arm.

    `need_n` is False for callers that were given an explicit `--n` and
    therefore do not consult N_DEFAULT.
    """
    caps = CAPS.get(dataset, {})
    out = [f"CAPS[{dataset!r}][{lvl!r}]"
           for lvl in sorted(levels) if caps.get(lvl, None) is None]
    if "direct" in levels and DIRECT_INSTRUCTION.get(dataset, None) is None:
        out.append(f"DIRECT_INSTRUCTION[{dataset!r}]")
    if need_n and N_DEFAULT.get(dataset, None) is None:
        out.append(f"N_DEFAULT[{dataset!r}]")
    return out


# -------------------------------------------------------------- ablation band

# Fractional depths on the paper's reindexed 0-100 scale, converted to layer
# indices by hooks.band_from_depth so the 36-layer numbers are never typed by
# hand. Ablation STRENGTH is band width -- not k, not magnitude.
#
# LIGHT is anchored: the paper ablates L38-54 for its experiential-report
# experiments and notes that larger k or later ranges impair coherence.
# HEAVY is the full workspace range L38-92.
# MEDIUM is INTERPOLATED BY US and does not correspond to anything in the
# paper. Say so in the report rather than implying three matched strengths.
BANDS = {
    "light":  (0.38, 0.54),   # -> layers 14-19
    "medium": (0.38, 0.73),   # -> layers 14-26   INTERPOLATED, not the paper's
    "heavy":  (0.38, 0.92),   # -> layers 14-33   collides with the motor region
}

# Primary band for the MVP. LIGHT, not HEAVY: the paper reports that on its
# smallest model (Haiku 4.5) ablation degraded coherence before producing any
# qualitative change, and Qwen3-4B is well below that. A band that induces
# incoherence cannot distinguish "J-space removed" from "model broken".
PRIMARY_BAND = "light"

# A fixed-width sliding window (14-19, 20-25, 26-31) plus a width sweep was run
# to test whether the band was the problem, because the directions probe found
# the direct answer is not readable in 14-19 at all (gold@10 = 0% through the
# whole band, first reaching 100% at layer 32). It was not the problem. At the
# pre-registered projection (each / gain-scaled) with the exclusion rule ON,
# every window looks the same:
#
#   window   ablation  random   correct-subset discordance (abl-only/ctl-only)
#   14-19      5/12     4/12                 0 / 0
#   20-25      4/12     4/12                 0 / 0
#   26-31      3/12     4/12                 1 / 0
#   14-33      5/12     5/12                 0 / 0   <- and both break the SAME 5
#
# So PRIMARY_BAND stays at the paper's anchor. That is not inertia: an earlier
# sweep run with mode="span" and the exclusion rule OFF showed 2/0 and 2/0
# discordance at 26-31 and 14-33 and looked like real selectivity, and moving
# the band on it would have been selection on outcome data. The clean sweep
# removed both the reason to move and the forking-paths problem with it.
# EXPLORED_N above still records that 12 problems were spent looking.


# -------------------------------------------------------------------- k values

# THREE tens, meaning three different things. Do not cross them.
K_ABLATE = 10       # top-k J-lens vectors zeroed at each position
EXCLUDE_TOPK = 10   # top-k tokens of the CLEAN pass that are exempt from that
K_OCCUPANCY = 25    # sparse-decomposition occupancy; NOT used for ablation

# K_ABLATE selects by inner product; K_OCCUPANCY selects by gradient pursuit.
# The paper is explicit that these give different, differently-redundant sets,
# so they are different methods and not just different numbers.


# --------------------------------- exclusion rule: DECIDED, AS THE PAPER STATES

# The paper exempts, at each position, the J-lens vectors of the tokens in the
# top-10 of a CLEAN forward pass, so the intervention targets internal
# reasoning rather than "the ability to report". We honour it. That costs a
# paired un-ablated forward pass at every generation step -- roughly 2x on the
# two most expensive cells -- and it is not optional.
#
# THIS REVERSES AN EARLIER DECISION, and the reversal is the point. We first
# set this False on an overlap measurement taken at band 14-19 on GSM8K, where
# the rule exempts 0.18-0.76 of 10 directions and looked inert. Two later
# measurements killed that:
#
#  1. WHERE. The exclusion-exposure probe at band 26-31 -- the band the
#     directions probe says the answer is actually computed in -- measures the
#     worst-layer overlap during generation at 5.05 of 10, not 0.76. The rule exempts HALF the selected
#     directions there. It rises monotonically with depth, so an overlap
#     measured at one band says nothing about another.
#
#  2. WHAT IT WAS DOING. The damage-floor probe, SST-2, band 14-33, ablation
#     flip rate with the rule OFF vs ON:
#
#             SST-2      MMLU
#       off     84%       24%
#        on      8%       12%
#
#     A 10x collapse on SST-2. With the rule off, the ablation was removing the
#     directions of the answer tokens themselves -- SST-2's answer is one of
#     two words, readable early, so it sits IN the top-10 at the final
#     position. That is destroying the readout, not the workspace, which is
#     exactly what the exemption is for. The 84% was an artifact of our
#     deviation, not a finding about Qwen3-4B.
#
# CONSEQUENCE: every flip count measured with this False is contaminated and
# must be re-measured. Cheapness was the only argument for dropping it, and
# cheapness is not a reason to run a different experiment than the one the
# hypothesis is about.
#
# SCOPE: the rule bites on the ABLATE selection only. The random controls
# draw uniformly from all 151,936 unembedding rows, where a draw landing on a
# clean top-10 token is a ~0.07% event per position -- negligible under either
# reading of the paper -- and hooks.make_ablation consumes the exclusion set
# for kind="ablate" alone. run.exclusion_for therefore skips the paired clean
# pass for the rand cells: paying it there doubled one of the two most
# expensive cells to compute a set nothing read, and skipping it leaves the
# generated tokens bit-identical.
USE_EXCLUSION = True


# -------------------------------------------------- DECIDED from measurement

# "each", the paper's literal wording: zero the projection onto EACH of the
# top-k vectors in turn. The worry was that ten overcomplete, non-orthogonal
# directions make this order-dependent and leave part of their span behind,
# which "span" would not. The directions probe measured how much that matters
# on Qwen3-4B under J = I:
#
#     eff_rank 9.13 of 10   mean |cos| 0.132   (random tokens: 9.54 / 0.127)
#
# The selected ten are as close to orthogonal as ten random unembedding rows,
# so the two modes are nearly the same operation -- measured displacement 0.145
# ("each") against 0.121 ("span"). Given that, take the literal reading.
#
# Two practical consequences, neither of them the reason but both worth
# recording: "span" runs an SVD per position per layer, MPS has no kernel for
# it (hooks.project_out falls back to CPU), and with USE_EXCLUSION=True the
# projection now runs at every generation step. "each" needs no decomposition.
PROJECTION_MODE = "each"

# True: remove W_U[t] * g, not the bare W_U[t]. The SELECTION must carry the
# gain (hooks.readout_scores) -- that was never a choice. This is about the
# direction REMOVED, and the deciding argument is self-consistency:
#
#     readout score for token t  =  <W_U[t] * g, h> / rms(h)
#
# so the component of h that produces t's readout lies along W_U[t] * g.
# Removing that component zeroes the score, by construction. Removing the bare
# W_U[t] does not -- it leaves a residual readout for a token the ablation
# claims to have removed, which would make "we zeroed the projection onto the
# J-lens vector" false of the quantity the J-lens actually reads.
#
# The literal-reading argument points the other way (the paper's J-lens vector
# is a row of W_U J_l, and under J = I that is W_U[t] with the norm sitting
# between it and h). We take self-consistency over literalism here because the
# ablation is defined by its EFFECT on the readout, and note the choice in the
# report. It costs displacement: 0.145 gain-scaled against 0.200 bare.
PROJECT_GAIN_SCALED = True

# The degeneration gate: before paying for the expensive cells, run ablated
# CoT on a few problems and stop to revise if generation has degenerated badly
# enough that the comparison is meaningless. Three things had to be settled --
# what it reads, against what baseline, and how big.
#
# WHAT IT READS: "unusable", i.e. outcome in {incomplete, unparsed, error}, NOT
# distinct10. On the M4 baseline a genuinely non-terminating trace (id 286,
# 3072 tokens of "Wait, but maybe...") scored distinct10 = 0.929 against a 0.5
# threshold, while a CORRECT trace scored lower at 0.877. distinct10 measures
# LEXICAL novelty and this looping is semantically repetitive but lexically
# fresh, so the detector is not merely mis-thresholded -- it is measuring the
# wrong thing and cannot be rescued by moving the number. hit_cap ->
# `incomplete` did catch id 286. distinct10 stays in the score file and in the
# report as a diagnostic; it is not the gate.
#
# AGAINST WHAT BASELINE: a DELTA over the matched intact cell, not an absolute
# rate. cot_intact already sits at 5% incomplete at n=20 (id 286 again), so an
# absolute 10% threshold leaves one problem of headroom and would fire on
# sampling noise. The question the gate exists to answer is comparative --
# "did ablation break generation so badly that the arms are not comparable" --
# so the measurement should be too.
#
# HOW BIG: 0.15. At the gate's n=20 one problem is 5 points, so this is 3
# problems. That is coarse, and it is stated as coarse rather than dressed up:
# the gate is a tripwire against catastrophe, not a test. GATE_N is separate
# from the run's n so that raising one does not silently move the other.
#
# WHERE IT RUNS: inside the cot_ablated cell, after its first n problems
# (run.ablated_gate) -- the gate observes the cell it protects. An earlier
# implementation fired after cot_random instead, a proxy that never saw
# ablated generation at all when "targeted ablation degenerates uniquely" is
# the hypothesis itself, and that a staged `--only cot_ablated` resume
# skipped entirely. The cot_random comparison survives as a secondary
# tripwire for breakage upstream of the hypothesis.
LOOP_GATE = {
    "signal": "unusable",     # outcome in {incomplete, unparsed, error}
    "mode": "delta",          # vs the matched intact cell
    "threshold": 0.15,
    "n": 20,
}
UNUSABLE_OUTCOMES = ("incomplete", "unparsed", "error")

# Only genuinely open items belong here. PROJECTION_MODE and
# PROJECT_GAIN_SCALED were settled from the directions probe and the readout
# algebra;
# their reasoning is above, where a later reader can disagree with it.
# Empty: every top-level pre-registration choice is settled, each with its
# measurement recorded above. require() still raises on any None, so a constant
# that is later unset -- or misspelled -- cannot pass silently.
#
# SCOPE. This dict holds module GLOBALS, which is all require() can reach. The
# per-dataset tables (CAPS, DIRECT_INSTRUCTION, N_DEFAULT) have open entries for
# math500 and aime24; those are nested, so they are gated by cap_for(),
# direct_prompt() and n_default() raising on their own behalf, and reported
# without raising by dataset_ready(). "Empty" here therefore does NOT mean the
# whole pre-registration is closed -- ask dataset_ready(d) per dataset.
# Currently open: every CAPS entry for math500 and aime24, and nothing else --
# measure them with `run.py --calibrate-caps`. The direct-condition prompts and
# the sample sizes are settled for all three.
_UNDECIDED = {}


def require(name: str):
    """Fetch a pre-registered value, refusing to proceed if it is unset.

    Raises on ANY None, not only on names still listed in _UNDECIDED. The
    narrower version had a hole that only opened once a choice was settled:
    removing PROJECTION_MODE from _UNDECIDED made require("PROJECTION_MODE")
    return None silently if the constant were ever unset again. _UNDECIDED then
    controls the QUALITY of the message, not whether there is one -- an unknown
    name also raises, so a typo cannot read as "nothing set here".
    """
    if name not in globals():
        raise ValueError(f"{name} is not a config constant; typo?")
    val = globals()[name]
    if val is None:
        why = _UNDECIDED.get(name)
        raise ValueError(
            f"{name} is unset"
            + (f" -- an UNDECIDED pre-registration choice ({why})" if why
               else " (it is not listed as undecided, so this is unexpected)")
            + ". Set it in config.py and commit before running ablated data; "
              "choosing it after seeing results is the post-hoc rescue this "
              "file exists to prevent.")
    return val


def projection(mode: str | None = None, gain: bool | None = None):
    """The projection choice, resolved in ONE place, as (mode, gain_scaled).

    Every probe and the run loop must agree on this or a measurement describes
    a different operation than the run it was taken to calibrate. That is not
    hypothetical: the calibration probe hardcoded ("span", True) as a working
    assumption while these were still None, the comment saying so went stale
    the moment they were settled, and an entire session's flip rates were
    measured with "span" against a pre-registration that says "each". Same
    failure as CAPS living in an unexecuted notebook cell -- a constant
    expressed twice, diverging silently.

    Explicit arguments override, for exploration only, and a caller that
    overrides should say so in its output.
    """
    return (require("PROJECTION_MODE") if mode is None else mode,
            require("PROJECT_GAIN_SCALED") if gain is None else gain)


def undecided() -> list[str]:
    """Which top-level pre-registration choices are still open.

    Globals only. For the nested per-dataset tables use dataset_ready(d).
    """
    return [k for k in _UNDECIDED if globals().get(k, None) is None]
