"""Pre-registered experimental parameters. The pre-registration, executable.

WHY THIS FILE EXISTS
--------------------
`CAPS` previously lived in an unexecuted cell of experiments.ipynb. That means
no test could reach it, a change to it produced no reviewable diff, and the
value recorded in the run manifest came from a notebook cell rather than from
anything under version control. The same was true of the k values and the
layer bands, which existed only as prose in the handoff document.

Everything here is a number that must be fixed BEFORE ablated data is seen,
and recorded in the manifest afterwards. Keeping them in one importable module
is the same discipline as decision 17 (condition names come from a grid, never
hand-typed): a constant that is written twice can drift, and drift here is
silent.

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

    NOTE: this returns DIFFERENT ids than the Milestone 4 run, which used
    random.sample. That is free now, because M4 is pilot data being re-run at
    n=150, and expensive later.
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
        # 32 -> 128. Decision 20 set 32 on the basis of "max observed was 7
        # tokens", which was measured on INTACT runs only. Milestone 6 showed
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


def cap_for(cond: str, dataset: str) -> int:
    """max_new_tokens for a condition name like 'direct_ablated' on `dataset`.

    `dataset` is REQUIRED and has no default. A default of "gsm8k" is precisely
    the bug this signature exists to close: m8_run.py accepted a --dataset flag
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
            f"post-hoc rescue that decision 25 exists to prevent.")
    return cap


# --------------------------------------------------- direct-condition prompt

# THE one definition of the direct condition's instruction and prefill. This
# string was copy-pasted into m5_probe, m6_probe, m7_calibration and m8_run --
# the same "a constant written twice can drift" failure as CAPS in a notebook
# cell, and worse here, because m5_probe asserts a prompt FINGERPRINT
# (683d8ea5f9e42c80) against the committed run manifest: a one-word edit to any
# single copy makes that probe describe a condition that never ran.
#
# The prefill is shared across datasets. All three want the answer in \boxed{},
# and scoring's extraction_mode="first_match" is built on that.
DIRECT_PREFILL = "\\boxed{"

DIRECT_INSTRUCTION = {
    # Byte-for-byte what the Milestone 4 and Milestone 8 GSM8K runs used. Do
    # NOT reword it: runs/gsm8k_manifest.json and m5_probe.DIRECT_FINGERPRINT
    # both pin this exact string, and the committed n=150 data stays comparable
    # to later runs only while it holds.
    "gsm8k": ("\n\nRespond with only the final numeric answer and nothing "
              "else. Do not show any reasoning."),

    # UNDECIDED. GSM8K's wording is WRONG here and must not be reused: it asks
    # for "the final numeric answer", and MATH-500 answers frequently are not
    # numeric -- \frac{3}{2}, 2\sqrt{2}, \frac{\pi}{2}, (2,5), intervals.
    # test_scoring.py already carries all of those shapes, so the SCORER
    # handles them; it is the PROMPT that would be instructing the model away
    # from the format its own gold uses. That inflates `unparsed`, and it does
    # so preferentially in the degraded (ablated) cells, which puts the
    # artifact straight into the interaction term.
    "math500": None,

    # UNDECIDED, but nearly free: AIME answers ARE integers 0-999 (which
    # scoring.validate_gold already enforces), so GSM8K's wording is
    # semantically correct as written and can most likely be adopted verbatim.
    # Left unset anyway, because "most likely" is not a pre-registration -- the
    # point of this file is that the choice is made deliberately, in a commit,
    # before the data exists.
    "aime24": None,
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
            f"wording scored better is decision 25's post-hoc rescue with "
            f"extra steps.")
    return suffix, DIRECT_PREFILL


def dataset_ready(dataset: str) -> list[str]:
    """Which per-dataset pre-registration choices are still unset.

    Empty means the dataset can be run. The per-dataset counterpart to
    `undecided()`, which covers top-level scalars: _UNDECIDED and require()
    operate on module globals, and a nested table entry is not one. The
    accessors above raise on their own behalf; this reports the same thing
    without raising, for a pre-flight or a report appendix.
    """
    out = [f"CAPS[{dataset!r}][{lvl!r}]"
           for lvl, cap in sorted(CAPS.get(dataset, {}).items()) if cap is None]
    if DIRECT_INSTRUCTION.get(dataset, None) is None:
        out.append(f"DIRECT_INSTRUCTION[{dataset!r}]")
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
# to test whether the band was the problem, because m7_directions found the
# direct answer is not readable in 14-19 at all (gold@10 = 0% through the whole
# band, first reaching 100% at layer 32). It was not the problem. At the
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
#  1. WHERE. m7_cot_exposure at band 26-31 -- the band m7_directions says the
#     answer is actually computed in -- measures the worst-layer overlap during
#     generation at 5.05 of 10, not 0.76. The rule exempts HALF the selected
#     directions there. It rises monotonically with depth, so an overlap
#     measured at one band says nothing about another.
#
#  2. WHAT IT WAS DOING. m7_damage_floor, SST-2, band 14-33, ablation flip
#     rate with the rule OFF vs ON:
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
USE_EXCLUSION = True


# -------------------------------------------------- DECIDED from measurement

# "each", the paper's literal wording: zero the projection onto EACH of the
# top-k vectors in turn. The worry was that ten overcomplete, non-orthogonal
# directions make this order-dependent and leave part of their span behind,
# which "span" would not. m7_directions measured how much that matters on
# Qwen3-4B under J = I:
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

# Decision 24's gate: after Milestone 7, run ablated CoT on a few problems and
# stop to revise if generation has degenerated badly enough that the comparison
# is meaningless. Three things had to be settled -- what it reads, against what
# baseline, and how big.
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
LOOP_GATE = {
    "signal": "unusable",     # outcome in {incomplete, unparsed, error}
    "mode": "delta",          # vs the matched intact cell
    "threshold": 0.15,
    "n": 20,
}
UNUSABLE_OUTCOMES = ("incomplete", "unparsed", "error")

# Only genuinely open items belong here. PROJECTION_MODE and
# PROJECT_GAIN_SCALED were settled from m7_directions and the readout algebra;
# their reasoning is above, where a later reader can disagree with it.
# Empty: every top-level pre-registration choice is settled, each with its
# measurement recorded above. require() still raises on any None, so a constant
# that is later unset -- or misspelled -- cannot pass silently.
#
# SCOPE. This dict holds module GLOBALS, which is all require() can reach. The
# per-dataset tables (CAPS, DIRECT_INSTRUCTION) have open entries for math500
# and aime24; those are nested, so they are gated by cap_for() and
# direct_prompt() raising on their own behalf, and reported without raising by
# dataset_ready(). "Empty" here therefore does NOT mean the whole
# pre-registration is closed -- ask dataset_ready(d) per dataset.
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
              "choosing it after seeing results is the post-hoc rescue that "
              "decision 25 exists to prevent.")
    return val


def projection(mode: str | None = None, gain: bool | None = None):
    """The projection choice, resolved in ONE place, as (mode, gain_scaled).

    Every probe and the run loop must agree on this or a measurement describes
    a different operation than the run it was taken to calibrate. That is not
    hypothetical: m7_calibration.py hardcoded ("span", True) as a working
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
