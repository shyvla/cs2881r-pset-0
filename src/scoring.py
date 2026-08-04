"""
Answer extraction and scoring for GSM8K / MATH-500 / AIME (Qwen3-4B).

ONE scoring path for all three datasets. The only per-dataset code is how to
READ a record -- where to load it from, which field holds the problem, and how
to pull the gold answer out (DATASETS, GOLD_FIELD). Everything downstream --
extraction, normalization, equivalence -- is identical, so a difference in
results across datasets cannot be an artifact of scoring.

The per-dataset PROMPT and CAPS live in config.py instead, not here: those are
numbers and wording that must be frozen before the data is seen, and config.py
is the file whose require()/cap_for()/direct_prompt() accessors refuse to run
when they are unset.

DESIGN PRINCIPLE FOR EVERY FIX BELOW
------------------------------------
The deliverable is a difference of differences. A scoring bias therefore only
hurts if it DIFFERS ACROSS CELLS. A uniform 3% extraction-failure rate is
harmless; 8% in one cell and 0% in another lands directly in the interaction
term and is indistinguishable from the effect we claim. Every fix here removes
a bias that was differential -- available preferentially to long outputs, or
to ablated (degraded) cells.

Verified against math-verify 0.9.0.
"""
import functools
import hashlib
import json
import os
import re
import subprocess
import time
from importlib.metadata import PackageNotFoundError, version

from math_verify import LatexExtractionConfig, parse, verify

# ====================================================== condition naming
# Condition names are built from a grid, never hand-typed, because
# "A"/"B"/"C"/"D" meant different things in the design table (A = direct
# intact) and in the baseline run script (A_cot = CoT intact). Mapping
# those backwards silently flips the sign of the headline number and nothing
# errors. There are no letters anywhere in this codebase now.
#
#   level = how much reasoning is externalised onto the page (most -> least)
#   state = what we did to the residual stream
#
# MVP grid = {cot, direct} x {intact, ablated, random} = 6 cells.
# `nothink` is the middle rung of the externalisation axis (stretch goal).
# `random` is the random-direction matched control.

LEVELS = ("cot", "nothink", "direct")
STATES = ("intact", "ablated", "random")


def cond_name(level: str, state: str) -> str:
    """Canonical condition string. Use this; never type a literal."""
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    if state not in STATES:
        raise ValueError(f"state must be one of {STATES}, got {state!r}")
    return f"{level}_{state}"


def parse_cond(cond: str) -> tuple[str, str]:
    """Inverse of cond_name. Raises on anything off-grid."""
    level, _, state = cond.rpartition("_")
    if level not in LEVELS or state not in STATES:
        raise ValueError(f"{cond!r} is not a valid condition name")
    return level, state


def unpack_cond(spec) -> tuple[bool, str, str]:
    """A condition spec is (thinking, suffix) or (thinking, suffix, prefill).

    `prefill` is text appended to the rendered chat template, so generation
    STARTS inside it. That is what makes the direct condition direct by
    construction: with prefill=r"\\boxed{" the first generated token is the
    answer, so there is no room for the model to externalise its reasoning.

    This matters beyond prompt hygiene. Asking a 4B model not to reason is
    instruction-following, and compliance degrades as problems get harder --
    which is the SAME axis as the research question, so leakage would be
    inseparable from a real difficulty effect. Prefilling is mechanical, so
    its compliance does not vary with difficulty.
    """
    if len(spec) == 2:
        thinking, suffix = spec
        prefill = ""
    elif len(spec) == 3:
        thinking, suffix, prefill = spec
    else:
        raise ValueError(
            f"condition spec must be (thinking, suffix) or "
            f"(thinking, suffix, prefill), got {len(spec)} items: {spec!r}")
    return bool(thinking), suffix, prefill


def render_prompt(tokenizer, question: str, thinking: bool, suffix: str = "",
                  prefill: str = "") -> str:
    """The ONE place a prompt is built. Generation and fingerprinting both
    call this, so the hash in the manifest cannot drift from what was run.

    Prompt construction used to be copy-pasted into every notebook cell,
    which is the precise failure prompt_fingerprint was added to detect --
    and a fingerprint computed by different code than the run is worthless.
    """
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question + suffix}],
        tokenize=False, add_generation_prompt=True,
        enable_thinking=thinking) + prefill


# ============================================================== layer 1
# Model-side text handling. Shared by all datasets: this is about Qwen's
# output conventions, not about the answer space.

END_TOKENS = ("<|im_end|>", "<|endoftext|>")


def strip_think(text: str) -> str:
    r"""Keep only what the model wrote AFTER closing its reasoning block.

    MANDATORY, not cosmetic. Reasoning traces contain discarded candidate
    answers, including \boxed{} ones, and math-verify prioritises boxed
    matches -- it will grade an abandoned guess. Measured: without this,
    "<think>The answer is \boxed{72}. No wait.</think> The answer is 144."
    scores CORRECT against gold 72.

    End tokens are TRUNCATED AT, not deleted. Deleting them left any
    text generated after <|im_end|> inside the body, so a run-on hallucinated
    second turn got graded. Measured with the old code:

        "Let me reconsider the problem.<|im_end|><|im_start|>assistant\n144"
        scored ('correct', '144') against gold 144

    -- the model stopped without answering and the grade came from a turn it
    invented. Running past EOS is an ablation-degraded behaviour, so this
    false positive was preferentially available to the ablated cells.
    """
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    for tok in END_TOKENS:
        text = text.split(tok)[0]
    return text.strip()


def think_trace(text: str) -> str:
    """The reasoning block itself, for degeneracy diagnostics.

    Handles all three shapes Qwen produces: an explicit <think>...</think>
    pair; an unclosed <think> (ran out of room mid-trace); and a trace whose
    opening tag was pre-filled by the chat template, so only </think> appears
    in the generated text. Returns "" when there is no trace.
    """
    if "<think>" in text:
        text = text.split("<think>", 1)[1]
    elif "</think>" not in text:
        return ""
    return text.split("</think>", 1)[0]


# --- markdown normalisation --------------------------------------------
# Qwen3 wraps final answers in markdown emphasis constantly, and a trailing
# "**" glues onto the number so that "72**" reads as a dangling Python-style
# exponentiation and fails to parse. Measured on math-verify 0.9.0:
#
#     '**72**'                  -> []   UNPARSED
#     '*72*'                    -> []   UNPARSED
#     '`72`'                    -> []   UNPARSED
#     'The answer is **72**.'   -> []   UNPARSED
#     '**Answer: 72**'          -> []   UNPARSED
#     '**72 clips**'            -> [72] fine   (the word breaks the run)
#     '**Answer:** 72'          -> [72] fine
#
# This bit the direct condition hardest: it is instructed to emit only a
# number, and therefore most often emits exactly "**72**". Left unfixed it
# would have driven the baseline direct accuracy artificially toward the
# "< 15% = floor problem" branch of the decision rule.
#
# A blanket strip of "*" is NOT safe -- it turns "3*4*5" into "345" -- so
# this runs ONLY AS A FALLBACK, after an unmodified parse has already failed.
# Regression is then impossible: it can rescue a failure, never alter a
# success. It also never looks at the gold answer, so it does not reintroduce
# the length-scaling leniency that extraction_mode="first_match" removes.

_MD_PATTERNS = (
    re.compile(r"\*\*(?!\s)(.+?)(?<!\s)\*\*", re.S),   # **bold**
    re.compile(r"(?<!\*)\*(?!\*)([^\s*]+)\*(?!\*)"),   # *italic*, no spaces
    re.compile(r"`+([^`\n]+?)`+"),                     # `code`
)


def unwrap_markdown(text: str) -> str:
    """Remove markdown emphasis wrappers, keeping their contents."""
    for pat in _MD_PATTERNS:
        text = pat.sub(lambda m: m.group(1), text)
    return text.strip()


# ============================================================== layer 2
# Extraction + equivalence, delegated wholesale to math-verify.
#
# Two non-default settings, both removing leniency that scales with output
# length (library defaults are tuned for RL reward, where generosity is fine):
#   extraction_mode="first_match" -- commit to the first parse, don't scan
#                                    the text hunting for something that
#                                    matches gold.
#   fallback_mode="no_fallback"   -- if nothing parses, say so; don't guess.

PARSE_KW = dict(
    fallback_mode="no_fallback",
    extraction_mode="first_match",
    parsing_timeout=5,
)
VERIFY_KW = dict(timeout_seconds=5)

# Parses that ran long enough to risk the timeout. A timeout returns an empty
# parse, which the LRU cache then makes PERMANENT for that string within the
# process -- and timing is machine-dependent, which quietly breaks the
# "identical output on macOS and Linux" property. Surfaced, not silenced.
SLOW_PARSES: list[tuple[float, str]] = []


@functools.lru_cache(maxsize=8192)
def _parse_gold(gold: str):
    # Wrapped in $...$ so it reads as LaTeX, matching MATH-500 gold format.
    return parse(f"${gold}$", extraction_config=[LatexExtractionConfig()])


@functools.lru_cache(maxsize=8192)
def _parse_pred(body: str):
    return parse(body, **PARSE_KW)


def parse_pred(body: str):
    """Cached parse with slow-parse accounting. A cache hit times at ~0."""
    t0 = time.perf_counter()
    out = _parse_pred(body)
    dt = time.perf_counter() - t0
    if dt >= 0.9 * PARSE_KW["parsing_timeout"]:
        SLOW_PARSES.append((round(dt, 2), body[:80]))
    return out


# ============================================================== layer 3
# Scoring. Five outcomes, not two.

OUTCOMES = ("correct", "incorrect", "incomplete", "unparsed", "error")


def score_detail(raw: str, gold: str, hit_cap: bool, thinking: bool) -> dict:
    """Grade one generation -> dict(outcome, extracted, normalized).

    `thinking` is REQUIRED and is not optional politeness: whether a closing
    </think> is mandatory depends on the condition, not on whether generation
    ran out of room. Conflating those two produced two real bugs:

      * a non-thinking generation that answered and THEN hit the cap was
        marked "incomplete" (false negative), and
      * a thinking generation that hit EOS without closing </think> was
        parsed as if the trace were a final answer -- "<think>I think it is
        72 but I am unsure" scored CORRECT (false positive).

    The second gate keys on the TEXT as well as the flag. The old
    condition was `thinking and "</think>" not in raw`, which left the very
    same false positive open on the other branch:

        score("<think>I think it is 72 but I am unsure", "72", False, False)
        -> ('correct', '72')

    A non-thinking condition should not emit <think> at all -- but under
    ablation it can, so the hole was reachable only in ablated cells, i.e.
    differentially. An unclosed trace is an unfinished answer no matter which
    condition produced it.

    `hit_cap` is therefore only a tiebreaker between "ran out of room"
    (incomplete) and "declined to answer" (unparsed).
    """
    if gold is None or str(gold).strip() == "":
        raise ValueError("empty gold answer")

    # The trace is not an answer. No closing tag => no answer, regardless of
    # why generation stopped or which condition asked for it.
    if ("<think>" in raw or thinking) and "</think>" not in raw:
        return dict(outcome="incomplete", extracted="", normalized=False)

    body = strip_think(raw)
    pred = parse_pred(body) if body else None

    # Fallback only: rescue markdown-wrapped answers without touching a body
    # that already parsed. See unwrap_markdown.
    normalized = False
    if not pred and body:
        cleaned = unwrap_markdown(body)
        if cleaned and cleaned != body:
            pred = parse_pred(cleaned)
            normalized = bool(pred)

    if not pred:
        return dict(outcome=("incomplete" if hit_cap else "unparsed"),
                    extracted="", normalized=False)

    ok = verify(_parse_gold(str(gold).strip()), pred, **VERIFY_KW)
    return dict(outcome=("correct" if ok else "incorrect"),
                extracted=str(pred[0]), normalized=normalized)


def score(raw: str, gold: str, hit_cap: bool, thinking: bool) -> tuple[str, str]:
    """(outcome, extracted_answer). Thin wrapper over score_detail."""
    d = score_detail(raw, gold, hit_cap, thinking)
    return d["outcome"], d["extracted"]


def is_correct(outcome: str) -> int:
    """Headline metric policy: anything that is not `correct` counts as
    not-correct -- incomplete, unparsed and error alike.

    Pre-register this. Report the full outcome distribution alongside, so a
    reader can see whether an accuracy collapse is really a termination
    collapse.
    """
    return int(outcome == "correct")


# ============================================================== diagnostics
def distinct_ngram_ratio(text: str, n: int = 10) -> float:
    """1.0 = no repetition, ->0 = degenerate looping.

    Ablated models loop. Tracking this turns a nuisance into a free secondary
    finding ("ablation degraded termination behaviour, not just accuracy")
    from data you are already collecting.
    """
    toks = text.split()
    if len(toks) < n:
        return 1.0
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams)


DEGENERATE_BELOW = 0.5


# ============================================== per-dataset: reading the data
# The ONLY place the three datasets differ. Verify field names against the
# loaded dataset before trusting them -- `run.py --check-data` does exactly
# that, for free and without a GPU.

GOLD_FIELD = {
    "gsm8k":   lambda r: r["answer"].split("####")[-1].strip(),
    "math500": lambda r: str(r["answer"]).strip(),
    "aime24":  lambda r: str(r["answer"]).strip(),
}

# How to LOAD each dataset and where the problem statement lives. Not a
# pre-registration choice -- these are facts about the datasets, so they belong
# here beside GOLD_FIELD rather than in config.py, which holds the numbers that
# must be frozen before the data is seen.
#
# `mirrors` is tried in order, matching the damage-floor probe's SST-2/MMLU
# loaders:
# hub repos get renamed and gated, and a run that dies on a 404 after the model
# is resident has wasted the load.
#
# `question` is a TUPLE OF CANDIDATE FIELD NAMES, resolved against the actual
# record by question_field() below, which raises listing the keys it did find.
# The alternative -- one hardcoded name -- is what run.py once did with
# ds[i]["question"], and it fails as a KeyError on the first problem of the
# first cell, after the model has loaded.
#
# UNVERIFIED: only gsm8k has ever been downloaded in this repo (the HF cache
# holds openai/gsm8k, cais/mmlu, stanfordnlp/sst2 and Qwen3-4B and nothing
# else). The math500 and aime24 repo ids, splits and field names below are the
# published conventions but have not been observed here. Run
# `python run.py --check-data --dataset math500` before trusting them; it
# resolves the field name and parses every gold, and costs nothing.
DATASETS = {
    "gsm8k": {
        "mirrors": (("openai/gsm8k", "main"), ("gsm8k", "main")),
        "split": "test",
        "question": ("question",),
        "rows": 1319,
    },
    "math500": {
        "mirrors": (("HuggingFaceH4/MATH-500", None),),
        "split": "test",
        "question": ("problem",),
        "rows": 500,
    },
    "aime24": {
        "mirrors": (("HuggingFaceH4/aime_2024", None),
                    ("Maxwell-Jia/AIME_2024", None)),
        "split": "train",
        "question": ("problem", "Problem"),
        "rows": 30,
    },
}


def load_problems(dataset: str):
    """Load a dataset's split, trying its mirrors in order.

    Returns (ds, path, name) -- the RESOLVED mirror, not the preferred one,
    because the manifest has to record where the data actually came from. Two
    mirrors of "the same" dataset can differ in row order or normalisation, and
    a pin file naming the primary while the fallback was used is exactly the
    kind of provenance that is worse than none.

    Raises with every mirror's failure rather than the last one's: a 404 from
    the fallback tells you nothing about why the primary was skipped.
    """
    from datasets import load_dataset
    spec = DATASETS[dataset]
    errs = []
    for path, name in spec["mirrors"]:
        try:
            return load_dataset(path, name, split=spec["split"]), path, name
        except Exception as e:                       # noqa: BLE001 - reported
            errs.append(f"{path}"
                        + (f"/{name}" if name else "")
                        + f": {type(e).__name__}: {e}")
    raise RuntimeError(f"could not load {dataset!r} from any mirror:\n  "
                       + "\n  ".join(errs))


def question_field(dataset: str, record) -> str:
    """Which field of `record` holds the problem statement.

    Resolved ONCE per run, not per problem, so a wrong field name is a
    pre-flight error rather than a crash on problem 1 after the model loads.
    """
    for f in DATASETS[dataset]["question"]:
        if f in record:
            return f
    raise KeyError(
        f"none of {DATASETS[dataset]['question']} is a field of {dataset!r}; "
        f"available: {sorted(record)}. Fix DATASETS[{dataset!r}]['question'] "
        f"in scoring.py -- the candidates there are published conventions, not "
        f"something this repo has verified for every mirror.")


def run_path(dataset: str, n: int, band: str) -> str:
    """The generations file for a run. ONE construction, because run.py
    writes it and analyze.py has to find it, and a filename built twice is
    a filename that eventually disagrees -- with the analysis silently reading
    a different run than the one just produced."""
    return f"runs/{dataset}_n{n}_{band}.jsonl".replace("(", "").replace(
        ")", "")


def validate_gold(dataset: str, records) -> list[int]:
    """Pre-flight: can we parse the dataset's OWN answers? Free, no GPU.

    Run before spending an hour generating. Returns failing indices.
    """
    get = GOLD_FIELD[dataset]
    bad = []
    for i, r in enumerate(records):
        g = get(r)
        if not g or not _parse_gold(g):
            bad.append(i)
            continue
        if dataset == "aime24":          # AIME answers are integers 0-999
            try:
                if not (0 <= int(g) <= 999):
                    bad.append(i)
            except ValueError:
                bad.append(i)
    return bad


# ============================================================== provenance
def _pkg(name: str):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _git_commit():
    """Short SHA, suffixed `-dirty` when the tree has uncommitted changes.

    Without the suffix the manifest records a commit whose contents are not
    what actually ran -- precisely the failure the reproducibility criterion
    is about.
    """
    def _run(*args):
        return subprocess.run(args, capture_output=True, text=True,
                              timeout=5).stdout.strip()
    try:
        sha = _run("git", "rev-parse", "--short", "HEAD")
        if not sha:
            return None
        return sha + ("-dirty" if _run("git", "status", "--porcelain") else "")
    except Exception:
        return None


def model_revision(model_or_config) -> str | None:
    """The resolved Hub commit for the checkpoint that is actually loaded.

    "Qwen/Qwen3-4B" is a moving target: it names a branch, not a snapshot. The
    assignment asks for exact dataset and model versions, and the GPU run
    spans ~6 cells over hours that may straddle an instance restart, so a
    checkpoint resolving differently between the first cell and the last is a
    confound with no signature anywhere in the data.

    Read from the loaded object rather than queried from the Hub, because the
    question is which weights ran, not which are current. Resolve it BEFORE
    the first generation, not after the last.
    """
    cfg = getattr(model_or_config, "config", model_or_config)
    return getattr(cfg, "_commit_hash", None)


def dataset_fingerprint(ds, ids=None, **extra) -> dict:
    """Identify the data a run used, strongly enough to detect a swap.

    Three layers, because each catches something the others miss:

      rows        the split's length. Catches a different split or a version
                  that added or removed problems.
      fingerprint the datasets library's own hash of the loaded table. Cheap
                  and exact, but a PRIVATE attribute -- fine only because
                  datasets is pinned exactly, and it degrades to None rather
                  than raising if the attribute moves.
      content     sha256 over the rows the run actually touched. The
                  definitive one: two releases can share a row count and
                  differ in a problem statement, and this is what would
                  notice. Restricted to `ids` because those are the only rows
                  the result depends on, which also keeps it fast on large
                  splits.
    """
    out = {"rows": int(ds.num_rows),
           "fingerprint": getattr(ds, "_fingerprint", None),
           "download_size": getattr(getattr(ds, "info", None),
                                    "download_size", None)}
    out.update(extra)
    if ids is not None:
        ids = sorted(int(i) for i in ids)
        h = hashlib.sha256()
        for i in ids:
            h.update(json.dumps(ds[i], sort_keys=True,
                                ensure_ascii=False).encode())
        out["ids"] = ids
        out["content_sha256"] = h.hexdigest()[:16]
    return out


def prompt_fingerprint(tokenizer, thinking: bool, suffix: str = "",
                       prefill: str = "") -> str:
    """Hash of the EXACT rendered prompt template for a canonical probe.

    Protects against the unanswerable question "did I change the direct
    instruction wording halfway through the run?". If the hash in your run
    manifest differs between two runs, the prompts differed. Full stop.

    `prefill` is part of the prompt, so it is part of the hash. Without it,
    the direct condition with and without r"\\boxed{" would fingerprint
    identically -- the manifest would show no change across the single most
    important prompt revision in this experiment.
    """
    return hashlib.sha256(
        render_prompt(tokenizer, "PROBE", thinking, suffix, prefill)
        .encode()).hexdigest()[:16]


def provenance(tokenizer=None, conditions: dict | None = None,
               dataset_fingerprints: dict | None = None,
               model_name: str | None = None,
               model_revision: str | None = None,
               seed: int | None = None,
               caps: dict | None = None,
               gen_config: dict | None = None) -> dict:
    """Everything needed to reproduce a run. Write this next to the results.

    `conditions` maps name -> (thinking, suffix) or (thinking, suffix,
    prefill), which is exactly what determines the prompt, so we hash each
    one and also record the parts in the clear. `caps` and `gen_config` are
    recorded because max_new_tokens and the sampling settings are part of
    the design, not implementation details -- Qwen3 ships do_sample=True, so
    "was this greedy?" must be answerable from the manifest.
    """
    prov = {
        "model": model_name,
        "model_revision": model_revision,
        "seed": seed,
        "git_commit": _git_commit(),
        "versions": {p: _pkg(p) for p in
                     ("math-verify", "datasets", "transformers", "torch",
                      "latex2sympy2_extended", "sympy", "numpy")},
        "parse_kw": dict(PARSE_KW),
        "verify_kw": dict(VERIFY_KW),
        "caps": dict(caps) if caps else {},
        "gen_config": dict(gen_config) if gen_config else {},
        "dataset_fingerprints": dataset_fingerprints or {},
    }
    if conditions:
        prov["conditions"] = {}
        for name, spec in conditions.items():
            thinking, suffix, prefill = unpack_cond(spec)
            entry = {"thinking": thinking, "suffix": suffix,
                     "prefill": prefill}
            if tokenizer is not None:
                entry["prompt_fingerprint"] = prompt_fingerprint(
                    tokenizer, thinking, suffix, prefill)
            prov["conditions"][name] = entry
    return prov


# ========================================================== persistence
def score_file(gen_path: str, scores_path: str, conditions: dict,
               verbose: bool = True, manifest_path: str | None = None,
               tokenizer=None, **prov_kw) -> list[dict]:
    """Score a generations JSONL into a separate scores JSONL.

    Kept separate from generation on purpose: a CoT generation costs ~100s,
    re-scoring costs ~2ms. Writing scores to their own file also means a
    change in scoring shows up as a diff instead of a silent re-grade.

    Each generation record needs: id, cond, raw, gold, hit_cap.
    `conditions` maps cond -> (thinking, suffix).

    Pass `manifest_path` (and optionally `tokenizer`, `caps`, `seed`,
    `model_name`) to write the provenance manifest in the same call. It used
    to be a function you had to remember to invoke separately, which is
    operationally the same as not having it.
    """
    out, errors = [], []
    with open(gen_path) as f:
        for line in f:
            r = json.loads(line)
            thinking = unpack_cond(conditions[r["cond"]])[0]
            try:
                d = score_detail(r["raw"], r["gold"], r["hit_cap"], thinking)
            except Exception as e:
                # One malformed record must not destroy a scoring pass over an
                # hour of generation. Recorded as its own outcome, never
                # silently dropped nor counted as incorrect by accident.
                d = dict(outcome="error", extracted="", normalized=False)
                errors.append((r.get("id"), r["cond"],
                               f"{type(e).__name__}: {e}"))
            body = strip_think(r["raw"])
            trace = think_trace(r["raw"])
            out.append({
                "id": r["id"], "cond": r["cond"],
                "outcome": d["outcome"], "correct": is_correct(d["outcome"]),
                "extracted": d["extracted"], "gold": r["gold"],
                "normalized": d["normalized"],
                "n_tok": r.get("n_tok"), "hit_cap": r["hit_cap"],
                # BOTH ratios. distinct10 used to be computed on the
                # stripped body only, so a CoT generation whose trace was 97%
                # repetition scored 1.000 -- the degeneracy detector was blind
                # in exactly the cell (ablated CoT) where the integration test
                # showed looping cuts a +20 interaction to +9. Looping lives
                # inside <think>; the post-trace body is usually under 10
                # tokens and returns 1.0 by the short-text branch anyway.
                "distinct10_body": round(distinct_ngram_ratio(body), 3),
                "distinct10_trace": round(distinct_ngram_ratio(trace), 3),
                "n_word_trace": len(trace.split()),
                "raw_sha": hashlib.sha256(r["raw"].encode()).hexdigest()[:12],
            })
    tmp = scores_path + ".tmp"
    with open(tmp, "w") as f:
        for rec in out:
            f.write(json.dumps(rec) + "\n")
    os.replace(tmp, scores_path)          # atomic: never a half-written file

    if manifest_path:
        prov = provenance(tokenizer=tokenizer, conditions=conditions, **prov_kw)
        tmp = manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(prov, f, indent=2, sort_keys=True)
        os.replace(tmp, manifest_path)

    if verbose:
        print(f"scored {len(out)} records -> {scores_path}")
        n_norm = sum(r["normalized"] for r in out)
        if n_norm:
            print(f"  {n_norm} answers ({n_norm / len(out):.1%}) recovered by "
                  f"markdown normalisation -- verify this rate is SIMILAR "
                  f"across cells; an asymmetric rate is a differential bias")
        if errors:
            print(f"  {len(errors)} SCORING ERRORS (outcome='error'):")
            for e in errors[:5]:
                print(f"    id={e[0]} cond={e[1]} {e[2]}")
        if SLOW_PARSES:
            print(f"  {len(SLOW_PARSES)} slow parses near the "
                  f"{PARSE_KW['parsing_timeout']}s timeout -- these may score "
                  f"differently on another machine")
        if manifest_path:
            print(f"  manifest -> {manifest_path}")
    return out
