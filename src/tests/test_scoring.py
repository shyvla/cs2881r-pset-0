"""Test suite for scoring.py. Run: python test_scoring.py"""
import sys

VERBOSE = "--quiet" not in sys.argv

from scoring import (DEGENERATE_BELOW, cond_name, distinct_ngram_ratio,
                     parse_cond, prompt_fingerprint, provenance, render_prompt,
                     score, score_detail, strip_think, think_trace,
                     unpack_cond, unwrap_markdown)

# (raw, gold, hit_cap, thinking, expected)
TESTS = [
    # ---- basic shapes -------------------------------------------------
    (r"Thus $\boxed{72}$.<|im_end|>",           "72", False, False, "correct"),
    ("**Answer:** **72 clips**.<|im_end|>",     "72", False, False, "correct"),
    ("144<|im_end|>",                           "72", False, False, "incorrect"),
    ("I cannot determine this.",                "72", False, False, "unparsed"),
    (r"\boxed{1,234}",                        "1234", False, False, "correct"),
    # ---- MATH-500 shapes: why math-verify replaces hand-rolled regex ---
    (r"\boxed{\frac{3}{2}}",                   "1.5", False, False, "correct"),
    (r"\boxed{0.5}",                   r"\frac{1}{2}", False, False, "correct"),
    (r"\boxed{72 \text{ clips}}",               "72", False, False, "correct"),
    (r"\boxed{2\sqrt{2}}",              r"2\sqrt{2}", False, False, "correct"),
    (r"\boxed{\frac{\pi}{2}}",       r"\frac{\pi}{2}", False, False, "correct"),
    (r"\boxed{\frac{1}{2}+\frac{\sqrt{5}}{2}}",     # same number, rewritten
                              r"\frac{1+\sqrt{5}}{2}", False, False, "correct"),
    (r"\boxed{(2,5)}",                       "(2,5)", False, False, "correct"),
    # ---- AIME shape ---------------------------------------------------
    (r"\boxed{204}",                           "204", False, False, "correct"),

    # ---- trace contamination (thinking mode) --------------------------
    # discarded boxed guess inside the trace, real answer differs
    (r"<think>The answer is \boxed{72}. No wait.</think>"
     "\n\nThe answer is 144.",                  "72", False, True, "incorrect"),
    # model reasons correctly then declines: must NOT mine the trace
    ("<think>48/2=24, so 48+24=72.</think>\n\nI'm not sure.",
                                                "72", False, True, "unparsed"),

    # ---- REGRESSION: the two holes found after v1 ---------------------
    # HOLE 1 was: non-thinking generation answered, THEN hit the cap.
    # Old gate marked this "incomplete". It is a completed answer.
    ("The answer is 72. We can also verify that 48 plus 24 equals 72, and",
                                                "72", True,  False, "correct"),
    # HOLE 2 was: thinking mode, EOS before closing </think>, hit_cap False.
    # Old gate never fired, so the raw trace got graded. Scored CORRECT.
    ("<think>I think it is 72 but I am unsure",  "72", False, True, "incomplete"),
    # same hole, cap-truncated variant
    ("<think>48+24=72. wait, 48*3=144",          "72", True,  True, "incomplete"),

    # ---- gate edge cases ----------------------------------------------
    # closed the trace, then ran out of room before answering
    ("<think>reasoning done.</think>\n\nTherefore we",
                                                "72", True,  True,  "incomplete"),
    # non-thinking, hit cap, never produced a number
    ("Let me think about how to approach this problem carefully and",
                                                "72", True,  False, "incomplete"),
    # non-thinking, clean stop, no number => declined, not truncated
    ("I don't know.",                            "72", False, False, "unparsed"),
    # thinking mode done properly
    ("<think>48+24=72</think>\n\n" + r"\boxed{72}", "72", False, True, "correct"),

    # ==== markdown emphasis broke extraction =====================
    # All six of these returned [] from math-verify 0.9.0 and were therefore
    # scored `unparsed` -- i.e. counted as wrong. The direct condition is
    # asked for a bare number and so emits exactly '**72**' most often, which
    # would have pushed baseline direct accuracy toward the "< 15% = floor
    # problem" branch of the decision rule for a purely cosmetic reason.
    ("**72**",                                   "72", False, False, "correct"),
    ("*72*",                                     "72", False, False, "correct"),
    ("`72`",                                     "72", False, False, "correct"),
    ("The answer is **72**.",                    "72", False, False, "correct"),
    ("**Answer: 72**",                           "72", False, False, "correct"),
    ("**144**",                                  "72", False, False, "incorrect"),
    # normalisation must not invent an answer where there is none
    ("**I don't know**",                         "72", False, False, "unparsed"),
    # bare asterisks as multiplication must survive untouched
    ("The product is 2*3 and 4*5, so 6.",         "6", False, False, "correct"),

    # ==== the gate keys on the TEXT, not just the flag ========
    # Previously ('correct', '72'): a non-thinking condition emitted <think>
    # (ablation makes this happen) and the unclosed trace got graded.
    ("<think>I think it is 72 but I am unsure",  "72", False, False, "incomplete"),
    ("<think>48+24=72 hmm",                      "72", True,  False, "incomplete"),
    # a closed trace from a non-thinking condition is still gradeable
    ("<think>48+24=72</think>\n\n72",            "72", False, False, "correct"),

    # ==== truncate at end tokens, don't delete them ==============
    # Previously ('correct', '144'): the model stopped without answering and
    # the grade came from a second turn it hallucinated past <|im_end|>.
    ("Let me reconsider the problem.<|im_end|><|im_start|>assistant\n144",
                                               "144", False, False, "unparsed"),
    ("72<|im_end|><|im_start|>assistant\n144",   "72", False, False, "correct"),
]


def test_scoring():
    """Every case is printed before the assert fires, so a failing run still
    shows the full table rather than stopping at the first bad case.

    The assert is load-bearing: this function used to accumulate into `ok`
    and RETURN it, which pytest ignores. A deliberately broken scorer (the text-gate fix
    reverted) produced "8 passed" under pytest while `python test_scoring.py`
    correctly exited 1 -- so a CI job wired to pytest would have been green
    on a broken scorer.
    """
    fails = []
    for raw, gold, cap, think, want in TESTS:
        got, extracted = score(raw, gold, cap, think)
        good = got == want
        if not good:
            fails.append((raw[:50], want, got))
        if VERBOSE or not good:
            print(f"{'ok  ' if good else 'FAIL'} think={int(think)} "
                  f"cap={int(cap)} want={want:10} got={got:10} "
                  f"ext={extracted!r:10} <- {raw[:40]!r}")
    assert not fails, f"{len(fails)} scoring case(s) failed: {fails}"


def test_strip_think():
    assert strip_think("<think>a</think>\n\nb<|im_end|>") == "b"
    assert strip_think("no tags here") == "no tags here"
    assert strip_think("<think>unclosed") == "<think>unclosed"
    # everything after the stop token is discarded, not just the token
    assert strip_think("b<|im_end|><|im_start|>assistant\nc") == "b"
    assert strip_think("b<|endoftext|>junk") == "b"
    print("ok   strip_think")


def test_think_trace():
    """Trace recovery: recover the trace in all three shapes Qwen emits."""
    assert think_trace("<think>abc</think>\n\nans") == "abc"
    assert think_trace("<think>unclosed abc") == "unclosed abc"
    # template pre-filled the opening tag, so only </think> is generated
    assert think_trace("abc</think>\n\nans") == "abc"
    assert think_trace("no trace at all") == ""
    print("ok   think_trace")


def test_unwrap_markdown():
    assert unwrap_markdown("**72**") == "72"
    assert unwrap_markdown("*72*") == "72"
    assert unwrap_markdown("`72`") == "72"
    assert unwrap_markdown("The answer is **72**.") == "The answer is 72."
    # must not eat asterisks used as multiplication
    assert unwrap_markdown("2*3 and 4*5") == "2*3 and 4*5"
    print("ok   unwrap_markdown")


def test_normalisation_is_fallback_only():
    """The markdown rescue must never fire on a body that already parsed.

    That is what makes it regression-proof, and what keeps it from
    reintroducing length-scaling leniency: it cannot change any existing
    result, only rescue a failure.
    """
    assert score_detail("72", "72", False, False)["normalized"] is False
    assert score_detail(r"\boxed{72}", "72", False, False)["normalized"] is False
    assert score_detail("**72**", "72", False, False)["normalized"] is True
    print("ok   normalisation is fallback-only")


def test_degeneracy():
    clean = " ".join(f"word{i}" for i in range(200))
    loop = "the answer is 72 so the answer is 72 " * 30
    assert distinct_ngram_ratio(clean) > 0.9
    assert distinct_ngram_ratio(loop) < DEGENERATE_BELOW
    assert distinct_ngram_ratio("short") == 1.0
    # the looping lives in the TRACE. Scoring the body alone is blind.
    cot = "<think>" + loop + "</think>\n\nThe answer is 72."
    body_ratio = distinct_ngram_ratio(strip_think(cot))
    trace_ratio = distinct_ngram_ratio(think_trace(cot))
    assert body_ratio == 1.0, body_ratio          # what the old code stored
    assert trace_ratio < DEGENERATE_BELOW, trace_ratio
    print(f"ok   degeneracy  clean={distinct_ngram_ratio(clean):.2f} "
          f"loop={distinct_ngram_ratio(loop):.2f} "
          f"cot(body={body_ratio:.2f} trace={trace_ratio:.2f})")


def test_cond_names():
    """Names come from a grid, so a typo cannot reach the analysis."""
    assert cond_name("cot", "intact") == "cot_intact"
    assert parse_cond("direct_ablated") == ("direct", "ablated")
    for bad in ("A_cot", "cot_intact_2", "direct", "cot_broken"):
        try:
            parse_cond(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} should not parse as a condition")
    for bad_call in (("A", "intact"), ("cot", "off")):
        try:
            cond_name(*bad_call)
        except ValueError:
            continue
        raise AssertionError(f"cond_name{bad_call} should have raised")
    print("ok   condition naming grid")


class _StubTok:
    """Stands in for a HF tokenizer so provenance is testable offline."""
    def apply_chat_template(self, msgs, tokenize, add_generation_prompt,
                            enable_thinking):
        think = "<think>\n\n</think>\n\n" if not enable_thinking else ""
        return f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n" \
               f"<|im_start|>assistant\n{think}"


def test_unpack_cond():
    """Condition specs may carry a prefill; both shapes must work."""
    assert unpack_cond((True, "")) == (True, "", "")
    assert unpack_cond((False, "S", "\\boxed{")) == (False, "S", "\\boxed{")
    for bad in ((True,), (True, "", "", "")):
        try:
            unpack_cond(bad)
        except ValueError:
            continue
        raise AssertionError(f"unpack_cond{bad} should have raised")
    print("ok   unpack_cond")


def test_render_prompt_and_prefill():
    """The prefill is part of the prompt, so it must be part of the hash.

    Without this, the direct condition with and without \\boxed{ would
    fingerprint identically -- the manifest would record no change across
    the single most important prompt revision in the experiment.
    """
    tok = _StubTok()
    plain = render_prompt(tok, "Q", False, "\n\nAnswer only.")
    boxed = render_prompt(tok, "Q", False, "\n\nAnswer only.", "\\boxed{")
    assert boxed == plain + "\\boxed{", boxed
    assert prompt_fingerprint(tok, False, "\n\nAnswer only.") != \
           prompt_fingerprint(tok, False, "\n\nAnswer only.", "\\boxed{")
    print("ok   render_prompt + prefill changes the fingerprint")


def test_dataset_registry_is_complete_and_consistent():
    """Every per-dataset table must cover the same three datasets.

    A dataset present in one table and missing from another is how --dataset
    used to half-work: the flag existed, DATASETS did not, and the loader named
    GSM8K literally. Keys agreeing is the cheapest possible guard.
    """
    import config
    from scoring import DATASETS, GOLD_FIELD
    assert set(DATASETS) == set(GOLD_FIELD) == set(config.CAPS) \
           == set(config.DIRECT_INSTRUCTION) == set(config.N_DEFAULT), (
        sorted(DATASETS), sorted(GOLD_FIELD), sorted(config.CAPS),
        sorted(config.DIRECT_INSTRUCTION), sorted(config.N_DEFAULT))
    for d, spec in DATASETS.items():
        assert spec["mirrors"] and all(len(m) == 2 for m in spec["mirrors"]), d
        assert spec["question"], d
        assert spec["rows"] > 0, d
        # A pre-registered n larger than the split is not a sample, it is an
        # impossibility -- and the point of check_n is that it costs nothing to
        # notice. aime24 is the whole split, so equality is allowed.
        n = config.N_DEFAULT[d]
        assert n is None or 0 < n <= spec["rows"], (d, n, spec["rows"])
    print("ok   dataset registry keys agree across scoring and config")


def test_level_tables_cover_every_level():
    """CAPS is keyed dataset->level and MEASURE_CAP level->ceiling. A level in
    one and missing from the other is how --calibrate-caps would KeyError after
    the model is resident, having resolved its cells from a third place."""
    import config
    from scoring import LEVELS
    for d, caps in config.CAPS.items():
        assert set(caps) == set(LEVELS), (d, sorted(caps), sorted(LEVELS))
    assert set(config.MEASURE_CAP) == set(LEVELS), sorted(config.MEASURE_CAP)
    assert set(config.RUN_LEVELS) <= set(LEVELS), config.RUN_LEVELS
    print("ok   CAPS and MEASURE_CAP cover scoring.LEVELS")


def test_question_field_resolves_or_names_the_keys_it_found():
    """Resolved once per run against a real record, so a wrong field name is a
    pre-flight error listing the available keys -- not a KeyError on problem 1
    with the model already resident, which is what ds[i]["question"] gave."""
    from scoring import question_field
    assert question_field("gsm8k", {"question": "q", "answer": "a"}) == "question"
    assert question_field("math500", {"problem": "p", "answer": "a"}) == "problem"
    # aime24 lists two candidates; either resolves
    assert question_field("aime24", {"Problem": "p"}) == "Problem"
    try:
        question_field("gsm8k", {"prompt": "q"})
    except KeyError as e:
        assert "prompt" in str(e), e     # must report what it DID find
        print("ok   question_field raises naming the available keys")
        return
    raise AssertionError("a missing question field must raise")


def test_run_path_is_the_one_filename_construction():
    """run.py writes it and analyze.py must find it. Built twice, they
    eventually disagree and the analysis reads a different run.

    Superseded runs made under the older m8_-prefixed convention live in
    runs/archive/ under their original names; analyze.py reads them via
    --file."""
    from scoring import run_path
    assert run_path("gsm8k", 150, "light") == \
           "runs/gsm8k_n150_light.jsonl"
    assert run_path("math500", 500, "light") == \
           "runs/math500_n500_light.jsonl"
    assert "(" not in run_path("gsm8k", 12, "range14, 20")
    print("ok   run_path")


def test_resolve_anchors_a_run_name_to_src_from_any_cwd():
    """The NAME is relative (above, pinned); what it means must not depend on
    where python was launched. Run from the repo root, "runs/..." used to be a
    fresh empty directory beside the caller: the run silently regenerated
    everything it had already produced, and analyze.py looked elsewhere for
    it."""
    import os

    from scoring import resolve, run_path
    here = os.path.dirname(os.path.abspath(__file__))
    src = os.path.dirname(here)
    got = resolve(run_path("gsm8k", 150, "light"))
    assert got == os.path.join(src, "runs/gsm8k_n150_light.jsonl"), got
    assert os.path.isabs(got)
    # An explicit --out means exactly what the caller typed.
    assert resolve("/tmp/x.jsonl") == "/tmp/x.jsonl"
    print("ok   resolve")


def test_provenance():
    tok = _StubTok()
    conds = {"cot_intact": (True, ""),
             "direct_intact": (False, "\n\nAnswer only.", "\\boxed{")}
    p = provenance(tokenizer=tok, conditions=conds, model_name="Qwen/Qwen3-4B",
                   seed=0, caps={"cot_intact": 2048, "direct_intact": 32},
                   gen_config={"do_sample": False})
    c = p["conditions"]
    assert len(c) == 2
    assert c["direct_intact"]["prefill"] == "\\boxed{"
    assert c["cot_intact"]["prompt_fingerprint"] != \
           c["direct_intact"]["prompt_fingerprint"]
    # a changed suffix must change the hash -- that is the whole point
    assert prompt_fingerprint(tok, False, "\n\nAnswer only.") != \
           prompt_fingerprint(tok, False, "\n\nAnswer ONLY.")
    assert p["versions"]["math-verify"] is not None
    assert p["caps"]["direct_intact"] == 32
    assert p["gen_config"]["do_sample"] is False   # Qwen3 defaults to True
    print(f"ok   provenance  math-verify={p['versions']['math-verify']} "
          f"git={p['git_commit']}")


def _run_all():
    # An assertion anywhere below propagates and exits non-zero on its own,
    # so this behaves identically under `python test_scoring.py` and pytest.
    test_scoring()
    test_strip_think()
    test_think_trace()
    test_unwrap_markdown()
    test_normalisation_is_fallback_only()
    test_degeneracy()
    test_cond_names()
    test_unpack_cond()
    test_render_prompt_and_prefill()
    test_dataset_registry_is_complete_and_consistent()
    test_level_tables_cover_every_level()
    test_question_field_resolves_or_names_the_keys_it_found()
    test_run_path_is_the_one_filename_construction()
    test_resolve_anchors_a_run_name_to_src_from_any_cwd()
    test_provenance()
    print(f"\nALL PASS ({len(TESTS)} scoring cases + 13 unit tests)")


if __name__ == "__main__":
    import os

    # `--quiet` suppresses the per-case table; failures still print.
    # The BrokenPipeError guard makes `... | head -N` safe: head closes the
    # pipe on exit and the next print would otherwise raise.
    try:
        _run_all()
        sys.stdout.flush()
    except BrokenPipeError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
