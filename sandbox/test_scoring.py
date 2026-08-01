"""Test suite for scoring.py. Run: python test_scoring.py"""
from scoring import (DEGENERATE_BELOW, cond_name, distinct_ngram_ratio,
                     parse_cond, prompt_fingerprint, provenance, score,
                     score_detail, strip_think, think_trace, unwrap_markdown)

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

    # ==== FIX 1: markdown emphasis broke extraction =====================
    # All six of these returned [] from math-verify 0.9.0 and were therefore
    # scored `unparsed` -- i.e. counted as wrong. The direct condition is
    # asked for a bare number and so emits exactly '**72**' most often, which
    # would have pushed Milestone-4 direct accuracy toward the "< 15% = floor
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

    # ==== FIX 2: the gate now keys on the TEXT, not just the flag ========
    # Previously ('correct', '72'): a non-thinking condition emitted <think>
    # (ablation makes this happen) and the unclosed trace got graded.
    ("<think>I think it is 72 but I am unsure",  "72", False, False, "incomplete"),
    ("<think>48+24=72 hmm",                      "72", True,  False, "incomplete"),
    # a closed trace from a non-thinking condition is still gradeable
    ("<think>48+24=72</think>\n\n72",            "72", False, False, "correct"),

    # ==== FIX 3: truncate at end tokens, don't delete them ==============
    # Previously ('correct', '144'): the model stopped without answering and
    # the grade came from a second turn it hallucinated past <|im_end|>.
    ("Let me reconsider the problem.<|im_end|><|im_start|>assistant\n144",
                                               "144", False, False, "unparsed"),
    ("72<|im_end|><|im_start|>assistant\n144",   "72", False, False, "correct"),
]


def test_scoring():
    ok = True
    for raw, gold, cap, think, want in TESTS:
        got, extracted = score(raw, gold, cap, think)
        good = got == want
        ok &= good
        print(f"{'ok  ' if good else 'FAIL'} think={int(think)} cap={int(cap)} "
              f"want={want:10} got={got:10} ext={extracted!r:10} "
              f"<- {raw[:40]!r}")
    return ok


def test_strip_think():
    assert strip_think("<think>a</think>\n\nb<|im_end|>") == "b"
    assert strip_think("no tags here") == "no tags here"
    assert strip_think("<think>unclosed") == "<think>unclosed"
    # FIX 3: everything after the stop token is discarded, not just the token
    assert strip_think("b<|im_end|><|im_start|>assistant\nc") == "b"
    assert strip_think("b<|endoftext|>junk") == "b"
    print("ok   strip_think")
    return True


def test_think_trace():
    """FIX 4 support: recover the trace in all three shapes Qwen emits."""
    assert think_trace("<think>abc</think>\n\nans") == "abc"
    assert think_trace("<think>unclosed abc") == "unclosed abc"
    # template pre-filled the opening tag, so only </think> is generated
    assert think_trace("abc</think>\n\nans") == "abc"
    assert think_trace("no trace at all") == ""
    print("ok   think_trace")
    return True


def test_unwrap_markdown():
    assert unwrap_markdown("**72**") == "72"
    assert unwrap_markdown("*72*") == "72"
    assert unwrap_markdown("`72`") == "72"
    assert unwrap_markdown("The answer is **72**.") == "The answer is 72."
    # must not eat asterisks used as multiplication
    assert unwrap_markdown("2*3 and 4*5") == "2*3 and 4*5"
    print("ok   unwrap_markdown")
    return True


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
    return True


def test_degeneracy():
    clean = " ".join(f"word{i}" for i in range(200))
    loop = "the answer is 72 so the answer is 72 " * 30
    assert distinct_ngram_ratio(clean) > 0.9
    assert distinct_ngram_ratio(loop) < DEGENERATE_BELOW
    assert distinct_ngram_ratio("short") == 1.0
    # FIX 4: the looping lives in the TRACE. Scoring the body alone is blind.
    cot = "<think>" + loop + "</think>\n\nThe answer is 72."
    body_ratio = distinct_ngram_ratio(strip_think(cot))
    trace_ratio = distinct_ngram_ratio(think_trace(cot))
    assert body_ratio == 1.0, body_ratio          # what the old code stored
    assert trace_ratio < DEGENERATE_BELOW, trace_ratio
    print(f"ok   degeneracy  clean={distinct_ngram_ratio(clean):.2f} "
          f"loop={distinct_ngram_ratio(loop):.2f} "
          f"cot(body={body_ratio:.2f} trace={trace_ratio:.2f})")
    return True


def test_cond_names():
    """FIX 6: names come from a grid, so a typo cannot reach the analysis."""
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
    return True


class _StubTok:
    """Stands in for a HF tokenizer so provenance is testable offline."""
    def apply_chat_template(self, msgs, tokenize, add_generation_prompt,
                            enable_thinking):
        think = "<think>\n\n</think>\n\n" if not enable_thinking else ""
        return f"<|im_start|>user\n{msgs[0]['content']}<|im_end|>\n" \
               f"<|im_start|>assistant\n{think}"


def test_provenance():
    tok = _StubTok()
    conds = {"cot_intact": (True, ""),
             "direct_intact": (False, "\n\nAnswer only.")}
    p = provenance(tokenizer=tok, conditions=conds, model_name="Qwen/Qwen3-4B",
                   seed=0, caps={"cot_intact": 2048, "direct_intact": 256})
    fps = p["prompt_fingerprints"]
    assert len(fps) == 2 and fps["cot_intact"] != fps["direct_intact"]
    # a changed suffix must change the hash -- that is the whole point
    assert prompt_fingerprint(tok, False, "\n\nAnswer only.") != \
           prompt_fingerprint(tok, False, "\n\nAnswer ONLY.")
    assert p["versions"]["math-verify"] is not None
    assert p["caps"]["direct_intact"] == 256       # caps are part of design
    print(f"ok   provenance  math-verify={p['versions']['math-verify']} "
          f"git={p['git_commit']} fingerprints={fps}")
    return True


if __name__ == "__main__":
    import sys
    res = [test_scoring(), test_strip_think(), test_think_trace(),
           test_unwrap_markdown(), test_normalisation_is_fallback_only(),
           test_degeneracy(), test_cond_names(), test_provenance()]
    print(f"\n{'ALL PASS' if all(res) else 'FAILURES PRESENT'} "
          f"({len(TESTS)} scoring cases + 7 unit tests)")
    sys.exit(0 if all(res) else 1)
