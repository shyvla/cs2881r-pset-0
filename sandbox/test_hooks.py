"""Tests for hooks.py.

Runs on a randomly-initialised Qwen3 built from config: no weights, no
download, no GPU, a few seconds. That is unusual for interpretability code and
worth exploiting -- it means the model contract can be a CI tripwire.

    python test_hooks.py [--quiet]
    pytest -q test_hooks.py

The contract tests (1-6) are not testing our code. They pin the behaviour of
transformers 5.14.1 that hooks.py is built on. If transformers is upgraded and
those go red, the experiment is wrong in a way that would otherwise be silent.
"""
import os
import sys

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from hooks import (Capture, LayerHooks, band_from_depth, decoder_layers,
                   final_norm, hook_census, logit_lens, n_layers)

NL = 6
D = 64
_CACHE = {}


def tiny_model():
    """Weightless Qwen3. Architecture -- and therefore the hook contract --
    is identical to Qwen3-4B; only the sizes differ."""
    if "m" not in _CACHE:
        torch.manual_seed(0)
        cfg = Qwen3Config(
            vocab_size=256, hidden_size=D, intermediate_size=128,
            num_hidden_layers=NL, num_attention_heads=4,
            num_key_value_heads=2, head_dim=16,
            max_position_embeddings=4096, pad_token_id=0,
        )
        m = Qwen3ForCausalLM(cfg).eval()
        m.generation_config.do_sample = False
        _CACHE["m"] = m
    return _CACHE["m"]


def ids(n=9):
    g = torch.Generator().manual_seed(1)
    return torch.randint(0, 256, (1, n), generator=g)


def _hook_count(model):
    return sum(len(l._forward_hooks) for l in decoder_layers(model))


# ===================================================== contract: transformers

def test_layer_returns_bare_tensor():
    """CONTRACT 1. v4 returned a tuple; v5 returns a tensor."""
    m = tiny_model()
    seen = {}
    with LayerHooks(m, [0], make_hook=lambda i: (
            lambda mod, a, o: seen.__setitem__("o", o))), torch.no_grad():
        m(ids())
    assert torch.is_tensor(seen["o"]), type(seen["o"])
    assert seen["o"].shape == (1, 9, D), seen["o"].shape


def test_output_index_zero_silently_strips_batch():
    """CONTRACT 1, the failure mode. output[0] does not raise."""
    m = tiny_model()
    seen = {}
    with LayerHooks(m, [0], make_hook=lambda i: (
            lambda mod, a, o: seen.__setitem__("o", o))), torch.no_grad():
        m(ids())
    assert seen["o"][0].shape == (9, D), "v4 habit output[0] should give 2-D"


def test_hook_matches_hidden_states_except_last():
    """CONTRACT 2. hook(layers[i]) == hidden_states[i+1] for i <= NL-2."""
    m = tiny_model()
    cap = Capture(m, range(NL))
    with cap, torch.no_grad():
        out = m(ids(), output_hidden_states=True)
    assert len(out.hidden_states) == NL + 1, len(out.hidden_states)
    for i in range(NL - 1):
        assert torch.equal(cap.first(i), out.hidden_states[i + 1].cpu()), i


def test_last_hidden_state_is_normed_not_raw():
    """CONTRACT 2, the exception. tie_last_hidden_states overwrites [-1]."""
    m = tiny_model()
    cap = Capture(m, [NL - 1])
    with cap, torch.no_grad():
        out = m(ids(), output_hidden_states=True)
    raw = cap.first(NL - 1)
    assert not torch.equal(out.hidden_states[-1].cpu(), raw), \
        "hidden_states[-1] should NOT be the raw last-layer output"
    normed = final_norm(m)(raw)
    assert torch.allclose(out.hidden_states[-1].cpu(), normed, atol=1e-5)


def test_logit_lens_reproduces_logits():
    """CONTRACT 2, independent oracle: routes through lm_head, which the
    capture mechanism never touches."""
    m = tiny_model()
    cap = Capture(m, [NL - 1])
    with cap, torch.no_grad():
        out = m(ids())
    assert torch.allclose(logit_lens(m, cap.first(NL - 1)), out.logits,
                          atol=1e-5)


def test_logit_lens_without_norm_is_wrong():
    """Dropping the norm does not raise; it silently returns garbage."""
    m = tiny_model()
    cap = Capture(m, [NL - 1])
    with cap, torch.no_grad():
        out = m(ids())
    assert not torch.allclose(m.lm_head(cap.first(NL - 1)), out.logits,
                              atol=1e-5)


def test_returning_tensor_replaces_output():
    """CONTRACT 3. This is the Half B intervention mechanism."""
    m = tiny_model()
    with torch.no_grad():
        base = m(ids()).logits.clone()
    with LayerHooks(m, [2], make_hook=lambda i: (
            lambda mod, a, o: o * 0.0)), torch.no_grad():
        zeroed = m(ids()).logits
    assert not torch.equal(base, zeroed), "hook return did not take effect"


def test_returning_tuple_raises():
    """CONTRACT 3. The v4 habit fails loudly, not silently. Good news."""
    m = tiny_model()
    try:
        with LayerHooks(m, [2], make_hook=lambda i: (
                lambda mod, a, o: (o * 0.0,))), torch.no_grad():
            m(ids())
    except AttributeError:
        return
    raise AssertionError("returning a tuple should raise AttributeError")


# ============================================================ contract: ours

def test_hooks_removed_on_exit():
    # delta, not zero: transformers permanently installs its own (contract 4)
    m = tiny_model()
    base = _hook_count(m)
    with Capture(m, [0, 1]) as cap:
        assert _hook_count(m) == base + 2
        assert cap.active
    assert _hook_count(m) == base
    assert not cap.active


def test_hooks_removed_on_exception():
    m = tiny_model()
    base = _hook_count(m)
    try:
        with Capture(m, [0, 1]):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert _hook_count(m) == base, "hooks leaked when the body raised"


def test_transformers_installs_permanent_capture_hooks():
    """CONTRACT 4. output_hidden_states=True is not side-effect free."""
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=3, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=16,
                      max_position_embeddings=64, pad_token_id=0)
    fresh = Qwen3ForCausalLM(cfg).eval()
    x = torch.zeros(1, 4, dtype=torch.long)
    with torch.no_grad():
        fresh(x)
    assert hook_census(fresh) == [0, 0, 0], hook_census(fresh)
    with torch.no_grad():
        fresh(x, output_hidden_states=True)
    assert hook_census(fresh) == [1, 1, 1], hook_census(fresh)
    with torch.no_grad():
        fresh(x)
    assert hook_census(fresh) == [1, 1, 1], "installed hooks are never removed"


def test_output_hidden_states_reports_pre_intervention():
    """CONTRACT 5, THE LANDMINE. The intervention takes effect on the forward
    pass, but hidden_states still shows the un-ablated value, because the
    transformers capture hook was registered first and runs first.

    If this ever goes green-by-equality, re-read contract 5 before trusting
    any Milestone 6 verification."""
    m = tiny_model()
    x = ids()
    with torch.no_grad():
        m(x, output_hidden_states=True)          # ensure capture hooks exist
        base_logits = m(x).logits.clone()
    with LayerHooks(m, [2], make_hook=lambda i: (
            lambda mod, a, o: o * 0.0)), torch.no_grad():
        out = m(x, output_hidden_states=True)
    assert not torch.equal(out.logits, base_logits), \
        "ablation had no effect on logits -- intervention is not firing"
    assert not bool((out.hidden_states[3] == 0).all()), \
        "hidden_states[3] is zero, so capture ran AFTER the intervention; " \
        "registration order differs from what contract 5 documents"


def test_batch_greater_than_one_raises():
    m = tiny_model()
    two = torch.cat([ids(), ids()], dim=0)
    try:
        with Capture(m, [0]), torch.no_grad():
            m(two)
    except ValueError as e:
        assert "batch size 2" in str(e), str(e)
        return
    raise AssertionError("batch 2 should have raised")


def test_bad_layer_indices_rejected():
    m = tiny_model()
    for bad in ([-1], [NL], [0, 0]):
        try:
            Capture(m, bad)
        except (IndexError, ValueError):
            continue
        raise AssertionError(f"{bad} should have been rejected")


def test_band_from_depth_matches_documented_band():
    assert band_from_depth(36) == range(14, 34), band_from_depth(36)
    assert list(band_from_depth(36))[-1] == 33
    assert band_from_depth(NL) == range(3, 6), band_from_depth(NL)


def test_generate_firing_pattern():
    """THE LANDMINE. One prefill pass over the prompt, then one pass per
    subsequent token. n_firings == n_new_tokens, NOT 1 + n_new."""
    m = tiny_model()
    x = ids(12)
    cap = Capture(m, [3], store=False)
    with cap, torch.no_grad():
        out = m.generate(x, max_new_tokens=5, use_cache=True)
    n_new = out.shape[1] - x.shape[1]
    lens = cap.seq_lens(3)
    assert cap.n_firings(3) == n_new, (cap.n_firings(3), n_new)
    assert lens[0] == 12, lens
    assert all(s == 1 for s in lens[1:]), lens
    assert cap.positions(3) == 12 + (n_new - 1)


def test_store_false_keeps_shapes_and_no_tensors():
    m = tiny_model()
    cap = Capture(m, [0], store=False)
    with cap, torch.no_grad():
        m(ids())
    assert cap.n_firings(0) == 1
    assert cap.nbytes == 0
    assert cap.acts == {}


def test_nbytes_matches_arithmetic():
    """seq x d_model x bytes-per-element, no hidden float32 upcast."""
    m = tiny_model()
    cap = Capture(m, range(NL))
    with cap, torch.no_grad():
        m(ids(9))
    expected = NL * 9 * D * cap.first(0).element_size()
    assert cap.nbytes == expected, (cap.nbytes, expected)


def test_n_layers_and_decoder_layers():
    m = tiny_model()
    assert n_layers(m) == NL
    assert len(decoder_layers(m)) == NL


# ===================================================================== runner

def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quiet = "--quiet" in argv
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            if not quiet:
                print(f"  ok    {name}")
        except Exception as e:
            failed.append((name, e))
            print(f"  FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - len(failed)}/{len(tests)} passed")
    if failed:
        print("hooks.py contract is NOT satisfied -- do not proceed to "
              "Milestone 6.")
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        code = main()
        sys.stdout.flush()
        sys.exit(code)
    except BrokenPipeError:
        # piping to `head` closes stdout early; a passing run must not look
        # like a failing build
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
