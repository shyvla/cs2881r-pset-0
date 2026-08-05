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

import config
from hooks import (Capture, Firing, Intervene, LayerHooks, add_noise,
                   band_from_depth, decoder_layers, directions_for,
                   draw_seed, final_norm, generate_ablated, hook_census,
                   logit_lens, make_ablation,
                   n_layers, project_out, random_directions, readout_gain,
                   readout_scores, topk_tokens)

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
    any intervention-placement verification."""
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


# ================================================= project_out (pure math)

def _dummy_firing(layer=0, call_idx=0):
    return Firing(layer=layer, call_idx=call_idx, is_prefill=(call_idx == 0),
                  pos_start=0, pos_end=1)


def test_project_out_orthonormal_modes_agree():
    torch.manual_seed(0)
    V = torch.linalg.qr(torch.randn(8, 3))[0].T          # 3 orthonormal rows
    h = torch.randn(2, 5, 8)
    a, b = project_out(h, V, "each"), project_out(h, V, "span")
    assert torch.allclose(a, b, atol=1e-5)
    assert (a.reshape(-1, 8) @ V.T).abs().max() < 1e-5


def test_project_out_modes_differ_when_non_orthogonal():
    """The ambiguity that has to be pre-registered. 'each' leaves part of the
    span behind; 'span' does not."""
    V = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])   # non-orthogonal
    h = torch.tensor([[3.0, 5.0, 7.0]])
    each, span = project_out(h, V, "each"), project_out(h, V, "span")
    assert not torch.allclose(each, span, atol=1e-4), (each, span)
    assert (span @ V.T).abs().max() < 1e-5, "span must leave nothing"
    assert (each @ V.T).abs().max() > 1e-3, "each is expected to leave residue"


def test_project_out_each_is_order_dependent():
    V = torch.tensor([[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    h = torch.tensor([[3.0, 5.0, 7.0]])
    assert not torch.allclose(project_out(h, V, "each"),
                              project_out(h, V.flip(0), "each"), atol=1e-4)


def test_residual_and_displacement_are_different_comparisons():
    """"span removes more" is ambiguous, and the two readings disagree.

    RESIDUAL, what is left afterwards: span <= each ALWAYS, because only span
    reaches true orthogonality to the directions.

    DISPLACEMENT, ||h - out||: ordering is NOT fixed. span is the MINIMAL
    displacement that achieves orthogonality, but `each` does not achieve it,
    so it is free to move further or less far.

    This distinction is why the calibration probe reports span displacing LESS
    (0.118 vs 0.142) while this file asserts span leaves a smaller residual.
    Both are true. An earlier test name said only "removes at least as much",
    which reads as contradicting the probe.

    On directions drawn independently of h the displacement ordering is close
    to a coin flip. On directions SELECTED by inner product with h -- which is
    exactly what the ablation does -- span displaces less in the large
    majority of cases, which is the regime the real model is in."""
    torch.manual_seed(1)
    d, both = 128, [0, 0]
    for _ in range(100):
        V, h = torch.randn(6, d), torch.randn(d)
        oe, os_ = project_out(h, V, "each"), project_out(h, V, "span")
        assert os_.norm() <= oe.norm() + 1e-4, "span must leave less behind"
        both[int((h - os_).norm() < (h - oe).norm())] += 1
    assert both[0] and both[1], f"displacement ordering looks fixed: {both}"

    sel_less = 0
    for _ in range(40):
        D = torch.randn(4000, d) / d ** 0.5
        h = torch.randn(d)
        V = D[(D @ h).topk(6).indices]
        sel_less += float((h - project_out(h, V, "span")).norm()
                          < (h - project_out(h, V, "each")).norm())
    assert sel_less / 40 > 0.7, f"expected span to displace less: {sel_less}/40"


def test_project_out_rank_deficient_is_safe():
    V = torch.tensor([[1.0, 0.0], [2.0, 0.0]])          # rank 1
    out = project_out(torch.tensor([[3.0, 4.0]]), V, "span")
    assert torch.allclose(out, torch.tensor([[0.0, 4.0]]), atol=1e-5), out


# ============================================ A: the plumbing is lossless

def test_null_intervention_is_bitwise_identical():
    m = tiny_model()
    with torch.no_grad():
        base = m(ids()).logits.clone()
    with Intervene(m, [2, 3], fn=lambda h, f: None), torch.no_grad():
        same = m(ids()).logits
    assert torch.equal(base, same), "a no-op hook perturbed the forward pass"


def test_alpha_zero_is_bitwise_identical():
    m = tiny_model()
    with torch.no_grad():
        base = m(ids()).logits.clone()
    with Intervene(m, [2, 3], fn=add_noise(0.0)), torch.no_grad():
        same = m(ids()).logits
    assert torch.equal(base, same)


def test_noise_is_seed_reproducible_and_seed_sensitive():
    m = tiny_model()
    outs = []
    for seed in (0, 0, 1):
        with Intervene(m, [3], fn=add_noise(0.05, seed=seed)), torch.no_grad():
            outs.append(m(ids()).logits.clone())
    assert torch.equal(outs[0], outs[1]), "same seed gave different noise"
    assert not torch.equal(outs[0], outs[2]), "different seeds gave same noise"


def test_bf16_rounds_away_tiny_alpha():
    """On THIS tensor, noise below bf16's relative epsilon (2^-8 = 0.0039)
    rounds away entirely: nothing changes at alpha <= 1e-3, 88% of elements
    change at 4e-3, all of them at 1e-2.

    DO NOT GENERALISE THIS TO A NO-OP FLOOR. An earlier version of this
    docstring claimed any per-element effect under ~0.4% of the residual
    magnitude is a no-op in bf16. The real model disproved it: logits moved at
    alpha = 1e-3. This tensor has 384 elements; a real forward pass has ~1.1M
    heavy-tailed ones, and with that many draws the lower tail always contains
    something small enough for a 0.1% perturbation to survive rounding. The
    cliff here is a small-sample artifact.

    What DOES hold, measured on the real model, is behavioural: KL stays at
    ~1e-4 from alpha 0.004 to 0.1, then rises ~6800x by 0.3. That flat region
    is the real hazard for calibration, and it is not a dtype property.

    h is seeded: with an unseeded draw one near-zero element flips a bitwise
    test, which is how this test first failed."""
    g = torch.Generator().manual_seed(0)
    h = torch.randn(1, 6, 64, generator=g).to(torch.bfloat16)
    f = _dummy_firing()
    for alpha in (1e-6, 1e-5, 1e-4, 1e-3):
        assert torch.equal(add_noise(alpha)(h, f), h), f"{alpha} should vanish"
    frac = (add_noise(4e-3)(h, f) != h).float().mean().item()
    assert 0.5 < frac < 1.0, f"expected a partial cliff at bf16 eps, got {frac}"
    assert (add_noise(1e-2)(h, f) != h).float().mean().item() == 1.0


# ============================================ B: it lands where you think

def test_noise_at_layer_i_changes_i_plus_2_not_i_plus_1():
    """Contract 5 as a diagnostic: hidden_states[i+1] is recorded BEFORE the
    intervention, so unchanged there is the confirmation, not a failure."""
    m = tiny_model()
    x = ids()
    with torch.no_grad():
        m(x, output_hidden_states=True)          # ensure capture hooks exist
        base = m(x, output_hidden_states=True)
    i = 2
    with Intervene(m, [i], fn=add_noise(0.5)), torch.no_grad():
        out = m(x, output_hidden_states=True)
    assert torch.equal(out.hidden_states[i + 1], base.hidden_states[i + 1])
    assert not torch.equal(out.hidden_states[i + 2], base.hidden_states[i + 2])


def test_position_mask_respects_causal_masking():
    """Perturb position p only: logits before p must be untouched, from p on
    must move. Causal masking guarantees it, so a failure means the position
    axis or the absolute-offset bookkeeping is wrong."""
    m = tiny_model()
    x = ids(10)
    p = 6
    with torch.no_grad():
        base = m(x).logits.clone()
    iv = Intervene(m, [2], fn=add_noise(0.5), positions={p})
    with iv, torch.no_grad():
        out = m(x).logits
    assert torch.equal(out[:, :p], base[:, :p]), "changed something before p"
    assert not torch.equal(out[:, p:], base[:, p:]), "changed nothing from p"
    assert iv.n_modified == 1, iv.n_modified


def test_exposure_counts_positions_not_firings():
    m = tiny_model()
    x = ids(10)
    iv = Intervene(m, [2, 3], fn=add_noise(0.1))
    with iv, torch.no_grad():
        m(x)
    assert iv.n_modified == 2 * 10, iv.n_modified


# ============================================ C: scope control

def test_generation_scope_cannot_change_the_first_new_token():
    """The first generated token comes out of the PREFILL pass, so a
    generation-only intervention cannot touch it. If it does, scope is
    broken."""
    m = tiny_model()
    x = ids(12)
    with torch.no_grad():
        base = m.generate(x, max_new_tokens=5, use_cache=True)
    with Intervene(m, [3], fn=add_noise(0.8), scope="generation"), torch.no_grad():
        gen = m.generate(x, max_new_tokens=5, use_cache=True)
    assert gen[0, x.shape[1]] == base[0, x.shape[1]], "scope leaked into prefill"


def test_prefill_scope_changes_the_first_new_token():
    m = tiny_model()
    x = ids(12)
    with torch.no_grad():
        base = m.generate(x, max_new_tokens=5, use_cache=True)
    with Intervene(m, [3], fn=add_noise(0.8), scope="prefill"), torch.no_grad():
        gen = m.generate(x, max_new_tokens=5, use_cache=True)
    assert gen[0, x.shape[1]] != base[0, x.shape[1]]


def test_firing_log_tracks_absolute_positions():
    """With a KV cache the tensor is (1,1,d) during generation and the shape
    does not tell you which position it is."""
    m = tiny_model()
    x = ids(12)
    iv = Intervene(m, [3], fn=lambda h, f: None)
    with iv, torch.no_grad():
        m.generate(x, max_new_tokens=4, use_cache=True)
    f = iv.firings(3)
    assert f[0].is_prefill and f[0].pos_start == 0 and f[0].pos_end == 12
    assert [x.pos_start for x in f[1:]] == [12, 13, 14], [y.pos_start for y in f]
    assert not any(x.is_prefill for x in f[1:])


def test_nocache_generation_is_refused_not_miscounted():
    """Absolute positions assume prefill-then-one-token-per-step.

    Without a KV cache every firing sees the whole prefix again, so
    `pos_start` runs past the true sequence length. That matters more here
    than it looks: `exclude` is keyed by ABSOLUTE position, so the paper's
    top-10 rule would go on quietly matching nothing. Refuse instead.
    """
    m = tiny_model()
    try:
        with Intervene(m, [2], fn=lambda h, f: None), torch.no_grad():
            m.generate(ids(10), max_new_tokens=3, use_cache=False)
    except RuntimeError as e:
        assert "seq_len" in str(e), e
        return
    raise AssertionError("a non-prefill firing with seq>1 should have raised")


def test_exclude_is_sliced_per_firing():
    m = tiny_model()
    x = ids(10)
    seen = {}
    def record(h, f):
        seen[f.call_idx] = f.exclude
        return None
    iv = Intervene(m, [2], fn=record, exclude={3: [11, 22], 7: [33]})
    with iv, torch.no_grad():
        m(x)
    assert seen[0] == {3: [11, 22], 7: [33]}, seen


def _ablate_logits(m, x, layer, exclude, kind="ablate", seed=0, problem=0):
    fn = make_ablation(m, 10, mode="span", gain_scaled=True, kind=kind,
                       seed=seed, problem=problem)
    with Intervene(m, [layer], fn=fn, scope="prefill",
                   exclude=exclude), torch.no_grad():
        return m(x).logits.clone()


def test_generate_ablated_matches_model_generate_when_fn_is_noop():
    """The reference test for run.py's decode loop.

    generate_ablated reimplements greedy decoding so the exclusion rule can
    interleave a clean forward pass per step. A reimplemented decoder is a
    place to get position ids, cache handling or the stopping rule subtly
    wrong, and every such bug would show up as an ablation "effect". With a
    no-op fn it must be bit-identical to transformers' own greedy decode --
    INCLUDING with the clean pass running, which is what proves the paired
    pass does not perturb the ablated trajectory.
    """
    m, x = tiny_model(), ids(9)
    with torch.no_grad():
        ref = m.generate(x, max_new_tokens=12, do_sample=False, use_cache=True,
                         eos_token_id=None)
    for ex in (None, 10):
        seq, n, iv = generate_ablated(m, x, [2, 3], lambda h, f: None, 12,
                                      exclude_topk=ex, eos_token_id=None)
        assert n == 12, (ex, n)
        assert torch.equal(seq, ref), f"exclude_topk={ex} diverged from generate"
        # one firing per step per layer, and positions must advance by one --
        # the clean pass must not have been counted
        fr = iv.firings(2)
        assert len(fr) == 12, (ex, len(fr))
        assert fr[0].is_prefill and fr[0].pos_end == 9, fr[0]
        assert [f.pos_start for f in fr[1:]] == list(range(9, 20)), \
            [f.pos_start for f in fr]


def test_generate_ablated_feeds_the_exclusion_set_per_step():
    """The rule is per position, and during generation the position is one
    token that did not exist when the loop started. Absolute keys, sliced by
    Intervene, arriving one per step -- if this regressed, the ablation would
    silently exempt nothing."""
    m, x = tiny_model(), ids(7)
    seen = {}

    def record(h, f):
        if f.layer == 2:
            seen[f.pos_start] = dict(f.exclude)
        return None

    _, n, _ = generate_ablated(m, x, [2], record, 5, exclude_topk=4,
                               eos_token_id=None)
    assert set(seen) == {0, 7, 8, 9, 10}, sorted(seen)
    assert sorted(seen[0]) == list(range(7)), "prefill needs every position"
    for p in (7, 8, 9, 10):
        assert list(seen[p]) == [p], (p, list(seen[p]))
        assert len(seen[p][p]) == 4, seen[p]


def test_generate_ablated_stops_on_eos():
    m, x = tiny_model(), ids(6)
    with torch.no_grad():
        first = int(m(x).logits[0, -1].argmax())
    _, n, _ = generate_ablated(m, x, [2], lambda h, f: None, 20,
                               eos_token_id=first)
    assert n == 1, f"should have stopped on the first token, got {n}"


def test_readouts_accept_captured_cpu_activations():
    """Capture detaches to CPU; the readouts must not require a manual move.

    HONEST LIMIT: on a CPU-only runner this asserts a contract it cannot
    falsify -- there is one device, so a missing `.to(...)` passes. The real
    failure is `Expected all tensors to be on the same device, mps:0 and cpu`
    raised from inside RMSNorm, and only an accelerator runner sees it. Kept
    because it pins the intended contract, and because both call sites that
    got this wrong looked exactly like the lines below.
    """
    m, x, L = tiny_model(), ids(8), 3
    cap = Capture(m, [L])
    with cap, torch.no_grad():
        m(x)
    h = cap.first(L)
    assert h.device.type == "cpu", "Capture is supposed to detach to CPU"
    dev = m.lm_head.weight.device
    with torch.no_grad():
        assert topk_tokens(m, h, 5).device == dev
        assert logit_lens(m, h).device == dev
        assert readout_scores(m, h).device == dev


def test_make_ablation_honours_the_exclusion_set():
    """The paper's rule -- tokens in the clean top-10 are exempt -- has to
    reach the selection, and it lives in VOCABULARY space.

    Two directions, because only the pair rules out a no-op. Excluding tokens
    that were never selected must change NOTHING; excluding the tokens that
    WERE selected must change something. An implementation that quietly
    ignores `exclude` passes the first and fails the second, which is exactly
    the failure this project spent a session not noticing.
    """
    m, x, L = tiny_model(), ids(8), 3
    cap = Capture(m, [L])
    with cap, torch.no_grad():
        m(x)
    with torch.no_grad():
        top = topk_tokens(m, cap.first(L), 10)

    base = _ablate_logits(m, x, L, {})
    unselected, selected = {}, {}
    for p in range(x.shape[1]):
        s = set(top[p].tolist())
        unselected[p] = [t for t in range(256) if t not in s][:10]
        selected[p] = top[p].tolist()
    assert torch.equal(base, _ablate_logits(m, x, L, unselected)), \
        "excluding never-selected tokens should be a no-op"
    assert not torch.equal(base, _ablate_logits(m, x, L, selected)), \
        "excluding the selected tokens should change what is removed"


def test_make_ablation_control_is_seed_reproducible():
    """The random-direction control has to reproduce across the MPS -> CUDA
    move, or the run's control is not the same control the calibration
    probe measured."""
    m, x, L = tiny_model(), ids(8), 3
    a = _ablate_logits(m, x, L, {}, kind="rand_tok", seed=5)
    assert torch.equal(a, _ablate_logits(m, x, L, {}, kind="rand_tok", seed=5))
    assert not torch.equal(a, _ablate_logits(m, x, L, {}, kind="rand_tok",
                                             seed=6))


def test_make_ablation_control_draws_differ_per_problem():
    """The control must AVERAGE over random selections. Keyed by (seed,
    layer, position) alone, every problem in a cell saw the same
    pseudo-random pattern -- decorrelated only by accident, where prompt
    lengths happened to differ -- so one unlucky draw biased the whole
    control arm instead of washing out across problems."""
    m, x, L = tiny_model(), ids(8), 3
    a = _ablate_logits(m, x, L, {}, kind="rand_tok", problem=0)
    assert torch.equal(a, _ablate_logits(m, x, L, {}, kind="rand_tok",
                                         problem=0))
    assert not torch.equal(a, _ablate_logits(m, x, L, {}, kind="rand_tok",
                                             problem=1)), \
        "problems must not share their random draws"


def test_draw_seed_is_deterministic_and_kills_the_stride_collision():
    """The old arithmetic (seed * 1_000_003 + layer * 1009 + pos) collided
    exactly at (layer, pos + 1009) == (layer + 1, pos) -- well inside a CoT
    trace, so adjacent band layers shared their 'random' draws at a fixed
    position offset."""
    assert draw_seed(0, 0, 14, 1509) == draw_seed(0, 0, 14, 1509)
    assert draw_seed(0, 0, 14, 1009) != draw_seed(0, 0, 15, 0), \
        "the pair the old stride arithmetic collided on"
    s = draw_seed(1, 2, 3, 4)
    assert isinstance(s, int) and 0 <= s < 2 ** 32


def test_generate_ablated_default_stop_set_is_the_models_own():
    """run.py leaves eos unspecified so the intervened cells stop on the SAME
    set model.generate uses. Qwen3's generation_config carries a LIST
    ([<|im_end|>, <|endoftext|>]) while tok.eos_token_id is the single first
    entry -- passing the latter let a degenerate intervened generation run
    through <|endoftext|> to the cap while its intact partner terminated,
    inflating hit_cap and `incomplete` in the intervened cells only."""
    m, x = tiny_model(), ids(6)
    with torch.no_grad():
        first = int(m(x).logits[0, -1].argmax())
    saved = m.generation_config.eos_token_id
    try:
        m.generation_config.eos_token_id = [9999, first]   # a list, as Qwen3
        _, n, _ = generate_ablated(m, x, [2], lambda h, f: None, 20)
        assert n == 1, f"the list's LATER entries must stop generation; " \
                       f"ran {n} tokens"
    finally:
        m.generation_config.eos_token_id = saved


def test_make_ablation_rejects_unknown_kind():
    try:
        make_ablation(tiny_model(), 10, mode="span", gain_scaled=True,
                      kind="noise")
    except ValueError:
        return
    raise AssertionError("an unknown kind should have been rejected")


def test_bad_scope_rejected():
    m = tiny_model()
    try:
        Intervene(m, [0], fn=lambda h, f: None, scope="sometimes")
    except ValueError:
        return
    raise AssertionError("bad scope should have been rejected")


# ======================================== gain-aware logit-lens readout

def _model_with_nontrivial_gain():
    """RMSNorm gain initialises to ones, so a randomly-initialised model
    cannot distinguish gain-aware from gain-blind. Give it a real gain."""
    m = tiny_model()
    if "gain_set" not in _CACHE:
        g = torch.Generator().manual_seed(7)
        with torch.no_grad():
            final_norm(m).weight.copy_(
                torch.rand(m.config.hidden_size, generator=g) * 2.0 + 0.25)
        _CACHE["gain_set"] = True
    return m


def _bf16_model():
    """bf16 with a vocabulary big enough for rank-boundary near-ties.

    The shared tiny_model is float32 with 256 tokens, where the rms shortcut
    is indistinguishable from the exact norm -- reverting the fix passed every
    other test in this file. Measured here: 93.8% top-10 set agreement between
    exact and shortcut in bf16, 100% in float32, against 94.1% on the real
    Qwen3-4B. Same mechanism, small enough to run in CI."""
    if "bf16" not in _CACHE:
        g = torch.Generator().manual_seed(3)
        torch.manual_seed(0)
        cfg = Qwen3Config(vocab_size=2000, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=16,
                          max_position_embeddings=64, pad_token_id=0)
        m = Qwen3ForCausalLM(cfg).to(torch.bfloat16).eval()
        with torch.no_grad():
            m.model.norm.weight.copy_(torch.rand(64, generator=g) * 2 + 0.25)
        _CACHE["bf16"] = m
    return _CACHE["bf16"]


def test_topk_uses_the_exact_norm_not_the_gain_shortcut():
    """REGRESSION for the calibration's 94.1% agreement failure.

    <W_U[t], g*h/rms(h)> and <W_U[t]*g, h> rank identically in real
    arithmetic. In bf16 they round differently, and enough near-ties sit at
    the rank-10 boundary to change which directions get ablated. Only the
    exact norm may be used."""
    m = _bf16_model()
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, 2000, (1, 16), generator=g)
    with torch.no_grad():
        h = m(x, output_hidden_states=True).hidden_states[-2]
        got = topk_tokens(m, h, 10)
        want = logit_lens(m, h).reshape(-1, 2000).topk(10, -1).indices
    assert torch.equal(got, want), "topk_tokens diverged from lm_head(norm(h))"


def test_readout_topk_matches_model_logits():
    """The check that catches a dropped gain. Applied to the LAST layer's
    output the gain-aware readout IS the model's logits, so its top-k must
    match exactly. Forget the gain and this fails."""
    m = _model_with_nontrivial_gain()
    cap = Capture(m, [NL - 1])
    with cap, torch.no_grad():
        out = m(ids())
    h = cap.first(NL - 1).to(next(m.parameters()).device)
    with torch.no_grad():
        got = topk_tokens(m, h, 5)
    want = out.logits.reshape(-1, out.logits.shape[-1]).topk(5, -1).indices
    assert torch.equal(got, want), (got[:2], want[:2])


def test_dropping_the_gain_changes_the_selection():
    """Not a rounding difference -- a different set of tokens."""
    m = _model_with_nontrivial_gain()
    cap = Capture(m, [NL - 1])
    with cap, torch.no_grad():
        m(ids())
    h = cap.first(NL - 1)
    with torch.no_grad():
        aware = topk_tokens(m, h, 10)
        blind = (h.reshape(-1, h.shape[-1]) @ m.lm_head.weight.T).topk(10, -1).indices
    assert not torch.equal(aware, blind), "gain made no difference to ranking"


def test_readout_scores_agree_with_topk_and_chunking():
    m = _model_with_nontrivial_gain()
    cap = Capture(m, [3])
    with cap, torch.no_grad():
        m(ids())
    h = cap.first(3)
    with torch.no_grad():
        full = readout_scores(m, h).reshape(-1, m.config.vocab_size)
        assert torch.equal(topk_tokens(m, h, 4, chunk=3),
                           topk_tokens(m, h, 4, chunk=1000)), "chunking differs"
        assert torch.equal(topk_tokens(m, h, 4, chunk=1000),
                           full.topk(4, -1).indices)


def test_exclusion_removes_tokens_from_topk():
    m = _model_with_nontrivial_gain()
    cap = Capture(m, [3])
    with cap, torch.no_grad():
        m(ids())
    h = cap.first(3)
    with torch.no_grad():
        base = topk_tokens(m, h, 5)
        banned = base[2, :3].tolist()
        after = topk_tokens(m, h, 5, exclude={2: banned})
    assert after.shape == base.shape, "exclusion must still return k survivors"
    assert not set(after[2].tolist()) & set(banned), after[2]
    assert torch.equal(after[0], base[0]), "exclusion leaked to another position"


def test_directions_for_shape_and_gain_flag():
    m = _model_with_nontrivial_gain()
    toks = [3, 11, 40]
    plain = directions_for(m, toks, gain_scaled=False)
    scaled = directions_for(m, toks, gain_scaled=True)
    assert plain.shape == (3, m.config.hidden_size), plain.shape
    assert not torch.allclose(plain, scaled)
    assert torch.allclose(scaled, plain * readout_gain(m))


# ================================================== config: pre-registration

def test_direct_cap_does_not_bind():
    """The placement probe saw the cap bind on 100% of intervened direct
    generations at 32. cap_warnings' own docstring prescribes 128-256."""
    assert config.CAPS["gsm8k"]["direct"] >= 128, config.CAPS["gsm8k"]["direct"]


def test_cap_for_is_keyed_by_level_so_ablated_states_need_no_entries():
    caps = config.CAPS["gsm8k"]
    for state in ("intact", "ablated", "random"):
        assert config.cap_for(f"direct_{state}", "gsm8k") == caps["direct"]
        assert config.cap_for(f"cot_{state}", "gsm8k") == caps["cot"]
    for bad, exc in ((("nonsense_intact", "gsm8k"), KeyError),
                     (("direct_intact", "nonsense"), KeyError)):
        try:
            config.cap_for(*bad)
        except exc:
            continue
        raise AssertionError(f"cap_for{bad} should have raised {exc.__name__}")


def test_cap_for_requires_a_dataset():
    """The signature IS the fix. run.py once accepted --dataset and then loaded
    GSM8K anyway, so a MATH-500 run was GSM8K problems under a MATH-500
    filename. A cap_for(cond) that defaulted to gsm8k would let exactly that
    class of bug back in, silently."""
    try:
        config.cap_for("direct_intact")
    except TypeError:
        return
    raise AssertionError("cap_for must not have a default dataset")


def test_unset_caps_raise_rather_than_reaching_generate():
    """max_new_tokens=None does not fail -- it generates to the context limit.
    So an undecided cap must raise here, not return None. Tested by UNSETTING
    a committed cap rather than pointing at whichever dataset happens to be
    open (math500's caps were committed and quietly retired the old version
    of this guard), so settling a pre-registration cannot retire it again."""
    saved = config.CAPS["gsm8k"]["direct"]
    try:
        config.CAPS["gsm8k"]["direct"] = None
        try:
            config.cap_for("direct_intact", "gsm8k")
        except ValueError:
            pass
        else:
            raise AssertionError("unset cap did not raise")
    finally:
        config.CAPS["gsm8k"]["direct"] = saved


def test_math500_pre_registration_is_closed_at_the_committed_values():
    """The caps the report will cite: cot at the 16384 measurement ceiling (a
    budget decision under the stopping rule -- the CAPS comment says why) and
    direct at the measured 128. A drift here is a different experiment."""
    assert config.dataset_ready("math500") == []
    for state in ("intact", "ablated", "random"):
        assert config.cap_for(f"cot_{state}", "math500") == 16384
        assert config.cap_for(f"direct_{state}", "math500") == 128


def test_gsm8k_direct_prompt_is_byte_identical_to_what_ran():
    """The committed n=150 data and the fingerprint assertion in
    probes/capture.py both pin this exact string. Centralising it into config
    must not have reworded it."""
    suffix, prefill = config.direct_prompt("gsm8k")
    assert suffix == ("\n\nRespond with only the final numeric answer and "
                      "nothing else. Do not show any reasoning."), repr(suffix)
    assert prefill == "\\boxed{", repr(prefill)


def test_math500_prompt_is_not_silently_gsm8ks():
    """GSM8K's wording asks for "the final numeric answer" and MATH-500
    answers are frequently \\frac{3}{2}, 2\\sqrt{2}, (2,5). Inheriting it would
    instruct the model away from the format its own gold uses, inflating
    `unparsed` preferentially in the degraded cells.

    This used to assert math500 was UNSET. Now that it is decided, the thing
    worth pinning is the property the decision had to satisfy -- a test that
    only says "undecided" turns settling the choice into a red build."""
    g, _ = config.direct_prompt("gsm8k")
    m, _ = config.direct_prompt("math500")
    assert "numeric" in g, g
    assert "numeric" not in m, m
    # ...and differs from GSM8K in nothing else. Any further rewording would
    # confound "MATH-500 is harder" with "MATH-500 was asked differently".
    assert m == g.replace(" numeric", ""), (m, g)


def test_aime_prompt_is_deliberately_gsm8ks_verbatim():
    """AIME answers are integers 0-999 (validate_gold enforces it), so "the
    final numeric answer" is correct as written and the byte-identical string
    keeps the two direct cells as close to one condition as two datasets
    allow. Pinned because it must be a DECISION, not a copy-paste nobody
    revisits."""
    assert config.direct_prompt("aime24") == config.direct_prompt("gsm8k")


def test_every_dataset_shares_the_boxed_prefill():
    """The prefill is what makes the direct condition direct, and
    scoring's extraction_mode="first_match" is built on it. A dataset whose
    prefill drifted would be a different condition under the same name."""
    for d in ("gsm8k", "math500", "aime24"):
        assert config.direct_prompt(d)[1] == "\\boxed{", d


def test_dataset_ready_reports_what_the_accessors_raise_on():
    """_UNDECIDED and require() reach globals only, so the nested per-dataset
    tables need their own reporting -- and it must agree with the accessors.

    Tests the AGREEMENT, not which items happen to be open, so settling a
    choice does not turn this red."""
    for d in ("gsm8k", "math500", "aime24"):
        open_items = config.dataset_ready(d)
        for level in config.RUN_LEVELS:
            raised = False
            try:
                config.cap_for(f"{level}_intact", d)
            except ValueError:
                raised = True
            assert raised == (f"CAPS[{d!r}][{level!r}]" in open_items), (
                d, level, open_items)
        raised = False
        try:
            config.n_default(d)
        except ValueError:
            raised = True
        assert raised == (f"N_DEFAULT[{d!r}]" in open_items), (d, open_items)


def test_dataset_ready_is_scoped_to_the_levels_being_run():
    """Two things were demanded that no run needed: a cap for `nothink`, which
    conditions() never builds, and the direct prompt for a cot-only staged run.
    Both blocked a dataset on a choice belonging to a cell that would not be
    generated."""
    saved = config.CAPS["gsm8k"]["nothink"]
    try:
        config.CAPS["gsm8k"]["nothink"] = None
        assert config.dataset_ready("gsm8k") == [], (
            "an unset nothink cap blocked a cot+direct run")
    finally:
        config.CAPS["gsm8k"]["nothink"] = saved

    # A cot-only stage must not need the direct arm's prompt.
    saved = config.DIRECT_INSTRUCTION["gsm8k"]
    try:
        config.DIRECT_INSTRUCTION["gsm8k"] = None
        assert config.dataset_ready("gsm8k", levels=("cot",)) == []
        assert config.dataset_ready("gsm8k") == \
               ["DIRECT_INSTRUCTION['gsm8k']"]
    finally:
        config.DIRECT_INSTRUCTION["gsm8k"] = saved

    # need_n=False for a caller that was handed an explicit --n.
    saved = config.N_DEFAULT["gsm8k"]
    try:
        config.N_DEFAULT["gsm8k"] = None
        assert "N_DEFAULT['gsm8k']" in config.dataset_ready("gsm8k")
        assert "N_DEFAULT['gsm8k']" not in config.dataset_ready(
            "gsm8k", need_n=False)
    finally:
        config.N_DEFAULT["gsm8k"] = saved


def test_n_default_refuses_to_borrow_another_datasets_n():
    """`--n 150` was an argparse literal: a GSM8K number, impossible on aime24
    (30 problems) and unjustified elsewhere. So an unset n must raise rather
    than fall back to another dataset's.

    Tests the MECHANISM against a temporarily-unset entry rather than against
    whichever dataset is currently open -- pinning "math500 is undecided" would
    make settling it a red build."""
    assert config.n_default("gsm8k") == 150
    assert config.n_default("aime24") == 30, "aime24 must be the whole split"
    saved = config.N_DEFAULT["gsm8k"]
    try:
        config.N_DEFAULT["gsm8k"] = None
        for bad, exc in (("gsm8k", ValueError), ("nonsense", KeyError)):
            try:
                config.n_default(bad)
            except exc:
                continue
            raise AssertionError(f"n_default({bad!r}) should have raised")
    finally:
        config.N_DEFAULT["gsm8k"] = saved


def test_suggest_cap_clears_the_distribution_it_was_given():
    """The suggestion has to exceed everything observed, or the cap it
    recommends binds on data already in hand."""
    for sample in ([7], [1, 2, 3], list(range(200)), [3000] * 5):
        assert config.suggest_cap(sample) > max(sample), sample
        assert config.suggest_cap(sample) % config.CAP_ROUNDING == 0, sample
    try:
        config.suggest_cap([])
    except ValueError:
        return
    raise AssertionError("an empty sample cannot size a cap")


def test_measure_cap_is_never_below_a_committed_cap():
    """MEASURE_CAP is a ceiling for calibration, so it must be at least as
    loose as any cap it could be used to size -- otherwise the measurement is
    censored below the answer.

    EQUALITY is allowed, because the stopping rule creates exactly one
    legitimate case of it: a tail that will not terminate cannot be chased to
    a cap (CEILING_RETRY_MAX_HIT), and the cap is then committed AT the
    measurement ceiling as a budget decision with the censoring disclosed --
    math500's cot cap is that case. Strictly-below stays the invariant for
    caps that claim to be measured."""
    for dataset, caps in config.CAPS.items():
        for level, cap in caps.items():
            if cap is not None:
                assert config.MEASURE_CAP[level] >= cap, (dataset, level)


def test_bands_translate_to_the_documented_layers():
    assert band_from_depth(36, *config.BANDS["light"]) == range(14, 20)
    assert band_from_depth(36, *config.BANDS["heavy"]) == range(14, 34)
    assert config.PRIMARY_BAND == "light"


def test_three_tens_are_distinct_constants():
    assert config.K_ABLATE == 10 and config.EXCLUDE_TOPK == 10
    assert config.K_OCCUPANCY == 25


def test_require_gates_unset_preregistration_choices():
    """Tests the MECHANISM, not which items happen to be open. Asserting
    PROJECTION_MODE is still unset would turn a legitimate decision into a
    red test, and a test that punishes progress gets deleted."""
    for name in config.undecided():
        try:
            config.require(name)
        except ValueError:
            continue
        raise AssertionError(f"{name} is unset but require() did not raise")
    saved = config.PROJECTION_MODE
    try:
        config.PROJECTION_MODE = None
        try:
            config.require("PROJECTION_MODE")
        except ValueError:
            pass
        else:
            raise AssertionError("require() did not gate an unset choice")
        config.PROJECTION_MODE = "span"
        assert config.require("PROJECTION_MODE") == "span"
    finally:
        config.PROJECTION_MODE = saved


# ================================================ random-direction control

def test_random_directions_shape_and_determinism():
    m = _model_with_nontrivial_gain()
    d = m.config.hidden_size
    for mode in ("tokens", "gaussian"):
        a1 = random_directions(m, 10, seed=5, mode=mode)
        a2 = random_directions(m, 10, seed=5, mode=mode)
        b = random_directions(m, 10, seed=6, mode=mode)
        assert a1.shape == (10, d), (mode, a1.shape)
        assert torch.equal(a1, a2), f"{mode} not reproducible"
        assert not torch.equal(a1, b), f"{mode} ignored the seed"


def test_random_tokens_control_uses_real_unembedding_rows():
    """The point of the tokens control: same direction DISTRIBUTION as the
    ablation, only the selection randomised. Gaussian directions are a
    different question and must not be confused with it."""
    m = _model_with_nontrivial_gain()
    V = random_directions(m, 6, seed=1, mode="tokens", gain_scaled=False)
    W = m.lm_head.weight
    for row in V:
        assert (W == row).all(dim=-1).any(), "not an unembedding row"
    G = random_directions(m, 6, seed=1, mode="gaussian")
    assert not any((W == r).all(dim=-1).any() for r in G)


def test_random_control_removes_a_comparable_fraction():
    """A control that removes a wildly different magnitude is not matched."""
    m = _model_with_nontrivial_gain()
    cap = Capture(m, [3])
    with cap, torch.no_grad():
        m(ids())
    h = cap.first(3)[0, 0].float()
    with torch.no_grad():
        sel = topk_tokens(m, cap.first(3), 10)[0]
        Va = directions_for(m, sel, gain_scaled=True).float()
        Vr = random_directions(m, 10, seed=2, mode="tokens",
                               gain_scaled=True).float()
    ra = (h - project_out(h, Va, "span")).norm() / h.norm()
    rr = (h - project_out(h, Vr, "span")).norm() / h.norm()
    assert 0 < rr < 1 and 0 < ra < 1, (ra, rr)


# ============================================ problem sampling

def test_problem_ids_is_pinned():
    """Once results depend on a sample, the sample must not drift."""
    assert config.problem_ids(20, 1319) == [
        120, 175, 192, 265, 350, 446, 474, 480, 554, 590,
        799, 883, 991, 1020, 1062, 1070, 1101, 1234, 1257, 1302]


def test_problem_ids_nest_for_every_dataset_size():
    """The property random.sample only had by accident. 500 is MATH-500,
    where random.sample's nesting demonstrably breaks."""
    for N in (200, 500, 1319, 7473):
        a, b, c = (set(config.problem_ids(k, N)) for k in (20, 50, 150))
        assert a <= b <= c, f"failed to nest at dataset_size={N}"
        assert len(a) == 20 and len(b) == 50 and len(c) == 150


def test_random_sample_does_not_nest_on_math500():
    """Documents WHY problem_ids exists. If this ever goes green, CPython
    changed its algorithm -- which is itself the fragility being avoided."""
    import random as _r
    _r.seed(0); small = set(_r.sample(range(500), 20))
    _r.seed(0); big = set(_r.sample(range(500), 150))
    assert not small <= big, "random.sample nested; the hazard may have moved"


def test_problem_ids_is_deterministic_and_bounded():
    assert config.problem_ids(10, 100) == config.problem_ids(10, 100)
    assert config.problem_ids(10, 100, seed=1) != config.problem_ids(10, 100)
    assert all(0 <= i < 100 for i in config.problem_ids(10, 100))
    for bad in ((101, 100), (-1, 100)):
        try:
            config.problem_ids(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should have raised")


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
        print("hooks.py contract is NOT satisfied -- do not run any "
              "intervention against it.")
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
