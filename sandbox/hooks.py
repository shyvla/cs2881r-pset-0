"""Forward-hook machinery for the J-space ablation experiment.

Milestone 5 scope: register hooks on decoder layers, capture the residual
stream, guarantee removal. No intervention. Half B's ablation reuses
`LayerHooks` with a different hook function -- nothing here changes for it.

THE CONTRACT (transformers 5.14.1)
----------------------------------
Verified in test_hooks.py against a randomly-initialised Qwen3 built from
config -- no weights, no download, no GPU, runs in seconds.

1. `Qwen3DecoderLayer.forward` returns a BARE TENSOR, not a tuple.
   `output[0]` does not raise. It silently hands you (seq, d_model) with the
   batch dimension stripped.

2. `output_hidden_states=True` is itself implemented with forward hooks
   (`@capture_outputs` + `_can_record_outputs`), so your hook and the model's
   own record are the same mechanism. The tuple is built as
   [embeddings, out(L0), out(L1), ... out(L_{n-1})], then
   `tie_last_hidden_states=True` POPS THE LAST ENTRY and substitutes
   `last_hidden_state`. Therefore:

       hidden_states[0]      = embeddings
       hidden_states[i + 1]  = raw output of layers[i]     for i <= n-2
       hidden_states[-1]     = norm(output of layers[n-1])  <-- NOT raw

   Because both sides are hooks, an equality test confirms module path and
   index but is NOT an independent oracle. `logit_lens` below is -- it routes
   through lm_head, which the capture mechanism never touches.

3. Returning a Tensor from a forward hook REPLACES the layer output. That is
   the Half B intervention mechanism. Returning a v4-style tuple raises
   AttributeError -- loudly, which is the good case: the ablation cannot
   quietly do nothing by returning the wrong container. It can only quietly do
   nothing by being on the wrong tensor, which is what Milestone 6 is for.

4. Calling `output_hidden_states=True` PERMANENTLY installs one forward hook
   per decoder layer, and never removes them. `maybe_install_capturing_hooks`
   sets a flag and returns early forever after. They are inert when nothing is
   collecting, but they are there, and `hook_census` will show them. A hook
   count of zero is therefore not the right leak test -- use a delta.

5. THE LANDMINE THAT FOLLOWS FROM 4. Forward hooks fire in registration order,
   and each receives the previous one's return value. Because the transformers
   capture hook was installed by an EARLIER validation call, it runs BEFORE an
   ablation hook registered later. So `output_hidden_states=True` reports the
   PRE-intervention activations while the intervention still takes effect on
   the forward pass. Verified: zeroing layer 2's output leaves
   `hidden_states[3]` unchanged, yet the logits change.

   Consequence: NEVER verify an intervention by reading `output_hidden_states`.
   It will show unchanged activations and you will conclude the ablation
   failed when it worked -- or, in the other registration order, the reverse.
   Verify interventions against logits, or against a Capture registered after
   the ablation hook. The registration order depends on whether anything
   earlier in the process happened to request hidden states, which is
   invisible, session-dependent, and exactly the sort of thing that makes a
   notebook behave differently from a script.

These are properties of transformers 5.x, NOT of transformers in general. The
v4 -> v5 change is exactly what made revision 2 of the handoff document wrong
on points 1 and 2. test_hooks.py pins them so a version bump goes red instead
of going quietly wrong -- the same reasoning that pins math-verify==0.9.0.

BATCH SIZE
----------
Everything here asserts batch 1 rather than assuming it. Batching would put
left-padding offsets, masked-position exposure miscounts, and batch-dependent
matmul reduction order underneath the one piece of code the whole result rests
on. If you later decide to batch, these assertions fail loudly instead of the
experiment succeeding quietly.
"""
import math

import torch

__all__ = [
    "decoder_layers", "n_layers", "band_from_depth", "resolve_band",
    "final_norm",
    "hook_census", "logit_lens", "LayerHooks", "Capture", "Intervene",
    "Firing", "add_noise", "project_out", "readout_gain", "readout_scores",
    "topk_tokens", "directions_for", "random_directions", "make_ablation",
    "generate_ablated",
    "BATCH1_MSG", "NOCACHE_MSG",
]

BATCH1_MSG = (
    "hook saw batch size {got}, expected 1. This code asserts batch-1 "
    "deliberately; see the BATCH SIZE note in hooks.py before relaxing it."
)

NOCACHE_MSG = (
    "layer {layer} saw seq_len={seq} on a non-prefill firing. Absolute "
    "position accounting assumes one prefill pass then one token per step, "
    "i.e. use_cache=True and no speculative decoding. Refusing to guess "
    "rather than silently miscount exposure."
)


# ------------------------------------------------------------------ topology

def decoder_layers(model):
    """The ONE place the module path is written.

    Same principle as scoring.render_prompt being the one place a prompt is
    built: if it is written twice it can drift, and drift here is silent.
    """
    return model.model.layers


def n_layers(model) -> int:
    return len(decoder_layers(model))


def band_from_depth(nl: int, lo: float = 0.38, hi: float = 0.92) -> range:
    """Translate the paper's fractional-depth band to layer indices.

    The paper's workspace band is ~L38-L92 on a reindexed 0-100 depth scale.
    Derived, never hand-typed: `range(14, 34)` written out by hand is a number
    that can be mistyped and will not error. For nl=36 this returns
    range(14, 34), i.e. layers 14..33 inclusive.

    Note for the report: Claude models are far deeper, so the paper's
    55-point-wide band compresses into ~20 Qwen layers. Since ablation
    strength is DEFINED as band width, our granularity is much coarser than
    the paper's, and the band's top collides with the motor region.
    """
    if nl <= 0:
        raise ValueError(f"n_layers must be positive, got {nl}")
    if not 0.0 <= lo < hi <= 1.0:
        raise ValueError(f"need 0 <= lo < hi <= 1, got lo={lo} hi={hi}")
    return range(math.ceil(lo * nl), math.floor(hi * nl) + 1)


def resolve_band(nl: int, fractions=None, layers: str | None = None):
    """A band from either depth fractions or an explicit "LO-HI", as
    (range, name). One parser, because two would drift.

    `layers` wins when given. It exists because every entry in config.BANDS
    starts at depth 0.38, so widening a band also moves it later and band
    POSITION cannot be varied independently of band WIDTH -- which is the
    confound a fixed-width sliding window is meant to break. Windows built
    this way are EXPLORATORY and are not pre-registered bands; the name is
    marked `custom(...)` so a printout cannot be mistaken for one.
    """
    if layers:
        try:
            lo, hi = (int(x) for x in layers.split("-"))
        except ValueError:
            raise ValueError(f"--layers wants LO-HI, got {layers!r}") from None
        if not 0 <= lo <= hi < nl:
            raise ValueError(f"--layers needs 0 <= lo <= hi < {nl}, "
                             f"got {lo}-{hi}")
        return range(lo, hi + 1), f"custom({lo}-{hi})"
    return band_from_depth(nl, *fractions), "band"


def hook_census(model):
    """Forward-hook count per decoder layer.

    Not a leak test on its own: see contract 4, transformers permanently
    installs one hook per layer the first time you ask for hidden states.
    Compare deltas around a `with` block, not against zero.
    """
    return [len(l._forward_hooks) for l in decoder_layers(model)]


def final_norm(model):
    """The RMSNorm applied after the last decoder layer."""
    return model.model.norm


def _to_head(model, h: torch.Tensor) -> torch.Tensor:
    """Place activations on the unembedding's device.

    `Capture` detaches to CPU on purpose -- 36 layers x 1500 positions is
    276 MB and holding that on a 16 GB unified-memory machine alongside 8 GB
    of weights is not free. So every readout below is routinely handed a CPU
    tensor while the head sits on mps, and the mismatch raises from inside
    RMSNorm with a message that names neither the caller nor the capture.

    Doing the move here rather than at each call site because the call sites
    got it wrong twice in one session, and the correct behaviour is not in
    doubt: reading activations through the model's head means putting them
    where the head is. A caller that already moved them pays nothing.
    """
    dev = model.lm_head.weight.device
    return h if h.device == dev else h.to(dev)


def logit_lens(model, h: torch.Tensor) -> torch.Tensor:
    """lm_head(norm(h)) -- the logit lens, and the J-lens special case J = I.

    The norm is NOT optional. `lm_head(h)` without it silently produces
    garbage, worst in early layers where the approximation is already least
    reliable. Applied to the LAST layer's output this reproduces the model's
    own logits exactly, which is what makes it usable as an independent check
    on the capture mechanism.
    """
    return model.lm_head(final_norm(model)(_to_head(model, h)))


# ------------------------------------------------------------- hook handling

class LayerHooks:
    """Own hook handles and guarantee their removal.

    A leaked hook that fires during the next experiment is a silent-corruption
    bug of exactly the kind this project keeps producing, so removal happens
    in __exit__ and therefore also on exception.

    Subclass or pass `make_hook`; Half B's ablation is a different `make_hook`
    over this same handle management.
    """

    def __init__(self, model, layers, make_hook=None):
        self.model = model
        self.layers = list(layers)
        nl = n_layers(model)
        for i in self.layers:
            # negative indices work on a ModuleList and would silently hook
            # the wrong end of the network
            if not isinstance(i, (int,)) or isinstance(i, bool):
                raise TypeError(f"layer index must be int, got {i!r}")
            if not 0 <= i < nl:
                raise IndexError(f"layer {i} out of range for {nl} layers")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError(f"duplicate layer indices: {self.layers}")
        if make_hook is not None:
            self.make_hook = make_hook
        self._handles = []

    def make_hook(self, layer: int):
        raise NotImplementedError("provide make_hook or subclass")

    @property
    def active(self) -> bool:
        return bool(self._handles)

    def __enter__(self):
        mods = decoder_layers(self.model)
        try:
            for i in self.layers:
                self._handles.append(
                    mods[i].register_forward_hook(self.make_hook(i)))
        except Exception:
            self.remove()
            raise
        return self

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def __exit__(self, *exc):
        self.remove()
        return False


class Capture(LayerHooks):
    """Record residual-stream activations. Read-only.

    `store=False` records shapes only and discards the tensors -- that is the
    mode Half B wants for counting ablated positions (exposure), where keeping
    the activations would be ruinous and is not needed.

    Tensors are detached and moved to CPU immediately, and kept in their
    native dtype. Do NOT .float() them: that doubles memory and makes measured
    size disagree with the 2-bytes-per-element arithmetic you planned with.
    """

    def __init__(self, model, layers, store: bool = True):
        super().__init__(model, layers)
        self.store = store
        self.acts = {}     # layer -> [tensor, ...] cpu, native dtype
        self.shapes = {}   # layer -> [(b, seq, d), ...]  always recorded

    def make_hook(self, layer: int):
        def hook(module, args, output):
            if not torch.is_tensor(output):
                raise TypeError(
                    f"layer {layer} returned {type(output).__name__}, expected "
                    f"Tensor. transformers 5.x returns a bare tensor; if this "
                    f"fires you are on a version with a different contract.")
            if output.shape[0] != 1:
                raise ValueError(BATCH1_MSG.format(got=output.shape[0]))
            self.shapes.setdefault(layer, []).append(tuple(output.shape))
            if self.store:
                self.acts.setdefault(layer, []).append(
                    output.detach().to("cpu"))
        return hook

    # ---- accessors

    def seq_lens(self, layer: int):
        """Sequence length of every firing, in order.

        With a KV cache this is [n_prompt, 1, 1, ...]: one prefill pass over
        the whole prompt, then one single-token pass per subsequent token.
        n_firings == n_new_tokens, because the prefill pass is what produces
        the first token.
        """
        return [s[1] for s in self.shapes.get(layer, [])]

    def n_firings(self, layer: int) -> int:
        return len(self.shapes.get(layer, []))

    def first(self, layer: int) -> torch.Tensor:
        """The prefill capture -- where the direct condition's compute lives."""
        return self.acts[layer][0]

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for ts in self.acts.values() for t in ts)

    def positions(self, layer: int) -> int:
        """Total token positions seen -- the exposure count for Half B."""
        return sum(self.seq_lens(layer))


# ============================================================ INTERVENTION
# Milestone 6 onward. The signature is fixed by the paper's ablation recipe
# (transformer-circuits.pub/2026/workspace, "J-space ablation leaves most
# capabilities intact"):
#
#   at each token position, across a band of layers, take the k=10 most
#   strongly activated J-lens vectors and zero the residual stream's
#   projection onto each -- EXCEPT that tokens appearing in the top-10 of a
#   clean forward pass are not ablated, so the intervention targets internal
#   reasoning rather than the ability to report.
#
# Two consequences for this interface:
#
#   1. The exclusion set lives in VOCABULARY space, not position space. It is
#      a per-position set of TOKEN IDS. A position mask cannot express it, so
#      `exclude` and `positions` are separate arguments and mean different
#      things.
#   2. Producing that set needs a paired un-ablated forward pass on the
#      CURRENT context, because ablating inside the band changes the final
#      logits. It is well defined however far the ablated run has drifted
#      from the clean one, but it costs a second forward pass per step.
#
# `positions` earns its place for scope control and for exposure counting
# (the difficulty-length confound: harder problems produce longer CoT and so
# more ablated positions; effect must be reported against exposure).

from dataclasses import dataclass, field


@dataclass
class Firing:
    """One hook activation. `pos_start`/`pos_end` are ABSOLUTE positions in
    the sequence, which with a KV cache the tensor shape does not tell you:
    during generation h is (1, 1, d) but that token is at position
    n_prompt + step."""
    layer: int
    call_idx: int
    is_prefill: bool
    pos_start: int
    pos_end: int
    exclude: dict = field(default_factory=dict)   # abs position -> token ids

    @property
    def n_pos(self) -> int:
        return self.pos_end - self.pos_start


class Intervene(LayerHooks):
    """Modify the residual stream. Reuses LayerHooks' handle management.

    fn(h, firing) -> Tensor. Returning a Tensor replaces the layer output
    (contract 3). Returning None leaves it untouched.

    scope     "prefill" | "generation" | "both". The direct condition is ~97%
              prefill, so an ablation that fires only during generation is
              barely an ablation and nothing errors -- this is the landmine
              from the handoff document, made switchable so it can be
              measured rather than assumed.
    positions None = every position, or a set of ABSOLUTE positions.
    exclude   dict {abs position -> iterable of token ids} handed to fn. Not
              interpreted here; the ablation in Milestone 7 consumes it.
    """

    def __init__(self, model, layers, fn, scope="both", positions=None,
                 exclude=None):
        super().__init__(model, layers)
        if scope not in ("prefill", "generation", "both"):
            raise ValueError(f"scope must be prefill/generation/both, got {scope!r}")
        self.fn = fn
        self.scope = scope
        self.positions = None if positions is None else set(positions)
        self.exclude = exclude or {}
        self.log = []          # every Firing, in order
        self.n_modified = 0    # exposure: positions actually changed
        # Suppress the intervention WITHOUT disturbing position accounting.
        # The paper's exclusion rule needs a clean forward pass on the ablated
        # run's own context at every step, and that pass goes through the same
        # model while these hooks are registered. Tearing the hooks down and
        # rebuilding them per step would reset _calls/_offset and every firing
        # would look like a fresh prefill at position 0.
        self.paused = False
        self._calls = {}       # layer -> call index
        self._offset = {}      # layer -> next absolute position

    def make_hook(self, layer: int):
        def hook(module, args, output):
            if not torch.is_tensor(output):
                raise TypeError(f"layer {layer} returned {type(output).__name__}")
            if output.shape[0] != 1:
                raise ValueError(BATCH1_MSG.format(got=output.shape[0]))
            if self.paused:
                return None          # before the counters, deliberately

            idx = self._calls.get(layer, 0)
            start = self._offset.get(layer, 0)
            # Position accounting assumes prefill-then-one-token-per-step. If
            # use_cache is off, or generation re-prefills, every firing sees
            # the whole sequence and `start` runs past the true length -- and
            # because `exclude` is keyed by ABSOLUTE position, every lookup
            # below would then silently miss. This class needs the guard more
            # than its predecessor did, which is where it comes from.
            if idx > 0 and output.shape[1] > 1:
                raise RuntimeError(NOCACHE_MSG.format(
                    layer=layer, seq=output.shape[1]))
            end = start + output.shape[1]
            self._calls[layer] = idx + 1
            self._offset[layer] = end

            # call index, not seq_len: a one-token prompt would make a
            # shape-based test ambiguous
            firing = Firing(layer=layer, call_idx=idx, is_prefill=(idx == 0),
                            pos_start=start, pos_end=end,
                            exclude={p: self.exclude[p]
                                     for p in range(start, end)
                                     if p in self.exclude})
            self.log.append(firing)

            in_scope = (self.scope == "both"
                        or (self.scope == "prefill") == firing.is_prefill)
            if not in_scope:
                return None

            if self.positions is None:
                mask = None
                n_hit = firing.n_pos
            else:
                hits = [p - start for p in range(start, end)
                        if p in self.positions]
                if not hits:
                    return None
                mask = torch.zeros(output.shape[1], dtype=torch.bool,
                                   device=output.device)
                mask[hits] = True
                n_hit = len(hits)

            new = self.fn(output, firing)
            if new is None:
                return None
            if new.shape != output.shape:
                raise ValueError(f"intervention changed shape "
                                 f"{tuple(output.shape)} -> {tuple(new.shape)}")
            self.n_modified += n_hit
            if mask is None:
                return new
            return torch.where(mask[None, :, None], new, output)
        return hook

    def firings(self, layer=None):
        return [f for f in self.log if layer is None or f.layer == layer]


# ------------------------------------------------------------ interventions

def add_noise(alpha: float, seed: int = 0):
    """Gaussian noise scaled to each position's residual norm.

    Scaled, not absolute: Milestone 5 measured mean residual norm rising from
    ~9 at L00 to ~545 at L31, a 60x range. Absolute noise would be negligible
    deep and catastrophic shallow, and the dose-response curve would be
    uninterpretable. alpha is the fraction of ||h|| added, per position.

    Generated on CPU in float32 from a seeded Generator, then moved. Costs a
    transfer; buys identical noise on MPS and CUDA, without which the
    random-direction control cannot be reproduced across the platform change
    between Milestone 6 and Milestone 8.
    """
    def fn(h, firing):
        if alpha == 0.0:
            return None
        g = torch.Generator().manual_seed(
            (seed * 1_000_003 + firing.call_idx * 1009 + firing.layer) % (2**31))
        n = torch.randn(tuple(h.shape), generator=g, dtype=torch.float32)
        n = n.to(device=h.device, dtype=h.dtype)
        scale = alpha * h.float().norm(dim=-1, keepdim=True) / math.sqrt(h.shape[-1])
        return h + n * scale.to(h.dtype)
    return fn


def project_out(h: torch.Tensor, V: torch.Tensor, mode: str = "span",
                tol: float = 1e-6) -> torch.Tensor:
    """Remove directions V from h.  h: (..., d)   V: (m, d)

    THE TWO MODES ARE NOT EQUIVALENT and the paper's wording does not settle
    which it used. It says to zero the projection onto EACH of the top-k
    vectors, and separately states that J-lens vectors are overcomplete and
    non-orthogonal. Sequentially projecting out ten non-orthogonal vectors is
    order-dependent and leaves part of their span behind; projecting out the
    span removes strictly more. Pre-register the choice.

      "each"  literal reading: subtract each direction in turn
      "span"  remove the whole subspace (SVD, rank-tolerant)
    """
    if V.ndim != 2 or V.shape[-1] != h.shape[-1]:
        raise ValueError(f"V must be (m, {h.shape[-1]}), got {tuple(V.shape)}")
    dt = h.dtype
    if mode == "each":
        out = h.float()
        for v in V.float():
            nv = v.norm()
            if nv < tol:
                continue
            u = v / nv
            out = out - (out @ u).unsqueeze(-1) * u
        return out.to(dt)
    if mode == "span":
        try:
            U, S, _ = torch.linalg.svd(V.float().T, full_matrices=False)
        except (RuntimeError, NotImplementedError):
            # MPS lacks some linalg kernels. V is (k, d_model) with k~10, so
            # the round trip is cheap and the alternative is a hard failure
            # deep inside a forward hook.
            U, S, _ = torch.linalg.svd(V.float().T.cpu(), full_matrices=False)
            U, S = U.to(h.device), S.to(h.device)
        if S.numel() == 0 or float(S[0]) < tol:
            return h
        r = int((S > tol * S[0]).sum())
        Q = U[:, :r]
        return (h.float() - (h.float() @ Q) @ Q.T).to(dt)
    raise ValueError(f"mode must be 'each' or 'span', got {mode!r}")


# ==================================================== LOGIT-LENS J-SPACE READ
# The J-lens vectors at layer l are the rows of W_U J_l, each tied to one
# vocabulary token. Under the feasibility approximation J_l = I they are the
# rows of the unembedding -- but NOT naively, because of the norm.
#
# The readout is  lens(h) = softmax(W_U norm(h)), and Qwen3RMSNorm carries a
# LEARNED per-dimension gain:  norm(h) = g * h / rms(h),  with
# g = model.model.norm.weight. Therefore
#
#     <W_U[t], norm(h)>  =  <W_U[t] * g, h> / rms(h)
#
# We apply the FULL norm rather than just the gain. An earlier version dropped
# the 1/rms(h) on the grounds that a positive per-position scalar cannot change
# a ranking. True in real arithmetic; false in bf16. Scaling h before the
# matmul changes the rounding, and at 151,936 tokens enough near-ties sit at
# the rank-10 boundary that the calibration measured only 94.1% top-10 set
# agreement with the model (top-1 was 100%, which is what rules out a gain
# bug). Matching the model's arithmetic exactly costs nothing and is exact.
#
# The gain itself is not optional either: dropping it selects a DIFFERENT ten
# directions and fails silently. `test_readout_topk_matches_model_logits`
# covers both.

def readout_gain(model):
    """The learned RMSNorm gain, in residual-stream coordinates."""
    return final_norm(model).weight


def readout_scores(model, h, chunk: int = 128):
    """Gain-aware logit-lens scores for every vocabulary token.

    Scales h rather than W_U: materialising W_U * g would be ~778 MB in bf16
    for Qwen3-4B, and h is one vector per position. Chunked over positions
    because the result is (n_pos, n_vocab) -- at 1385 CoT positions that is
    ~420 MB, so callers wanting only the top-k should use `topk_tokens`.
    """
    W = model.lm_head.weight
    h = _to_head(model, h)
    flat = final_norm(model)(h).reshape(-1, h.shape[-1]).to(W.dtype)
    parts = [flat[i:i + chunk] @ W.T for i in range(0, flat.shape[0], chunk)]
    return torch.cat(parts, 0).reshape(*h.shape[:-1], W.shape[0])


def topk_tokens(model, h, k: int, exclude=None, chunk: int = 128):
    """Top-k token ids per position, never materialising the full score array.

    `exclude` is keyed by LOCAL position index within `h`, not by absolute
    sequence position -- a hook sees (1, 1, d) during generation and the
    caller holds `firing.pos_start` needed to convert. Excluded tokens are
    masked to -inf before the top-k, so k survivors are always returned.

    Returns (n_pos, k) on h's device.
    """
    W = model.lm_head.weight
    h = _to_head(model, h)
    flat = final_norm(model)(h).reshape(-1, h.shape[-1]).to(W.dtype)
    out = []
    for i in range(0, flat.shape[0], chunk):
        s = (flat[i:i + chunk] @ W.T).float()
        if exclude:
            for j in range(s.shape[0]):
                ex = exclude.get(i + j)
                if ex is not None and len(ex):
                    s[j, torch.as_tensor(list(ex), device=s.device)] = -math.inf
        out.append(s.topk(k, dim=-1).indices)
    return torch.cat(out, 0)


def directions_for(model, token_ids, gain_scaled: bool):
    """J-lens directions for the given tokens, as (m, d_model).

    `gain_scaled` is a PRE-REGISTRATION CHOICE (config.PROJECT_GAIN_SCALED),
    not a convenience flag, and it is a different question from selection:

      True   remove W_U[t] * g, the direction whose inner product with the
             raw residual stream IS the readout score. Zeroing it zeroes the
             token's readout contribution.
      False  remove W_U[t], the paper's J-lens vector as literally defined
             (a row of W_U J_l, with the norm sitting between it and h).

    They differ whenever the gain is not uniform, which for a trained model it
    is not. There is no default here on purpose.
    """
    V = model.lm_head.weight[torch.as_tensor(token_ids)].detach()
    return V * readout_gain(model).detach() if gain_scaled else V


def generate_ablated(model, input_ids, layers, fn, max_new_tokens,
                     exclude_topk=None, eos_token_id=None):
    """Greedy decode under ablation, honouring the paper's exclusion rule.

    Returns (sequence, n_new, intervene). `intervene.n_modified` is the
    exposure count the difficulty-length confound has to be reported against.

    WHY THIS EXISTS RATHER THAN model.generate INSIDE AN Intervene BLOCK. The
    exclusion rule exempts, at each position, the J-lens vectors of the tokens
    in the top-10 of a CLEAN forward pass on the CURRENT context. Inside a
    generation loop that context is the ablated run's own output, so the clean
    pass cannot be precomputed -- it has to interleave, one clean step per
    ablated step, each with its own KV cache over the same token sequence.
    `model.generate` gives no hook into the loop where that could happen.

    Two caches, ~144 KB/token each (measured in M5), so a 3072-token CoT holds
    about 885 MB of cache. That is the whole cost of the rule, and it is why
    dropping it was tempting -- see config.USE_EXCLUSION for the measurements
    that show dropping it changes what the ablation does.

    `exclude_topk=None` skips the clean pass entirely and runs single-pass.
    That is not a speed knob: it is a different experiment, and the config
    constant is what decides it.

    The clean pass runs with `intervene.paused = True` rather than by removing
    the hooks, so the ablated run's absolute-position accounting is untouched.
    Removing and re-registering per step would restart it at position 0 and
    every firing would claim to be a prefill.
    """
    if input_ids.shape[0] != 1:
        raise ValueError(BATCH1_MSG.format(got=input_ids.shape[0]))
    if max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be >= 1, got {max_new_tokens}")

    eos = eos_token_id
    if eos is None:
        eos = getattr(model.generation_config, "eos_token_id", None)
    stop = set()
    if eos is not None:
        stop = {int(e) for e in (eos if isinstance(eos, (list, tuple))
                                 else [eos])}

    iv = Intervene(model, list(layers), fn=fn, scope="both", exclude={})
    new, clean_cache, abl_cache = [], None, None
    step, pos = input_ids, 0
    with iv, torch.no_grad():
        while True:
            if exclude_topk:
                iv.paused = True
                oc = model(input_ids=step, past_key_values=clean_cache,
                           use_cache=True)
                iv.paused = False
                clean_cache = oc.past_key_values
                top = oc.logits[0].topk(exclude_topk, -1).indices
                # Only the current firing's positions are ever looked up, so
                # clearing keeps this O(step) instead of O(sequence) for a
                # 3072-token trace.
                iv.exclude.clear()
                for j in range(step.shape[1]):
                    iv.exclude[pos + j] = top[j].tolist()

            oa = model(input_ids=step, past_key_values=abl_cache,
                       use_cache=True)
            abl_cache = oa.past_key_values
            pos += step.shape[1]
            nxt = int(oa.logits[0, -1].argmax())
            new.append(nxt)
            if nxt in stop or len(new) >= max_new_tokens:
                break
            step = torch.tensor([[nxt]], device=input_ids.device,
                                dtype=input_ids.dtype)

    seq = torch.cat([input_ids, torch.tensor([new], device=input_ids.device,
                                             dtype=input_ids.dtype)], 1)
    return seq, len(new), iv


def make_ablation(model, k: int, mode: str, gain_scaled: bool,
                  kind: str = "ablate", seed: int = 0, track=None):
    """The intervention itself, as an `Intervene` fn -- ONE definition.

    Milestone 7's calibration, the automatic-task damage floor and the
    Milestone 8 run loop all apply the same operation, and the same reasoning
    that put the module path in `decoder_layers` and the prompt in
    `render_prompt` applies here with more force: an ablation written twice
    can drift, and the drift would be between the measurement and its own
    control.

    kind        "ablate"      top-k J-lens directions by inner product
                "rand_tok"    k random unembedding rows -- the paper's
                              control: same operation, same k, same layers,
                              only the SELECTION randomised
                "rand_gauss"  k isotropic directions -- does removing ANY k
                              dimensions matter?
    mode        project_out mode, config.PROJECTION_MODE
    gain_scaled config.PROJECT_GAIN_SCALED. No default: both are
                pre-registration choices, not conveniences.
    track       optional dict; per-position ||dh||/||h|| is appended under
                `kind` for the displacement tables.

    The exclusion set is read from `firing.exclude`, which `Intervene`
    populates and slices, so it must be supplied there as
    {absolute position: token ids} -- vocabulary space, not position space.
    Keys are converted to firing-local indices here because a hook sees
    (1, 1, d) during generation and `topk_tokens` indexes locally.
    """
    if kind not in ("ablate", "rand_tok", "rand_gauss"):
        raise ValueError(f"kind must be ablate/rand_tok/rand_gauss, got {kind!r}")

    def fn(h, firing):
        sel = None
        if kind == "ablate":
            loc = {p - firing.pos_start: ids
                   for p, ids in firing.exclude.items()}
            with torch.no_grad():
                sel = topk_tokens(model, h, k, exclude=loc)
        out = h.clone()
        for q in range(h.shape[1]):
            if sel is not None:
                V = directions_for(model, sel[q], gain_scaled=gain_scaled)
            else:
                V = random_directions(
                    model, k,
                    seed=seed * 1_000_003 + firing.layer * 1009
                         + firing.pos_start + q,
                    mode="tokens" if kind == "rand_tok" else "gaussian",
                    gain_scaled=gain_scaled)
            hq = h[0, q].float()
            new = project_out(hq, V.float(), mode=mode)
            if track is not None:
                track.setdefault(kind, []).append(
                    float((hq - new).norm() / hq.norm()))
            out[0, q] = new.to(h.dtype)
        return out

    return fn


def random_directions(model, k: int, seed: int, mode: str = "tokens",
                      gain_scaled: bool = False):
    """The paper's random-direction control: SAME operation, same k, same
    layers, only the SELECTION randomised.

    This is not additive noise, and the difference is not cosmetic. Noise
    matched on total displacement spreads over d_model dimensions while the
    ablation removes k of them, so the diffuse version touches more
    output-relevant directions and flips MORE often. Measured on Qwen3-4B at
    matched norm: noise 46% against the ablation's 30%. That comparison
    measures the gap between two operations, not the role of the J-space.

    mode="tokens"    k random unembedding rows. Holds the direction
                     distribution fixed and varies only WHICH tokens, so it
                     isolates the J-space selection. This is the control that
                     tests the hypothesis.
    mode="gaussian"  k isotropic directions -- does removing ANY k dimensions
                     matter? Scale is irrelevant, project_out normalises.
    """
    g = torch.Generator().manual_seed(int(seed) % (2 ** 31))
    W = model.lm_head.weight
    if mode == "tokens":
        ids = torch.randint(0, W.shape[0], (k,), generator=g)
        return directions_for(model, ids, gain_scaled=gain_scaled)
    if mode == "gaussian":
        return torch.randn(k, W.shape[1], generator=g).to(
            device=W.device, dtype=W.dtype)
    raise ValueError(f"mode must be 'tokens' or 'gaussian', got {mode!r}")
