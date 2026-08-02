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
    "decoder_layers", "n_layers", "band_from_depth", "final_norm",
    "logit_lens", "LayerHooks", "Capture", "BATCH1_MSG",
]

BATCH1_MSG = (
    "hook saw batch size {got}, expected 1. This code asserts batch-1 "
    "deliberately; see the BATCH SIZE note in hooks.py before relaxing it."
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


def logit_lens(model, h: torch.Tensor) -> torch.Tensor:
    """lm_head(norm(h)) -- the logit lens, and the J-lens special case J = I.

    The norm is NOT optional. `lm_head(h)` without it silently produces
    garbage, worst in early layers where the approximation is already least
    reliable. Applied to the LAST layer's output this reproduces the model's
    own logits exactly, which is what makes it usable as an independent check
    on the capture mechanism.
    """
    return model.lm_head(final_norm(model)(h))


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
