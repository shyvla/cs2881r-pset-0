"""Capture probe -- read one layer's activation and PROVE it is the right tensor.

    python -m probes.capture          # real Qwen3-4B, best available device
    python -m probes.capture --tiny   # weightless smoke test, seconds, no GPU

No intervention anywhere in this file. Every check is falsifiable: it can come
back red, and if it does, the reason is named.

`--tiny` builds a randomly-initialised Qwen3 from config. The numbers are
meaningless but the CONTRACT is identical, so it verifies the probe itself
runs before you spend model-loading time on it.
"""
import argparse
import sys

import torch

import config
from hooks import (Capture, band_from_depth, decoder_layers, final_norm,
                   hook_census, logit_lens, n_layers)

from loaders import MODEL, load_real, load_tiny, pick_device
# Must match runs/archive/gsm8k_manifest.json -> conditions.direct_intact, or the
# probe is measuring a condition that never ran. That is now enforced by
# construction rather than by a copied string: config owns the one definition,
# and DIRECT_FINGERPRINT below is the assertion that it still hashes to what
# the committed manifest recorded.
DIRECT_SUFFIX, DIRECT_PREFILL = config.direct_prompt("gsm8k")
DIRECT_FINGERPRINT = "683d8ea5f9e42c80"
QUESTION = ("Kelly has 5 quarters and 2 dimes. If she buys a can of pop for "
            "55 cents, how many cents will she have left?")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f"\n          {detail}" if detail and not ok else ""))
    return ok


# Loading comes from loaders, and no longer from a copy here. The copy loaded
# the checkpoint by BRANCH while loaders pins a revision, so this probe could
# assert a fingerprint against weights the run had never seen -- and its tiny
# model left model.norm.weight at ones, where loaders gives it a real gain. A
# probe that measures a different model than the run cannot calibrate it.


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--long-tokens", type=int, default=1500)
    a = ap.parse_args(argv)
    device = pick_device(a.device, a.tiny)

    model, tok = (load_tiny if a.tiny else load_real)(device)
    NL = n_layers(model)
    D = model.config.hidden_size
    band = band_from_depth(NL)
    print(f"model     : {'TINY (random weights)' if a.tiny else MODEL}")
    print(f"layers={NL}  d_model={D}  device={device}")
    print(f"band      : layers {band.start}..{band.stop - 1} "
          f"(derived from 0.38-0.92 depth, not hand-typed)\n")

    # ---- prompt: the direct condition. 0.7 s per run, and its prefill /
    # generation split is maximally lopsided -- the property Half B needs.
    if a.tiny:
        enc = {"input_ids": torch.randint(0, 256, (1, 60)).to(device)}
    else:
        from scoring import prompt_fingerprint, render_prompt
        fp = prompt_fingerprint(tok, thinking=False, suffix=DIRECT_SUFFIX,
                                prefill=DIRECT_PREFILL)
        check("prompt matches the direct_intact condition in the manifest",
              fp == DIRECT_FINGERPRINT,
              f"got {fp}, manifest {DIRECT_FINGERPRINT}"
              if fp != DIRECT_FINGERPRINT else "")
        text = render_prompt(tok, QUESTION, thinking=False,
                             suffix=DIRECT_SUFFIX, prefill=DIRECT_PREFILL)
        enc = tok(text, return_tensors="pt").to(device)
    n_prompt = enc["input_ids"].shape[1]
    print(f"n_prompt  : {n_prompt}\n")

    probe = band[len(band) // 2]

    # =============================================== 1. shape, dtype, device
    print("1. the captured tensor")
    cap = Capture(model, [probe])
    with cap, torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    got = cap.first(probe)
    check(f"layer {probe} capture is (1, {n_prompt}, {D})",
          tuple(got.shape) == (1, n_prompt, D), f"got {tuple(got.shape)}")
    check("dtype preserved (no silent float32 upcast)",
          got.dtype == next(model.parameters()).dtype,
          f"{got.dtype} vs model {next(model.parameters()).dtype}")
    check(f"len(hidden_states) == {NL + 1}", len(out.hidden_states) == NL + 1,
          f"got {len(out.hidden_states)}")

    # ================================= 2. equality across the whole band
    print(f"\n2. hook == hidden_states[i+1] across the band")
    cap = Capture(model, band)
    with cap, torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    bad = [i for i in band
           if not torch.equal(cap.first(i), out.hidden_states[i + 1].cpu())]
    check(f"all {len(band)} band layers match their reference", not bad,
          f"mismatched: {bad}")

    # ================================= 3. the last layer is the exception
    print("\n3. the last layer, where the doc was wrong")
    cap = Capture(model, [NL - 1])
    with cap, torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    raw = cap.first(NL - 1)
    ref = out.hidden_states[-1].cpu()
    check("hidden_states[-1] is NOT the raw last-layer output",
          not torch.equal(ref, raw))
    check("hidden_states[-1] IS norm(raw)",
          torch.allclose(final_norm(model)(raw.to(device)).cpu().float(),
                         ref.float(), atol=1e-2))

    # ================================= 4. independent oracle
    print("\n4. independent oracle (routes through lm_head, not the capture)")
    with torch.no_grad():
        recon = logit_lens(model, raw.to(device)).cpu().float()
        no_norm = model.lm_head(raw.to(device)).cpu().float()
    check("lm_head(norm(h_last)) reproduces the model's logits",
          torch.allclose(recon, out.logits.cpu().float(), atol=5e-2),
          f"max abs diff "
          f"{float((recon - out.logits.cpu().float()).abs().max()):.3e}")
    check("dropping the norm does NOT reproduce them (it fails silently)",
          not torch.allclose(no_norm, out.logits.cpu().float(), atol=5e-2))

    # ================================= 5. norm growth with depth
    print("\n5. residual-stream norm by depth")
    cap = Capture(model, range(NL))
    with cap, torch.no_grad():
        model(**enc)
    norms = [cap.first(i).float().norm(dim=-1).mean().item() for i in range(NL)]
    for i in range(0, NL, 9):
        print(f"          L{i:02d}={norms[i]:9.2f}   L{min(i + 4, NL - 1):02d}="
              f"{norms[min(i + 4, NL - 1)]:9.2f}")
    check("later layers have larger activations than early ones",
          norms[-1] > norms[0], f"L00={norms[0]:.2f} L{NL-1:02d}={norms[-1]:.2f}")

    # ================================= 6. THE LANDMINE: prefill vs generation
    print("\n6. where does the direct condition's computation live?")
    cap = Capture(model, [probe], store=False)
    with cap, torch.no_grad():
        gen = model.generate(**enc, max_new_tokens=4 if a.tiny else 32)
    n_new = gen.shape[1] - n_prompt
    lens = cap.seq_lens(probe)
    print(f"          new tokens={n_new}  firings={len(lens)}  seq_lens={lens}")
    check("n_firings == n_new_tokens (prefill produces the first token)",
          cap.n_firings(probe) == n_new, f"{cap.n_firings(probe)} vs {n_new}")
    check("first firing is the whole prompt, the rest are single tokens",
          lens[0] == n_prompt and all(s == 1 for s in lens[1:]))
    share = lens[0] / sum(lens)
    print(f"          prefill {lens[0]} of {sum(lens)} ablatable positions "
          f"= {share:.1%}")
    check("prefill dominates the ablatable positions (>90%)", share > 0.90,
          f"only {share:.1%} -- if this is low the direct condition's "
          f"computation is not where Half B assumes it is")
    if not a.tiny:
        print(f"          completion: "
              f"{tok.decode(gen[0, n_prompt:], skip_special_tokens=True)!r}")

    # ================================= 7. memory
    print(f"\n7. memory: one forward pass over {a.long_tokens} tokens, all layers")
    long_ids = (enc["input_ids"][:, :1].repeat(1, a.long_tokens)
                if a.tiny else
                tok(QUESTION * 200, return_tensors="pt").input_ids[:, :a.long_tokens].to(device))
    cap = Capture(model, range(NL))
    with cap, torch.no_grad():
        model(input_ids=long_ids)
    esz = cap.first(0).element_size()
    predicted = NL * long_ids.shape[1] * D * esz
    print(f"          measured {cap.nbytes / 1e6:7.1f} MB   "
          f"predicted {predicted / 1e6:7.1f} MB   "
          f"({NL} x {long_ids.shape[1]} x {D} x {esz}B)")
    check("measured capture size matches the arithmetic",
          cap.nbytes == predicted)
    cap_one = Capture(model, [probe])
    with cap_one, torch.no_grad():
        model(input_ids=long_ids)
    print(f"          one layer only: {cap_one.nbytes / 1e6:.1f} MB "
          f"({NL}x less -- hook only the band you need)")

    # ---- leak check, and the census that shows why zero is the wrong test
    print(f"\n          hook census now: {set(hook_census(model))} per layer "
          f"(transformers installed its own; see contract 4)")
    check("no hooks of ours left behind",
          all(c <= 1 for c in hook_census(model)),
          f"census {hook_census(model)}")

    # ---- summary
    npass = sum(ok for _, ok in RESULTS)
    print(f"\n{npass}/{len(RESULTS)} checks passed")
    if npass != len(RESULTS):
        print("Capture is NOT verified. Do not run any intervention -- an "
              "intervention on a tensor you have not verified is the exact "
              "failure this probe exists to prevent.")
    return 0 if npass == len(RESULTS) else 1


if __name__ == "__main__":
    sys.exit(main())
