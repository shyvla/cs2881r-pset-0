"""Milestone 6 -- trivial intervention, and prove it lands where you think.

    python m6_probe.py                # real Qwen3-4B on mps
    python m6_probe.py --tiny         # weightless smoke test, no GPU

"Output changed" is NOT the deliverable. Any perturbation anywhere changes
the output. Section 11 of the handoff document says to expect the ablation to
fire on the wrong tensor while everything appears to work, so every check
below is one a MISPLACED intervention would fail.

Bands (derived, never hand-typed):
  LIGHT  0.38-0.54 -> layers 14-19. The paper's coherence-preserving choice
         (it ablates L38-54 for the experiential-report experiments and notes
         that later ranges impair coherence).
  FULL   0.38-0.92 -> layers 14-33. The whole workspace range, and the top of
         it collides with the motor region.
"""
import argparse
import math
import sys

import torch

import config
from hooks import (Capture, Intervene, add_noise, band_from_depth,
                   hook_census, n_layers)

from loaders import MODEL
# The one definition, from config. GSM8K-only probe, so it names the dataset.
DIRECT_SUFFIX, DIRECT_PREFILL = config.direct_prompt("gsm8k")
# GSM8K test id 733: direct-correct AND cot-correct in 630 tokens, nothink in
# 99. Chosen so degradation has somewhere to go -- id 286, the M5 probe
# question, is already wrong in both conditions.
Q733_FALLBACK = ("Greg found $20 while walking down the street with his 3 "
                 "younger siblings. He wants to split the money equally "
                 "among them. How much money did each of them get?")
Q733_GOLD = "5"

RESULTS = []


def check(name, ok, detail="", gating=True):
    """gating=False: calibration, reported but does not block Milestone 7."""
    RESULTS.append((name, bool(ok), gating))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}"
          + (f"\n          {detail}" if detail and not ok else ""))
    return ok


def kl(p_logits, q_logits):
    p = torch.log_softmax(p_logits.float(), -1)
    q = torch.log_softmax(q_logits.float(), -1)
    return float((p.exp() * (p - q)).sum())


def load_real(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16).to(device).eval()
    gc = model.generation_config
    gc.do_sample = False
    gc.temperature = gc.top_p = gc.top_k = None   # match the M4 manifest
    return model, tok


def load_tiny(device):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=36, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=4096, pad_token_id=0)
    m = Qwen3ForCausalLM(cfg).to(device).eval()
    m.generation_config.do_sample = False
    return m, None


def get_question(tok, tiny):
    if tiny:
        return Q733_FALLBACK
    from datasets import load_dataset
    for path in ("openai/gsm8k", "gsm8k"):
        try:
            return load_dataset(path, "main", split="test")[733]["question"]
        except Exception as e:
            last = e
    print(f"  (could not load GSM8K: {type(last).__name__}; using fallback "
          f"wording for id 733 -- NOT byte-identical to the M4 run)")
    return Q733_FALLBACK


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    device = a.device or ("cpu" if a.tiny else "mps")

    model, tok = (load_tiny if a.tiny else load_real)(device)
    NL = n_layers(model)
    LIGHT = band_from_depth(NL, 0.38, 0.54)
    FULL = band_from_depth(NL, 0.38, 0.92)
    print(f"model : {'TINY (random weights)' if a.tiny else MODEL}  "
          f"layers={NL} device={device}")
    print(f"LIGHT : {LIGHT.start}-{LIGHT.stop - 1}    "
          f"FULL : {FULL.start}-{FULL.stop - 1}\n")

    question = get_question(tok, a.tiny)
    if a.tiny:
        enc = {"input_ids": torch.randint(0, 256, (1, 60)).to(device)}
    else:
        from scoring import render_prompt
        text = render_prompt(tok, question, thinking=False,
                             suffix=DIRECT_SUFFIX, prefill=DIRECT_PREFILL)
        enc = tok(text, return_tensors="pt").to(device)
    n_prompt = enc["input_ids"].shape[1]

    with torch.no_grad():
        clean = model(**enc).logits.clone()
        clean_gen = model.generate(**enc, max_new_tokens=32 if not a.tiny else 4)
    print(f"n_prompt={n_prompt}  clean completion: "
          f"{tok.decode(clean_gen[0, n_prompt:], skip_special_tokens=True)!r}"
          if not a.tiny else f"n_prompt={n_prompt}")

    # ============================================ A: the plumbing is lossless
    print("\nA. lossless plumbing")
    with Intervene(model, list(FULL), fn=lambda h, f: None), torch.no_grad():
        null = model(**enc).logits
    check("a no-op hook is bitwise identical to no hook",
          torch.equal(clean, null))
    with Intervene(model, list(FULL), fn=add_noise(0.0)), torch.no_grad():
        zero = model(**enc).logits
    check("alpha=0 through the real code path is bitwise identical",
          torch.equal(clean, zero))
    outs = []
    for seed in (0, 0, 1):
        with Intervene(model, list(LIGHT), fn=add_noise(0.05, seed=seed)), \
                torch.no_grad():
            outs.append(model(**enc).logits.clone())
    check("same seed reproduces, different seed does not",
          torch.equal(outs[0], outs[1]) and not torch.equal(outs[0], outs[2]))

    # ============================================ B: it lands where you think
    print("\nB. correct target")
    i = LIGHT.start
    with torch.no_grad():
        model(**enc, output_hidden_states=True)          # install capture hooks
        base_hs = model(**enc, output_hidden_states=True)
    with Intervene(model, [i], fn=add_noise(0.5)), torch.no_grad():
        pert = model(**enc, output_hidden_states=True)
    check(f"hidden_states[{i+1}] unchanged (recorded pre-intervention)",
          torch.equal(pert.hidden_states[i + 1], base_hs.hidden_states[i + 1]))
    check(f"hidden_states[{i+2}] changed (intervention propagated)",
          not torch.equal(pert.hidden_states[i + 2], base_hs.hidden_states[i + 2]))

    p = n_prompt // 2
    iv = Intervene(model, [i], fn=add_noise(0.5), positions={p})
    with iv, torch.no_grad():
        one = model(**enc).logits
    check(f"perturbing position {p} leaves logits before it untouched",
          torch.equal(one[:, :p], clean[:, :p]))
    check("and moves logits from that position on",
          not torch.equal(one[:, p:], clean[:, p:]))
    check("exposure counted 1 position", iv.n_modified == 1, str(iv.n_modified))

    # ============================================ C: THE LANDMINE, behavioural
    print("\nC. prefill vs generation (M5 said 97% of positions are prefill)")
    res = {}
    for scope in ("prefill", "generation", "both"):
        iv = Intervene(model, list(FULL), fn=add_noise(0.3), scope=scope)
        with iv, torch.no_grad():
            g = model.generate(**enc, max_new_tokens=32 if not a.tiny else 4)
        res[scope] = (g, iv.n_modified)
        txt = ("" if a.tiny else
               f"  {tok.decode(g[0, n_prompt:], skip_special_tokens=True)!r}")
        print(f"          {scope:<11} positions modified={iv.n_modified:<6}{txt}")
    first = n_prompt
    check("generation-only cannot change the first new token",
          res["generation"][0][0, first] == clean_gen[0, first])
    check("prefill-only does change it",
          res["prefill"][0][0, first] != clean_gen[0, first])
    check("prefill-only modifies far more positions than generation-only",
          res["prefill"][1] > 10 * res["generation"][1],
          f"{res['prefill'][1]} vs {res['generation'][1]}")

    # ============================================ D: dose-response
    print("\nD. dose-response in the LIGHT band")
    print(f"          {'alpha':>8}{'KL(clean||noised)':>20}{'top1 kept':>11}"
          f"{'  completion' if not a.tiny else ''}")
    moved_logits = moved_text = None
    for alpha in (0.0, 1e-3, 4e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0):
        with Intervene(model, list(LIGHT), fn=add_noise(alpha)), torch.no_grad():
            lg = model(**enc).logits
            g = model.generate(**enc, max_new_tokens=32 if not a.tiny else 4)
        d = kl(clean[0, -1], lg[0, -1])
        same = bool(lg[0, -1].argmax() == clean[0, -1].argmax())
        txt = ("" if a.tiny else
               "  " + repr(tok.decode(g[0, n_prompt:], skip_special_tokens=True)))
        print(f"          {alpha:>8.4g}{d:>20.5f}{str(same):>11}{txt}")
        if moved_logits is None and d > 1e-6:
            moved_logits = alpha
        if moved_text is None and not torch.equal(g, clean_gen):
            moved_text = alpha
    check("nothing at all happens below bf16 epsilon (2^-8 = 0.0039)",
          moved_logits is None or moved_logits >= 4e-3,
          f"logits first moved at alpha={moved_logits}", gating=False)
    print(f"          logits first move at alpha={moved_logits}; "
          f"text first moves at alpha={moved_text}")
    check("there is a regime where logits move but greedy text does not",
          moved_text is None or moved_logits is None or moved_text >= moved_logits,
          "text moved before logits, which should be impossible", gating=False)

    # ============================================ E: coherence, light vs full
    print("\nE. does the wider band break coherence? (paper: later ranges do,"
          "\n   and its smallest model degraded before showing any effect)")
    if a.tiny:
        print("          skipped in --tiny (no tokenizer)")
    else:
        from scoring import render_prompt
        t2 = render_prompt(tok, question, thinking=False, suffix="", prefill="")
        e2 = tok(t2, return_tensors="pt").to(device)
        for label, band in (("clean", None), ("light 14-19", LIGHT),
                            ("full  14-33", FULL)):
            if band is None:
                with torch.no_grad():
                    g = model.generate(**e2, max_new_tokens=512)
            else:
                with Intervene(model, list(band), fn=add_noise(0.1)), \
                        torch.no_grad():
                    g = model.generate(**e2, max_new_tokens=512)
            out = tok.decode(g[0, e2["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            toks = out.split()
            grams = [tuple(toks[j:j + 10]) for j in range(max(0, len(toks) - 9))]
            d10 = len(set(grams)) / len(grams) if grams else 1.0
            print(f"\n          --- {label}  ({g.shape[1] - e2['input_ids'].shape[1]} tok, "
                  f"distinct10={d10:.3f}, gold={Q733_GOLD}) ---")
            print("          " + out[:400].replace("\n", "\n          "))
        print("\n          Read the text. Do not trust distinct10: on the M4 "
              "baseline\n          a genuine non-terminating loop scored 0.929.")

    print(f"\n          hook census: {set(hook_census(model))} per layer")
    check("no hooks left behind", all(c <= 1 for c in hook_census(model)))

    gate = [(k, ok) for k, ok, g in RESULTS if g]
    info = [(k, ok) for k, ok, g in RESULTS if not g]
    ng = sum(ok for _, ok in gate)
    print(f"\ngating   {ng}/{len(gate)} passed  (A, B, C -- these block M7)")
    print(f"calibration {sum(ok for _, ok in info)}/{len(info)} passed  "
          f"(D -- informational)")
    if ng != len(gate):
        print("\nMilestone 6 is NOT complete. Do not start Milestone 7.")
        for k, ok in gate:
            if not ok:
                print(f"  failed: {k}")
    return 0 if ng == len(gate) else 1


if __name__ == "__main__":
    sys.exit(main())
