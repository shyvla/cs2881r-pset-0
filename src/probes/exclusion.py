"""Exclusion-exposure probe -- does the paper's exclusion rule actually bite
during CoT generation?

    python -m probes.exclusion            # real Qwen3-4B on mps
    python -m probes.exclusion --tiny     # weightless smoke test

WHY THIS DECIDES SOMETHING. The paper exempts, at each position, the J-lens
vectors of the tokens in the top-10 of a CLEAN forward pass. Getting that set
during generation needs a paired un-ablated pass at every step, which roughly
doubles the two most expensive cells of the main run. At the time of this
measurement nothing in the repo implemented it (run.py now does), and there
were three ways forward:

  (a) build the two-cache decode loop            faithful, ~2x the cost
  (b) drop the exclusion rule in BOTH arms       cheap, a stated deviation
  (c) apply it in prefill only, both arms        cheap, and WRONG in a
                                                 specific way -- see below

Option (c) is the trap. The direct condition is ~97% prefill and CoT is ~2%,
so a prefill-only rule would apply to almost all of one arm of the contrast
and almost none of the other -- asymmetric across exactly the axis the
interaction is defined on.

Option (b) is defensible only if the rule is weak during generation. The
calibration measured it eating 0.12-0.47 of 10 directions, but that was on
DIRECT PREFILL. Deep inside a reasoning chain the layer-19 readout may align
with the output far more, and nobody has looked. This does.

METHOD, and the one thing to know about it. We do not regenerate. The M4
baseline already holds greedy CoT completions, so we teacher-force the stored
text in a single forward pass: for a greedy decode the resulting residual
stream is the same one generation produced, up to KV-cache numerics in bf16.
That turns ~1 GPU-hour of regeneration into seconds. The numerics caveat is
real but harmless here -- we are measuring how often two top-10 sets
intersect, not reproducing logits bitwise.

The clean top-10 comes from topk_tokens on the last layer's residual stream,
never from model(...).logits: at 3150 positions the full logit array is
~950 MB in bf16 and topk_tokens chunks instead. logit_lens applied to the
last layer reproduces the model's own logits (contract 2), so this is the
same number by a cheaper route.
"""
import argparse
import json
import sys

import torch

import config
from hooks import Capture, n_layers, resolve_band, topk_tokens
from loaders import load_real, load_tiny

BASELINE = "runs/gsm8k_baseline.jsonl"
# From the calibration run: direct condition, PREFILL positions, band 14-19.
# The comparison this probe exists to make.
#
# These replace an earlier set (0.12 ... 0.47) that came from a hand-typed
# paraphrase prompt the run never used -- the same "measured a prompt that
# never ran" fault the calibration had. Measured on problem 120, the first
# problem of config.problem_ids. Keyed by layer, so a --layers window outside
# 14-19 correctly prints no reference rather than a misleading one.
DIRECT_PREFILL_OVERLAP = {14: 0.23, 15: 0.27, 16: 0.38,
                          17: 0.47, 18: 0.59, 19: 0.73}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--file", default=BASELINE)
    ap.add_argument("--n", type=int, default=6, help="CoT traces to measure")
    ap.add_argument("--layers", default=None, metavar="LO-HI",
                    help="explicit window, overriding the primary band. The "
                         "overlap measured at 14-19 does NOT transfer to a "
                         "deeper band -- it rises with depth.")
    ap.add_argument("--max-pos", type=int, default=2048,
                    help="truncate long traces; activations are 7.7 MB/layer "
                         "per 1500 positions")
    a = ap.parse_args(argv)
    device = a.device or ("cpu" if a.tiny else "mps")

    model, tok = (load_tiny if a.tiny else load_real)(device)
    NL = n_layers(model)
    try:
        band, nm = resolve_band(NL, config.BANDS[config.PRIMARY_BAND],
                                a.layers)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    KX = config.EXCLUDE_TOPK
    K = config.K_ABLATE

    # (sequence, n_prompt) pairs. n_prompt is the prefill/generation boundary,
    # which is the whole point of the split below.
    seqs = []
    if a.tiny:
        for i in range(min(a.n, 3)):
            g = torch.Generator().manual_seed(i)
            seqs.append((torch.randint(0, 256, (1, 120), generator=g).to(device),
                         40, i))
    else:
        from datasets import load_dataset
        from scoring import render_prompt, unpack_cond
        ds = None
        for path in ("openai/gsm8k", "gsm8k"):
            try:
                ds = load_dataset(path, "main", split="test"); break
            except Exception:
                pass
        recs = [json.loads(l) for l in open(a.file)]
        recs = [r for r in recs if r["cond"] == "cot_intact"][:a.n]
        if not recs:
            print(f"no cot_intact records in {a.file}", file=sys.stderr)
            return 2
        for r in recs:
            p = tok(render_prompt(tok, ds[r["id"]]["question"], True, "", ""),
                    return_tensors="pt").input_ids
            gen = tok(r["raw"], add_special_tokens=False,
                      return_tensors="pt").input_ids
            ids = torch.cat([p, gen], 1)[:, :a.max_pos].to(device)
            seqs.append((ids, p.shape[1], r["id"]))

    print(f"model {'TINY' if a.tiny else 'Qwen/Qwen3-4B'}   "
          f"band={config.PRIMARY_BAND if nm == 'band' else nm} "
          f"{band.start}-{band.stop - 1}   "
          f"k={K} exclude_top={KX}")
    print(f"{len(seqs)} teacher-forced CoT traces from {a.file}\n")

    # layer -> ("prefill"|"generated") -> list of per-position overlaps
    acc = {L: {"prefill": [], "generated": []} for L in band}
    n_pre = n_gen = 0
    for ids, npr, pid in seqs:
        cap = Capture(model, list(band) + [NL - 1])
        with cap, torch.no_grad():
            model(input_ids=ids)
        with torch.no_grad():
            clean = topk_tokens(model, cap.first(NL - 1), KX)
        T = clean.shape[0]
        npr = min(npr, T)
        n_pre += npr
        n_gen += T - npr
        clean_sets = [set(clean[t].tolist()) for t in range(T)]
        for L in band:
            with torch.no_grad():
                sel = topk_tokens(model, cap.first(L), K)
            ov = [len(set(sel[t].tolist()) & clean_sets[t]) for t in range(T)]
            acc[L]["prefill"] += ov[:npr]
            acc[L]["generated"] += ov[npr:]
        del cap
        print(f"   problem {pid:<6} {T:>5} positions "
              f"({npr} prefill, {T - npr} generated)")

    print(f"\nexclusion overlap /10, CoT condition  "
          f"({n_pre} prefill + {n_gen} generated positions)")
    print(f"   {'layer':>6}{'prefill':>10}{'generated':>11}{'gen >=5':>9}"
          f"{'gen ==10':>10}   direct-prefill ref")
    gen_means = {}
    for L in band:
        pre = torch.tensor(acc[L]["prefill"], dtype=torch.float32)
        gen = torch.tensor(acc[L]["generated"], dtype=torch.float32)
        gen_means[L] = float(gen.mean()) if gen.numel() else float("nan")
        ref = DIRECT_PREFILL_OVERLAP.get(L)
        print(f"   {L:>6}{pre.mean():>10.2f}{gen.mean():>11.2f}"
              f"{float((gen >= 5).float().mean()):>9.1%}"
              f"{float((gen >= K).float().mean()):>10.1%}"
              f"{'' if ref is None else f'{ref:>19.2f}'}")

    worst = max(gen_means.values())
    print("\n" + "=" * 68)
    print(f"worst-layer mean overlap during GENERATION: {worst:.2f} of {K}")
    if worst < 1.0:
        print(f"  OPTION (b) IS DEFENSIBLE. The rule exempts under one "
              f"direction in ten\n  where CoT spends ~98% of its positions, "
              f"so dropping it in BOTH arms costs\n  almost nothing "
              f"behaviourally and saves the paired forward pass. Pre-register\n"
              f"  the deviation and cite this table.")
    elif worst < 3.0:
        print(f"  MARGINAL. The rule bites more during generation than it "
              f"does on direct\n  prefill, but not hard. Option (b) is still "
              f"arguable; say so explicitly\n  rather than leaving it to the "
              f"reader.")
    else:
        print(f"  OPTION (a) IS REQUIRED. The rule exempts {worst:.1f} of "
              f"{K} directions during\n  generation -- dropping it would "
              f"change what the ablation does, in the arm\n  the interaction "
              f"is most sensitive to. Build the two-cache decode loop and\n"
              f"  re-run power.py --secs against the doubled cost.")
    print("  OPTION (c), prefill-only, stays ruled out regardless: it would "
          "apply to\n  ~97% of direct and ~2% of CoT, asymmetrically across "
          "the contrast.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
