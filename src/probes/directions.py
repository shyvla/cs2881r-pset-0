"""Milestone 7 diagnostic -- WHAT does the top-10 select, and how many
directions are really in it?

    python m7_directions.py           # real Qwen3-4B on mps
    python m7_directions.py --tiny    # weightless smoke test

NOTHING IS GENERATED, SCORED, OR INTERVENED ON. This is pure measurement.

WHY THIS EXISTS. The Milestone 7 calibration found the random-token control
flipping the answer at least as often as the ablation (4/12 vs 3/12) while
removing LESS of the residual stream (0.051 vs 0.080). So the flips are not a
magnitude artifact -- something is wrong with WHICH ten directions we pick.
Two candidates, and they want opposite fixes:

  H1  The ten are redundant. Under the feasibility approximation J = I the
      J-lens vector for token t is a row of the unembedding, and the top ten
      rows by inner product tend to be spellings of one token ("5", " 5",
      " five"). Ten nearly parallel directions span ~3 dimensions, so "k=10"
      is a misleading number and the paper's k does not transfer. The paper
      says as much: top-k by inner product gives a MORE REDUNDANT set than
      the gradient pursuit it uses for occupancy.
      -> fix is k, or the selection rule. The band is fine.

  H2  The logit lens is bad here. The paper states it degrades in early
      layers and agrees closely with the J-lens only in the last several.
      Our primary band 14-19 is L38-54 on the paper's scale -- the EARLIEST
      workspace region. We picked it because it is the coherence-safest, and
      it is also the approximation-hostile one.
      -> fix is the band. k is fine.

Both are visible in the same two measurements, which is why they are in one
probe:

  IDENTITY   decode the ten selected tokens. Spellings of one word means H1.
             Rare/high-norm junk means the readout is not tracking meaning
             here at all, which is H2.
  REDUNDANCY effective rank of the ten directions. Near 10 kills H1 outright.

Both are computed in the primary band AND in a MATCHED-WIDTH band at the top
of the network, where the paper says the logit lens is trustworthy. The late
band is the control for the early band: H1 predicts the same redundancy in
both, H2 predicts the late band looks qualitatively better. Widths are matched
because effective rank is being compared across the two and a wider band would
sample more layers, not more structure.

CAUTION ON THE LATE BAND. Layers 30-35 include the motor region (~33-35),
where the readout aligning with the output is expected and is not evidence of
a healthy workspace. It is a diagnostic reference point for the arithmetic,
not a candidate ablation band.
"""
import argparse
import sys

import torch

import config
from hooks import (Capture, band_from_depth, directions_for, n_layers,
                   topk_tokens)
from loaders import load_real, load_tiny
from probes.calibrate import DIRECT_PREFILL, DIRECT_SUFFIX


def spectrum(V):
    """Redundancy of the rows of V, as (eff_rank, rank_1pct, mean|cos|).

    eff_rank is the participation ratio of the singular values,
    (sum s)^2 / sum s^2. It is 10 for ten orthogonal equal-length directions
    and 1 for ten copies of one direction, and unlike a hard threshold it
    degrades smoothly, so it does not need a tolerance argument to be
    meaningful. rank_1pct is the hard count above 1% of the top singular
    value -- reported alongside because the participation ratio is pulled
    down by unequal lengths as well as by collinearity, and the two agreeing
    is what rules that out.

    mean|cos| is the mean absolute off-diagonal cosine, which is the same
    story without any decomposition: it is ~0.02 for random directions in
    2560 dimensions and approaches 1 for spellings of one token.
    """
    Vf = V.float()
    try:
        S = torch.linalg.svdvals(Vf.T)
    except (RuntimeError, NotImplementedError):
        S = torch.linalg.svdvals(Vf.T.cpu())      # MPS lacks the kernel
    S = S.cpu()
    eff = float(S.sum() ** 2 / (S ** 2).sum().clamp_min(1e-30))
    hard = int((S > 0.01 * S[0]).sum())
    U = Vf / Vf.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    G = (U @ U.T).abs().float().cpu()
    m = G.shape[0]
    off = float((G.sum() - G.diagonal().sum()) / max(1, m * (m - 1)))
    return eff, hard, off


def depth_profile(model, tok, encs, golds, K, band):
    """At the ANSWER position only, how readable is the answer at each depth?

    Added after the first run of this probe found the band reading junk at
    exactly the position that sets the direct answer while reading sensible
    tokens everywhere else. Effective rank is blind to that -- it measures
    geometry, and this is semantics -- so it needs its own instrument.

    The answer position is the LAST prefill position: logits[t] predict token
    t+1, so for the direct condition (prefill ends in "\\boxed{") that
    position's readout IS the answer distribution. Two statistics per layer:

      numeric   fraction of the top-10 that decode to a bare number. Cheap,
                needs no gold, and separates "the model is thinking about
                quantity here" from "this is multilingual filler".
      gold@10   how often the gold's FIRST token is in the top-10. Stricter,
                and the one that actually says the answer is present.

    A curve that stays flat and low through 14-19 and rises late says the
    direct answer is computed above the primary band, which would make a
    coherence-safe band structurally unable to damage the direct condition.
    """
    NL, n = n_layers(model), len(encs)
    num = torch.zeros(NL)
    gold = torch.zeros(NL)
    ex = [[] for _ in range(NL)]
    norms = [[] for _ in range(NL)]     # per-position ||h||, all problems
    first = {}                          # problem 0 only, position identity kept
    # One problem at a time: holding 36 layers x n problems of activations is
    # avoidable, and M5 measured this exact arithmetic at 7.7 MB per layer.
    for pi, (e, g) in enumerate(zip(encs, golds)):
        gf = tok.encode(g)[0]
        c = Capture(model, list(range(NL)))
        with c, torch.no_grad():
            model(**e)
        for L in range(NL):
            h = c.first(L)
            with torch.no_grad():
                sel = topk_tokens(model, h[:, -1:], K)[0]
            toks = [tok.decode([int(t)]).strip() for t in sel]
            num[L] += sum(t.isdigit() for t in toks) / K
            gold[L] += int(gf in {int(t) for t in sel})
            if len(ex[L]) < 3:
                ex[L].append(toks[0] or " ")
            pn = h.float()[0].norm(dim=-1).cpu()
            norms[L] += pn.tolist()
            if pi == 0:
                first[L] = (pn, e["input_ids"][0])
        del c

    print("3. DEPTH PROFILE -- when does the answer become readable?")
    print(f"   answer position only (last prefill), over {n} problems\n")
    print(f"   {'layer':>6}{'numeric/10':>12}{'gold@10':>10}   top-1 examples")
    for L in range(NL):
        print(f"   {L:>6}{num[L] / n:>12.2f}{gold[L] / n:>10.0%}   "
              + " ".join(repr(t) for t in ex[L]))

    # ---------------------------------------------------------------- norms
    # M5 reported the MEAN residual norm per layer. Qwen-family models are
    # known to develop a few positions with vastly larger activations, and a
    # mean hides them -- while the ablation is applied PER POSITION, and
    # project_out removes a fixed fraction of whatever norm it finds. If
    # max/median runs to 10x somewhere in the band, "we remove 12% of ||h||"
    # is an average over two very different regimes.
    print("\n4. RESIDUAL NORM PER POSITION -- does the mean hide outliers?")
    print(f"   {'layer':>6}{'median':>10}{'p99':>10}{'max':>10}"
          f"{'max/median':>12}")
    for L in range(NL):
        t = torch.tensor(norms[L])
        md = float(t.median())
        print(f"   {L:>6}{md:>10.1f}{float(t.quantile(0.99)):>10.1f}"
              f"{float(t.max()):>10.1f}{float(t.max()) / max(md, 1e-9):>12.1f}")

    # WHICH positions, and are they the same ones every time? p99 == max in
    # the table above means a handful of positions carry it, and the identity
    # decides what to do: Qwen's massive activations sit on attention-sink
    # tokens, and an ablation that removes a fixed FRACTION of ||h|| there is
    # removing ~300x the absolute magnitude it removes anywhere else.
    # Intervene(positions=...) can scope them out; this says whether it should.
    print("\n   outlier positions (norm > 10x median), problem 0")
    for L in (band.start, band.stop - 1):
        pn, ids_ = first[L]
        md = pn.median()
        idx = (pn > 10 * md).nonzero().flatten().tolist()
        shown = ", ".join(
            f"p{i}={tok.decode([int(ids_[i])])!r}({pn[i] / md:.0f}x)"
            for i in idx[:6])
        print(f"   layer {L:>2}: {len(idx)}/{len(pn)} positions   "
              + (shown or "none"))
    return (num / n).tolist(), (gold / n).tolist()


def show(tok, ids, width=86):
    """Decoded token strings, repr'd so whitespace and byte junk are visible."""
    if tok is None:
        return " ".join(str(int(i)) for i in ids)
    s = " ".join(repr(tok.decode([int(i)])) for i in ids)
    return s if len(s) <= width else s[:width - 1] + "…"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=12,
                    help="problem pool, mirrors m7_calibration.py --n")
    ap.add_argument("--pid", type=int, default=0,
                    help="index into that pool; the prompt to inspect")
    ap.add_argument("--n-pos", type=int, default=6,
                    help="prompt positions sampled per layer")
    ap.add_argument("--depth-profile", action="store_true",
                    help="step 3: all layers, answer position, whole pool")
    a = ap.parse_args(argv)
    device = a.device or ("cpu" if a.tiny else "mps")

    model, tok = (load_tiny if a.tiny else load_real)(device)
    NL = n_layers(model)
    band = band_from_depth(NL, *config.BANDS[config.PRIMARY_BAND])
    late = range(NL - len(band), NL)          # matched width, see docstring
    K = config.K_ABLATE

    # The prompt is a real dataset problem drawn by config.problem_ids, never
    # hardcoded. m7_calibration.py steps 0-2 used a hand-typed paraphrase and
    # therefore characterised a prompt that never ran -- it reported a
    # displacement of 0.118 for a run that applied 0.080.
    encs, golds = [], []
    if a.tiny:
        enc = {"input_ids": torch.randint(0, 256, (1, 60)).to(device)}
        qid = None
    else:
        from datasets import load_dataset
        from scoring import GOLD_FIELD, render_prompt
        ds = None
        for path in ("openai/gsm8k", "gsm8k"):
            try:
                ds = load_dataset(path, "main", split="test"); break
            except Exception:
                pass
        ids = config.problem_ids(a.n, len(ds))
        for i in ids:
            encs.append(tok(render_prompt(tok, ds[i]["question"],
                                          thinking=False, suffix=DIRECT_SUFFIX,
                                          prefill=DIRECT_PREFILL),
                            return_tensors="pt").to(device))
            golds.append(GOLD_FIELD["gsm8k"](ds[i]))
        qid, enc = ids[a.pid], encs[a.pid]
    n_pos = enc["input_ids"].shape[1]

    print(f"model {'TINY' if a.tiny else 'Qwen/Qwen3-4B'}   "
          f"band={config.PRIMARY_BAND} {band.start}-{band.stop - 1}   "
          f"late {late.start}-{late.stop - 1} (matched width {len(band)})")
    print(f"problem {qid}   n_pos={n_pos}   k={K}\n")

    layers = list(band) + list(late)
    cap = Capture(model, layers)
    with cap, torch.no_grad():
        model(**enc)

    # Evenly spaced, plus the LAST prefill position always. For the direct
    # condition that position emits the answer, so it is the one place where
    # a bad selection turns directly into a wrong number.
    step = max(1, n_pos // a.n_pos)
    sample = sorted(set(list(range(0, n_pos, step))[:a.n_pos - 1] + [n_pos - 1]))

    sel = {}
    for L in layers:
        with torch.no_grad():
            sel[L] = topk_tokens(model, cap.first(L).to(device), K)

    # ------------------------------------------------------------- identity
    print("1. IDENTITY -- what are the ten tokens?")
    print("   spellings of one token => H1 (redundant).  rare/byte junk => "
          "H2 (lens bad here).\n")
    ctx = enc["input_ids"][0]
    for group, name in ((band, "BAND"), (late, "LATE")):
        for L in (group.start, group.stop - 1):
            print(f"   {name} layer {L}")
            for p in sample:
                here = show(tok, [ctx[p]], 14) if tok else str(int(ctx[p]))
                tag = "  <- answer pos" if p == n_pos - 1 else ""
                print(f"     p{p:<4}{here:>16}  {show(tok, sel[L][p])}{tag}")
            print()

    # ----------------------------------------------------------- redundancy
    # gain_scaled is PROJECT_GAIN_SCALED, now True. It does not change
    # the SELECTION -- topk_tokens always uses the full norm including the
    # gain -- only which direction gets removed once selected. Both are
    # reported because the redundancy of the removed set is evidence for that
    # pre-registration choice too.
    print("2. REDUNDANCY -- how many directions are really in the ten?")
    print("   eff_rank is the participation ratio of the singular values: 10 "
          "for ten\n   orthogonal directions, 1 for ten copies of one. "
          "Median over sampled positions.\n")
    print(f"   {'layer':>6}{'gain':>7}{'eff_rank':>10}{'rank@1%':>9}"
          f"{'mean|cos|':>11}")
    med = {}
    for L in layers:
        for gain in (True, False):
            rows = []
            for p in sample:
                V = directions_for(model, sel[L][p], gain_scaled=gain)
                rows.append(spectrum(V))
            t = torch.tensor(rows)
            m = t.median(0).values.tolist()
            med[(L, gain)] = m
            mark = "  <-- band" if L in band and gain else ""
            print(f"   {L:>6}{str(gain):>7}{m[0]:>10.2f}{m[1]:>9.0f}"
                  f"{m[2]:>11.3f}{mark}")

    # A random-token baseline for the same statistic. Without it "eff_rank
    # 4.2" is a number with no scale: the question is not whether the ten are
    # redundant in the abstract but whether they are MORE redundant than the
    # control the M8 design already compares against.
    from hooks import random_directions
    rnd = torch.tensor([spectrum(random_directions(model, K, seed=s,
                                                   mode="tokens",
                                                   gain_scaled=True))
                        for s in range(len(sample))])
    r = rnd.median(0).values.tolist()
    print(f"   {'random':>6}{'True':>7}{r[0]:>10.2f}{r[1]:>9.0f}"
          f"{r[2]:>11.3f}  <-- control, k random unembedding rows")

    # ---------------------------------------------------------------- depth
    prof = None
    if a.depth_profile and not a.tiny:
        print()
        prof = depth_profile(model, tok, encs, golds, K, band)

    # -------------------------------------------------------------- summary
    b = sum(med[(L, True)][0] for L in band) / len(band)
    l = sum(med[(L, True)][0] for L in late) / len(late)
    cb = sum(med[(L, True)][2] for L in band) / len(band)
    cl = sum(med[(L, True)][2] for L in late) / len(late)
    print("\n" + "=" * 68)
    print(f"mean eff_rank   band {b:.2f}   late {l:.2f}   random {r[0]:.2f}")
    print(f"mean |cos|      band {cb:.3f}   late {cl:.3f}   random {r[2]:.3f}")
    print()
    if a.tiny:
        print("TINY: meaningless -- k/d is 10/64 here against 10/2560 real.")
    else:
        if b >= 0.9 * r[0]:
            print("H1 NOT SUPPORTED: the ten are about as spread out as ten "
                  "random tokens.\n  Redundancy is not why the control keeps "
                  "up. Look at the band (H2) and\n  at whether k=10 is simply "
                  "too small a fraction of 2560.")
        else:
            print(f"H1 SUPPORTED: the ten span ~{b:.1f} effective dimensions, "
                  f"not 10.\n  'k=10' does not mean on Qwen3-4B what it means "
                  f"in the paper, and that is\n  a reportable limitation of "
                  f"J = I. Read the tokens above to confirm they\n  are "
                  f"spellings of one thing.")
        print()
        if l > b * 1.25:
            print(f"H2 SUPPORTED: the late band is markedly less redundant "
                  f"({l:.2f} vs {b:.2f}).\n  Consistent with the logit lens "
                  f"degrading early. A fixed-width sliding\n  window is the "
                  f"next measurement -- but note the late band overlaps the\n"
                  f"  motor region, so it is a reference point, not a "
                  f"candidate band.")
        else:
            print(f"H2 NOT SUPPORTED HERE: late is not cleaner than the band "
                  f"({l:.2f} vs {b:.2f}),\n  so depth alone does not fix the "
                  f"selection.")
    if prof:
        num, gold = prof
        bg = sum(gold[L] for L in band) / len(band)
        top = max(range(len(gold)), key=lambda L: gold[L])
        print(f"\ngold token in the top-10 at the answer position:"
              f"  band {bg:.0%}   best layer {top} {gold[top]:.0%}")
        if bg < 0.25 and top >= band.stop:
            print(f"  The direct answer is not readable in the band and "
                  f"becomes readable at\n  layer {top}. A band chosen for "
                  f"coherence safety sits BELOW where this\n  condition's "
                  f"answer is computed, which is a mechanism for the null "
                  f"that\n  is about the task, not about the ablation.")
    print("=" * 68)
    return 0


if __name__ == "__main__":
    sys.exit(main())
