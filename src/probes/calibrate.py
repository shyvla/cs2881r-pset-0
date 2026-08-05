"""Calibration -- can the ablation move the outcome measure at all?

    python -m probes.calibrate           # real Qwen3-4B, best available device
    python -m probes.calibrate --tiny    # weightless smoke test

NO ABLATED DATA IS PRODUCED HERE. Nothing is generated and nothing is scored.
This measures the SIZE of the intervention, because the placement probe found
that size is not a smooth dial:

    alpha 0.004 -> 0.1   KL ~1e-4, greedy text unchanged   (dead zone)
    alpha 0.1   -> 0.3   KL rises ~6800x                   (cliff)

If projecting out the top-10 J-lens directions removes under ~10% of ||h||,
this calibration will measure no accuracy effect from an ablation working exactly
as specified, and that null is indistinguishable from the hypothesis being
false. If it removes more than ~30%, the model may simply be broken, and that
is indistinguishable too. Either way you want to know for the price of a few
forward passes rather than 3 GPU-hours.

It also closes the one gap the unit tests cannot: `topk_tokens` is verified
against a 256-token vocabulary in float32, while the real thing has 151,936
tokens in bf16. We compute (h * g) @ W_U.T where the model computes
lm_head(g * h / rms(h)) -- same ranking mathematically, different rounding,
and vastly more near-ties available to flip at real vocabulary size.

The ablation applied here is PREFILL-ONLY, which for the direct condition is
very nearly the whole recipe: the capture probe measured 97% of its positions
as prefill, and the answer is the token straight after the "\\boxed{" prefill.

The exclusion rule follows config.USE_EXCLUSION -- now True. An earlier
revision of this docstring said the rule had been dropped; that decision was
reversed on the measurements recorded in config.py, and the reversal is why
the main run loop pays for a paired clean forward pass at every generation
step instead of being Intervene wrapped around model.generate.
"""
import argparse
import math
import sys

import torch

import config
from hooks import (Capture, Intervene, band_from_depth, directions_for,
                   make_ablation, n_layers, project_out, resolve_band,
                   topk_tokens)
from loaders import MODEL, load_real, load_tiny, pick_device

# From config, not a fourth copy of the string. These probes are GSM8K-only
# measurements, so they ask for the GSM8K prompt explicitly. The directions
# probe imports both names from here, which is why they stay module-level.
DIRECT_SUFFIX, DIRECT_PREFILL = config.direct_prompt("gsm8k")

# Measured by the placement probe on the real model, LIGHT band, direct prompt.
NOISE_CURVE = [(0.004, 0.00011), (0.01, 0.00022), (0.03, 0.00060),
               (0.1, 0.00058), (0.3, 3.95382), (1.0, 8.15946)]


def pct(x, q):
    return float(torch.quantile(x.float(), q))


def wilson(k, n, z=1.96):
    """Score interval. A zero count is not certainty: 0/8 has an upper bound
    near 32%, which is why the verdict must not read a zero as 'no effect'."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    r = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return max(0.0, (c - r) / d), min(1.0, (c + r) / d)


def mcnemar(b, c):
    """Exact two-sided McNemar on the discordant pairs.

    Only the discordant counts carry information: problems that flip under
    both conditions, or neither, say nothing about whether the SELECTION
    matters. Under the null "ablation and control are equally disruptive"
    each discordant pair is a fair coin, so this is a binomial test at p=0.5.
    Exact rather than chi-square because b+c here is single digits.
    """
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=50,
                    help="problems for the behavioural check (step 3)")
    ap.add_argument("--band", default=config.PRIMARY_BAND,
                    choices=sorted(config.BANDS))
    ap.add_argument("--mode", default=None, choices=("each", "span"),
                    help="override config.PROJECTION_MODE; exploratory only")
    ap.add_argument("--gain", default=None, type=lambda v: v == "true",
                    choices=(True, False),
                    help="override config.PROJECT_GAIN_SCALED as true|false")
    ap.add_argument("--exclusion", default=None, choices=("on", "off"),
                    help="override config.USE_EXCLUSION for this run. "
                         "EXPLORATORY: the config value is the "
                         "pre-registration, this is for measuring whether it "
                         "matters on a band or task where it was not "
                         "justified.")
    ap.add_argument("--skip-sink", action="store_true",
                    help="spare position 0. The directions probe measured it as the "
                         "ONLY position above 10x the median residual norm -- "
                         "the <|im_start|> attention sink, at 266-363x. "
                         "project_out removes a fixed FRACTION of ||h||, so "
                         "ablating there removes ~300x the absolute magnitude "
                         "it removes elsewhere, and 'break the attention sink' "
                         "is not 'remove the J-space'.")
    ap.add_argument("--layers", default=None, metavar="LO-HI",
                    help="explicit layer window, overriding --band. "
                         "EXPLORATORY, not a pre-registered band. Exists "
                         "because all three config.BANDS start at depth 0.38, "
                         "so widening one also moves it later -- band POSITION "
                         "and band WIDTH cannot be separated without this.")
    a = ap.parse_args(argv)
    device = pick_device(a.device, a.tiny)
    use_ex = (config.USE_EXCLUSION if a.exclusion is None
              else a.exclusion == "on")
    # Resolved here, beside the other overrides, so the banner and the
    # intervention cannot disagree about what this run did.
    CAL_MODE, CAL_GAIN = config.projection(a.mode, a.gain)

    model, tok = (load_tiny if a.tiny else load_real)(device)
    NL = n_layers(model)
    try:
        band, nm = resolve_band(NL, config.BANDS[a.band], a.layers)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    band_name = a.band if nm == "band" else nm
    K, KX = config.K_ABLATE, config.EXCLUDE_TOPK
    print(f"model {'TINY' if a.tiny else MODEL}   band={band_name} "
          f"{band.start}-{band.stop - 1} (width {len(band)})   k_ablate={K}")
    if a.skip_sink:
        print("scope: position 0 SPARED (attention sink, 266-363x median norm)")
    print(f"projection={CAL_MODE}/gain={CAL_GAIN}"
          + ("   [override]" if (a.mode or a.gain is not None) else ""))
    print(f"exclusion rule: "
          + (f"ON, top-{KX} of the clean pass exempt" if use_ex else
             "OFF (a stated deviation)")
          + ("" if a.exclusion is None else "   [--exclusion override]"))
    print(f"still undecided in config.py: {config.undecided()}\n")

    # Problems for step 3. Drawn by config.problem_ids, NOT hardcoded: a
    # hand-typed list is the run's sampling logic expressed a second time,
    # and it can drift from the run it is meant to calibrate.
    probs, encs, golds = [], [], {}
    if a.tiny:
        for i, pid in enumerate(config.problem_ids(min(a.n, 4), 100)):
            g = torch.Generator().manual_seed(i)
            probs.append(pid)
            encs.append({"input_ids": torch.randint(
                0, 256, (1, 60), generator=g).to(device)})
    else:
        from datasets import load_dataset
        from scoring import GOLD_FIELD, render_prompt
        ds = None
        for path in ("openai/gsm8k", "gsm8k"):
            try:
                ds = load_dataset(path, "main", split="test"); break
            except Exception:
                pass
        for pid in config.problem_ids(a.n, len(ds)):
            probs.append(pid)
            golds[pid] = GOLD_FIELD["gsm8k"](ds[pid])
            encs.append(tok(render_prompt(tok, ds[pid]["question"],
                                          thinking=False, suffix=DIRECT_SUFFIX,
                                          prefill=DIRECT_PREFILL),
                            return_tensors="pt").to(device))

    # Steps 0-2 characterise encs[0] -- the SAME prompt step 3 measures, and a
    # real dataset problem. They used to use a hand-typed paraphrase, so they
    # described a prompt that never ran: step 2 reported a displacement of
    # 0.118 for a step 3 that applied 0.080. Anything measured up here has to
    # be about the run, or it is decoration.
    enc = encs[0]
    n_pos = enc["input_ids"].shape[1]

    cap = Capture(model, list(band) + [NL - 1])
    with cap, torch.no_grad():
        clean_logits = model(**enc).logits
    print(f"steps 0-2 prompt: problem {probs[0]}   n_pos={n_pos}")
    print(f"step 3: {len(probs)} problems "
          f"(config.problem_ids, seed {config.SEED})\n")

    # ---------------------------------------------------------------- step 0
    print("0. does our readout agree with the model at real vocabulary size?")
    h_last = cap.first(NL - 1).to(device)
    with torch.no_grad():
        ours = topk_tokens(model, h_last, 10)
    theirs = clean_logits.reshape(-1, clean_logits.shape[-1]).topk(10, -1).indices
    top1 = float((ours[:, 0] == theirs[:, 0]).float().mean())
    setsame = sum(set(ours[p].tolist()) == set(theirs[p].tolist())
                  for p in range(n_pos)) / n_pos
    print(f"   top-1 agreement {top1:.1%}   top-10 set agreement {setsame:.1%}")
    ok_read = setsame > 0.98
    print(f"   {'PASS' if ok_read else 'FAIL'} -- a disagreeing top-10 means "
          f"the ablation removes different directions than intended\n")

    # ---------------------------------------------------------------- step 1
    # logits[t] predicts token t+1, so the clean top-10 AT position t is
    # exactly the set of tokens the model was about to emit there.
    clean_top = clean_logits[0].topk(KX, -1).indices
    exclude = {p: clean_top[p].tolist() for p in range(n_pos)}
    exclude_by_prob = {}
    for pid, e in zip(probs, encs):
        with torch.no_grad():
            ct = model(**e).logits[0].topk(KX, -1).indices
        exclude_by_prob[pid] = {p: ct[p].tolist() for p in range(ct.shape[0])}

    print("1. how much of the top-10 does the exclusion rule eat?")
    print("   (high overlap in the band would mean the band is in the motor "
          "regime,\n    where the J-lens readout aligns with the output)")
    print(f"   {'layer':>6}{'mean overlap /10':>18}{'positions fully eaten':>24}")
    overlap = {}
    for L in band:
        h = cap.first(L).to(device)
        with torch.no_grad():
            free = topk_tokens(model, h, K, exclude=exclude)
            raw = topk_tokens(model, h, K)
        ov = torch.tensor([len(set(raw[p].tolist()) & set(clean_top[p].tolist()))
                           for p in range(n_pos)], dtype=torch.float32)
        overlap[L] = (free, ov)
        print(f"   {L:>6}{ov.mean():>18.2f}{int((ov >= K).sum()):>24}")

    # ---------------------------------------------------------------- step 2
    stride = max(1, n_pos // 24)
    print(f"\n2. ||dh||/||h|| from projecting out {K} directions, by variant")
    print(f"   every {stride}th position; step 3 reports the MEDIAN of the "
          f"same\n   quantity over all positions, so the two are comparable.")
    print(f"   {'mode':>6}{'gain':>7}{'median':>10}{'p90':>10}{'max':>10}")
    ratios = {}
    for mode in ("each", "span"):
        for gain in (True, False):
            allr = []
            for L in band:
                h = cap.first(L).to(device).float()[0]
                sel = overlap[L][0]
                for p in range(0, n_pos, stride):
                    V = directions_for(model, sel[p], gain_scaled=gain).float()
                    d = h[p] - project_out(h[p], V, mode=mode)
                    allr.append((d.norm() / h[p].norm()).item())
            r = torch.tensor(allr)
            ratios[(mode, gain)] = r
            print(f"   {mode:>6}{str(gain):>7}{pct(r,0.5):>10.3f}"
                  f"{pct(r,0.9):>10.3f}{r.max():>10.3f}")

    ref = ratios[("span", True)]
    print(f"\n   reference variant span/gain-scaled: median {pct(ref,0.5):.3f}")
    print(f"   per-position spread matters -- the capture probe found the mean residual norm "
          f"hides\n   outlier positions. p99={pct(ref,0.99):.3f} "
          f"max={ref.max():.3f}")

    med = pct(ref, 0.5)
    # ---------------------------------------------------------------- step 3
    # THE VERDICT CANNOT REST ON ||dh||/||h||: those thresholds came from
    # RANDOM noise, and removing ten specific directions is not the same
    # operation as adding isotropic noise of equal norm. Criterion is
    # behavioural. For the DIRECT condition that is nearly the whole story --
    # the answer is the token straight after the prefill, so a top-1 flip at
    # the final prefill position IS the accuracy effect.
    #
    # The controls are the paper's: SAME operation, same k, same layers, only
    # the selection randomised. An earlier version used additive noise, which
    # flipped MORE often than the ablation (46% vs 30%) because diffuse
    # displacement touches more output-relevant directions than a rank-10
    # projection does. That measured the gap between two operations.
    CAL_CAP = 16          # calibration only; the run uses config.cap_for
    ratios = {}

    # hooks.make_ablation, not a local copy: the damage floor and the M8 run
    # loop apply the same operation, and an ablation written more than once
    # can drift between a measurement and its own control. The exclusion set
    # is handed to Intervene as {abs position: token ids} and reaches the
    # function through firing.exclude.
    #
    # CAL_MODE / CAL_GAIN come from config.projection(), resolved at parse
    # time above. They were hardcoded ("span", True) as a "working assumption"
    # while the config values were None; the comment saying so went stale the
    # moment they were settled, and a session of flip rates got measured with
    # span against a pre-registration that says each. Step 2 still reports all
    # four variants, which is what the choice was made on.

    def kl(p_l, q_l):
        p = torch.log_softmax(p_l.float(), -1)
        q = torch.log_softmax(q_l.float(), -1)
        return float((p.exp() * (p - q)).sum())

    KINDS = [("ablate", "ablation top-10"),
             ("rand_tok", "random tokens  "),
             ("rand_gauss", "random gaussian")]
    print(f"\n3. behavioural effect over {len(probs)} problems "
          f"(direct condition)")
    print("   controls are the paper's: same operation, same k, same layers,"
          "\n   only the selection randomised.\n")

    if not a.tiny:
        from scoring import score
    res = {k: {"flip": 0, "kl": [], "sub": 0} for k, _ in KINDS}
    # WHICH problems flipped, not just how many. The three conditions run on
    # the same problems, so the design is paired and the marginal counts throw
    # that away: "3/12 and 4/12" is a different result depending on whether
    # the three are a subset of the four or disjoint from them.
    hit = {k: set() for k, _ in KINDS}
    correct_ids = []
    for pid, e in zip(probs, encs):
        with torch.no_grad():
            cl = model(**e).logits[0, -1]
        c = int(cl.argmax())
        ok = None
        if not a.tiny:
            with torch.no_grad():
                g = model.generate(**e, max_new_tokens=CAL_CAP)
            n_new = g.shape[1] - e["input_ids"].shape[1]
            raw = DIRECT_PREFILL + tok.decode(
                g[0, e["input_ids"].shape[1]:], skip_special_tokens=True)
            # hit_cap is not decoration: score() uses it to tell "ran out of
            # room" from "declined to answer". It was hardcoded False here.
            ok = score(raw, golds[pid], hit_cap=(n_new >= CAL_CAP),
                       thinking=False)[0] == "correct"
            if ok:
                correct_ids.append(pid)
        for kind, _ in KINDS:
            fn = make_ablation(model, K, mode=CAL_MODE, gain_scaled=CAL_GAIN,
                               kind=kind, track=ratios)
            # Step 1 above still MEASURES the overlap -- that measurement is
            # what justified config.USE_EXCLUSION, and would overturn it on a
            # band where the rule bites harder. What changes here is whether
            # the ablation ACTS on it, which must match what run.py does.
            ex = exclude_by_prob[pid] if use_ex else None
            # positions is SCOPE, a different axis from `exclude`, which is
            # vocabulary space. Sparing the sink is a position decision.
            pos = (set(range(1, e["input_ids"].shape[1]))
                   if a.skip_sink else None)
            with Intervene(model, list(band), fn=fn, scope="prefill",
                           exclude=ex, positions=pos), torch.no_grad():
                lg = model(**e).logits[0, -1]
            flipped = int(lg.argmax()) != c
            res[kind]["flip"] += flipped
            res[kind]["kl"].append(kl(cl, lg))
            if flipped:
                hit[kind].add(pid)
            if ok and flipped:
                res[kind]["sub"] += 1

    n, nc = len(probs), len(correct_ids)
    print(f"   {'condition':>17}{'|dh|/|h|':>10}{'flips':>9}{'rate':>7}"
          f"{'95% CI':>13}{'medKL':>9}{'p90 KL':>9}{'of correct':>12}")
    for kind, label in KINDS:
        r = res[kind]
        lo, hi = wilson(r["flip"], n)
        # MEDIAN, matching step 2. This was a mean, and comparing a mean here
        # against a median there is how 0.080 and 0.118 sat in the same
        # report without anyone noticing they measure different prompts too.
        mag = pct(torch.tensor(ratios.get(kind, [0.0])), 0.5)
        kt = torch.tensor(r["kl"])
        sub = f"{r['sub']}/{nc}" if nc else "n/a"
        print(f"   {label:>17}{mag:>10.3f}{r['flip']:>6}/{n:<3}"
              f"{r['flip']/n:>6.0%}{f'[{lo:.0%},{hi:.0%}]':>13}"
              f"{pct(kt,0.5):>9.4f}{pct(kt,0.9):>9.4f}{sub:>12}")
    print(f"\n   M6 noise dose-response for scale: "
          + "  ".join(f"a={al}:{k:.4f}" for al, k in NOISE_CURVE))

    # ------------------------------------------------- paired, exact McNemar
    ok_set = set(correct_ids)
    print(f"\n   PAIRED -- ablation vs random tokens on the SAME problems.")
    print(f"   {'subset':>12}{'both':>7}{'abl only':>10}{'ctl only':>10}"
          f"{'neither':>9}{'McNemar p':>12}")
    paired = {}
    for label, keep in (("all", set(probs)), ("correct", ok_set)):
        if not keep:
            continue
        A, B = hit["ablate"] & keep, hit["rand_tok"] & keep
        both, b, c = len(A & B), len(A - B), len(B - A)
        paired[label] = (both, b, c, mcnemar(b, c))
        print(f"   {label:>12}{both:>7}{b:>10}{c:>10}"
              f"{len(keep) - both - b - c:>9}{mcnemar(b, c):>12.3f}")
    print("   Only the discordant columns carry information. A large "
          "'both' means the\n   two operations are hitting the same fragile "
          "problems, which is itself\n   evidence that fragility is the "
          "variable and selection is not.")

    abl, rt = res["ablate"], res["rand_tok"]
    print(f"\n   If random tokens flip as often as the top-10, the effect is "
          f"not about\n   WHICH directions are removed, and the M8 control "
          f"will say so.")

    # ---------------------------------------- implied direct-drop FLOOR
    # THIS IS A FLOOR, NOT A CEILING, and it was labelled the wrong way round.
    # `sub` counts correct problems whose FIRST GENERATED TOKEN flipped. A
    # correct problem that survives its first token can still go wrong later
    # in the sequence, so (nc - sub)/n bounds direct_ablated from ABOVE, and
    # the drop it implies is therefore a LOWER bound. The old text read
    # "the interaction cannot exceed D points" off a number that says the
    # opposite, and the sample-size choice was about to be made against it.
    #
    # What this probe genuinely bounds:
    #   direct_ablated <= DA        so   direct drop >= DI - DA
    #   interaction    <= direct drop      (CoT damage only subtracts)
    # and the direct drop is bounded only from below here, so the interaction
    # is not bounded above at all. Use the floor to decide whether the effect
    # is worth powering for, never to conclude that it is too small.
    if not a.tiny and nc:
        di = nc / n
        da = (nc - abl["sub"]) / n
        drop = di - da
        single = sum(len(tok.encode(golds[p])) == 1 for p in probs)
        print(f"\n   IMPLIED DIRECT-DROP FLOOR")
        print(f"     direct_intact      {di:.0%}   ({nc}/{n})")
        print(f"     direct_ablated    <={da:.0%}   ({nc - abl['sub']}/{n}; "
              f"{abl['sub']} of the correct ones flipped)")
        print(f"     direct drop       >={drop * 100:.0f} points")
        print(f"     A FLOOR: 'flip' is the first generated token only, so an "
              f"answer can go\n     wrong without it -- {single}/{n} golds "
              f"here are single-token, and for the\n     rest this "
              f"undercounts. The interaction is at most the direct drop, "
              f"and\n     this probe does not bound the direct drop from "
              f"above.")
        print(f"     pessimistic power check:  python power.py {di:.2f} "
              f"{da:.2f} 0.90 0.90 --rho 0 --ns 150")

    # ---------------------------------------------------------------- verdict
    rate = abl["flip"] / n
    lo, hi = wilson(abl["flip"], n)
    print("\n" + "=" * 68)
    if a.tiny:
        print("TINY: numbers are meaningless -- k/d is 10/64 here against "
              "10/2560 on the\nreal model. This only shows the code runs.")
    elif abl["flip"] == 0:
        print(f"VERDICT: no detected effect -- 0/{n}, 95% CI [0%, {hi:.0%}].")
        print(f"  NOT 'the interaction is zero'. The ceiling is {hi:.0%}. "
              f"Raise --n before\n  concluding, then widen the band if it "
              f"holds.")
    elif rt["flip"] >= abl["flip"]:
        both, b, c, p = paired.get("all", (0, 0, 0, 1.0))
        print(f"VERDICT: the control flips at least as often as the ablation "
              f"({rt['flip']}/{n} vs\n  {abl['flip']}/{n}). Removing ten "
              f"RANDOM token directions does as much as removing\n  the "
              f"top-10, so this is not evidence for the J-space.")
        print(f"\n  PAIRED, which is what the design supports: {b} problems "
              f"flipped under the\n  ablation only, {c} under the control "
              f"only, {both} under both, exact McNemar\n  p = {p:.3f}. "
              + ("Underpowered -- this cannot separate the two conditions, "
                 "and\n  the marginal comparison above should not be read as "
                 "if it could."
                 if p > 0.10 else
                 "The two conditions differ, in the control's favour."))
    else:
        print(f"VERDICT: ablation {rate:.0%} against random-token control "
              f"{rt['flip']/n:.0%}.")
        print(f"  The selection matters, which is the precondition for the "
              f"J-space story.\n  Whether the gap is large enough to detect "
              f"is the power question above.")
    print("=" * 68)
    print(f"\nM8 control: random_directions(k={K}, mode='tokens') -- the "
          f"paper's control,\nNOT add_noise. Same operation, same k, same "
          f"layers, selection randomised.")
    if config.undecided():
        print(f"Still unset before any ablated data: {config.undecided()}")
    return 0 if ok_read else 1


if __name__ == "__main__":
    sys.exit(main())
