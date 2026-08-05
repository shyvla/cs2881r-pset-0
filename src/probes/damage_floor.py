"""The automatic-task damage floor -- is the model ablated, or just broken?

    python -m probes.damage_floor            # real Qwen3-4B, best available device
    python -m probes.damage_floor --tiny     # weightless smoke test

WHAT THIS IS FOR. The assignment requires controls that separate a real
J-space effect from broad degradation, and the paper's central claim has two
halves: ablating the J-space collapses internal multi-step reasoning AND
leaves automatic tasks intact. The random-direction control tests the first
half -- does it matter WHICH directions we remove. This tests the second --
does it matter WHAT WE ASK. Without it, "GSM8K direct drops N points" is
equally consistent with a workspace being removed and with a 4B model being
mildly lobotomised.

It becomes load-bearing rather than optional under the negative result: if the
window between "no effect" and "incoherent" turns out to be empty on Qwen3-4B,
this is the measurement that establishes which end we are at.

THE TASKS. Both are single-token-answer by construction, which is not a
convenience -- it is what makes them the SAME measurement as the GSM8K direct
condition. There the answer is the token straight after the "\\boxed{"
prefill, so a top-1 flip at the last prefill position IS the accuracy effect,
and one forward pass replaces a generation. The same holds here, so the floor
costs seconds and reuses machinery the calibration probe already verified.

  SST-2    sentiment. The cleanest example of the paper's "automatic"
           category: one pass, no intermediate results, nothing to write down.
  MMLU     four-way multiple choice. Knowledge retrieval rather than
           classification, so the floor covers two different non-reasoning
           capabilities instead of one. Some MMLU items do involve reasoning
           -- that cuts against us, making the floor conservative, and it is
           reported separately from SST-2 rather than pooled.

READING IT. The comparison is against the GSM8K direct-condition flip rate
from probes/calibrate.py at the SAME band, k and projection settings:

  reasoning flips >> automatic flips     the ablation is selective
  reasoning flips ~= automatic flips     broad degradation; the J-space story
                                         is not supported by this evidence
  both ~= 0                              the intervention is too weak to say
                                         anything, whatever the band is

The random-token control runs on the automatic tasks too. It has to: if
random directions damage SST-2 as much as the top-10 do, the floor tells us
about magnitude, not selection, exactly as on GSM8K.
"""
import argparse
import sys

import torch

import config
from hooks import Intervene, make_ablation, n_layers, resolve_band
from loaders import load_real, load_tiny, pick_device
from probes.calibrate import mcnemar, wilson

SST2_SUFFIX = ("\n\nIs the sentiment of this sentence positive or negative? "
               "Answer with one word.")
MMLU_SUFFIX = ("\n\nAnswer with the single letter of the correct option and "
               "nothing else.")
LETTERS = ["A", "B", "C", "D"]


def sst2_items(n):
    from datasets import load_dataset
    ds = None
    for path in (("stanfordnlp/sst2", None), ("glue", "sst2")):
        try:
            ds = load_dataset(path[0], path[1], split="validation"); break
        except Exception:
            pass
    if ds is None:
        return []
    out = []
    for i in config.problem_ids(n, len(ds)):
        r = ds[i]
        # label 1 = positive in both mirrors of SST-2.
        out.append((i, r["sentence"].strip() + SST2_SUFFIX,
                    "positive" if r["label"] == 1 else "negative"))
    return out


def mmlu_items(n):
    from datasets import load_dataset
    ds = None
    for args in (("cais/mmlu", "all"), ("hails/mmlu_no_train", "all")):
        try:
            ds = load_dataset(*args, split="test"); break
        except Exception:
            pass
    if ds is None:
        return []
    out = []
    for i in config.problem_ids(n, len(ds)):
        r = ds[i]
        body = r["question"].strip() + "\n" + "\n".join(
            f"{L}. {c}" for L, c in zip(LETTERS, r["choices"]))
        out.append((i, body + MMLU_SUFFIX, LETTERS[int(r["answer"])]))
    return out


def is_correct(task, text):
    """Did the single top-1 token answer correctly?

    Deliberately strict and deliberately NOT math-verify: these are not math
    answers, and scoring.py's "one scoring path" principle is about the three
    math datasets sharing a scorer, not about forcing a sentiment label
    through it.
    A first token that is neither label counts as wrong, which is the same
    convention the headline GSM8K accuracy uses for `incomplete`.
    """
    t = text.strip().lower()
    if task == "sst2":
        return ("positive" if t.startswith("pos") else
                "negative" if t.startswith("neg") else None)
    return t.upper() if t.upper() in LETTERS else None


def run_task(model, tok, task, items, band, K, mode, gain, skip_sink,
             use_ex):
    """Clean pass, then one intervened pass per condition. No generation."""
    from scoring import render_prompt
    KINDS = [("ablate", "ablation top-10"),
             ("rand_tok", "random tokens  "),
             ("rand_gauss", "random gaussian")]
    res = {k: {"flip": 0, "sub": 0} for k, _ in KINDS}
    hit = {k: set() for k, _ in KINDS}
    ncorrect = 0
    for pid, body, gold in items:
        enc = tok(render_prompt(tok, body, thinking=False, suffix="",
                                prefill=""), return_tensors="pt").to(model.device)
        with torch.no_grad():
            clean = model(**enc).logits[0, -1]
        top1 = int(clean.argmax())
        ok = is_correct(task, tok.decode([top1])) == gold
        ncorrect += ok

        # Honour config.USE_EXCLUSION, because the floor is only a floor if it
        # is the SAME operation the reasoning cells get. An exclusion rule that
        # applied here and not there would make the comparison meaningless in
        # the direction that flatters the hypothesis.
        exclude = None
        if use_ex:
            with torch.no_grad():
                ct = model(**enc).logits[0].topk(config.EXCLUDE_TOPK,
                                                 -1).indices
            exclude = {p: ct[p].tolist() for p in range(ct.shape[0])}

        for kind, _ in KINDS:
            # problem=pid, matching run.py: per-problem draws for the random
            # kinds, so the floor's controls are the run's controls.
            fn = make_ablation(model, K, mode=mode, gain_scaled=gain,
                               kind=kind, problem=pid)
            pos = (set(range(1, enc["input_ids"].shape[1]))
                   if skip_sink else None)
            with Intervene(model, list(band), fn=fn, scope="prefill",
                           exclude=exclude, positions=pos), torch.no_grad():
                lg = model(**enc).logits[0, -1]
            if int(lg.argmax()) != top1:
                res[kind]["flip"] += 1
                hit[kind].add(pid)
                if ok:
                    res[kind]["sub"] += 1
    return res, hit, ncorrect, KINDS


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiny", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n", type=int, default=25, help="items per task")
    ap.add_argument("--band", default=config.PRIMARY_BAND,
                    choices=sorted(config.BANDS))
    ap.add_argument("--exclusion", default=None, choices=("on", "off"),
                    help="override config.USE_EXCLUSION for this run. "
                         "EXPLORATORY: the config value is the "
                         "pre-registration, this is for measuring whether it "
                         "matters on a band or task where it was not "
                         "justified.")
    ap.add_argument("--skip-sink", action="store_true",
                    help="spare position 0, the <|im_start|> attention sink "
                         "(266-363x median residual norm). Must match the "
                         "probes/calibrate.py run being compared against.")
    ap.add_argument("--layers", default=None, metavar="LO-HI",
                    help="explicit window, overriding --band. Must match the "
                         "probes/calibrate.py run being compared against.")
    ap.add_argument("--mode", default=None, choices=("each", "span"),
                    help="override config.PROJECTION_MODE; exploratory only")
    ap.add_argument("--gain", default=None, choices=("true", "false"),
                    help="override config.PROJECT_GAIN_SCALED")
    a = ap.parse_args(argv)
    device = pick_device(a.device, a.tiny)
    mode, gain = config.projection(
        a.mode, None if a.gain is None else a.gain == "true")
    use_ex = (config.USE_EXCLUSION if a.exclusion is None
              else a.exclusion == "on")

    model, tok = (load_tiny if a.tiny else load_real)(device)
    NL = n_layers(model)
    try:
        band, nm = resolve_band(NL, config.BANDS[a.band], a.layers)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    band_name = a.band if nm == "band" else nm
    K = config.K_ABLATE

    print(f"model {'TINY' if a.tiny else 'Qwen/Qwen3-4B'}   band={band_name} "
          f"{band.start}-{band.stop - 1}   k={K}   "
          f"projection={mode}/gain={gain}   "
          f"exclusion={'on' if use_ex else 'off'}")
    print("Match these settings to the probes/calibrate.py run you compare "
          "against;\nthe floor is only a floor at the same dose.\n")

    if a.tiny:
        print("TINY: the tasks need real data and a real tokenizer. This "
              "checks imports\nand argument wiring only.")
        return 0

    tasks = [("sst2", sst2_items(a.n)), ("mmlu", mmlu_items(a.n))]
    rows = {}
    for task, items in tasks:
        if not items:
            print(f"{task}: dataset unavailable, skipped\n")
            continue
        res, hit, nc, KINDS = run_task(model, tok, task, items, band, K,
                                       mode, gain, a.skip_sink,
                                       use_ex)
        n = len(items)
        rows[task] = (res, hit, nc, n)
        print(f"{task.upper()}  n={n}   intact accuracy {nc / n:.0%} ({nc}/{n})")
        print(f"   {'condition':>17}{'flips':>9}{'rate':>7}{'95% CI':>13}"
              f"{'of correct':>12}")
        for kind, label in KINDS:
            r = res[kind]
            lo, hi = wilson(r["flip"], n)
            ci = f"[{lo:.0%},{hi:.0%}]"
            sub = f"{r['sub']}/{nc}" if nc else "n/a"
            print(f"   {label:>17}{r['flip']:>6}/{n:<3}{r['flip'] / n:>6.0%}"
                  f"{ci:>13}{sub:>12}")
        A, B = hit["ablate"], hit["rand_tok"]
        b, c = len(A - B), len(B - A)
        print(f"   paired vs random tokens: {b} ablation-only, {c} "
              f"control-only, exact McNemar p={mcnemar(b, c):.3f}\n")

    if not rows:
        print("no tasks ran -- nothing to conclude")
        return 2

    print("=" * 68)
    print("Compare the ablation rows against the GSM8K direct-condition flip")
    print("rate from probes/calibrate.py at the same band, k and projection.")
    print("A floor well BELOW the reasoning rate is the paper's result. A "
          "floor AT\nthe reasoning rate is broad degradation and the "
          "interaction, if any, is\nnot evidence for a workspace.")
    for task, (res, _, nc, n) in rows.items():
        print(f"   {task:>6}: ablation {res['ablate']['flip'] / n:.0%}   "
              f"random tokens {res['rand_tok']['flip'] / n:.0%}")
    print("=" * 68)
    if config.undecided():
        print(f"Still unset before any ablated data: {config.undecided()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
