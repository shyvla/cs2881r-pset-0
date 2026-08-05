"""Model loading, in ONE place.

`load_real`/`load_tiny` began life inside the calibration probe and were
imported from there by every later script -- which meant the main experiment
imported from a diagnostic. They live here so the dependency points the right
way: probes and the run loop both depend on loading, never on each other.

Greedy decoding is set at load time, not per-call: every consumer of these
models relies on generation being a deterministic function of (problem,
condition), and a loader that forgot to set it would corrupt every paired
comparison downstream.
"""
import torch

MODEL = "Qwen/Qwen3-4B"

# WHICH WEIGHTS. "Qwen/Qwen3-4B" names a BRANCH, not a snapshot, so
# from_pretrained resolves it against whatever the Hub serves at download time
# -- and a warm local cache and a freshly rented instance can therefore load
# different weights from the identical command line. This experiment is a
# difference of differences whose arms may be generated days apart on different
# machines, so that is the same class of confound as an unpinned dataset: it
# lands in the headline number with no signature anywhere in the data.
#
# This is the commit the committed GSM8K n=150 run and both calibration files
# were generated against (runs/*_pin.json). Passing it to from_pretrained makes
# the pin a REQUEST rather than a post-hoc observation -- previously the hash
# was read back off the loaded model, which records the drift faithfully but
# only after the GPU hours have been spent.
#
# TO MOVE IT: change this constant in a commit of its own, and treat data
# generated either side of that commit as two experiments. `--model-revision`
# is deliberately not a flag; see load_real.
MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"

# Backends this repo has actually been run on. `cuda:N` is accepted too -- see
# _parse_device.
BACKENDS = ("cpu", "cuda", "mps")


def backend_of(device: str) -> str:
    """'cuda:1' -> 'cuda'. The BACKEND is what determines the numerics, so it
    is what device-drift checks compare: moving a run between two cards of the
    same model is not the confound that moving it from MPS to CUDA is."""
    return device.split(":")[0]


def _parse_device(name: str) -> str:
    """Validate an explicit --device, or stop.

    Checked here rather than left to `.to(device)` because `.to()` is reached
    AFTER from_pretrained has downloaded and materialised 8GB of weights: a
    typo ("--device gpu", "--device cuda" on a Mac) then costs a model load
    before it says so. Availability is checked too, not just spelling, for the
    same reason.

    RAISES SystemExit, not ValueError, so that all seven entry points get the
    one-line message rather than a traceback without seven copies of the same
    try/except -- the duplication this module was created to remove. Same
    reasoning as load_real's revision check: every caller is a CLI.
    """
    dev = name.strip().lower()
    base = backend_of(dev)
    if base not in BACKENDS:
        raise SystemExit(f"unknown device {name!r}; expected one of "
                         f"{', '.join(BACKENDS)} (or cuda:N)")
    idx = dev.split(":")[1] if ":" in dev else None
    if idx is not None and (base != "cuda" or not idx.isdigit()):
        raise SystemExit(f"unknown device {name!r}; only cuda takes an index")
    if base == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(f"--device {name!r} but torch reports no CUDA "
                             f"device on this machine")
        n = torch.cuda.device_count()
        if idx is not None and int(idx) >= n:
            raise SystemExit(f"--device {name!r} but this machine has {n} "
                             f"CUDA device(s), i.e. cuda:0..cuda:{n - 1}")
    if base == "mps" and not (getattr(torch.backends, "mps", None)
                              and torch.backends.mps.is_available()):
        raise SystemExit(f"--device {name!r} but torch reports no MPS backend "
                         f"on this machine")
    return dev


def pick_device(explicit: str | None = None, tiny: bool = False) -> str:
    """Where to put the model. ONE resolution, for the same reason everything
    else here is centralised.

    `("cpu" if tiny else "mps")` was copy-pasted into run.py and all six
    probes -- seven literals encoding "this repo runs on a Mac". That is fine
    until it isn't: on any CUDA box every one of them raises, and the fix has
    to be applied seven times or the entry points disagree about where the
    experiment ran.

    CUDA first, then MPS, then CPU. An explicit --device always wins, because
    a machine with both should still be pinnable to one -- but it is validated
    (see _parse_device) rather than passed through, so a typo costs nothing.

    NOTE FOR THE REPORT: greedy decoding is deterministic ON A GIVEN BACKEND,
    not across backends -- bf16 kernels differ between MPS and CUDA, so the
    same prompt can decode to different text. Cells that will be differenced
    against each other must therefore be generated on the SAME device. run.py
    records the device in the pin file AND on every record, and refuses to
    resume a file onto a different backend without --allow-device-change.
    """
    if explicit:
        return _parse_device(explicit)
    if tiny:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def hardware(device: str) -> dict:
    """What actually ran the generation, for the manifest.

    Recorded because this experiment is a difference of differences and the
    arms may be generated hours or days apart -- potentially on different
    machines. A backend switch between the direct arm and the cot arm lands
    directly in the headline number with no signature anywhere in the data,
    which is the same class of confound as an unpinned checkpoint.

    Takes the RESOLVED device rather than re-detecting one. It used to detect
    independently of pick_device, which meant the manifest reported where the
    run could have gone rather than where it went: `--device cpu` on a CUDA box
    recorded "cuda". Two answers to one question is the failure this module
    exists to prevent.
    """
    base = backend_of(device)
    idx = int(device.split(":")[1]) if ":" in device else 0
    return {"device": device,
            "backend": base,
            "torch": torch.__version__,
            "gpu": (torch.cuda.get_device_name(idx) if base == "cuda"
                    else None)}


def load_real(device, revision: str = MODEL_REVISION):
    """The real Qwen3-4B, at the pinned revision, on ONE device.

    SINGLE-DEVICE BY DESIGN. No device_map, no sharding, no quantization:
    Qwen3-4B in bf16 is ~8GB of weights and fits on any card worth renting, and
    every alternative changes the numbers. Quantization changes the logits
    outright, so no cell produced under it is comparable to the committed bf16
    data; `device_map="auto"` keeps the numerics but splits the module tree
    across devices, which the capture and intervention hooks have never been
    run against. If a box ever needs either, it is a pre-registered change to
    this function and a re-generation, not a flag.

    `revision` is an argument only so a test can exercise the mismatch branch.
    It is not exposed as a CLI flag anywhere: the checkpoint is pinned in a
    commit, so that data generated either side of a bump is separable by
    reading the repo rather than by trusting a shell history.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, revision=revision)
    m = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=revision, dtype=torch.bfloat16).to(device).eval()
    # Belt and braces: `revision` is what we ASKED for, `_commit_hash` is what
    # transformers resolved. They can disagree -- a stale cache entry keyed
    # under a moved tag, or a mirror that ignores the parameter -- and the
    # whole point of the pin is that the answer is not taken on trust.
    got = getattr(m.config, "_commit_hash", None)
    if revision and got and got != revision:
        raise SystemExit(
            f"asked for {MODEL} at revision {revision} but transformers "
            f"loaded {got}. Every number produced against these weights would "
            f"be labelled with a checkpoint that did not generate it. Clear "
            f"the HF cache for this model, or change loaders.MODEL_REVISION "
            f"deliberately and treat the data as a separate experiment.")
    m.generation_config.do_sample = False
    m.generation_config.temperature = None
    m.generation_config.top_p = m.generation_config.top_k = None
    return m, tok


def load_tiny(device):
    """A randomly-initialised Qwen3 built from config: no weights, no
    download, no GPU. The numbers are meaningless but the CONTRACT is
    identical, which is what the weightless smoke tests and CI rely on."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=36, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=16,
                      max_position_embeddings=4096, pad_token_id=0)
    m = Qwen3ForCausalLM(cfg).to(device).eval()
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():                       # trained models have a real gain
        m.model.norm.weight.copy_(
            torch.rand(cfg.hidden_size, generator=g) * 2 + 0.25)
    m.generation_config.do_sample = False
    return m, None
