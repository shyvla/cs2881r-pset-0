# Running on a rented GPU

Bare instance to generating data. Every step before step 6 is free — run them
all before the meter starts, because each one fails for a reason that would
otherwise surface minutes into a paid session.

The repo runs on Apple Silicon (MPS) and on CUDA from the same command line;
`loaders.pick_device` resolves the backend and `--device` pins it. What it will
*not* do is let one output file span two backends without you saying so — see
[Device rules](#device-rules), which is the part of this document that is not
boilerplate.

---

## 1. Instance

- **Any card with ≥16 GB.** Qwen3-4B in bf16 is ~8 GB of weights, plus KV cache
  for a 3072-token CoT trace. An L4 or A10 (24 GB) is comfortable. Loading is
  single-device by design — no sharding, no quantization (`loaders.load_real`
  says why), so a second GPU buys nothing.
- **~20 GB free disk**: ~9 GB model cache, plus the datasets.
- **Python 3.12** (what CI uses).

## 2. Install

```bash
git clone <this repo> && cd cs2881r-pset-0
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cuXXX torch==2.13.0
pip install -r requirements.txt
```

The torch index URL is the one line that differs from a local install: replace
`cuXXX` with the CUDA build matching the instance's driver (`nvidia-smi` reports
it; https://download.pytorch.org/whl/ lists the builds that exist for 2.13.0). `requirements.txt` pins the version,
not the platform wheel — deliberately, since the repo runs on both, and it is
why the backend is recorded per record rather than assumed.

**Do not skip that line and let PyPI choose.** The default `torch==2.13.0`
wheel is a `+cu130` build, which needs a 13.x driver; against the 12.x drivers
common on rented instances, torch disables CUDA *silently* and
`loaders.pick_device` then falls back to CPU. That is a ~40x slowdown that
looks like a working run — measured here at ~1 tok/s against ~1.5 tok/s on an
M-series laptop. Of the 12.x builds, only **cu126** exists for 2.13.0
(`cu124` and `cu128` have no wheels for this version), and a 12.8 driver runs a
12.6 runtime fine under CUDA minor-version compatibility.

Verify before generating anything, and prefer an explicit `--device cuda` on
every command afterwards — `loaders._parse_device` then refuses in one second
with "torch reports no CUDA device on this machine" instead of quietly
succeeding on the wrong backend:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. Cache location

The model cache belongs on the instance's large volume, not the root disk:

```bash
export HF_HOME=/workspace/.hf
```

Qwen3-4B is a public checkpoint, so no token is needed. `export HF_TOKEN=...`
only if you are pulling through a gated mirror.

## 4. Verify the wiring, free

From `src/` (all later commands assume it):

```bash
cd src
python -m pytest -q
python run.py --tiny --n 1 --smoke-cap 8 --out /tmp/smoke.jsonl
```

The suite needs no weights, no download and no GPU: it builds a randomly
initialised Qwen3 from config and pins the five `transformers` 5.14.1 contracts
`hooks.py` depends on. If it is red, nothing downstream means anything.

## 5. Verify the data, free

```bash
python run.py --check-data --dataset gsm8k
```

Loads the split, resolves which field holds the problem, parses every gold, and
reports the pre-registration choices still unset — with no model in memory. Run
it for each dataset you intend to generate.

`math500` and `aime24` will report unset `CAPS`. That is expected: the caps are
measured, not guessed (step 6).

## 6. First paid command

The model downloads on the first real load (~9 GB, several minutes).

**If the dataset's caps are unset** (`math500`, `aime24`) — measure them first.
This generates intact cells only, at a deliberately generous ceiling, and
writes to a separate `runs/calib_*` namespace that can never be analysed as run
data:

```bash
python run.py --calibrate-caps --dataset math500
```

Commit the suggested cap into `config.CAPS` by hand before generating anything
ablated. The script deliberately will not write it for you.

**Time the run before committing to it:**

```bash
python run.py --n 1 --dataset gsm8k
```

Prints per-cell seconds and extrapolates to the full n. The CoT cells dominate
by two orders of magnitude, and `--only direct` exists so the cheap arm can be
paid for first.

## 7. The run

```bash
python run.py --dataset gsm8k              # all six cells at the committed n
python run.py --dataset gsm8k --only cot   # or one arm at a time
```

Resumable and keyed by `(id, cond)`: re-running appends only what is missing.
The loop gate runs inside the most expensive cell and stops it after its
first ~20 problems if ablated generation has degenerated; a free tripwire
after `cot_random` catches breakage upstream of the hypothesis.

## 8. Bring it home

Copy back `src/runs/<name>.jsonl` **and** `src/runs/<name>_pin.json` — the pin
records the checkpoint, the dataset fingerprint and the hardware, and
`analyze.py` reads the split length from it.

Scoring and analysis need no GPU. Do them locally:

```bash
python analyze.py --dataset gsm8k
```

---

## Device rules

Greedy decoding is deterministic **on a backend, not across backends**: bf16
kernels differ between MPS and CUDA, so the same prompt can decode to different
text. This experiment's headline number is a difference of differences across
six cells, so a file whose cells were generated on different machines has the
backend folded into the result with nothing else to show for it.

Therefore:

- The resolved device is written to the pin file and stamped on **every
  record** (`device`), and `analyze.py` reads it back: a file spanning two
  backends is named cell by cell in the header, and if the *arms* differ the
  interaction line says so, since that difference is inside the headline
  number.
- Resuming a file onto a **different backend is refused**. Pass
  `--allow-device-change` if the mixing is deliberate — a laptop that died
  mid-run, say. The pin then records both devices in `hardware_history`, the
  per-record stamps say which cell came from where, and the report has to
  disclose it.
- Moving between two cards of the same backend (`cuda:0` → `cuda:1`) is not
  refused. The backend is what determines the numerics.
- A **checkpoint or dataset change** on resume is refused with no override.
  There is no reading under which pooling those into one interaction is
  correct.

`loaders.MODEL_REVISION` pins the exact Hub commit and is passed to
`from_pretrained`, so a fresh instance loads the same weights as a warm local
cache rather than whatever the branch points at today.

### The files already in `runs/`

Three runs are committed part-generated, from before hardware was recorded:
`gsm8k_n150_light` (28 of 150 `direct_intact`), `calib_math500_n25` and
`calib_aime24_n10` (their direct arms only). Their pins carry no `hardware`
key, so resuming them on a rented GPU prints a note and proceeds — the guard
cannot verify a backend that was never written down.

Those existing records were generated on Apple Silicon — the archived probe
logs in `runs/archive/` carry MPS fallback warnings. Continuing any of these
files on CUDA therefore *is* a cross-backend run, silently: nothing on disk can
refuse it for you. Either regenerate the file from scratch on one machine
(cheapest by far for the 28-record GSM8K file), or continue it knowingly and
say so in the report.
