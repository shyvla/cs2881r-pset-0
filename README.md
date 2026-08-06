# CS 2881R Homework Zero

CS 2881R Homework Zero — [assignment](https://boazbk.github.io/mltheoryseminar/hw0-2026/), on
[*Verbalizable Representations Form a Global Workspace in Language Models*](https://transformer-circuits.pub/2026/workspace/index.html)
(Anthropic, 2026).

The paper's claim is that a small, selective set of verbalizable representations — the
**J-space** — acts as a global workspace that flexible internal reasoning is routed
through. If that is true, ablating it should cost the model most where it has to hold
reasoning *inside* the residual stream, and least where the reasoning has been written
out onto the page as chain-of-thought tokens.

That is the one number this repo is built to measure:

```
interaction = (direct_intact - direct_ablated) - (cot_intact - cot_ablated)
```

positive if ablation hurts direct answering more than it hurts CoT. Plus the paper's own
control — the same arithmetic with **ten random unembedding directions** substituted for
the ten J-lens directions — which must come out at ~0, or what we measured is broad
degradation rather than a J-space effect.

Model: **Qwen3-4B** (36 layers), greedy, checkpoint pinned by Hub commit.
Datasets: **GSM8K**, **MATH-500**, **AIME 2024**.

---

## Design

Six cells, `{level} x {state}`, built from a grid in `scoring.py` so a condition name can
never be hand-typed and drift:

|            | intact | ablated (top-10 J-lens) | random (10 random unembedding rows) |
|------------|--------|-------------------------|-------------------------------------|
| **cot**    | ✓      | ✓                       | ✓                                   |
| **direct** | ✓      | ✓                       | ✓                                   |

* **cot** — normal thinking mode, reasoning externalised onto the page.
* **direct** — `\boxed{` prefill plus "respond with only the final answer", so the answer
  token is computed internally. This is the arm the hypothesis predicts should break.
* **ablated** — at every generation position and every layer in the band, zero the
  residual-stream projection onto each of the top-`k=10` J-lens directions, *exempting*
  the J-lens vectors of the tokens in the top-10 of a paired **clean** forward pass, as
  the paper prescribes. Under the feasibility approximation `J = I` the J-lens vector for
  token `t` is a row of the unembedding, gain-scaled.
* **random** — the same operation on ten directions drawn uniformly from all 151,936
  unembedding rows. The matched control for "does it matter *which* directions we remove".

Two contrasts are reported per dataset:

* **selectivity** `= (intact - ablated) - (intact - random) = random - ablated`, within an arm.
* **interaction**, above, across arms — with the random-control interaction printed beside it.

Ablation **strength is band width**, not `k` and not magnitude. The primary band is the
paper's coherence-preserving anchor L38–54 on its reindexed 0–100 scale, converted to
Qwen3-4B's 36 layers by `hooks.band_from_depth` → **layers 14–19** (`light`). `heavy`
(0.38–0.92) and an interpolated `medium` exist in `config.BANDS` but were not run.

Everything above is fixed in [`src/config.py`](src/config.py) **before** ablated data is
seen — sample sizes (from `power.py`), token caps, the band, `k`, the projection mode, the
exclusion rule, the degeneration gate — and every accessor *raises* on an undecided value
rather than defaulting. That file is the pre-registration, executable; its comments carry
the measurement or the argument behind each number, including the ones we got wrong first.

---

## Results so far

Full transcripts: [`src/data_analysis.md`](src/data_analysis.md). Figures:
[`figures/`](figures/).

| dataset | n | CoT intact / ablated / random | Direct intact / ablated / random | selectivity (CoT arm) | interaction (ablation) | interaction (random control) |
|---|---|---|---|---|---|---|
| GSM8K | 150 | 80.7 / 74.7 / 76.7 | 27.3 / 24.7 / 24.7 | +2.0 [−3.3, +7.3] | **−3.3 [−10.7, +4.0]** | −1.3 [−7.3, +4.7] |
| MATH-500 | 150 (level-balanced) | 94.0 / — / 96.0 | 34.0 / 30.7 / 28.7 | — | *regenerating, see below* | **+7.3 [+2.7, +12.7]**, p=0.003 |
| AIME 2024 | 30 | 66.7 / 55.0 (n=20) / 73.3 | 0.0 / 0.0 / 0.0 | +20.0 [0.0, +40.0] | −20.0 [−40.0, +0.0] | 0.0 [−15.0, +15.0] |

Accuracies in %, contrasts in percentage points with 95% paired-bootstrap CIs.

**Read this before reading the table.**

* **GSM8K is the one complete, clean result, and it is a null.** No selectivity in either
  arm, and the interaction points the *wrong way* (ablation cost CoT slightly more than
  direct) with a CI spanning zero. The random control is also ~0, so this is a null rather
  than a mess.
* **MATH-500's `cot_ablated` cell was generated and then lost.** All 150 problems ran to
  completion — 707 minutes of H100 time, ~283 s/problem — and I then destroyed the records
  by accident before they were merged into `runs/math500_n150_light.jsonl`. Stupid, and
  entirely my own doing: nothing about the pipeline or the gate is implicated. The cell is
  being regenerated, and the records plus the updated analysis, table row and figures
  **will be in this repo by noon on Thursday 6 August 2026**. Until then MATH-500 stands at
  5 of 6 cells and has no ablation interaction.

  <details>
  <summary>Proof the run completed (terminal log, 2026-08-05 23:34)</summary>

  <br>

  <img src="figures/math500_cot_ablated_run_log.png" width="400"
       alt="Terminal log of the completed MATH-500 cot_ablated run">

  Tail of that log:

  ```
  cot_ablated      id=497   12259 tok   680.5s  mod=73902
    cot_ablated: 150 problems in 707.0 min

  per-generation seconds
    cot_ablated        282.8 s  x150  generations
    TOTAL              282.8 s/problem   -> 11.8 h at n=150
  ```

  The per-problem lines also corroborate the disclosed censoring: several problems
  (ids 400, 416, 439, 477, 490, 491, 486) terminate at exactly `16384 tok`, i.e. on
  MATH-500's committed CoT cap.

  </details>

* **MATH-500's random-control interaction is significantly non-zero** — the quantity that
  is supposed to be ~0 is not — which on its own says the direct arm degrades under *any*
  ten removed directions on this dataset. The regenerated ablation interaction cannot be
  read without that beside it, and must not be read against zero.
* **AIME 2024's `cot_ablated` cell is 20 of 30 problems because the pre-registered
  degeneration gate fired and stopped it**: 40% unusable against `cot_intact`'s 5%, a
  +35pt delta against a 15% threshold, with 8 cap hits at 32768 and 6 of those ending in
  verbatim repetition loops. The +20 selectivity and the −20 interaction are computed over
  that stopped cell, at n=20, with 40-point CIs. They are not evidence of anything yet.
  The cell stays at 20/30: `src/shard_ablated_tail.sh` documents exactly how the remaining
  10 problems *could* be generated by blinding the gate, and why doing so would oblige a
  disclosure that the pre-registered protocol for that cell was overridden.
* **AIME direct accuracy is 0% in all three states.** A floor cannot move, so that arm
  contributes no information about ablation on this dataset.

See [Disclosures](#disclosures) for the rest.

---

## Repository layout

```
CLOUD.md                  renting a GPU: install, cache, verify-before-you-pay, device rules
figures/                  fig1_accuracy.png, fig2_contrasts.png, and the lost-run log screenshot
requirements.txt          exact pins; two of them are load-bearing, see the comments
src/
  config.py               THE PRE-REGISTRATION. Every fixed number, with its reasoning.
  hooks.py                capture + intervention. Encodes 5 transformers-5.14.1 contracts.
  loaders.py              model loading in one place; pinned revision; device resolution
  run.py                  generation — the six cells. Resumable, shardable, gated.
  scoring.py              answer extraction + scoring, one path for all three datasets
  analysis.py             paired bootstrap, McNemar, the interaction
  analyze.py              scoring + reporting over a generations file (no GPU)
  power.py                how many problems per cell — run BEFORE spending GPU hours
  merge_runs.py           merge staged per-pod files into one file, refusing mismatches
  make_figures.py         the two report figures
  shard_ablated_tail.sh   code to complete aime24's stopped cell across two cards
  data_analysis.md        analyze.py transcripts the figures and the report are read off
  probes/                 run-once diagnostics; each one's conclusion is frozen into config.py
  tests/                  186 tests, no weights, no GPU, ~7s
  runs/                   generations + pins (below)
  experiments.ipynb       historical, superseded by the scripts above
```

`report.pdf` — submitted alongside this repo

### `src/runs/`

`run.py` writes `runs/<dataset>_n<n>_<band>.jsonl` plus a sidecar `_pin.json`, and
`analyze.py` reads the same path (`scoring.run_path`, so the name is constructed in one
place). The pin records the model revision, the dataset fingerprint + content hash + the
exact problem ids, the hardware, and the file's `hardware_history` / `merged_from`
provenance — `analyze.py` reads the split length back out of it.

| path | what |
|---|---|
| `runs/{gsm8k_n150,math500_n150,aime24_n30}_light.jsonl` | the canonical merged run files. Every number above comes from these. |
| `runs/calib_{math500_n25,aime24_n10}.jsonl` | cap calibrations. A **separate namespace** on purpose: intact cells only, generated at `config.MEASURE_CAP` rather than at a committed cap, stamped `calibration: true`, and `analyze.py` refuses to read them as run data. |
| `runs/incoming/`, `runs/saved_runs/`, `runs/final/` | the per-pod staged files exactly as they came off the rented GPUs, kept so `merge_runs.py`'s inputs stay auditable after the merge. |
| `runs/archive/` | superseded artifacts, including the n=20 pilot and runs made under the old `random.sample` problem sampler. See `runs/archive/README.md`. Never pool these with the above. |

One generations record:

```json
{"id": 294, "cond": "direct_intact", "dataset": "gsm8k", "seed": 0,
 "raw": "\\boxed{5}<|im_end|>", "gold": "3", "n_tok": 3, "cap": 128,
 "hit_cap": false, "device": "cuda", "secs": 4.8, "n_modified": 0,
 "band": "14-19", "calibration": false, "difficulty": null, "calib_role": null}
```

Scores are deliberately *not* stored beside them: generation costs hours and scoring costs
milliseconds, so a change to the scorer is a re-run of `analyze.py`, never of the GPU job.

---

## Setup

Python **3.12** (what CI uses).

```bash
git clone <this repo> && cd cs2881r-pset-0 && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

`requirements.txt` pins the torch **version** but not the platform wheel — the repo runs on
Apple Silicon (MPS) and on CUDA from the same command line. On a rented GPU, install the
matching CUDA wheel *first*: **[`CLOUD.md`](CLOUD.md) is the file to read before paying for
an instance**, and that wheel line is the one whose omission silently costs a 40x slowdown
that looks like a working run.

Point the model cache at a volume with ~9 GB free:

```bash
export HF_HOME=/workspace/.hf
```

`Qwen/Qwen3-4B` is public, so no token is needed. `loaders.MODEL_REVISION` pins Hub commit
`1cfa9a7208912126459214e8b04321603b3df60c` — the commit every committed run was generated
against — and passes it to `from_pretrained`, so a fresh instance loads the same weights a
warm cache does.

---

## Reproducing the tables and figures (no GPU, seconds)

Everything in the results table and both figures regenerates from the committed
generations files. From `src/`:

```bash
python -m pytest -q
```

```bash
python analyze.py --dataset gsm8k
python analyze.py --dataset math500
python analyze.py --dataset aime24
```

Per dataset that prints: per-cell accuracy and outcome composition
(`correct`/`incorrect`/`incomplete`/`unparsed`/`error`, plus how many answers the markdown
fallback rescued), selectivity per arm with McNemar, the interaction with its random
control beside it, and — for GSM8K only — the exploratory-vs-holdout split.
[`src/data_analysis.md`](src/data_analysis.md) is a transcript of exactly these commands,
and it is the table above's source.

The figures are hand-transcribed from that transcript into `DATA` / `CONTRASTS` at the top
of `make_figures.py`, which is why the plotting script needs only matplotlib and is not
part of the pinned experiment environment:

```bash
python make_figures.py            # -> ../figures/fig1_accuracy.png, fig2_contrasts.png
```

Sample-size arithmetic (a pre-registration input, not a post-hoc excuse) —
`direct_intact direct_ablated cot_intact cot_ablated`, plus the measured
problem-difficulty correlation:

```bash
python power.py 0.34 0.30 0.94 0.90 --rho 0.5
```

---

## Regenerating the data (GPU)

Read [`CLOUD.md`](CLOUD.md) first — it orders these so that everything that can fail for
free fails before the meter starts. From `src/`:

```bash
python run.py --check-data --dataset math500
```

```bash
python run.py --tiny --n 1 --smoke-cap 8 --out /tmp/smoke.jsonl
```

```bash
python run.py --n 1 --dataset gsm8k
```

```bash
python run.py --dataset gsm8k
```

`--check-data` resolves the loader, the question field and every gold answer with no model
in memory; `--tiny` builds a randomly-initialised Qwen3 from config and checks the wiring
weightlessly; `--n 1` prints per-cell seconds and extrapolates to the full n. Then
`--only cot`, `--only cot_ablated` etc. stage the arms so the cheap ones can be paid for
first.

Notes that matter:

* **`--n` defaults from `config.N_DEFAULT[dataset]`**, not from argparse. A dataset whose n
  has not been derived from `power.py` refuses to run.
* **Resumable**, keyed by `(id, cond)` — re-running appends only what is missing.
* **A file may not silently span two backends.** Greedy decoding is deterministic on a
  backend, not across them (bf16 kernels differ), and this experiment differences six
  cells against each other. Resuming onto a different backend is refused;
  `--allow-device-change` overrides and is recorded in `hardware_history`. A checkpoint or
  dataset change on resume is refused with no override.
* **Caps are measured, not guessed.** For a dataset with unset caps,
  `python run.py --calibrate-caps --dataset math500` generates intact cells only, at the
  generous `config.MEASURE_CAP` ceiling, into the `runs/calib_*` namespace, and *prints* a
  suggested cap. It will not write it for you — a pre-registration a script can edit is not
  a pre-registration.
* **The degeneration gate fires inside `cot_ablated`**, after its first 20 problems, on the
  `unusable` rate relative to the matched intact cell. Exit code 3 is the gate firing. That
  is not a bug; see AIME above.
* Merging staged per-pod files: `python merge_runs.py podA.jsonl podB.jsonl --out
  runs/<name>.jsonl`. It refuses inputs that disagree about weights, dataset, sample,
  backend or GPU model, and refuses duplicate `(id, cond)` records whose contents differ.

The `probes/` diagnostics run weightlessly with `--tiny`
(`python -m probes.directions --tiny`) and against real weights without it. Each answered
one question — is the intervention on the right tensor, is it large enough to move anything
at all, does the exclusion rule bite during generation, is the model ablated or merely
broken, what do the top-10 directions actually select — and each conclusion is now a
constant in `config.py` with its measurement written next to it.

CI ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)) runs the suite, every
probe's `--tiny` path and the `run.py` smoke tests on each push. It is a tripwire on the
model contract rather than just a test run: `hooks.py` depends on five properties of
transformers 5.14.1 that are *not* properties of transformers in general (the decoder layer
returning a bare tensor, hidden-state capture being itself hook-based, hook firing order
making `output_hidden_states` report *pre*-intervention activations, …), and
`tests/test_hooks.py` pins all five against a randomly-initialised Qwen3 built from config
— no weights, no download, no GPU.

---

## Disclosures

What a reader should know before believing any number above. Each is expanded, with its
reasoning, in the file named.

* **The band was selected on outcome data**, by comparing flip rates across five layer
  windows on 12 GSM8K problems (`config.EXPLORED_N`). `analyze.py` always reports those 12
  apart from the holdout and never pools them — they score `direct_intact` 50% against the
  holdout's 25%. A later clean sweep found every window equivalent, so the band stayed at
  the paper's anchor; the 12 problems were spent regardless.
* **The band was selected on GSM8K and transferred** to MATH-500 and AIME
  (`config.EXPLORED_ON`). On those datasets that is an assumption, not a measurement.
* **MATH-500's sample is level-balanced** — 30 problems from each of the five difficulty
  levels — so its cell accuracies are level-balanced averages and are *not* comparable to
  published split-weighted MATH-500 numbers.
* **MATH-500's CoT cap (16384) is a budget cap at the ceiling it was measured at**, not a
  cleared tail: the re-measurement still censored 15% of the hard-end sample, exactly on
  the pre-committed stopping-rule boundary, so we stopped rather than doubling again. 7 of
  150 intact CoT generations are `incomplete` at that cap.
* **AIME 2024's caps were committed without any calibration.** 32768 is a budget ceiling
  bounded by the checkpoint's own `max_position_embeddings` (40960), not a measured tail,
  and the `incomplete` rate at it was unknowable in advance — it came out at 8 of the 20
  `cot_ablated` generations.
* **AIME's `cot_ablated` cell is gate-stopped at 20/30**, as above.
* **MATH-500's `cot_ablated` records were lost after a completed run** and are being
  regenerated; due in this repo by noon on Thursday 6 August 2026. See
  [Results so far](#results-so-far) for the log of the lost run. This is operator error, not
  a pipeline or protocol failure, and the regenerated cell is the same pre-registered
  condition at the same committed cap — but it *was* generated after the other five cells'
  results were known, which is worth stating even though nothing about the cell is chosen
  at run time.
* **MATH-500's random-control interaction is significantly non-zero** (+7.3 pts, p=0.003),
  which is the signature of broad degradation rather than a J-space effect.
* **One deliberate deviation from a literal reading of the paper**: the removed direction is
  gain-scaled (`W_U[t] * g`, not the bare `W_U[t]`), because the J-lens readout score lies
  along the gain-scaled direction, so removing anything else leaves a residual readout for a
  token the ablation claims to have removed. It costs displacement — 0.145 against 0.200.
  `config.PROJECT_GAIN_SCALED`.
* **The `medium` band width is interpolated by us** and corresponds to nothing in the paper.
* An earlier configuration ran with the exclusion rule **off**, which on SST-2 produced an
  84% flip rate that collapsed to 8% once the rule was restored: with the rule off, the
  ablation was removing the answer tokens' own directions — destroying the readout, not the
  workspace. Every flip count measured that way is contaminated and was re-measured.
  `config.USE_EXCLUSION`.

---

## External resources

* Assignment — <https://boazbk.github.io/mltheoryseminar/hw0-2026/>
* Paper — Anthropic, *Verbalizable Representations Form a Global Workspace in Language
  Models* — <https://transformer-circuits.pub/2026/workspace/index.html>
* Model — `Qwen/Qwen3-4B`, revision `1cfa9a7208912126459214e8b04321603b3df60c` —
  <https://huggingface.co/Qwen/Qwen3-4B>
* GSM8K — `openai/gsm8k`, config `main`, split `test` (1319 rows) —
  <https://huggingface.co/datasets/openai/gsm8k>
* MATH-500 — `HuggingFaceH4/MATH-500`, split `test` (500 rows) —
  <https://huggingface.co/datasets/HuggingFaceH4/MATH-500>
* AIME 2024 — `HuggingFaceH4/aime_2024`, split `train` (30 rows) —
  <https://huggingface.co/datasets/HuggingFaceH4/aime_2024>
* Answer checking — `math-verify` 0.9.0 — <https://github.com/huggingface/Math-Verify>
