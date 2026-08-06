// Builds report.docx from the analyze.py numbers (src/data_analysis.md) and
// the figures in figures/. Re-run after MATH-500 lands and after editing.
//   node make_report.js
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, BorderStyle, ImageRun,
  ShadingType, LevelFormat, convertInchesToTwip,
} = require("docx");

const FIG = (f) => fs.readFileSync(path.join(__dirname, "figures", f));

// ---------------------------------------------------------------- helpers
const body = (text, opts = {}) =>
  new Paragraph({
    spacing: { after: 120, line: 264 },
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, size: 21, ...opts })],
  });

const rich = (runs, opts = {}) =>
  new Paragraph({
    spacing: { after: 120, line: 264 },
    alignment: AlignmentType.JUSTIFIED,
    ...opts,
    children: runs.map((r) =>
      typeof r === "string" ? new TextRun({ text: r, size: 21 })
        : new TextRun({ size: 21, ...r })),
  });

const bullet = (runs) =>
  new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80, line: 264 },
    alignment: AlignmentType.JUSTIFIED,
    children: (Array.isArray(runs) ? runs : [runs]).map((r) =>
      typeof r === "string" ? new TextRun({ text: r, size: 21 })
        : new TextRun({ size: 21, ...r })),
  });

const h1 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, size: 26, bold: true, color: "1a1a19" })],
  });

const h2 = (text) =>
  new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 180, after: 100 },
    children: [new TextRun({ text, size: 22, bold: true, color: "1a1a19" })],
  });

const placeholder = (text) =>
  new Paragraph({
    spacing: { before: 100, after: 140, line: 264 },
    shading: { type: ShadingType.CLEAR, fill: "FFF3C4" },
    border: {
      left: { style: BorderStyle.SINGLE, size: 12, color: "EDA100" },
    },
    children: [new TextRun({ text, size: 21, italics: true, color: "6b5900" })],
  });

const caption = (text) =>
  new Paragraph({
    spacing: { before: 60, after: 200 },
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, size: 18, italics: true, color: "555555" })],
  });

const figure = (file, widthIn, heightIn) =>
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 40 },
    children: [new ImageRun({
      type: "png", data: FIG(file),
      transformation: { width: widthIn * 96, height: heightIn * 96 },
    })],
  });

// table helper: header row + data rows, all DXA widths
function makeTable(colWidths, header, rows) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  const cell = (text, { bold = false, fill = null, align = AlignmentType.LEFT } = {}, w) =>
    new TableCell({
      width: { size: w, type: WidthType.DXA },
      shading: fill ? { type: ShadingType.CLEAR, fill } : undefined,
      margins: { top: 40, bottom: 40, left: 80, right: 80 },
      children: [new Paragraph({
        alignment: align,
        children: [new TextRun({ text, size: 19, bold })],
      })],
    });
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [
      new TableRow({
        tableHeader: true,
        children: header.map((t, i) =>
          cell(t, { bold: true, fill: "EEEDE6", align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER }, colWidths[i])),
      }),
      ...rows.map((r) => new TableRow({
        children: r.map((t, i) =>
          cell(String(t), { align: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER }, colWidths[i])),
      })),
    ],
  });
}

const spacer = () => new Paragraph({ spacing: { after: 120 }, children: [] });

// ---------------------------------------------------------------- content
const cellCols = [1900, 700, 900, 950, 1000, 1100];
const cellHeader = ["Condition", "n", "Acc.", "Correct", "Incorrect", "Incomplete"];

const gsmRows = [
  ["cot_intact", 150, "80.7%", 121, 5, 24],
  ["cot_ablated", 150, "74.7%", 112, 8, 30],
  ["cot_random", 150, "76.7%", 115, 7, 28],
  ["direct_intact", 150, "27.3%", 41, 109, 0],
  ["direct_ablated", 150, "24.7%", 37, 113, 0],
  ["direct_random", 150, "24.7%", 37, 113, 0],
];

const aimeRows = [
  ["cot_intact", 30, "66.7%", 20, 5, 5],
  ["cot_ablated", "20*", "55.0%", 11, 1, 8],
  ["cot_random", 30, "73.3%", 22, 7, 1],
  ["direct_intact", 30, "0.0%", 0, 30, 0],
  ["direct_ablated", 30, "0.0%", 0, 30, 0],
  ["direct_random", 30, "0.0%", 0, 30, 0],
];

const statCols = [2400, 1450, 1900, 1350, 1000];
const statHeader = ["Contrast", "Point (pts)", "95% CI (pts)", "p", "n pairs"];

const gsmStats = [
  ["CoT: ablated vs intact (McNemar)", "−6.0", "—", "0.035", 150],
  ["CoT: random vs intact (McNemar)", "−4.0", "—", "0.180", 150],
  ["Selectivity, CoT arm", "+2.0", "[−3.3, +7.3]", "0.532", 150],
  ["Selectivity, direct arm", "+0.0", "[−5.3, +5.3]", "1.000", 150],
  ["Interaction (ablation)", "−3.3", "[−10.7, +4.0]", "0.407", 150],
  ["Interaction (random control)", "−1.3", "[−7.3, +4.7]", "0.678", 150],
];

const aimeStats = [
  ["CoT: ablated vs intact (McNemar)", "−11.7", "—", "0.219", 20],
  ["CoT: random vs intact (McNemar)", "+6.7", "—", "1.000", 30],
  ["Selectivity, CoT arm", "+20.0", "[+0.0, +40.0]", "0.115", 20],
  ["Selectivity, direct arm", "+0.0", "[+0.0, +0.0]", "1.000", 30],
  ["Interaction (ablation)", "−20.0", "[−40.0, +0.0]", "0.113", 20],
  ["Interaction (random control)", "+0.0", "[−15.0, +15.0]", "1.000", 20],
];

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 200 } } },
      }],
    }],
  },
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: {
          top: convertInchesToTwip(0.9), bottom: convertInchesToTwip(0.9),
          left: convertInchesToTwip(1.0), right: convertInchesToTwip(1.0),
        },
      },
    },
    children: [
      // ---------------------------------------------------------- title
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 60 },
        children: [new TextRun({
          text: "Does J-Space Ablation Trade Off Against External Chain of Thought?",
          size: 32, bold: true,
        })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 40 },
        children: [new TextRun({
          text: "CS 2881r (AI Safety) — Homework 0", size: 22, color: "444444",
        })],
      }),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 240 },
        children: [new TextRun({
          text: "Shyla Nguyen · shylanguyen@college.harvard.edu · August 2026",
          size: 20, color: "666666",
        })],
      }),

      // ---------------------------------------------------------- 1. hypothesis
      h1("1. Hypothesis"),
      rich([
        "The workspace paper identifies a small set of verbalizable directions (the ",
        { text: "J-space", italics: true },
        ") that carries a model's intermediate reasoning state, and shows that ablating it devastates multi-hop reasoning while leaving shallow extraction intact. Chain of thought externalizes intermediate state into the context, where an activation-space ablation cannot reach it. Our pre-registered hypothesis (recorded in ",
        { text: "src/config.py", font: "Consolas", size: 19 },
        " before any ablated data was generated) is a substitution account:",
      ]),
      rich([
        { text: "H1 (substitution). ", bold: true },
        "J-space ablation hurts ",
        { text: "direct", italics: true },
        " answering (no chain of thought) more than ",
        { text: "CoT", italics: true },
        " answering, because CoT can re-externalize the work the workspace was doing. The pre-registered headline statistic is the interaction",
      ]),
      new Paragraph({
        alignment: AlignmentType.CENTER,
        spacing: { after: 120 },
        children: [new TextRun({
          text: "Δ = (direct_intact − direct_ablated) − (cot_intact − cot_ablated),",
          size: 21, italics: true,
        })],
      }),
      rich([
        "with H1 predicting Δ > 0. A Δ ≈ 0 with a tight interval means no measurable trade-off; Δ < 0 would indicate the opposite of substitution — that the workspace remains load-bearing ",
        { text: "during", italics: true },
        " chain of thought. Both outcomes are visible under the same design, which is what makes the test fair.",
      ]),
      rich([
        { text: "H2 (difficulty scaling). ", bold: true },
        "As problem difficulty rises (GSM8K → MATH-500 → AIME 2024), substitution predicts the CoT arm stays robust to ablation even as problems harden, since more of the computation is externalized. The complementary account predicts the ablation deficit in the CoT arm grows with difficulty, because harder problems make the model lean on internal workspace state between emitted tokens.",
      ]),
      rich([
        { text: "Selectivity requirement. ", bold: true },
        "Under either hypothesis, a deficit only counts as a J-space effect if it exceeds a matched random-direction control; a deficit shared with the control is broad degradation, not workspace removal.",
      ]),

      // ---------------------------------------------------------- 2. design
      h1("2. Experiment design"),
      rich([
        "Per dataset, a 2×3 factorial: an ",
        { text: "externalization axis", italics: true },
        " (cot: thinking mode enabled; direct: thinking disabled, an “answer only, no reasoning” instruction, and a ",
        { text: "\\boxed{", font: "Consolas", size: 19 },
        " prefill that commits the model to answering immediately) crossed with an ",
        { text: "intervention axis", italics: true },
        " (intact / J-space ablated / random-direction control). All six cells run the same problems, so every comparison is paired at the problem level.",
      ]),
      h2("Controls that separate J-space effects from broad degradation"),
      bullet([
        { text: "Matched random control (rand_tok). ", bold: true },
        "The identical operation — project out k = 10 directions drawn from the unembedding, in the same layer band, with the same projection mode — with only the selection randomized: k uniformly random unembedding rows instead of the top-k readout tokens, re-drawn per (problem, layer, position). Anything the ablation does that the control also does is degradation, not a workspace effect.",
      ]),
      bullet([
        { text: "Control interaction. ", bold: true },
        "The full interaction Δ is recomputed with the random cells substituted for the ablated ones. It should be ≈ 0; if it is not, the pipeline itself (not J-space) moves the headline number, and the two must be read together.",
      ]),
      bullet([
        { text: "Selectivity per arm. ", bold: true },
        "(intact − ablated) − (intact − random) = random − ablated, with a paired bootstrap CI: the targeted-minus-matched-control deficit within each arm.",
      ]),
      bullet([
        { text: "Degeneration gate. ", bold: true },
        "Before paying for the expensive cells, ablated CoT runs on the first 20 problems and the run stops if its unusable rate (incomplete / unparsed / error) exceeds the matched intact cell's by more than 15 points — a tripwire against comparing arms when generation has simply broken.",
      ]),
      h2("Statistics and pre-registration"),
      rich([
        "Within-arm comparisons use McNemar's exact test on paired outcomes; selectivity and interaction estimates use a percentile bootstrap (10,000 resamples) over ",
        { text: "problems", italics: true },
        ", not observations, preserving the pairing. Sample sizes came from a power analysis at pilot accuracies (ρ = 0.5): n = 150 for GSM8K; n = 150 for MATH-500 (amended from 100 before any MATH-500 data existed); all 30 problems for AIME 2024. Every analysis choice — layer band, k, projection mode, exclusion rule, token caps, sample sizes, prompts — was fixed in a committed config module before ablated data was generated, and accessors raise on any undecided value so a choice cannot be made silently at the keyboard. Twelve GSM8K problems were spent selecting the layer band; the analysis flags them and reports the exploratory-vs-holdout split alongside the headline numbers, since selection on outcome data inflates whatever it selected for.",
      ]),
      rich([
        "Why this design answers the question: substitution, complementarity, and no-effect each produce a distinct, visible signature — positive, negative, or null interaction — and the matched control plus control interaction ensure that whatever signature appears cannot be explained by the intervention simply damaging the model.",
      ]),

      // ---------------------------------------------------------- 3. details
      h1("3. Experimental details"),
      rich([
        { text: "Model and decoding. ", bold: true },
        "Qwen3-4B at a pinned checkpoint revision, bf16, greedy decoding, batch size 1, single device per run. Greedy bf16 decoding is deterministic only within a backend (MPS vs CUDA kernels differ), so the device is recorded on every record and a run refuses to resume onto a different backend.",
      ]),
      rich([
        { text: "J-lens under the feasibility approximation J = I. ", bold: true },
        "The paper's J-lens requires per-layer Jacobians averaged over positions and a 1,000-prompt corpus, which is out of reach at our compute. We use the special case J = I: the J-lens vector for token t at every layer is the unembedding row W_U[t] scaled by the final RMSNorm's learned gain, and the readout is the gain-aware logit lens. Calibration against the model's own logits at the final layer: 100% top-1 agreement, 94.1% top-10 set agreement (the discrepancy is bf16 rounding at the rank-10 boundary; the full norm is applied to make the arithmetic exact). This is the single largest deviation from the paper and is discussed in §5.",
      ]),
      rich([
        { text: "Ablation. ", bold: true },
        "At every position in the band, the top k = 10 vocabulary tokens by gain-aware readout are selected and their J-lens directions projected out of the residual stream (“each” mode — sequential per-direction projection, the paper's literal wording; a directions probe measured the selected ten as near-orthogonal, effective rank 9.13/10, so “each” and “span” differ little). The band is the paper's ",
        { text: "light", italics: true },
        " band L38–54 on its 0–100 fractional-depth scale, mapped to layers 14–19 of Qwen3-4B's 36. Light rather than heavy because the paper reports that on its smallest model ablation degraded coherence before producing any qualitative change, and Qwen3-4B is far smaller still.",
      ]),
      rich([
        { text: "Exclusion rule. ", bold: true },
        "Following the paper, J-lens vectors of tokens in the top-10 of a clean forward pass on the same context are exempted at each position, so the intervention targets internal state rather than the readout of the token being emitted. This requires an interleaved clean forward pass at every generation step (≈ 2× cost on ablated cells). It is not optional: with the rule off, a damage-floor probe showed an 84% flip rate on SST-2 collapsing to 8% with it on — the rule-off “effect” was destruction of the answer readout itself.",
      ]),
      rich([
        { text: "Datasets and sampling. ", bold: true },
        "GSM8K test split, n = 150 by seeded shuffle prefix (samples nest across pilot and full runs by construction). AIME 2024, all 30 problems. MATH-500, n = 150 stratified 30 per difficulty level — note this makes cell accuracies level-balanced averages, not comparable to published split-weighted MATH-500 numbers. The layer band was selected on 12 GSM8K problems and transferred unchanged to the other datasets; on those datasets the transfer is an assumption, not a measurement.",
      ]),
      rich([
        { text: "Token caps. ", bold: true },
        "Per dataset and condition, calibrated on intact generations where possible: GSM8K cot 3072 / direct 128; MATH-500 cot 16,384 / direct 128 (a disclosed budget cap — ≈15% of level-5 intact CoT generations still hit it, per the stopping rule committed before the measurement was read); AIME cot 32,768 / direct 512 (an uncalibrated budget ceiling bounded by the checkpoint's 40,960-position limit). Generations that hit the cap score as “incomplete” (counted incorrect); compositions are reported per cell because a cap that binds asymmetrically would leak into the interaction.",
      ]),
      rich([
        { text: "Scoring. ", bold: true },
        "First \\boxed{} match extracted and checked with a pinned math-verify; the direct condition's prefill has already opened the box, so trailing junk cannot change a score.",
      ]),
      rich([
        { text: "Compute limitations. ", bold: true },
        "Runs were generated on a consumer Apple-silicon laptop (MPS) and rented CUDA pods, never mixing backends within a comparison. The AIME cot_ablated cell completed only 20 of 30 problems before the compute budget ran out; paired analyses use the 20 available pairs. The MATH-500 run is generated but not yet analyzed.",
      ]),

      // ---------------------------------------------------------- 4. results
      h1("4. Experimental results"),
      h2("GSM8K (n = 150)"),
      makeTable(cellCols, cellHeader, gsmRows),
      caption("Table 1. GSM8K cell accuracies and outcome composition (n = 150 per cell)."),
      makeTable(statCols, statHeader, gsmStats),
      caption("Table 2. GSM8K paired contrasts. Points in percentage points; CIs are 10,000-resample paired percentile bootstraps over problems; McNemar p-values are exact."),
      rich([
        "Ablation lowered CoT accuracy by 6.0 points (McNemar p = 0.035), but the matched random control lowered it by 4.0 points (p = 0.18): selectivity is +2.0 points with a 95% CI of [−3.3, +7.3]. The direct arm shows identical 2.7-point drops under ablation and control (selectivity exactly 0). The headline interaction is −3.3 points, CI [−10.7, +4.0] — the wrong sign for H1 and consistent with zero — and the control interaction is −1.3 points, CI [−7.3, +4.7], confirming the pipeline itself does not manufacture an interaction. The 12 problems the band was selected on are reported apart from the 138-problem holdout: they are measurably unrepresentative (direct_intact 50% vs the holdout's 25%) and show no ablation deficit at all (83% in all three CoT cells vs 80/74/76% on the holdout), so the headline estimates — which include them as 12 of 150 pairs — are, if anything, diluted rather than inflated by the band-selection set.",
      ]),
      h2("MATH-500 (n = 150, stratified 30/level)"),
      placeholder("[MATH-500 results to be added: cell table, contrast table, and per-level breakdown from analyze.py --dataset math500 once the run is scored. Also report the observed incomplete rate against the disclosed ~15% level-5 censoring expectation, and re-run power.py at the observed accuracies.]"),
      h2("AIME 2024 (n = 30; cot_ablated n = 20)"),
      makeTable(cellCols, cellHeader, aimeRows),
      caption("Table 3. AIME 2024 cell accuracies. *cot_ablated completed 20 of 30 problems (compute limit); its paired contrasts use those 20 pairs."),
      makeTable(statCols, statHeader, aimeStats),
      caption("Table 4. AIME 2024 paired contrasts. The direct arm is at floor (0%) in all cells, so its rows carry no information."),
      rich([
        "Every direct cell scored 0/30: at AIME difficulty Qwen3-4B cannot answer without reasoning, so the direct arm — and therefore the interaction — is uninformative by floor effect. In the CoT arm, ablation cost 11.7 points on the 20 paired problems while the random control ",
        { text: "gained", italics: true },
        " 6.7, giving a selectivity point estimate of +20.0 — the largest targeted deficit observed anywhere in the study — but with a CI of [+0.0, +40.0] (p = 0.115). The analysis tool's own pre-registered rule flags any CI wider than 25 points as unable to distinguish “no effect” from “too few problems”, and this one is 40 wide. Composition shifted toward incomplete under ablation (8/20 vs 5/30 intact), consistent with ablated traces failing to terminate.",
      ]),
      figure("fig1_accuracy.png", 6.5, 2.6),
      caption("Figure 1. Accuracy by cell. *AIME cot_ablated is n = 20 of 30 (compute limit). MATH-500 panel to be added."),
      figure("fig2_contrasts.png", 5.6, 4.2),
      caption("Figure 2. Pre-registered contrasts with 95% bootstrap CIs. H1 predicts a positive interaction; the random-control interaction should sit at zero. MATH-500 rows to be added."),

      // ---------------------------------------------------------- 5. analysis
      h1("5. Analysis of results"),
      rich([
        { text: "The substitution hypothesis is not supported. ", bold: true },
        "On GSM8K — the only dataset where both arms are off floor and fully powered — the interaction is slightly negative and its CI excludes the large positive effect H1 predicted: a substitution effect of ≥8 points would have been visible, and none was. The nominally significant CoT ablation drop (p = 0.035) does not survive the matched control; at this band and strength, projecting out the top-10 readout directions is not measurably worse than projecting out 10 random unembedding rows. We report this as an informative null for large effects, not merely an absence of evidence.",
      ]),
      rich([
        { text: "The difficulty trend runs opposite to substitution, but is inconclusive. ", bold: true },
        "The only suggestive targeted effect in the study is in the AIME CoT arm (+20 points selectivity), i.e., ablation biting harder exactly where reasoning is longest — the direction H2's complementary account predicts, not H1. Three things stop us from claiming it: the CI touches zero and is 40 points wide; the cell is missing its last 10 problems, and because problems complete in index order the missingness may correlate with difficulty; and the direct arm's floor removes the within-dataset interaction that would anchor the comparison. MATH-500, which sits between the two in difficulty and has both arms plausibly off floor, is the natural adjudicator.",
      ]),
      placeholder("[MATH-500 analysis to be added: does the CoT-arm selectivity grow monotonically GSM8K → MATH-500 → AIME? Is the interaction estimable with both arms off floor? Per-level breakdown as a within-dataset difficulty test of H2.]"),
      h2("Alternative explanations that remain open"),
      bullet([
        { text: "The J = I approximation may miss the real J-space. ", bold: true },
        "Our “J-lens” is the gain-aware logit lens, which reads what an activation is disposed to say ",
        { text: "now", italics: true },
        ", not the paper's Jacobian-averaged disposition. A directions probe found the gold answer is not linearly readable in the ablated band at all (gold@10 = 0% through layers 14–19, first reaching 100% at layer 32), so the top-10 readout there may be tokenizer noise rather than workspace content. A null under J = I does not falsify the paper's claim.",
      ]),
      bullet([
        { text: "Scale. ", bold: true },
        "The paper's clean effects are on frontier-scale models and it reports that its smallest model degraded in coherence before showing qualitative workspace effects. Qwen3-4B may simply be below the scale at which a verbalizable workspace is cleanly separable.",
      ]),
      bullet([
        { text: "Band granularity and placement. ", bold: true },
        "The paper's 16-point-deep light band maps to just 6 of 36 layers, and a fixed-width sliding-window sweep (14–19, 20–25, 26–31, plus full-width 14–33) run at the pre-registered settings found no window with selective damage — but all windows shared the band's coarseness. Strength was never escalated to the medium/heavy bands, deliberately, to avoid the incoherence confound.",
      ]),
      bullet([
        { text: "Cap asymmetries. ", bold: true },
        "Incomplete counts rise under both ablation and control (GSM8K CoT: 24 → 30/28), so part of every drop is non-termination rather than wrong reasoning. The caps are shared across states, and the control absorbs the same penalty, so this biases the selectivity estimates toward zero rather than manufacturing effects — but on AIME the 8/20 ablated incomplete rate is a real limitation.",
      ]),
      bullet([
        { text: "Control choice. ", bold: true },
        "Our control matches the operation (same k, band, projection) with randomized selection. The paper also uses matched-norm random perturbations; a norm-matched noise control (implemented but not run) could separate “removing any structured directions” from “removing this much signal”.",
      ]),
      h2("What we would test next"),
      bullet("Finish the AIME cot_ablated cell (10 problems) to remove the missingness caveat, and complete the MATH-500 analysis (placeholders above)."),
      bullet("A dose–response curve over band width (light → medium → heavy) with the coherence gate as the stopping rule, testing whether selectivity appears before degeneration does."),
      bullet([
        "The ",
        { text: "nothink", italics: true },
        " middle rung (thinking disabled, no answer-only instruction), pre-registered but unrun — it separates “no externalization” from “forced immediate answer”.",
      ]),
      bullet("Estimate actual per-layer Jacobians on a subset of prompts to test whether J ≠ I selects materially different directions in the workspace band — the direct test of the biggest approximation in this study."),
      bullet("Report observed power: re-run the power analysis at the measured accuracies and correlation, so the widths of the null CIs are themselves interpretable."),
      spacer(),
      rich([
        { text: "Reproducibility. ", bold: true },
        "All prompts, pre-registered constants, run manifests, and scored records are in the repository; analyze.py regenerates every number above from the committed .jsonl run files.",
      ], { alignment: AlignmentType.LEFT }),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(path.join(__dirname, "report.docx"), buf);
  console.log("wrote report.docx");
});
