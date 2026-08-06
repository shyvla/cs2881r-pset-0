"""Report figures, from the analyze.py output transcribed in data_analysis.md.

Numbers are HAND-TRANSCRIBED from analyze.py runs (see src/data_analysis.md)
rather than recomputed here, so this script needs only matplotlib -- it is not
part of the pinned experiment environment. When the MATH-500 analysis lands,
add its entry to DATA and CONTRASTS below and re-run.

Usage:  python make_figures.py [outdir]      (default: ../figures)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Categorical palette: first three slots of a CVD-validated 8-hue order
# (adjacent + all-pairs safe). intact/ablated/random keep these hues in every
# figure -- color follows the condition, never the panel.
C_INTACT, C_ABLATED, C_RANDOM = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#1a1a19", "#6b6a63", "#e5e4dd"

STATES = ("intact", "ablated", "random")

# dataset -> arm -> {state: (accuracy %, n)}
DATA = {
    "GSM8K (n=150)": {
        "CoT":    {"intact": (80.7, 150), "ablated": (74.7, 150), "random": (76.7, 150)},
        "Direct": {"intact": (27.3, 150), "ablated": (24.7, 150), "random": (24.7, 150)},
    },
    # MATH-500 goes here when analyzed: "MATH-500 (n=150, level-balanced)": {...},
    "AIME 2024 (n=30)": {
        "CoT":    {"intact": (66.7, 30), "ablated": (55.0, 20), "random": (73.3, 30)},
        "Direct": {"intact": (0.0, 30),  "ablated": (0.0, 30),  "random": (0.0, 30)},
    },
}

# Pre-registered contrasts: label -> (point, lo, hi) in percentage points.
# Selectivity = (intact-ablated) - (intact-random); positive = targeted
# ablation hurts more than matched random. Interaction = (direct drop) -
# (CoT drop); positive = ablation hurts direct answering more than CoT.
CONTRASTS = {
    "GSM8K": [
        ("Selectivity, CoT arm",        2.0,  -3.3,  7.3, C_ABLATED),
        ("Selectivity, direct arm",     0.0,  -5.3,  5.3, C_ABLATED),
        ("Interaction (ablation)",     -3.3, -10.7,  4.0, C_INTACT),
        ("Interaction (random ctrl)",  -1.3,  -7.3,  4.7, C_RANDOM),
    ],
    # "MATH-500": [...],
    "AIME 2024": [
        ("Selectivity, CoT arm",       20.0,   0.0, 40.0, C_ABLATED),
        ("Selectivity, direct arm",     0.0,   0.0,  0.0, C_ABLATED),
        ("Interaction (ablation)",    -20.0, -40.0,  0.0, C_INTACT),
        ("Interaction (random ctrl)",   0.0, -15.0, 15.0, C_RANDOM),
    ],
}


def style_ax(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK)


def fig_accuracy(outdir: Path):
    n_panels = len(DATA)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.0 * n_panels, 3.1),
                             sharey=True)
    if n_panels == 1:
        axes = [axes]
    colors = {"intact": C_INTACT, "ablated": C_ABLATED, "random": C_RANDOM}
    width, gap = 0.26, 0.02
    for ax, (ds, arms) in zip(axes, DATA.items()):
        style_ax(ax)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.8)
        for gi, (arm, cells) in enumerate(arms.items()):
            for si, state in enumerate(STATES):
                acc, n = cells[state]
                x = gi + (si - 1) * (width + gap)
                ax.bar(x, acc, width=width, color=colors[state], zorder=3)
                note = f"{acc:.0f}" if acc else "0"
                if n != max(c[1] for c in cells.values()):
                    note += f"*"
                ax.text(x, acc + 1.5, note, ha="center", va="bottom",
                        fontsize=8, color=INK)
        ax.set_xticks(range(len(arms)))
        ax.set_xticklabels(list(arms), fontsize=10)
        ax.set_title(ds, fontsize=10.5, color=INK, pad=8)
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Accuracy (%)", fontsize=10, color=INK)
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[s]) for s in STATES]
    fig.legend(handles, STATES, ncol=3, frameon=False, fontsize=9,
               loc="lower center", bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(outdir / "fig1_accuracy.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def fig_contrasts(outdir: Path):
    rows = []
    for ds, items in CONTRASTS.items():
        rows.append((ds, None))
        rows.extend((None, it) for it in items)
    fig, ax = plt.subplots(figsize=(6.8, 0.42 * len(rows) + 0.8))
    style_ax(ax)
    ax.axvline(0, color=MUTED, linewidth=1, zorder=2)
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    y = len(rows)
    yticks, ylabels = [], []
    for header, item in rows:
        y -= 1
        if header is not None:
            ax.text(-0.02, y, header, transform=ax.get_yaxis_transform(),
                    ha="right", va="center", fontsize=10, color=INK,
                    fontweight="bold")
            continue
        label, pt, lo, hi, color = item
        ax.plot([lo, hi], [y, y], color=color, linewidth=2,
                solid_capstyle="round", zorder=3)
        ax.plot(pt, y, "o", color=color, markersize=7,
                markeredgecolor="white", markeredgewidth=1.2, zorder=4)
        yticks.append(y)
        ylabels.append(label)
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=9)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("Percentage points (95% CI)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_contrasts.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    outdir = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path(__file__).resolve().parent.parent / "figures"
    outdir.mkdir(exist_ok=True)
    fig_accuracy(outdir)
    fig_contrasts(outdir)
    print(f"wrote {outdir}/fig1_accuracy.png and fig2_contrasts.png")
