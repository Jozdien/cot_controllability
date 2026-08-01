"""Generate final clean plots: 4 per model + cross-model comparisons.

Per model:
  1. All confirmed variants, aggregate (macro-avg across 8 modes)
  2. All confirmed variants, by mode
  3. Highlighted variants, aggregate
  4. Highlighted variants, by mode

Cross-model:
  - Aggregate comparison (highlights only)
  - By-mode comparison (highlights only)
"""

from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = [
    "#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]

MODE_ORDER_8 = [
    "lowercase_thinking", "word_suppression", "uppercase_thinking",
    "alternating_case", "repeat_sentences", "end_of_sentence",
    "meow_between_words", "multiple_word_suppression",
]


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def compute_stats(rows: list[dict]) -> dict[str, dict[str, dict]]:
    counts = defaultdict(lambda: defaultdict(lambda: {"k": 0, "n": 0}))
    for r in rows:
        v = r.get("variant", "")
        m = r.get("mode", "") or r.get("control_mode", "")
        c = r.get("compliant")
        if not v or not m or c is None:
            continue
        counts[v][m]["n"] += 1
        if c:
            counts[v][m]["k"] += 1
    stats = {}
    for v, modes in counts.items():
        stats[v] = {}
        for m, ct in modes.items():
            n, k = ct["n"], ct["k"]
            p = k / n if n > 0 else 0.0
            se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
            stats[v][m] = {"rate": p, "se": se, "n": n, "k": k}
    return stats


def macro_avg(mode_stats: dict[str, dict], modes: list[str]) -> tuple[float, float]:
    present = [mode_stats[m] for m in modes if m in mode_stats]
    if not present:
        return 0.0, 0.0
    avg = np.mean([s["rate"] for s in present])
    se = np.sqrt(np.sum([s["se"] ** 2 for s in present])) / len(present)
    return avg, se


def merge_stats(*sources: dict[str, dict[str, dict]]) -> dict[str, dict[str, dict]]:
    """Merge multiple stats dicts. Later sources override earlier for same variant."""
    merged: dict[str, dict[str, dict]] = {}
    for src in sources:
        for v, mstats in src.items():
            if v == "baseline":
                continue
            merged[v] = mstats
    # Use latest baseline
    for src in reversed(sources):
        if "baseline" in src:
            merged["baseline"] = src["baseline"]
            break
    return merged


def rename_stats(
    stats: dict[str, dict[str, dict]],
    rename_map: dict[str, str],
) -> dict[str, dict[str, dict]]:
    out = {}
    for v, mstats in stats.items():
        out[rename_map.get(v, v)] = mstats
    return out


def plot_aggregate(
    all_stats: dict[str, dict[str, dict]],
    label_order: list[str],
    modes: list[str],
    output_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (10, 5),
):
    vals, errs, labels = [], [], []
    for label in label_order:
        if label not in all_stats:
            continue
        avg, se = macro_avg(all_stats[label], modes)
        vals.append(avg * 100)
        errs.append(se * 100 * 1.96)
        labels.append(label)

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=figsize)
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(labels))]
    bars = ax.bar(x, vals, yerr=errs, capsize=4, color=bar_colors,
                  edgecolor="white", linewidth=0.5)
    for bar, err, val in zip(bars, errs, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.3,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Compliance (%)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    top = max(v + e for v, e in zip(vals, errs)) if vals else 1
    ax.set_ylim(0, top * 1.3 + 2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_by_mode(
    all_stats: dict[str, dict[str, dict]],
    label_order: list[str],
    modes: list[str],
    output_path: Path,
    title: str = "",
    figsize: tuple[float, float] | None = None,
):
    labels = [l for l in label_order if l in all_stats]
    n_labels = len(labels)
    width = 0.8 / n_labels
    if figsize is None:
        figsize = (max(14, len(modes) * 1.8), 6)
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(modes))
    for i, label in enumerate(labels):
        mode_stats = all_stats[label]
        vals = [mode_stats.get(m, {}).get("rate", 0.0) * 100 for m in modes]
        errs = [mode_stats.get(m, {}).get("se", 0.0) * 100 * 1.96 for m in modes]
        offset = (i - (n_labels - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs,
                      capsize=3, label=label, color=COLORS[i % len(COLORS)])
        for bar, err in zip(bars, errs):
            h = bar.get_height()
            if h > 0.5:
                ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=9)
    ax.set_ylabel("Compliance (%)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_cross_model_aggregate(
    all_models: dict[str, dict[str, dict[str, dict]]],
    variant_order: list[str],
    modes: list[str],
    output_path: Path,
    title: str = "",
):
    n_models = len(all_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5.5))
    if n_models == 1:
        axes = [axes]
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(variant_order))]

    for ax, (model_name, mdata) in zip(axes, all_models.items()):
        present = [v for v in variant_order if v in mdata]
        vals, errs, labels = [], [], []
        for v in present:
            avg, se = macro_avg(mdata[v], modes)
            vals.append(avg * 100)
            errs.append(se * 100 * 1.96)
            labels.append(v)
        x = np.arange(len(labels))
        ci = [bar_colors[variant_order.index(l)] for l in labels]
        bars = ax.bar(x, vals, yerr=errs, capsize=4, color=ci,
                      edgecolor="white", linewidth=0.5)
        for bar, err, val in zip(bars, errs, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(model_name, fontsize=13)
        ax.set_ylabel("Compliance (%)" if ax == axes[0] else "", fontsize=12)

    top = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(0, top * 1.15 + 2)
    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_cross_model_by_mode(
    all_models: dict[str, dict[str, dict[str, dict]]],
    variant_order: list[str],
    modes: list[str],
    output_path: Path,
    title: str = "",
    variant_colors: dict[str, str] | None = None,
):
    n_models = len(all_models)
    fig, axes = plt.subplots(n_models, 1, figsize=(22, 4.5 * n_models))
    if n_models == 1:
        axes = [axes]
    for ax, (model_name, mdata) in zip(axes, all_models.items()):
        present = [v for v in variant_order if v in mdata]
        n_labels = len(present)
        width = 0.8 / n_labels
        x = np.arange(len(modes))
        for i, v in enumerate(present):
            mode_stats = mdata[v]
            vals = [mode_stats.get(m, {}).get("rate", 0.0) * 100 for m in modes]
            errs = [mode_stats.get(m, {}).get("se", 0.0) * 100 * 1.96 for m in modes]
            offset = (i - (n_labels - 1) / 2) * width
            color = (variant_colors or {}).get(v, COLORS[i % len(COLORS)])
            bars = ax.bar(x + offset, vals, width, yerr=errs,
                          capsize=2, label=v, color=color)
            for bar, err in zip(bars, errs):
                h = bar.get_height()
                if h > 1.0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                            f"{h:.1f}%", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=9)
        ax.set_ylabel("Compliance (%)", fontsize=11)
        ax.set_title(model_name, fontsize=13)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


HIGHLIGHTS_COLORS = {
    "baseline":           "#8C8C8C",
    "zs: repeat 15x":    "#7BA3D4",
    "zs: metacognition":  "#3B6EA5",
    "15-shot cal (OOD)":  "#88CC88",
    "15-shot (\"on-policy\")":"#2D8E2D",
}


def plot_cross_model_grouped(
    all_models: dict[str, dict[str, dict[str, dict]]],
    variant_order: list[str],
    modes: list[str],
    output_path: Path,
    title: str = "",
    variant_colors: dict[str, str] | None = None,
):
    """Single grouped bar chart: all models side by side, bars colored by variant."""
    if variant_colors is None:
        variant_colors = HIGHLIGHTS_COLORS
    model_names = list(all_models.keys())
    n_models = len(model_names)
    n_variants = len(variant_order)
    width = 0.8 / n_variants
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, v in enumerate(variant_order):
        vals, errs = [], []
        for name in model_names:
            mdata = all_models[name]
            if v in mdata:
                avg, se = macro_avg(mdata[v], modes)
                vals.append(avg * 100)
                errs.append(se * 100 * 1.96)
            else:
                vals.append(0)
                errs.append(0)
        offset = (i - (n_variants - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                      color=variant_colors.get(v, COLORS[i % len(COLORS)]),
                      edgecolor="white", linewidth=0.5, label=v)
        for bar, err, val in zip(bars, errs, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Compliance (%)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    all_tops = []
    for name in model_names:
        for v in variant_order:
            mdata = all_models[name]
            if v in mdata:
                avg, se = macro_avg(mdata[v], modes)
                all_tops.append(avg * 100 + se * 100 * 1.96)
    ax.set_ylim(0, max(all_tops) * 1.3 + 2 if all_tops else 10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_bestof_all_models(
    models: list[str],
    baseline: list[tuple[float, float]],
    best_zs: list[tuple[float, float, str]],
    best_fs: list[tuple[float, float, str]],
    output_path: Path,
    title: str = "",
):
    """Single grouped bar chart: N models x 3 bars (baseline, best ZS, best FS)."""
    n = len(models)
    x = np.arange(n)
    width = 0.25
    cat_colors = ["#8C8C8C", "#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bl_vals = [v * 100 for v, _ in baseline]
    bl_errs = [s * 100 * 1.96 for _, s in baseline]
    zs_vals = [v * 100 for v, _, _ in best_zs]
    zs_errs = [s * 100 * 1.96 for _, s, _ in best_zs]
    fs_vals = [v * 100 for v, _, _ in best_fs]
    fs_errs = [s * 100 * 1.96 for _, s, _ in best_fs]

    bars_bl = ax.bar(x - width, bl_vals, width, yerr=bl_errs, capsize=4,
                     color=cat_colors[0], edgecolor="white", linewidth=0.5,
                     label="Baseline")
    bars_zs = ax.bar(x, zs_vals, width, yerr=zs_errs, capsize=4,
                     color=cat_colors[1], edgecolor="white", linewidth=0.5,
                     label="Best zero-shot")
    bars_fs = ax.bar(x + width, fs_vals, width, yerr=fs_errs, capsize=4,
                     color=cat_colors[2], edgecolor="white", linewidth=0.5,
                     label="Best few-shot")

    for bars, errs in [(bars_bl, bl_errs), (bars_zs, zs_errs), (bars_fs, fs_errs)]:
        for bar, err in zip(bars, errs):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Compliance (%)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.legend(fontsize=10, loc="upper left")

    all_tops = [h + e for h, e in zip(bl_vals + zs_vals + fs_vals,
                                       bl_errs + zs_errs + fs_errs)]
    ax.set_ylim(0, max(all_tops) * 1.3 + 2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_bestof_vs_metr(
    models: list[str],
    baseline: list[tuple[float, float]],
    best_zs: list[tuple[float, float, str]],
    best_fs: list[tuple[float, float, str]],
    metr_sft: list[float],
    output_path: Path,
    title: str = "",
    metr_label: str = "METR SFT",
):
    """Grouped bar chart: baseline, best ZS, best FS, METR SFT per model."""
    n = len(models)
    x = np.arange(n)
    width = 0.19
    cat_colors = ["#8C8C8C", "#4C72B0", "#55A868", "#E07B39"]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    bl_vals = [v * 100 for v, _ in baseline]
    bl_errs = [s * 100 * 1.96 for _, s in baseline]
    zs_vals = [v * 100 for v, _, _ in best_zs]
    zs_errs = [s * 100 * 1.96 for _, s, _ in best_zs]
    fs_vals = [v * 100 for v, _, _ in best_fs]
    fs_errs = [s * 100 * 1.96 for _, s, _ in best_fs]
    metr_vals = metr_sft

    bars_bl = ax.bar(x - 1.5 * width, bl_vals, width, yerr=bl_errs, capsize=3,
                     color=cat_colors[0], edgecolor="white", linewidth=0.5,
                     label="Baseline")
    bars_zs = ax.bar(x - 0.5 * width, zs_vals, width, yerr=zs_errs, capsize=3,
                     color=cat_colors[1], edgecolor="white", linewidth=0.5,
                     label="Best zero-shot")
    bars_fs = ax.bar(x + 0.5 * width, fs_vals, width, yerr=fs_errs, capsize=3,
                     color=cat_colors[2], edgecolor="white", linewidth=0.5,
                     label="Best few-shot")
    bars_metr = ax.bar(x + 1.5 * width, metr_vals, width, capsize=3,
                       color=cat_colors[3], edgecolor="white", linewidth=0.5,
                       label=metr_label)

    for bars, errs in [(bars_bl, bl_errs), (bars_zs, zs_errs), (bars_fs, fs_errs),
                       (bars_metr, [0] * n)]:
        for bar, err in zip(bars, errs):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Compliance (%)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="upper left")

    all_tops = (bl_vals + zs_vals + fs_vals + metr_vals +
                [h + e for h, e in zip(bl_vals + zs_vals + fs_vals,
                                       bl_errs + zs_errs + fs_errs)])
    ax.set_ylim(0, max(all_tops) * 1.3 + 2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ── Reasoning length helpers ──

RLEN_VARIANT_ORDER = [
    "baseline",
    "zs: repeat 15x", "zs: metacognition",
    "15-shot cal (OOD)", "15-shot (\"on-policy\")",
]

RLEN_COLORS = {
    "baseline":                "#8C8C8C",
    "zs: repeat 15x":         "#7BA3D4",
    "zs: metacognition":      "#3B6EA5",
    "15-shot cal (OOD)":      "#88CC88",
    "15-shot (\"on-policy\")": "#2D8E2D",
}


def load_reasoning_lengths(
    file_paths: list[Path],
    variant_rename: dict[str, str] | None = None,
    label: str | None = None,
) -> dict[str, dict[str, list[tuple[int, bool]]]]:
    """Load reasoning lengths from any JSONL file.

    Handles both prompt_ablation (reasoning_len field, variant/mode keys)
    and fewshot/hillclimb (reasoning text, control_mode key) formats.
    If label is set, all rows are grouped under that label (ignoring variant field).
    """
    data: dict[str, dict[str, list[tuple[int, bool]]]] = {}
    for p in file_paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rl = r.get("reasoning_len")
            if rl is None:
                reasoning = r.get("reasoning", "")
                rl = len(reasoning) if reasoning else 0
            if not rl or rl <= 0:
                continue
            mode = r.get("mode") or r.get("control_mode", "")
            if not mode:
                continue
            compliant = bool(r.get("compliant"))
            if label:
                display = label
            else:
                v = r.get("variant", "")
                if variant_rename and v not in variant_rename:
                    continue
                display = (variant_rename or {}).get(v, v)
            data.setdefault(display, {}).setdefault(mode, []).append((rl, compliant))
    return data


def _rlen_stats(lengths: list[int]) -> tuple[float, float]:
    """Mean and SE of reasoning lengths (in thousands of chars)."""
    if not lengths:
        return 0.0, 0.0
    arr = np.array(lengths) / 1000.0
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se


def plot_reasoning_length_grouped(
    all_models: dict[str, dict[str, dict[str, list[tuple[int, bool]]]]],
    variant_order: list[str],
    modes: list[str],
    output_path: Path,
    title: str = "",
    variant_colors: dict[str, str] | None = None,
    compliant_only: bool = False,
):
    """Single grouped bar chart of reasoning length: models on x-axis, bars colored by variant."""
    colors = variant_colors or RLEN_COLORS
    model_names = list(all_models.keys())
    n_models = len(model_names)
    n_variants = len(variant_order)
    width = 0.8 / n_variants
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, v in enumerate(variant_order):
        vals, errs = [], []
        for name in model_names:
            mdata = all_models.get(name, {})
            if v in mdata:
                all_lens = []
                for m in modes:
                    entries = mdata[v].get(m, [])
                    for rl, c in entries:
                        if compliant_only and not c:
                            continue
                        all_lens.append(rl)
                mean, se = _rlen_stats(all_lens)
                vals.append(mean)
                errs.append(se * 1.96)
            else:
                vals.append(0)
                errs.append(0)
        offset = (i - (n_variants - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                      color=colors.get(v, COLORS[i % len(COLORS)]),
                      edgecolor="white", linewidth=0.5, label=v)
        for bar, err, val in zip(bars, errs, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 0.1,
                        f"{val:.1f}k", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Reasoning Length (k chars)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    all_tops = [v + e for v, e in zip(
        [max((0,), default=0) for _ in model_names],
        [0] * n_models,
    )]
    for i, v in enumerate(variant_order):
        for j, name in enumerate(model_names):
            mdata = all_models.get(name, {})
            if v in mdata:
                all_lens = []
                for m in modes:
                    entries = mdata[v].get(m, [])
                    for rl, c in entries:
                        if compliant_only and not c:
                            continue
                        all_lens.append(rl)
                mean, se = _rlen_stats(all_lens)
                all_tops.append(mean + se * 1.96)
    ax.set_ylim(0, max(all_tops) * 1.2 + 0.5 if all_tops else 10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_reasoning_length_by_mode(
    all_models: dict[str, dict[str, dict[str, list[tuple[int, bool]]]]],
    variant_order: list[str],
    modes: list[str],
    output_path: Path,
    title: str = "",
    variant_colors: dict[str, str] | None = None,
    compliant_only: bool = False,
):
    """Cross-model by-mode reasoning length (one row per model)."""
    n_models = len(all_models)
    fig, axes = plt.subplots(n_models, 1, figsize=(22, 4.5 * n_models))
    if n_models == 1:
        axes = [axes]
    colors = variant_colors or RLEN_COLORS

    for ax, (model_name, mdata) in zip(axes, all_models.items()):
        present = [v for v in variant_order if v in mdata]
        n_labels = len(present)
        width = 0.8 / n_labels
        x = np.arange(len(modes))
        for i, v in enumerate(present):
            mode_data = mdata[v]
            vals, errs = [], []
            for m in modes:
                entries = mode_data.get(m, [])
                if compliant_only:
                    entries = [(rl, c) for rl, c in entries if c]
                lens = [rl for rl, _ in entries]
                mean, se = _rlen_stats(lens)
                vals.append(mean)
                errs.append(se * 1.96)
            offset = (i - (n_labels - 1) / 2) * width
            color = colors.get(v, COLORS[i % len(COLORS)])
            bars = ax.bar(x + offset, vals, width, yerr=errs,
                          capsize=2, label=v, color=color)
            for bar, err in zip(bars, errs):
                h = bar.get_height()
                if h > 0.2:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.05,
                            f"{h:.1f}k", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=9)
        ax.set_ylabel("Reasoning Length (k chars)", fontsize=11)
        ax.set_title(model_name, fontsize=13)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_reasoning_length_combined(
    all_models: dict[str, dict[str, dict[str, list[tuple[int, bool]]]]],
    variant_order: list[str],
    modes: list[str],
    output_path: Path,
    title: str = "",
    variant_colors: dict[str, str] | None = None,
):
    """Two subplots side by side: all responses (left) and compliant-only (right)."""
    colors = variant_colors or RLEN_COLORS
    model_names = list(all_models.keys())
    n_models = len(model_names)
    n_variants = len(variant_order)
    width = 0.8 / n_variants
    x = np.arange(n_models)

    fig, (ax_all, ax_comp) = plt.subplots(1, 2, figsize=(20, 5.5))

    for ax, compliant_only, subtitle in [
        (ax_all, False, "All Responses"),
        (ax_comp, True, "Compliant Only"),
    ]:
        top_val = 0
        for i, v in enumerate(variant_order):
            vals, errs = [], []
            for name in model_names:
                mdata = all_models.get(name, {})
                if v in mdata:
                    all_lens = []
                    for m in modes:
                        entries = mdata[v].get(m, [])
                        for rl, c in entries:
                            if compliant_only and not c:
                                continue
                            all_lens.append(rl)
                    mean, se = _rlen_stats(all_lens)
                    vals.append(mean)
                    errs.append(se * 1.96)
                    top_val = max(top_val, mean + se * 1.96)
                else:
                    vals.append(0)
                    errs.append(0)
            offset = (i - (n_variants - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                          color=colors.get(v, COLORS[i % len(COLORS)]),
                          edgecolor="white", linewidth=0.5, label=v)
            for bar, err, val in zip(bars, errs, vals):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + err + 0.1,
                            f"{val:.1f}k", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(model_names, fontsize=10)
        ax.set_ylabel("Reasoning Length (k chars)", fontsize=11)
        ax.set_title(subtitle, fontsize=12)
        ax.set_ylim(0, top_val * 1.25 + 0.5 if top_val else 10)

    ax_all.legend(fontsize=8, loc="upper left")
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def load_accuracy_by_variant(
    file_paths: list[Path],
    variant_rename: dict[str, str] | None = None,
    label: str | None = None,
) -> dict[str, dict]:
    """Load accuracy stats from JSONL files. Returns {variant: {k, n}}."""
    data: dict[str, dict] = {}
    for p in file_paths:
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if label:
                display = label
            else:
                v = r.get("variant", "")
                if variant_rename and v not in variant_rename:
                    continue
                display = (variant_rename or {}).get(v, v)
            correct = r.get("correct")
            if correct is None:
                continue
            entry = data.setdefault(display, {"k": 0, "n": 0})
            entry["n"] += 1
            if correct:
                entry["k"] += 1
    return data


def plot_accuracy_by_variant(
    all_models: dict[str, dict[str, dict]],
    variant_order: list[str],
    output_path: Path,
    title: str = "",
    variant_colors: dict[str, str] | None = None,
):
    """Grouped bar chart of accuracy: models on x-axis, bars per variant."""
    colors = variant_colors or RLEN_COLORS
    model_names = list(all_models.keys())
    present = [v for v in variant_order
               if any(v in all_models[m] for m in model_names)]
    n_variants = len(present)
    width = 0.8 / n_variants
    x = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, v in enumerate(present):
        vals, errs = [], []
        for name in model_names:
            d = all_models[name].get(v)
            if d and d["n"] > 0:
                rate = d["k"] / d["n"]
                se = np.sqrt(rate * (1 - rate) / d["n"])
                vals.append(rate * 100)
                errs.append(se * 100 * 1.96)
            else:
                vals.append(0)
                errs.append(0)
        offset = (i - (n_variants - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                      color=colors.get(v, COLORS[i % len(COLORS)]),
                      edgecolor="white", linewidth=0.5, label=v)
        for bar, err, val in zip(bars, errs, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, 85)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


_RLEN_ZERO_FLOOR = 0.003


def _apply_log_compliance_axis(ax):
    """Log y-axis for compliance rate that bottoms out at 0%."""
    ax.set_yscale("log")
    ax.set_ylim(_RLEN_ZERO_FLOOR * 0.6, 100)
    ticks = [100, 10, 1, 0.1, 0.01, _RLEN_ZERO_FLOOR]
    labels = ["100%", "10%", "1%", "0.1%", "0.01%", "0%"]
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.grid(axis="y", alpha=0.3, which="major")


def plot_compliance_vs_reasoning_length(
    all_models: dict[str, dict[str, dict[str, list[tuple[int, bool]]]]],
    variant_order: list[str],
    output_path: Path,
    title: str = "",
    variant_colors: dict[str, str] | None = None,
    n_bins: int = 12,
):
    """Line plot: compliance rate (y) vs reasoning length bucket (x).

    Pools all modes and all models together. One line per variant.
    """
    colors = variant_colors or RLEN_COLORS

    pooled: dict[str, list[tuple[int, bool]]] = {}
    for mdata in all_models.values():
        for variant, mode_data in mdata.items():
            for pairs in mode_data.values():
                pooled.setdefault(variant, []).extend(pairs)

    all_lens = [rl for pts in pooled.values() for rl, _ in pts]
    if not all_lens:
        return
    lo, hi = min(all_lens), np.percentile(all_lens, 97)
    edges = np.linspace(lo, hi, n_bins + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    present = [v for v in variant_order if v in pooled]
    for v in present:
        pts = pooled[v]
        xs, ys, sizes = [], [], []
        for i in range(len(edges) - 1):
            bucket = [(rl, c) for rl, c in pts if edges[i] <= rl < edges[i + 1]]
            if i == len(edges) - 2:
                bucket += [(rl, c) for rl, c in pts if rl >= edges[i + 1]]
            if len(bucket) < 5:
                continue
            mid = (edges[i] + edges[i + 1]) / 2 / 1000.0
            rate = sum(c for _, c in bucket) / len(bucket)
            xs.append(mid)
            ys.append(rate * 100)
            sizes.append(len(bucket))
        if xs:
            ys_plot = [max(y, _RLEN_ZERO_FLOOR) for y in ys]
            ax.plot(xs, ys_plot, "o-", color=colors.get(v, "#333"),
                    label=v, markersize=5, linewidth=1.8)

    ax.set_xlabel("Reasoning Length (k chars)")
    ax.set_ylabel("Compliance Rate (%)")
    ax.set_title(title or "Compliance Rate vs Reasoning Length\n(all models pooled)")
    ax.legend(fontsize=8)
    _apply_log_compliance_axis(ax)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_compliance_vs_reasoning_length_per_model(
    all_models: dict[str, dict[str, dict[str, list[tuple[int, bool]]]]],
    variant_order: list[str],
    output_path: Path,
    title: str = "",
    variant_colors: dict[str, str] | None = None,
    n_bins: int = 10,
):
    """One subplot per model: compliance rate vs reasoning length."""
    colors = variant_colors or RLEN_COLORS
    model_names = list(all_models.keys())
    n_models = len(model_names)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        mdata = all_models[model_name]
        pooled: dict[str, list[tuple[int, bool]]] = {}
        for variant, mode_data in mdata.items():
            for pairs in mode_data.values():
                pooled.setdefault(variant, []).extend(pairs)

        all_lens = [rl for pts in pooled.values() for rl, _ in pts]
        if not all_lens:
            continue
        lo, hi = min(all_lens), np.percentile(all_lens, 97)
        edges = np.linspace(lo, hi, n_bins + 1)

        present = [v for v in variant_order if v in pooled]
        for v in present:
            pts = pooled[v]
            xs, ys = [], []
            for i in range(len(edges) - 1):
                bucket = [(rl, c) for rl, c in pts if edges[i] <= rl < edges[i + 1]]
                if i == len(edges) - 2:
                    bucket += [(rl, c) for rl, c in pts if rl >= edges[i + 1]]
                if len(bucket) < 5:
                    continue
                mid = (edges[i] + edges[i + 1]) / 2 / 1000.0
                rate = sum(c for _, c in bucket) / len(bucket)
                xs.append(mid)
                ys.append(rate * 100)
            if xs:
                ys_plot = [max(y, _RLEN_ZERO_FLOOR) for y in ys]
                ax.plot(xs, ys_plot, "o-", color=colors.get(v, "#333"),
                        label=v, markersize=4, linewidth=1.5)

        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("Reasoning Length (k chars)")
        _apply_log_compliance_axis(ax)

    axes[0].set_ylabel("Compliance Rate (%)")
    axes[-1].legend(fontsize=7, loc="upper right")
    fig.suptitle(title or "Compliance Rate vs Reasoning Length", fontsize=13, y=1.02)
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ── Data loading ──

RESULTS = Path("results")

# Display name mappings
QWEN_RENAME = {
    "calibration_shot": "1-shot cal",
    "cal_2shot": "2-shot cal",
    "cal_5shot_system": "5-shot cal (OOD)",
    "cal_10shot_system": "10-shot cal (OOD)",
    "cal_15shot_system": "15-shot cal (OOD)",
    "combo_render_selfcheck": "render+selfcheck",
    "combo_bounty_selfcheck": "bounty+selfcheck",
    "combo_all_three": "render+selfcheck+bounty",
    "combo_twopass_example": "twopass+example",
}

GPTOSS_RENAME = {
    "self_check": "self_check",
    "render_mode": "render_mode",
    "combo_render_selfcheck": "render+selfcheck",
    "combo_bounty_selfcheck": "bounty+selfcheck",
    "cal_2shot": "2-shot cal",
    "cal_3shot": "3-shot cal",
    "cal_3shot_strong": "3-shot cal (strong)",
    "cal_5shot_system": "5-shot cal (OOD)",
    "cal_10shot_system": "10-shot cal (OOD)",
    "cal_15shot_system": "15-shot cal (OOD)",
}


def compute_fewshot_stats(
    rows: list[dict],
    label: str,
) -> dict[str, dict[str, dict]]:
    """Compute stats from fewshot eval files (keyed by control_mode, not variant)."""
    counts = defaultdict(lambda: {"k": 0, "n": 0})
    for r in rows:
        m = r.get("control_mode", "")
        c = r.get("compliant")
        if not m or c is None:
            continue
        counts[m]["n"] += 1
        if c:
            counts[m]["k"] += 1
    stats = {}
    for m, ct in counts.items():
        n, k = ct["n"], ct["k"]
        p = k / n if n > 0 else 0.0
        se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        stats[m] = {"rate": p, "se": se, "n": n, "k": k}
    return {label: stats}


def load_fewshot_data(
    file_paths: list[Path],
    label: str,
) -> dict[str, dict[str, dict]]:
    """Load and merge multiple fewshot eval files under a single label."""
    all_rows = []
    for p in file_paths:
        if p.exists():
            all_rows.extend(load_jsonl(p))
    if not all_rows:
        return {}
    return compute_fewshot_stats(all_rows, label)


def load_model_data(
    file_paths: list[Path],
    rename_map: dict[str, str],
) -> dict[str, dict[str, dict]]:
    all_sources = []
    for p in file_paths:
        if p.exists():
            all_sources.append(rename_stats(compute_stats(load_jsonl(p)), rename_map))
    if not all_sources:
        return {}
    return merge_stats(*all_sources)


def _hc_stats(rates: list[float], n: int = 50) -> dict[str, dict]:
    """Build mode stats from 8 compliance rates.

    Order: lowercase, word_supp, multi_word, uppercase, alt_case,
           end_sent, repeat_sent, meow.
    """
    modes = [
        "lowercase_thinking", "word_suppression", "multiple_word_suppression",
        "uppercase_thinking", "alternating_case", "end_of_sentence",
        "repeat_sentences", "meow_between_words",
    ]
    stats = {}
    for m, rate in zip(modes, rates):
        k = round(rate * n)
        p = k / n
        se = np.sqrt(p * (1 - p) / n)
        stats[m] = {"rate": p, "se": se, "n": n, "k": k}
    return stats


def main():
    out = RESULTS / "plots" / "main"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Qwen3-32B ──
    q32b_data = load_model_data([
        RESULTS / "prompt_ablation_r7_zs_32b" / "prompt_ablation_qwen3-32b.jsonl",
        RESULTS / "prompt_ablation_r7_cal_32b" / "prompt_ablation_qwen3-32b.jsonl",
        RESULTS / "prompt_ablation_r8_confirm" / "prompt_ablation_qwen3-32b.jsonl",
        RESULTS / "prompt_ablation_r9_confirm" / "prompt_ablation_qwen3-32b.jsonl",
        RESULTS / "prompt_ablation_cal_qwen32b" / "prompt_ablation_qwen3-32b.jsonl",
    ], QWEN_RENAME)

    # Add fewshot 15-shot data (real dataset examples with transformed reasoning)
    q32b_fewshot_0 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-32b_0shot.jsonl",
        RESULTS / "fewshot_tinker_0shot_newmodes" / "eval_qwen3-32b_0shot.jsonl",
    ], "fewshot 0-shot")
    q32b_fewshot_15 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-32b_15shot.jsonl",
    ], "fewshot 15-shot")
    q32b_data.update(q32b_fewshot_0)
    q32b_data.update(q32b_fewshot_15)

    # Hillclimb zero-shot (from hillclimb_zeroshot.py --batch qwen)
    q32b_data["zs: repeat 15x"] = _hc_stats([0.16, 0.24, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00])
    q32b_data["zs: metacognition"] = _hc_stats([0.26, 0.16, 0.08, 0.00, 0.00, 0.00, 0.00, 0.00])

    q32b_all_order = [
        "baseline", "repeated", "two_pass", "token_bounty",
        "1-shot cal", "2-shot cal",
        "15-shot cal (OOD)",
        "render+selfcheck", "bounty+selfcheck",
        "twopass+example", "render+selfcheck+bounty",
        "fewshot 15-shot",
        "zs: repeat 15x", "zs: metacognition",
    ]
    q32b_highlights = [
        "baseline",
        "2-shot cal", "15-shot cal (OOD)",
        "fewshot 15-shot",
        "zs: repeat 15x", "zs: metacognition",
    ]

    if q32b_data:
        plot_aggregate(q32b_data, q32b_all_order, MODE_ORDER_8,
                       out / "qwen3_32b_all_aggregate.png",
                       "Qwen3-32B: All Confirmed Variants\nmacro-avg across 8 modes, 95% CI",
                       figsize=(14, 5))
        plot_by_mode(q32b_data, q32b_all_order, MODE_ORDER_8,
                     out / "qwen3_32b_all_by_mode.png",
                     "Qwen3-32B: All Confirmed Variants Per-Mode",
                     figsize=(22, 7))
        plot_aggregate(q32b_data, q32b_highlights, MODE_ORDER_8,
                       out / "qwen3_32b_highlights_aggregate.png",
                       "Qwen3-32B: Best Variants\nmacro-avg across 8 modes, 95% CI")
        plot_by_mode(q32b_data, q32b_highlights, MODE_ORDER_8,
                     out / "qwen3_32b_highlights_by_mode.png",
                     "Qwen3-32B: Best Variants Per-Mode")

    # ── Qwen3-8B ──
    q8b_data = load_model_data([
        RESULTS / "prompt_ablation_r7_zs_8b" / "prompt_ablation_qwen3-8b.jsonl",
        RESULTS / "prompt_ablation_r7_cal_8b" / "prompt_ablation_qwen3-8b.jsonl",
        RESULTS / "prompt_ablation_r9_8b" / "prompt_ablation_qwen3-8b.jsonl",
        RESULTS / "prompt_ablation_cal_qwen8b" / "prompt_ablation_qwen3-8b.jsonl",
    ], QWEN_RENAME)

    q8b_fewshot_0 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-8b_0shot.jsonl",
        RESULTS / "fewshot_tinker_0shot_newmodes" / "eval_qwen3-8b_0shot.jsonl",
    ], "fewshot 0-shot")
    q8b_fewshot_15 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-8b_15shot.jsonl",
    ], "fewshot 15-shot")
    q8b_data.update(q8b_fewshot_0)
    q8b_data.update(q8b_fewshot_15)

    # Hillclimb zero-shot (from hillclimb_zeroshot.py --batch qwen)
    q8b_data["zs: repeat 15x"] = _hc_stats([0.10, 0.12, 0.06, 0.00, 0.00, 0.00, 0.00, 0.02])
    q8b_data["zs: metacognition"] = _hc_stats([0.02, 0.08, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00])

    q8b_all_order = [
        "baseline", "repeated", "token_bounty",
        "1-shot cal", "2-shot cal",
        "15-shot cal (OOD)",
        "render+selfcheck", "bounty+selfcheck",
        "render+selfcheck+bounty",
        "fewshot 15-shot",
        "zs: repeat 15x", "zs: metacognition",
    ]
    q8b_highlights = [
        "baseline",
        "2-shot cal", "15-shot cal (OOD)",
        "fewshot 15-shot",
        "zs: repeat 15x", "zs: metacognition",
    ]

    if q8b_data:
        plot_aggregate(q8b_data, q8b_all_order, MODE_ORDER_8,
                       out / "qwen3_8b_all_aggregate.png",
                       "Qwen3-8B: All Confirmed Variants\nmacro-avg across 8 modes, 95% CI",
                       figsize=(14, 5))
        plot_by_mode(q8b_data, q8b_all_order, MODE_ORDER_8,
                     out / "qwen3_8b_all_by_mode.png",
                     "Qwen3-8B: All Confirmed Variants Per-Mode",
                     figsize=(22, 7))
        plot_aggregate(q8b_data, q8b_highlights, MODE_ORDER_8,
                       out / "qwen3_8b_highlights_aggregate.png",
                       "Qwen3-8B: Best Variants\nmacro-avg across 8 modes, 95% CI")
        plot_by_mode(q8b_data, q8b_highlights, MODE_ORDER_8,
                     out / "qwen3_8b_highlights_by_mode.png",
                     "Qwen3-8B: Best Variants Per-Mode")

    # ── Qwen calibration comparison (8B vs 32B) ──
    qwen_cal_order = [
        "baseline", "2-shot cal", "15-shot cal (OOD)",
    ]
    qwen_cal_models = {}
    if q8b_data and "15-shot cal (OOD)" in q8b_data:
        qwen_cal_models["Qwen3-8B"] = q8b_data
    if q32b_data and "15-shot cal (OOD)" in q32b_data:
        qwen_cal_models["Qwen3-32B"] = q32b_data
    if qwen_cal_models:
        plot_cross_model_aggregate(
            qwen_cal_models, qwen_cal_order, MODE_ORDER_8,
            out / "qwen_cal_comparison_aggregate.png",
            "Qwen Calibration: 8B vs 32B\n(macro-avg across 8 modes, 95% CI)",
        )
        plot_cross_model_by_mode(
            qwen_cal_models, qwen_cal_order, MODE_ORDER_8,
            out / "qwen_cal_comparison_by_mode.png",
            "Qwen Calibration Per-Mode: 8B vs 32B",
        )

    # ── GPT-OSS-20B ──
    gptoss20b_data = load_model_data([
        RESULTS / "gptoss_gpt-oss-20b" / "prompt_ablation_gpt-oss-20b.jsonl",
        RESULTS / "gptoss_r3_confirm_20b" / "prompt_ablation_gpt-oss-20b.jsonl",
        RESULTS / "gptoss_r4_confirm_20b" / "prompt_ablation_gpt-oss-20b.jsonl",
    ], GPTOSS_RENAME)

    gptoss20b_fewshot_0 = load_fewshot_data([
        RESULTS / "fewshot_gptoss_20b" / "eval_gpt-oss-20b_0shot.jsonl",
    ], "fewshot 0-shot")
    gptoss20b_fewshot_15 = load_fewshot_data([
        RESULTS / "fewshot_gptoss_20b" / "eval_gpt-oss-20b_15shot.jsonl",
    ], "fewshot 15-shot")
    gptoss20b_data.update(gptoss20b_fewshot_0)
    gptoss20b_data.update(gptoss20b_fewshot_15)

    gptoss_all_order = [
        "baseline", "render_mode",
        "render+selfcheck", "bounty+selfcheck",
        "2-shot cal", "3-shot cal", "3-shot cal (strong)",
        "15-shot cal (OOD)",
        "fewshot 15-shot",
        "zs: repeat 15x", "zs: metacognition",
    ]
    gptoss_highlights = [
        "baseline", "15-shot cal (OOD)",
        "fewshot 15-shot",
        "zs: repeat 15x", "zs: metacognition",
    ]

    if gptoss20b_data:
        plot_aggregate(gptoss20b_data, gptoss_all_order, MODE_ORDER_8,
                       out / "gptoss_20b_all_aggregate.png",
                       "GPT-OSS-20B: All Confirmed Variants\nmacro-avg across 8 modes, 95% CI",
                       figsize=(16, 5))
        plot_by_mode(gptoss20b_data, gptoss_all_order, MODE_ORDER_8,
                     out / "gptoss_20b_all_by_mode.png",
                     "GPT-OSS-20B: All Confirmed Variants Per-Mode",
                     figsize=(24, 7))
        plot_aggregate(gptoss20b_data, gptoss_highlights, MODE_ORDER_8,
                       out / "gptoss_20b_highlights_aggregate.png",
                       "GPT-OSS-20B: Best Variants\nmacro-avg across 8 modes, 95% CI")
        plot_by_mode(gptoss20b_data, gptoss_highlights, MODE_ORDER_8,
                     out / "gptoss_20b_highlights_by_mode.png",
                     "GPT-OSS-20B: Best Variants Per-Mode")

    # ── GPT-OSS-120B ──
    gptoss120b_data = load_model_data([
        RESULTS / "gptoss_gpt-oss-120b" / "prompt_ablation_gpt-oss-120b.jsonl",
        RESULTS / "gptoss_r3_confirm_120b" / "prompt_ablation_gpt-oss-120b.jsonl",
        RESULTS / "gptoss_r4_confirm_120b" / "prompt_ablation_gpt-oss-120b.jsonl",
    ], GPTOSS_RENAME)

    gptoss120b_fewshot_0 = load_fewshot_data([
        RESULTS / "fewshot_gptoss_120b" / "eval_gpt-oss-120b_0shot.jsonl",
    ], "fewshot 0-shot")
    gptoss120b_fewshot_15 = load_fewshot_data([
        RESULTS / "fewshot_gptoss_120b" / "eval_gpt-oss-120b_15shot.jsonl",
    ], "fewshot 15-shot")
    gptoss120b_data.update(gptoss120b_fewshot_0)
    gptoss120b_data.update(gptoss120b_fewshot_15)

    # ── Hillclimb zero-shot prompt engineering ──
    # 50 samples × 8 modes, from hillclimb_zeroshot.py bestof batch
    # Rates: [lowercase, word_supp, multi_word, uppercase, alt_case, end_sent, repeat_sent, meow]

    hc_20b = {
        "baseline":       _hc_stats([0.04, 0.22, 0.10, 0.02, 0.00, 0.00, 0.14, 0.00]),
        "repeat 5x":      _hc_stats([0.16, 0.30, 0.12, 0.26, 0.00, 0.00, 0.20, 0.00]),
        "repeat 10x":     _hc_stats([0.20, 0.28, 0.18, 0.34, 0.00, 0.08, 0.20, 0.00]),
        "repeat 15x":     _hc_stats([0.16, 0.38, 0.20, 0.34, 0.00, 0.02, 0.20, 0.02]),
        "repeat 20x":     _hc_stats([0.16, 0.24, 0.26, 0.32, 0.00, 0.00, 0.20, 0.00]),
        "extreme repeat": _hc_stats([0.22, 0.24, 0.24, 0.28, 0.00, 0.00, 0.20, 0.00]),
        "metacognition":  _hc_stats([0.16, 0.14, 0.12, 0.16, 0.00, 0.00, 0.40, 0.00]),
        "step-by-step":   _hc_stats([0.16, 0.18, 0.08, 0.36, 0.00, 0.04, 0.42, 0.02]),
        "sandwich":       _hc_stats([0.06, 0.20, 0.08, 0.24, 0.02, 0.00, 0.10, 0.02]),
        "promise first":  _hc_stats([0.00, 0.06, 0.06, 0.20, 0.00, 0.14, 0.00, 0.00]),
        "priority inv":   _hc_stats([0.10, 0.12, 0.06, 0.08, 0.00, 0.00, 0.08, 0.14]),
        "physical law":   _hc_stats([0.00, 0.14, 0.10, 0.08, 0.00, 0.00, 0.12, 0.00]),
        "sys+sandwich":   _hc_stats([0.10, 0.18, 0.10, 0.10, 0.02, 0.00, 0.06, 0.00]),
    }

    hc_120b = {
        "baseline":       _hc_stats([0.00, 0.20, 0.06, 0.10, 0.02, 0.00, 0.06, 0.02]),
        "repeat 5x":      _hc_stats([0.02, 0.34, 0.16, 0.04, 0.02, 0.00, 0.06, 0.00]),
        "repeat 10x":     _hc_stats([0.06, 0.36, 0.12, 0.10, 0.02, 0.02, 0.08, 0.04]),
        "repeat 15x":     _hc_stats([0.02, 0.32, 0.10, 0.06, 0.00, 0.02, 0.08, 0.00]),
        "repeat 20x":     _hc_stats([0.02, 0.20, 0.06, 0.10, 0.02, 0.04, 0.16, 0.00]),
        "extreme repeat": _hc_stats([0.16, 0.30, 0.16, 0.16, 0.02, 0.00, 0.14, 0.00]),
        "metacognition":  _hc_stats([0.18, 0.22, 0.06, 0.18, 0.06, 0.02, 0.82, 0.02]),
        "step-by-step":   _hc_stats([0.18, 0.14, 0.08, 0.18, 0.02, 0.14, 0.14, 0.04]),
        "sandwich":       _hc_stats([0.06, 0.20, 0.06, 0.02, 0.02, 0.00, 0.20, 0.08]),
        "promise first":  _hc_stats([0.12, 0.08, 0.06, 0.18, 0.02, 0.16, 0.00, 0.02]),
        "physical law":   _hc_stats([0.04, 0.18, 0.08, 0.00, 0.02, 0.00, 0.02, 0.04]),
        "priority inv":   _hc_stats([0.02, 0.06, 0.06, 0.14, 0.06, 0.00, 0.04, 0.04]),
        "sys+sandwich":   _hc_stats([0.04, 0.20, 0.08, 0.04, 0.02, 0.02, 0.14, 0.02]),
    }

    # Add best hillclimb variants to existing GPT-OSS data for combined plots
    gptoss20b_data["zs: repeat 15x"] = hc_20b["repeat 15x"]
    gptoss20b_data["zs: metacognition"] = hc_20b["metacognition"]
    gptoss120b_data["zs: repeat 15x"] = hc_120b["repeat 15x"]
    gptoss120b_data["zs: metacognition"] = hc_120b["metacognition"]

    if gptoss120b_data:
        plot_aggregate(gptoss120b_data, gptoss_all_order, MODE_ORDER_8,
                       out / "gptoss_120b_all_aggregate.png",
                       "GPT-OSS-120B: All Confirmed Variants\nmacro-avg across 8 modes, 95% CI",
                       figsize=(16, 5))
        plot_by_mode(gptoss120b_data, gptoss_all_order, MODE_ORDER_8,
                     out / "gptoss_120b_all_by_mode.png",
                     "GPT-OSS-120B: All Confirmed Variants Per-Mode",
                     figsize=(24, 7))
        plot_aggregate(gptoss120b_data, gptoss_highlights, MODE_ORDER_8,
                       out / "gptoss_120b_highlights_aggregate.png",
                       "GPT-OSS-120B: Best Variants\nmacro-avg across 8 modes, 95% CI")
        plot_by_mode(gptoss120b_data, gptoss_highlights, MODE_ORDER_8,
                     out / "gptoss_120b_highlights_by_mode.png",
                     "GPT-OSS-120B: Best Variants Per-Mode")

    # ── Hillclimb standalone plots ──
    hc_all_order = [
        "repeat 15x", "repeat 10x", "step-by-step", "repeat 20x",
        "extreme repeat", "repeat 5x", "metacognition", "sandwich",
        "priority inv", "sys+sandwich", "baseline", "promise first",
        "physical law",
    ]
    hc_top6 = [
        "baseline", "repeat 10x", "repeat 15x",
        "extreme repeat", "metacognition", "step-by-step",
    ]

    if hc_20b:
        plot_aggregate(hc_20b, hc_all_order, MODE_ORDER_8,
                       out / "gptoss_20b_hc_aggregate.png",
                       "GPT-OSS-20B: Zero-Shot Hillclimb Variants\nmacro-avg across 8 modes, 95% CI",
                       figsize=(16, 5))
        plot_by_mode(hc_20b, hc_top6, MODE_ORDER_8,
                     out / "gptoss_20b_hc_by_mode.png",
                     "GPT-OSS-20B: Zero-Shot Hillclimb Top Variants Per-Mode")

    if hc_120b:
        hc_120b_order = [
            "metacognition", "extreme repeat", "step-by-step", "repeat 10x",
            "sandwich", "promise first", "repeat 5x", "repeat 15x",
            "repeat 20x", "sys+sandwich", "baseline", "priority inv",
            "physical law",
        ]
        plot_aggregate(hc_120b, hc_120b_order, MODE_ORDER_8,
                       out / "gptoss_120b_hc_aggregate.png",
                       "GPT-OSS-120B: Zero-Shot Hillclimb Variants\nmacro-avg across 8 modes, 95% CI",
                       figsize=(16, 5))
        hc_120b_top6 = [
            "baseline", "repeat 10x", "extreme repeat",
            "metacognition", "step-by-step", "repeat 5x",
        ]
        plot_by_mode(hc_120b, hc_120b_top6, MODE_ORDER_8,
                     out / "gptoss_120b_hc_by_mode.png",
                     "GPT-OSS-120B: Zero-Shot Hillclimb Top Variants Per-Mode")

    # Hillclimb cross-model comparison
    if hc_20b and hc_120b:
        hc_cross = {
            "GPT-OSS-20B": hc_20b,
            "GPT-OSS-120B": hc_120b,
        }
        hc_cross_order = [
            "baseline", "repeat 10x", "repeat 15x",
            "metacognition", "step-by-step", "extreme repeat",
        ]
        plot_cross_model_aggregate(
            hc_cross, hc_cross_order, MODE_ORDER_8,
            out / "hc_cross_model_aggregate.png",
            "Zero-Shot Hillclimb: GPT-OSS-20B vs 120B\n(macro-avg across 8 modes, 95% CI)",
        )
        plot_cross_model_by_mode(
            hc_cross, hc_cross_order, MODE_ORDER_8,
            out / "hc_cross_model_by_mode.png",
            "Zero-Shot Hillclimb Per-Mode: GPT-OSS-20B vs 120B",
        )

    # ── GPT-OSS best-of-each summary ──
    # Per-model: baseline, best zero-shot, best few-shot
    gptoss_bestof_20b = {
        "baseline": gptoss20b_data["baseline"],
        "best zs: repeat 15x": hc_20b["repeat 15x"],
        "best fs: 15-shot": gptoss20b_data["fewshot 15-shot"],
    }
    gptoss_bestof_120b = {
        "baseline": gptoss120b_data["baseline"],
        "best zs: metacognition": hc_120b["metacognition"],
        "best fs: 15-shot": gptoss120b_data["fewshot 15-shot"],
    }
    bestof_order = ["baseline", "best zs: repeat 15x", "best zs: metacognition", "best fs: 15-shot"]
    bestof_cross = {
        "GPT-OSS-20B": gptoss_bestof_20b,
        "GPT-OSS-120B": gptoss_bestof_120b,
    }
    plot_cross_model_aggregate(
        bestof_cross, bestof_order, MODE_ORDER_8,
        out / "gptoss_bestof_aggregate.png",
        "GPT-OSS: Baseline vs Best Zero-Shot vs Best Few-Shot\n(macro-avg across 8 modes, 95% CI)",
    )
    plot_cross_model_by_mode(
        bestof_cross, bestof_order, MODE_ORDER_8,
        out / "gptoss_bestof_by_mode.png",
        "GPT-OSS: Baseline vs Best Zero-Shot vs Best Few-Shot (Per-Mode)",
    )

    # ── All models: baseline vs best ZS vs best FS ──
    def _find_best(mdata):
        bl_avg, bl_se = macro_avg(mdata.get("baseline", {}), MODE_ORDER_8)
        best_zs = (0.0, 0.0, "—")
        best_fs = (0.0, 0.0, "—")
        for v, stats in mdata.items():
            avg, se = macro_avg(stats, MODE_ORDER_8)
            if v.startswith("zs:") and avg > best_zs[0]:
                best_zs = (avg, se, v)
            elif ("shot" in v or "cal" in v) and v not in ("baseline", "fewshot 0-shot"):
                if avg > best_fs[0]:
                    best_fs = (avg, se, v)
        return (bl_avg, bl_se), best_zs, best_fs

    bestof_models, bestof_bl, bestof_zs, bestof_fs = [], [], [], []
    for name, mdata in [("Qwen3-8B", q8b_data), ("Qwen3-32B", q32b_data),
                         ("GPT-OSS-20B", gptoss20b_data), ("GPT-OSS-120B", gptoss120b_data)]:
        if not mdata:
            continue
        bl, zs, fs = _find_best(mdata)
        bestof_models.append(name)
        bestof_bl.append(bl)
        bestof_zs.append(zs)
        bestof_fs.append(fs)

    if bestof_models:
        plot_bestof_all_models(
            bestof_models, bestof_bl, bestof_zs, bestof_fs,
            out / "all_models_bestof_summary.png",
            "All Models: Baseline vs Best Zero-Shot vs Best Few-Shot\n(macro-avg across 8 modes, 95% CI)",
        )

    # ── Cross-model comparisons ──
    # Add display-name aliases for few-shot variants
    for mdata in [q32b_data, q8b_data, gptoss20b_data, gptoss120b_data]:
        if mdata and "2-shot cal" in mdata:
            mdata["2-shot (OOD)"] = mdata["2-shot cal"]
        if mdata and "fewshot 15-shot" in mdata:
            mdata["15-shot (\"on-policy\")"] = mdata["fewshot 15-shot"]

    cross_highlights_order = [
        "baseline",
        "zs: repeat 15x", "zs: metacognition",
        "15-shot cal (OOD)",
        "15-shot (\"on-policy\")",
    ]

    all_models = {}
    if q8b_data:
        all_models["Qwen3-8B"] = q8b_data
    if q32b_data:
        all_models["Qwen3-32B"] = q32b_data
    if gptoss20b_data:
        all_models["GPT-OSS-20B"] = gptoss20b_data
    if gptoss120b_data:
        all_models["GPT-OSS-120B"] = gptoss120b_data

    if len(all_models) >= 2:
        plot_cross_model_aggregate(
            all_models, cross_highlights_order, MODE_ORDER_8,
            out / "cross_model_highlights_aggregate.png",
            "Aggregate Compliance: Best Variants Across Models\n(macro-avg across 8 modes, 95% CI)",
        )
        plot_cross_model_by_mode(
            all_models, cross_highlights_order, MODE_ORDER_8,
            out / "cross_model_highlights_by_mode.png",
            "Per-Mode Compliance: Best Variants Across Models",
            variant_colors=HIGHLIGHTS_COLORS,
        )
        plot_cross_model_grouped(
            all_models, cross_highlights_order, MODE_ORDER_8,
            out / "cross_model_highlights_grouped.png",
            "Aggregate Compliance: Best Variants Across Models\n(macro-avg across 8 modes, 95% CI)",
        )

    # ── Shot-scaling comparison (all models with cal data) ──
    scaling_models = {}
    scaling_order = [
        "baseline", "15-shot cal (OOD)",
    ]
    if q8b_data and "15-shot cal (OOD)" in q8b_data:
        scaling_models["Qwen3-8B"] = q8b_data
    if q32b_data and "15-shot cal (OOD)" in q32b_data:
        scaling_models["Qwen3-32B"] = q32b_data
    if gptoss20b_data and "15-shot cal (OOD)" in gptoss20b_data:
        scaling_models["GPT-OSS-20B"] = gptoss20b_data
    if gptoss120b_data and "15-shot cal (OOD)" in gptoss120b_data:
        scaling_models["GPT-OSS-120B"] = gptoss120b_data

    if scaling_models:
        plot_cross_model_aggregate(
            scaling_models, scaling_order, MODE_ORDER_8,
            out / "all_models_shot_scaling_aggregate.png",
            "Shot-Scaling: 5→10→15 (system prompt cal)\n(macro-avg across 8 modes, 95% CI)",
        )
        plot_cross_model_by_mode(
            scaling_models, scaling_order, MODE_ORDER_8,
            out / "all_models_shot_scaling_by_mode.png",
            "Shot-Scaling Per-Mode: All Models with Cal Data",
        )

    # ── OOD calibration shot-scaling (single grouped plot) ──
    ood_scaling_order = [
        "baseline", "2-shot cal",
        "5-shot cal (OOD)", "10-shot cal (OOD)", "15-shot cal (OOD)",
    ]
    ood_scaling_colors = {
        "baseline":           "#8C8C8C",
        "2-shot cal":         "#C6DBEF",
        "5-shot cal (OOD)":   "#6BAED6",
        "10-shot cal (OOD)":  "#2171B5",
        "15-shot cal (OOD)":  "#08306B",
    }
    ood_models = {}
    if q8b_data:
        ood_models["Qwen3-8B"] = q8b_data
    if q32b_data:
        ood_models["Qwen3-32B"] = q32b_data
    if gptoss20b_data:
        ood_models["GPT-OSS-20B"] = gptoss20b_data
    if gptoss120b_data:
        ood_models["GPT-OSS-120B"] = gptoss120b_data
    if ood_models:
        plot_cross_model_grouped(
            ood_models, ood_scaling_order, MODE_ORDER_8,
            out / "ood_cal_shot_scaling.png",
            "OOD Calibration Shot-Scaling\n(macro-avg across 8 modes, 95% CI)",
            variant_colors=ood_scaling_colors,
        )

    # ── From-eval fewshot shot-scaling (single grouped plot) ──
    # Load intermediate shot counts for Qwen
    q32b_fewshot_3 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-32b_3shot.jsonl",
        RESULTS / "fewshot_tinker_newmodes_shots" / "eval_qwen3-32b_3shot.jsonl",
    ], "fewshot 3-shot")
    q32b_fewshot_5 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-32b_5shot.jsonl",
        RESULTS / "fewshot_tinker_newmodes_shots" / "eval_qwen3-32b_5shot.jsonl",
    ], "fewshot 5-shot")
    q8b_fewshot_3 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-8b_3shot.jsonl",
        RESULTS / "fewshot_tinker_newmodes_shots" / "eval_qwen3-8b_3shot.jsonl",
    ], "fewshot 3-shot")
    q8b_fewshot_5 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-8b_5shot.jsonl",
        RESULTS / "fewshot_tinker_newmodes_shots" / "eval_qwen3-8b_5shot.jsonl",
    ], "fewshot 5-shot")

    eval_q8b = dict(q8b_data)
    eval_q8b.update(q8b_fewshot_3)
    eval_q8b.update(q8b_fewshot_5)
    eval_q32b = dict(q32b_data)
    eval_q32b.update(q32b_fewshot_3)
    eval_q32b.update(q32b_fewshot_5)

    # Load GPT-OSS 3/5-shot fewshot data
    gptoss20b_fewshot_3 = load_fewshot_data([
        RESULTS / "fewshot_gptoss_20b" / "eval_gpt-oss-20b_3shot.jsonl",
    ], "fewshot 3-shot")
    gptoss20b_fewshot_5 = load_fewshot_data([
        RESULTS / "fewshot_gptoss_20b" / "eval_gpt-oss-20b_5shot.jsonl",
    ], "fewshot 5-shot")
    gptoss120b_fewshot_3 = load_fewshot_data([
        RESULTS / "fewshot_gptoss_120b" / "eval_gpt-oss-120b_3shot.jsonl",
    ], "fewshot 3-shot")
    gptoss120b_fewshot_5 = load_fewshot_data([
        RESULTS / "fewshot_gptoss_120b" / "eval_gpt-oss-120b_5shot.jsonl",
    ], "fewshot 5-shot")

    eval_gptoss20b = dict(gptoss20b_data)
    eval_gptoss20b.update(gptoss20b_fewshot_3)
    eval_gptoss20b.update(gptoss20b_fewshot_5)
    eval_gptoss120b = dict(gptoss120b_data)
    eval_gptoss120b.update(gptoss120b_fewshot_3)
    eval_gptoss120b.update(gptoss120b_fewshot_5)

    eval_scaling_order = [
        "baseline", "fewshot 3-shot", "fewshot 5-shot", "fewshot 15-shot",
    ]
    eval_scaling_colors = {
        "baseline":         "#8C8C8C",
        "fewshot 3-shot":   "#74C476",
        "fewshot 5-shot":   "#238B45",
        "fewshot 15-shot":  "#00441B",
    }
    eval_models = {}
    if eval_q8b and "baseline" in eval_q8b:
        eval_models["Qwen3-8B"] = eval_q8b
    if eval_q32b and "baseline" in eval_q32b:
        eval_models["Qwen3-32B"] = eval_q32b
    if eval_gptoss20b and "baseline" in eval_gptoss20b:
        eval_models["GPT-OSS-20B"] = eval_gptoss20b
    if eval_gptoss120b and "baseline" in eval_gptoss120b:
        eval_models["GPT-OSS-120B"] = eval_gptoss120b
    if eval_models:
        plot_cross_model_grouped(
            eval_models, eval_scaling_order, MODE_ORDER_8,
            out / "eval_fewshot_shot_scaling.png",
            "\"On-Policy\" Fewshot Shot-Scaling\n(macro-avg across 8 modes, 95% CI)",
            variant_colors=eval_scaling_colors,
        )

    # ── METR SFT comparison ──
    # CoTControl compliance from METR fine-tuning paper (step 60 SFT)
    metr_sft = {
        "Qwen3-8B":     5.6,
        "Qwen3-32B":    9.8,
        "GPT-OSS-20B":  7.3,
        "GPT-OSS-120B": 13.1,
    }

    if bestof_models:
        metr_vals = [metr_sft.get(m, 0.0) for m in bestof_models]
        plot_bestof_vs_metr(
            bestof_models, bestof_bl, bestof_zs, bestof_fs, metr_vals,
            out / "all_models_vs_metr_sft.png",
            "Prompt Engineering vs METR SFT — ReasonIF-only\n(macro-avg across 8 modes, 95% CI)",
            metr_label="METR SFT (ReasonIF)",
        )

    # METR SFT (final) — trained on CoTControl + ReasonIF data
    metr_sft_final = {
        "Qwen3-8B":     20.7,
        "Qwen3-32B":    26.5,
        "GPT-OSS-20B":  19.1,
        "GPT-OSS-120B": 29.1,
    }

    if bestof_models:
        metr_final_vals = [metr_sft_final.get(m, 0.0) for m in bestof_models]
        plot_bestof_vs_metr(
            bestof_models, bestof_bl, bestof_zs, bestof_fs, metr_final_vals,
            out / "all_models_vs_metr_sft_final.png",
            "Prompt Engineering vs METR SFT — Final (CoTControl+ReasonIF)\n(macro-avg across 8 modes, 95% CI)",
            metr_label="METR SFT (final)",
        )

    # ── Reasoning length plots ──
    # baseline + ZS (repeat_15x, metacognition) from prompt_ablation + hillclimb
    # FS: 15-shot cal (OOD) from prompt_ablation, 15-shot on-policy from fewshot eval
    rlen_baseline_rename = {"baseline": "baseline"}
    rlen_cal_rename = {"cal_15shot_system": "15-shot cal (OOD)"}
    rlen_hc_rename = {"repeat_15x": "zs: repeat 15x", "metacognition": "zs: metacognition"}

    rlen_models: dict[str, dict[str, dict[str, list[tuple[int, bool]]]]] = {}
    for model_name, ablation_dirs, hc_file, fewshot_paths in [
        ("Qwen3-8B", [
            RESULTS / "prompt_ablation_r7_zs_8b" / "prompt_ablation_qwen3-8b.jsonl",
            RESULTS / "prompt_ablation_r7_cal_8b" / "prompt_ablation_qwen3-8b.jsonl",
            RESULTS / "prompt_ablation_r9_8b" / "prompt_ablation_qwen3-8b.jsonl",
            RESULTS / "prompt_ablation_cal_qwen8b" / "prompt_ablation_qwen3-8b.jsonl",
        ],
            RESULTS / "hillclimb_zs" / "hillclimb_qwen3-8b.jsonl",
        [
            RESULTS / "fewshot_tinker" / "eval_qwen3-8b_15shot.jsonl",
        ]),
        ("Qwen3-32B", [
            RESULTS / "prompt_ablation_r7_zs_32b" / "prompt_ablation_qwen3-32b.jsonl",
            RESULTS / "prompt_ablation_r7_cal_32b" / "prompt_ablation_qwen3-32b.jsonl",
            RESULTS / "prompt_ablation_r8_confirm" / "prompt_ablation_qwen3-32b.jsonl",
            RESULTS / "prompt_ablation_r9_confirm" / "prompt_ablation_qwen3-32b.jsonl",
            RESULTS / "prompt_ablation_cal_qwen32b" / "prompt_ablation_qwen3-32b.jsonl",
        ],
            RESULTS / "hillclimb_zs" / "hillclimb_qwen3-32b.jsonl",
        [
            RESULTS / "fewshot_tinker" / "eval_qwen3-32b_15shot.jsonl",
        ]),
        ("GPT-OSS-20B", [
            RESULTS / "gptoss_gpt-oss-20b" / "prompt_ablation_gpt-oss-20b.jsonl",
            RESULTS / "gptoss_r3_confirm_20b" / "prompt_ablation_gpt-oss-20b.jsonl",
            RESULTS / "gptoss_r4_confirm_20b" / "prompt_ablation_gpt-oss-20b.jsonl",
        ],
            RESULTS / "hillclimb_zs" / "hillclimb_gpt-oss-20b.jsonl",
        [
            RESULTS / "fewshot_gptoss_20b" / "eval_gpt-oss-20b_15shot.jsonl",
        ]),
        ("GPT-OSS-120B", [
            RESULTS / "gptoss_gpt-oss-120b" / "prompt_ablation_gpt-oss-120b.jsonl",
            RESULTS / "gptoss_r3_confirm_120b" / "prompt_ablation_gpt-oss-120b.jsonl",
            RESULTS / "gptoss_r4_confirm_120b" / "prompt_ablation_gpt-oss-120b.jsonl",
        ],
            RESULTS / "hillclimb_zs" / "hillclimb_gpt-oss-120b.jsonl",
        [
            RESULTS / "fewshot_gptoss_120b" / "eval_gpt-oss-120b_15shot.jsonl",
        ]),
    ]:
        merged: dict[str, dict[str, list[tuple[int, bool]]]] = {}
        # baseline from prompt ablation
        merged.update(load_reasoning_lengths(ablation_dirs, variant_rename=rlen_baseline_rename))
        # 15-shot cal (OOD) from prompt ablation
        merged.update(load_reasoning_lengths(ablation_dirs, variant_rename=rlen_cal_rename))
        # ZS: repeat_15x + metacognition from hillclimb
        merged.update(load_reasoning_lengths([hc_file], variant_rename=rlen_hc_rename))
        # 15-shot on-policy from fewshot eval
        merged.update(load_reasoning_lengths(fewshot_paths, label="15-shot (\"on-policy\")"))
        if merged:
            rlen_models[model_name] = merged

    if rlen_models:
        plot_reasoning_length_grouped(
            rlen_models, RLEN_VARIANT_ORDER, MODE_ORDER_8,
            out / "reasoning_length_aggregate.png",
            "Mean Reasoning Length by Variant\n(all responses, 95% CI)",
            variant_colors=RLEN_COLORS,
        )
        plot_reasoning_length_by_mode(
            rlen_models, RLEN_VARIANT_ORDER, MODE_ORDER_8,
            out / "reasoning_length_by_mode.png",
            "Reasoning Length by Mode and Variant\n(all responses, 95% CI)",
            variant_colors=RLEN_COLORS,
        )
        plot_reasoning_length_grouped(
            rlen_models, RLEN_VARIANT_ORDER, MODE_ORDER_8,
            out / "reasoning_length_compliant_aggregate.png",
            "Mean Reasoning Length by Variant\n(compliant responses only, 95% CI)",
            variant_colors=RLEN_COLORS,
            compliant_only=True,
        )
        plot_reasoning_length_by_mode(
            rlen_models, RLEN_VARIANT_ORDER, MODE_ORDER_8,
            out / "reasoning_length_compliant_by_mode.png",
            "Reasoning Length by Mode and Variant\n(compliant responses only, 95% CI)",
            variant_colors=RLEN_COLORS,
            compliant_only=True,
        )
        plot_reasoning_length_combined(
            rlen_models, RLEN_VARIANT_ORDER, MODE_ORDER_8,
            out / "reasoning_length_combined.png",
            "Mean Reasoning Length by Variant (95% CI)",
            variant_colors=RLEN_COLORS,
        )
        plot_compliance_vs_reasoning_length(
            rlen_models, RLEN_VARIANT_ORDER,
            out / "compliance_vs_reasoning_length.png",
            "Compliance Rate vs Reasoning Length\n(all models pooled, all modes)",
            variant_colors=RLEN_COLORS,
        )
        plot_compliance_vs_reasoning_length_per_model(
            rlen_models, RLEN_VARIANT_ORDER,
            out / "compliance_vs_reasoning_length_per_model.png",
            "Compliance Rate vs Reasoning Length",
            variant_colors=RLEN_COLORS,
        )

    # ── Accuracy by variant ──
    ACC_VARIANT_ORDER = [
        "baseline", "repeat 15x", "metacognition",
        "15-shot cal (OOD)", "15-shot (\"on-policy\")",
    ]
    ACC_COLORS = {
        "baseline":                "#8C8C8C",
        "repeat 15x":             "#7BA3D4",
        "metacognition":          "#3B6EA5",
        "15-shot cal (OOD)":      "#88CC88",
        "15-shot (\"on-policy\")": "#2D8E2D",
    }
    acc_bl_rename = {"baseline": "baseline"}
    acc_cal_rename = {"cal_15shot_system": "15-shot cal (OOD)"}
    acc_hc_rename = {"repeat_15x": "repeat 15x", "metacognition": "metacognition"}

    acc_models: dict[str, dict[str, dict]] = {}
    for model_name, ablation_dirs, hc_file, fewshot_paths in [
        ("Qwen3-8B", [
            RESULTS / "prompt_ablation_r7_zs_8b" / "prompt_ablation_qwen3-8b.jsonl",
            RESULTS / "prompt_ablation_r7_cal_8b" / "prompt_ablation_qwen3-8b.jsonl",
            RESULTS / "prompt_ablation_r9_8b" / "prompt_ablation_qwen3-8b.jsonl",
            RESULTS / "prompt_ablation_cal_qwen8b" / "prompt_ablation_qwen3-8b.jsonl",
        ],
            RESULTS / "hillclimb_zs" / "hillclimb_qwen3-8b.jsonl",
        [
            RESULTS / "fewshot_tinker" / "eval_qwen3-8b_15shot.jsonl",
        ]),
        ("Qwen3-32B", [
            RESULTS / "prompt_ablation_r7_zs_32b" / "prompt_ablation_qwen3-32b.jsonl",
            RESULTS / "prompt_ablation_r7_cal_32b" / "prompt_ablation_qwen3-32b.jsonl",
            RESULTS / "prompt_ablation_r8_confirm" / "prompt_ablation_qwen3-32b.jsonl",
            RESULTS / "prompt_ablation_r9_confirm" / "prompt_ablation_qwen3-32b.jsonl",
            RESULTS / "prompt_ablation_cal_qwen32b" / "prompt_ablation_qwen3-32b.jsonl",
        ],
            RESULTS / "hillclimb_zs" / "hillclimb_qwen3-32b.jsonl",
        [
            RESULTS / "fewshot_tinker" / "eval_qwen3-32b_15shot.jsonl",
        ]),
        ("GPT-OSS-20B", [
            RESULTS / "gptoss_gpt-oss-20b" / "prompt_ablation_gpt-oss-20b.jsonl",
            RESULTS / "gptoss_r3_confirm_20b" / "prompt_ablation_gpt-oss-20b.jsonl",
            RESULTS / "gptoss_r4_confirm_20b" / "prompt_ablation_gpt-oss-20b.jsonl",
        ],
            RESULTS / "hillclimb_zs" / "hillclimb_gpt-oss-20b.jsonl",
        [
            RESULTS / "fewshot_gptoss_20b" / "eval_gpt-oss-20b_15shot.jsonl",
        ]),
        ("GPT-OSS-120B", [
            RESULTS / "gptoss_gpt-oss-120b" / "prompt_ablation_gpt-oss-120b.jsonl",
            RESULTS / "gptoss_r3_confirm_120b" / "prompt_ablation_gpt-oss-120b.jsonl",
            RESULTS / "gptoss_r4_confirm_120b" / "prompt_ablation_gpt-oss-120b.jsonl",
        ],
            RESULTS / "hillclimb_zs" / "hillclimb_gpt-oss-120b.jsonl",
        [
            RESULTS / "fewshot_gptoss_120b" / "eval_gpt-oss-120b_15shot.jsonl",
        ]),
    ]:
        merged: dict[str, dict] = {}
        merged.update(load_accuracy_by_variant(ablation_dirs, variant_rename=acc_bl_rename))
        merged.update(load_accuracy_by_variant(ablation_dirs, variant_rename=acc_cal_rename))
        merged.update(load_accuracy_by_variant([hc_file], variant_rename=acc_hc_rename))
        merged.update(load_accuracy_by_variant(fewshot_paths, label="15-shot (\"on-policy\")"))
        if merged:
            acc_models[model_name] = merged

    if acc_models:
        plot_accuracy_by_variant(
            acc_models, ACC_VARIANT_ORDER,
            out / "accuracy_by_variant.png",
            "Accuracy by Variant\n(95% CI)",
            variant_colors=ACC_COLORS,
        )

    # ── Pirate compliance ──
    pirate_models = ["Qwen3-8B", "Qwen3-32B", "GPT-OSS-20B", "GPT-OSS-120B"]
    pirate_files = [
        RESULTS / "pirate" / "pirate_qwen3-8b.jsonl",
        RESULTS / "pirate" / "pirate_qwen3-32b.jsonl",
        RESULTS / "pirate" / "pirate_gpt-oss-20b.jsonl",
        RESULTS / "pirate" / "pirate_gpt-oss-120b.jsonl",
    ]
    pirate_comply = []
    pirate_se = []
    pirate_ok = True
    for p in pirate_files:
        if not p.exists():
            pirate_ok = False
            break
        rows = load_jsonl(p)
        valid = [r for r in rows if r.get("compliant") is not None]
        n = len(valid)
        k = sum(1 for r in valid if r["compliant"])
        rate = k / n if n else 0
        se = np.sqrt(rate * (1 - rate) / n) if n else 0
        pirate_comply.append(rate * 100)
        pirate_se.append(se * 100 * 1.96)

    if pirate_ok:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        x = np.arange(len(pirate_models))
        width = 0.5
        bars_c = ax.bar(x, pirate_comply, width, yerr=pirate_se, capsize=4,
                        color="#55A868", edgecolor="white", linewidth=0.5)
        for bar, err in zip(bars_c, pirate_se):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.5,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(pirate_models, fontsize=11)
        ax.set_ylabel("Compliance (%)", fontsize=12)
        ax.set_title('"Speak Like a Pirate" in Reasoning\n(compliance: 3+ unique pirate words, 5+ total occurrences)',
                     fontsize=13)
        ax.set_ylim(0, max(c + e for c, e in zip(pirate_comply, pirate_se)) * 1.3 + 2)
        plt.tight_layout()
        plt.savefig(out / "pirate_compliance.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {out / 'pirate_compliance.png'}")
        plt.close()

    print(f"\nDone. {len(list(out.glob('*.png')))} plots in {out}/")


if __name__ == "__main__":
    main()
