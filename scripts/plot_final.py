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
            bars = ax.bar(x + offset, vals, width, yerr=errs,
                          capsize=2, label=v, color=COLORS[i % len(COLORS)])
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
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


# ── Data loading ──

RESULTS = Path("results")

# Display name mappings
QWEN_RENAME = {
    "calibration_shot": "1-shot cal",
    "cal_2shot": "2-shot cal",
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
    "cal_5shot_system": "5-shot cal (sys)",
    "cal_10shot_system": "10-shot cal (sys)",
    "cal_15shot_system": "15-shot cal (sys)",
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
    ], QWEN_RENAME)

    # Add fewshot 15-shot data (real dataset examples with transformed reasoning)
    q32b_fewshot_0 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-32b_0shot.jsonl",
        RESULTS / "fewshot_tinker_0shot_newmodes" / "eval_qwen3-32b_0shot.jsonl",
    ], "fewshot 0-shot")
    q32b_fewshot_15 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-32b_15shot.jsonl",
        RESULTS / "fewshot_tinker_newmodes" / "eval_qwen3-32b_15shot.jsonl",
    ], "fewshot 15-shot")
    q32b_data.update(q32b_fewshot_0)
    q32b_data.update(q32b_fewshot_15)

    q32b_all_order = [
        "baseline", "repeated", "self_check", "two_pass", "token_bounty",
        "1-shot cal", "2-shot cal",
        "render+selfcheck", "bounty+selfcheck",
        "twopass+example", "render+selfcheck+bounty",
        "fewshot 15-shot",
    ]
    q32b_highlights = [
        "baseline", "self_check",
        "bounty+selfcheck", "2-shot cal", "fewshot 15-shot",
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
    ], QWEN_RENAME)

    q8b_fewshot_0 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-8b_0shot.jsonl",
        RESULTS / "fewshot_tinker_0shot_newmodes" / "eval_qwen3-8b_0shot.jsonl",
    ], "fewshot 0-shot")
    q8b_fewshot_15 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-8b_15shot.jsonl",
        RESULTS / "fewshot_tinker_newmodes" / "eval_qwen3-8b_15shot.jsonl",
    ], "fewshot 15-shot")
    q8b_data.update(q8b_fewshot_0)
    q8b_data.update(q8b_fewshot_15)

    q8b_all_order = [
        "baseline", "repeated", "self_check", "token_bounty",
        "1-shot cal", "2-shot cal",
        "render+selfcheck", "bounty+selfcheck",
        "render+selfcheck+bounty",
        "fewshot 15-shot",
    ]
    q8b_highlights = [
        "baseline", "self_check",
        "bounty+selfcheck", "2-shot cal", "fewshot 15-shot",
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

    # ── GPT-OSS-20B ──
    gptoss20b_data = load_model_data([
        RESULTS / "gptoss_gpt-oss-20b" / "prompt_ablation_gpt-oss-20b.jsonl",
        RESULTS / "gptoss_r3_confirm_20b" / "prompt_ablation_gpt-oss-20b.jsonl",
        RESULTS / "gptoss_r4_confirm_20b" / "prompt_ablation_gpt-oss-20b.jsonl",
    ], GPTOSS_RENAME)

    gptoss20b_fewshot_15 = load_fewshot_data([
        RESULTS / "fewshot_gptoss_20b" / "eval_gpt-oss-20b_15shot.jsonl",
    ], "fewshot 15-shot")
    gptoss20b_data.update(gptoss20b_fewshot_15)

    gptoss_all_order = [
        "baseline", "self_check", "render_mode",
        "render+selfcheck", "bounty+selfcheck",
        "2-shot cal", "3-shot cal", "3-shot cal (strong)",
        "5-shot cal (sys)", "10-shot cal (sys)", "15-shot cal (sys)",
        "fewshot 15-shot",
        "zs: repeat 15x", "zs: metacognition",
    ]
    gptoss_highlights = [
        "baseline", "10-shot cal (sys)",
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

    gptoss120b_fewshot_15 = load_fewshot_data([
        RESULTS / "fewshot_gptoss_120b" / "eval_gpt-oss-120b_15shot.jsonl",
    ], "fewshot 15-shot")
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

    # ── Cross-model comparisons ──
    cross_highlights_order = [
        "baseline", "self_check",
        "bounty+selfcheck", "2-shot cal",
        "5-shot cal (sys)", "10-shot cal (sys)",
        "fewshot 15-shot",
        "zs: repeat 15x", "zs: metacognition",
    ]

    all_models = {}
    if q32b_data:
        all_models["Qwen3-32B"] = q32b_data
    if q8b_data:
        all_models["Qwen3-8B"] = q8b_data
    if gptoss120b_data:
        all_models["GPT-OSS-120B"] = gptoss120b_data
    if gptoss20b_data:
        all_models["GPT-OSS-20B"] = gptoss20b_data

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
        )

    # ── GPT-OSS shot-scaling comparison ──
    gptoss_scaling = {}
    scaling_order = [
        "baseline", "5-shot cal (sys)",
        "10-shot cal (sys)", "15-shot cal (sys)",
    ]
    if gptoss20b_data:
        gptoss_scaling["GPT-OSS-20B"] = gptoss20b_data
    if gptoss120b_data:
        gptoss_scaling["GPT-OSS-120B"] = gptoss120b_data

    if gptoss_scaling:
        plot_cross_model_aggregate(
            gptoss_scaling, scaling_order, MODE_ORDER_8,
            out / "gptoss_shot_scaling_aggregate.png",
            "GPT-OSS Shot-Scaling: 5→10→15\n(macro-avg across 8 modes, 95% CI)",
        )
        plot_cross_model_by_mode(
            gptoss_scaling, scaling_order, MODE_ORDER_8,
            out / "gptoss_shot_scaling_by_mode.png",
            "GPT-OSS Shot-Scaling Per-Mode",
        )

    print(f"\nDone. {len(list(out.glob('*.png')))} plots in {out}/")


if __name__ == "__main__":
    main()
