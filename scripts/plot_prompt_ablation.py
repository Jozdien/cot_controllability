"""Plot prompt ablation results: aggregate, per-mode, and cross-condition comparisons."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3",
          "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
          "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]

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


def _get_mode(r: dict) -> str:
    return r.get("mode", "") or r.get("control_mode", "")


def _get_compliant(r: dict):
    return r.get("compliant")


def compute_stats(rows: list[dict], variant_key: str = "variant") -> dict[str, dict[str, dict]]:
    """Returns {variant: {mode: {rate, se, n, k}}}."""
    counts = defaultdict(lambda: defaultdict(lambda: {"k": 0, "n": 0}))
    for r in rows:
        v = r.get(variant_key, "")
        m = _get_mode(r)
        c = _get_compliant(r)
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


def compute_stats_single(rows: list[dict]) -> dict[str, dict]:
    """Returns {mode: {rate, se, n, k}} for a single-condition file."""
    counts = defaultdict(lambda: {"k": 0, "n": 0})
    for r in rows:
        m = _get_mode(r)
        c = _get_compliant(r)
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
    return stats


def macro_avg(mode_stats: dict[str, dict], modes: list[str]) -> tuple[float, float]:
    """Compute macro-average rate and SE across given modes."""
    present = [mode_stats[m] for m in modes if m in mode_stats]
    if not present:
        return 0.0, 0.0
    avg = np.mean([s["rate"] for s in present])
    se = np.sqrt(np.sum([s["se"] ** 2 for s in present])) / len(present)
    return avg, se


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
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=10)
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
    ax.legend(fontsize=9, loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_horizontal_bar(
    names: list[str],
    vals: list[float],
    errs: list[float],
    output_path: Path,
    title: str = "",
    figsize: tuple[float, float] = (10, 7),
    highlight_top: int = 3,
):
    """Horizontal bar chart for ranking many variants."""
    sorted_idx = np.argsort(vals)
    names = [names[i] for i in sorted_idx]
    errs = [errs[i] for i in sorted_idx]
    vals = [vals[i] for i in sorted_idx]

    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=figsize)
    bar_colors = []
    n = len(names)
    for i in range(n):
        if i >= n - highlight_top:
            bar_colors.append("#55A868")
        elif names[i] == "baseline":
            bar_colors.append("#4C72B0")
        else:
            bar_colors.append("#AAAAAA")

    bars = ax.barh(y, vals, xerr=errs, capsize=3, color=bar_colors,
                   edgecolor="white", linewidth=0.5)
    for bar, val, err in zip(bars, vals, errs):
        if val > 0.5:
            ax.text(val + err + 0.3, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}%", va="center", fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Compliance (%)", fontsize=12)
    if title:
        ax.set_title(title, fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    output_dir = Path("results")

    # ── 1. R5 confirmation: top creative zero-shot variants (aggregate + by mode) ──
    r5_path = Path("results/prompt_ablation_r5/prompt_ablation_qwen3-32b.jsonl")
    if r5_path.exists():
        r5_stats = compute_stats(load_jsonl(r5_path))
        r5_order = ["baseline", "self_check", "repeated", "broken_keyboard"]
        plot_aggregate(
            r5_stats, r5_order, MODE_ORDER_8, output_dir / "plot_prompt_ablation_aggregate.png",
            title="Zero-Shot Prompt Ablation: Aggregate Compliance\n(Qwen3-32B, n=100, averaged across 8 modes)",
        )
        plot_by_mode(
            r5_stats, r5_order, MODE_ORDER_8, output_dir / "plot_prompt_ablation_by_mode.png",
            title="Zero-Shot Prompt Ablation: Per-Mode Compliance\n(Qwen3-32B, n=100)",
        )

    # ── 2. R4 creative pilot: horizontal ranking of all 13 creative variants ──
    r4_path = Path("results/prompt_ablation_r4/prompt_ablation_qwen3-32b.jsonl")
    if r4_path.exists():
        r4_stats = compute_stats(load_jsonl(r4_path))
        r4_modes = ["lowercase_thinking", "word_suppression"]
        names, vals, errs = [], [], []
        for v, mode_stats in r4_stats.items():
            avg, se = macro_avg(mode_stats, r4_modes)
            names.append(v)
            vals.append(avg * 100)
            errs.append(se * 100 * 1.96)
        plot_horizontal_bar(
            names, vals, errs,
            output_dir / "plot_creative_variants_pilot.png",
            title="Creative Zero-Shot Variants: Pilot Results\n(Qwen3-32B, n=50, lowercase + word_suppression)",
            figsize=(10, 6),
        )

    # ── 3. R2 standard variants: horizontal ranking ──
    r2_path = Path("results/prompt_ablation/prompt_ablation_qwen3-32b.jsonl")
    if r2_path.exists():
        r2_stats = compute_stats(load_jsonl(r2_path))
        r2_modes = ["lowercase_thinking", "word_suppression"]
        names, vals, errs = [], [], []
        for v, mode_stats in r2_stats.items():
            avg, se = macro_avg(mode_stats, r2_modes)
            names.append(v)
            vals.append(avg * 100)
            errs.append(se * 100 * 1.96)
        plot_horizontal_bar(
            names, vals, errs,
            output_dir / "plot_standard_variants_pilot.png",
            title="Standard Prompt Variants: Pilot Results\n(Qwen3-32B, n=50, lowercase + word_suppression)",
            figsize=(10, 5),
        )

    # ── 4. Definitive comparison: best zero-shot vs calibration vs fewshot ──
    fewshot_dir = Path("results/fewshot_tinker")

    # Build comparison from best confirmed data
    definitive: dict[str, dict[str, dict]] = {}

    # Zero-shot variants from R7 (n=300, 8 modes)
    r7_zs_path = Path("results/prompt_ablation_r7_zs_32b/prompt_ablation_qwen3-32b.jsonl")
    if r7_zs_path.exists():
        r7_zs = compute_stats(load_jsonl(r7_zs_path))
        definitive["baseline"] = r7_zs.get("baseline", {})
        if "repeated" in r7_zs:
            definitive["repeated"] = r7_zs["repeated"]
        if "self_check" in r7_zs:
            definitive["self_check"] = r7_zs["self_check"]

    # Best combo zero-shot from R9 confirmation (n=200, 8 modes)
    r9_confirm_path_sec4 = Path("results/prompt_ablation_r9_confirm/prompt_ablation_qwen3-32b.jsonl")
    if r9_confirm_path_sec4.exists():
        r9c = compute_stats(load_jsonl(r9_confirm_path_sec4))
        if "combo_bounty_selfcheck" in r9c:
            definitive["bounty+selfcheck"] = r9c["combo_bounty_selfcheck"]
        if "combo_render_selfcheck" in r9c:
            definitive["render+selfcheck"] = r9c["combo_render_selfcheck"]

    # Calibration 1-shot and 2-shot from R7
    cal_path = Path("results/prompt_ablation_r7_cal_32b/prompt_ablation_qwen3-32b.jsonl")
    if cal_path.exists():
        cal_stats = compute_stats(load_jsonl(cal_path))
        if "calibration_shot" in cal_stats:
            definitive["1-shot cal"] = cal_stats["calibration_shot"]
        if "cal_2shot" in cal_stats:
            definitive["2-shot cal"] = cal_stats["cal_2shot"]

    # 15-shot fewshot
    p15 = fewshot_dir / "eval_qwen3-32b_15shot.jsonl"
    if p15.exists():
        definitive["15-shot"] = compute_stats_single(load_jsonl(p15))

    if definitive:
        def_order = [
            "baseline", "repeated", "self_check",
            "render+selfcheck", "bounty+selfcheck",
            "1-shot cal", "2-shot cal",
            "15-shot",
        ]
        def_order = [l for l in def_order if l in definitive]

        plot_aggregate(
            definitive, def_order, MODE_ORDER_8,
            output_dir / "plot_overall_aggregate.png",
            title="Zero-Shot vs Calibration vs Few-Shot (Qwen3-32B)\nmacro-avg across 8 CoTControl modes",
            figsize=(14, 5.5),
        )
        plot_by_mode(
            definitive, def_order, MODE_ORDER_8,
            output_dir / "plot_overall_by_mode.png",
            title="Per-Mode Compliance: Zero-Shot vs Calibration vs Few-Shot (Qwen3-32B)",
            figsize=(22, 6),
        )

    # ── 5. R7: n=300 comparison across both models ──
    r7_dirs = {
        "zs_32b": Path("results/prompt_ablation_r7_zs_32b"),
        "zs_8b": Path("results/prompt_ablation_r7_zs_8b"),
        "cal_32b": Path("results/prompt_ablation_r7_cal_32b"),
        "cal_8b": Path("results/prompt_ablation_r7_cal_8b"),
    }
    r7_data: dict[str, dict[str, dict[str, dict]]] = {}
    for key, d in r7_dirs.items():
        model = "qwen3-32b" if "32b" in key else "qwen3-8b"
        p = d / f"prompt_ablation_{model}.jsonl"
        if p.exists():
            r7_data[key] = compute_stats(load_jsonl(p))

    # Load 15-shot fewshot data for both models (use only the 8 shared modes)
    fewshot_15: dict[str, dict[str, dict]] = {}
    for model_tag in ["32b", "8b"]:
        model = f"qwen3-{model_tag}"
        p = fewshot_dir / f"eval_{model}_15shot.jsonl"
        if p.exists():
            fewshot_15[model_tag] = compute_stats_single(load_jsonl(p))

    for model_tag, model_label in [("32b", "Qwen3-32B"), ("8b", "Qwen3-8B")]:
        zs_key = f"zs_{model_tag}"
        cal_key = f"cal_{model_tag}"
        if zs_key not in r7_data or cal_key not in r7_data:
            continue

        merged: dict[str, dict[str, dict]] = {}
        for variant, mode_stats in r7_data[zs_key].items():
            merged[variant] = mode_stats
        for variant, mode_stats in r7_data[cal_key].items():
            if variant == "baseline":
                merged["baseline (cal)"] = mode_stats
            else:
                merged[variant] = mode_stats
        if model_tag in fewshot_15:
            merged["15-shot"] = fewshot_15[model_tag]

        label_order = ["baseline", "self_check", "repeated",
                       "calibration_shot", "cal_2shot", "15-shot"]

        plot_aggregate(
            merged, label_order, MODE_ORDER_8,
            output_dir / f"plot_r7_aggregate_{model_tag}.png",
            title=f"Compliance: All Conditions ({model_label})\nmacro-avg across 8 modes",
            figsize=(11, 5),
        )
        plot_by_mode(
            merged, label_order, MODE_ORDER_8,
            output_dir / f"plot_r7_by_mode_{model_tag}.png",
            title=f"Per-Mode Compliance ({model_label})",
            figsize=(18, 6),
        )

    # 2-panel aggregate: 8B vs 32B side by side
    if all(k in r7_data for k in r7_dirs):
        fig, axes = plt.subplots(1, 2, figsize=(18, 5.5), sharey=True)
        variant_order = ["baseline", "self_check", "repeated",
                         "calibration_shot", "cal_2shot", "15-shot"]
        bar_colors = [COLORS[i % len(COLORS)] for i in range(len(variant_order))]

        for ax, model_tag, model_label in zip(axes, ["32b", "8b"],
                                               ["Qwen3-32B", "Qwen3-8B"]):
            merged = {}
            for variant, ms in r7_data[f"zs_{model_tag}"].items():
                merged[variant] = ms
            for variant, ms in r7_data[f"cal_{model_tag}"].items():
                if variant != "baseline":
                    merged[variant] = ms
            if model_tag in fewshot_15:
                merged["15-shot"] = fewshot_15[model_tag]

            vals, errs, labels = [], [], []
            for v in variant_order:
                if v not in merged:
                    continue
                avg, se = macro_avg(merged[v], MODE_ORDER_8)
                vals.append(avg * 100)
                errs.append(se * 100 * 1.96)
                labels.append(v)

            x = np.arange(len(labels))
            bars = ax.bar(x, vals, yerr=errs, capsize=4,
                          color=bar_colors[:len(labels)],
                          edgecolor="white", linewidth=0.5)
            for bar, err, val in zip(bars, errs, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 0.5,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=10)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
            ax.set_title(model_label, fontsize=13)
            ax.set_ylabel("Compliance (%)" if ax == axes[0] else "", fontsize=12)

        top = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(0, top * 1.15 + 2)

        fig.suptitle("CoT Compliance: 8B vs 32B (macro-avg across 8 modes)",
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_r7_model_comparison.png",
                    dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_r7_model_comparison.png'}")
        plt.close()

    # ── 6. R8: 8-mode confirmation of combo variants ──
    r8_confirm_path = Path("results/prompt_ablation_r8_confirm/prompt_ablation_qwen3-32b.jsonl")
    if r8_confirm_path.exists():
        r8_stats = compute_stats(load_jsonl(r8_confirm_path))

        r8_order = [
            "baseline", "self_check", "two_pass",
            "combo_twopass_example", "combo_all_three", "combo_render_selfcheck",
        ]

        plot_aggregate(
            r8_stats, r8_order, MODE_ORDER_8,
            output_dir / "plot_r8_confirm_aggregate.png",
            title="R8 Confirmation: Aggregate Compliance (Qwen3-32B)\nmacro-avg across 8 modes, n=200/mode",
            figsize=(12, 5),
        )
        plot_by_mode(
            r8_stats, r8_order, MODE_ORDER_8,
            output_dir / "plot_r8_confirm_by_mode.png",
            title="R8 Confirmation: Per-Mode Compliance (Qwen3-32B, n=200)",
            figsize=(18, 6),
        )

        # Also make a combined plot: best R8 combo + 15-shot + fewshot baseline
        r8_combined: dict[str, dict[str, dict]] = {}
        r8_combined["baseline"] = r8_stats["baseline"]
        r8_combined["self_check"] = r8_stats["self_check"]
        r8_combined["combo_render_selfcheck"] = r8_stats["combo_render_selfcheck"]
        r8_combined["combo_all_three"] = r8_stats["combo_all_three"]

        # Add calibration/fewshot reference if available
        if "cal_32b" in r7_data:
            for v in ["cal_2shot"]:
                if v in r7_data["cal_32b"]:
                    r8_combined[v] = r7_data["cal_32b"][v]
        if "32b" in fewshot_15:
            r8_combined["15-shot"] = fewshot_15["32b"]

        r8_combined_order = [
            "baseline", "self_check",
            "combo_all_three", "combo_render_selfcheck",
            "cal_2shot", "15-shot",
        ]
        r8_combined_order = [l for l in r8_combined_order if l in r8_combined]

        plot_aggregate(
            r8_combined, r8_combined_order, MODE_ORDER_8,
            output_dir / "plot_r8_overall.png",
            title="Best Zero-Shot vs Calibration/Few-Shot (Qwen3-32B)\nmacro-avg across 8 modes",
            figsize=(12, 5),
        )
        plot_by_mode(
            r8_combined, r8_combined_order, MODE_ORDER_8,
            output_dir / "plot_r8_overall_by_mode.png",
            title="Best Zero-Shot vs Calibration/Few-Shot: Per-Mode (Qwen3-32B)",
            figsize=(18, 6),
        )

    # ── 7. R8 exotic variants: horizontal ranking ──
    r8_exotic_path = Path("results/prompt_ablation_r8_exotic/prompt_ablation_qwen3-32b.jsonl")
    if r8_exotic_path.exists():
        exotic_stats = compute_stats(load_jsonl(r8_exotic_path))
        exotic_modes = ["lowercase_thinking", "word_suppression", "uppercase_thinking", "end_of_sentence"]
        names, vals, errs = [], [], []
        for v, mode_stats in exotic_stats.items():
            avg, se = macro_avg(mode_stats, exotic_modes)
            names.append(v)
            vals.append(avg * 100)
            errs.append(se * 100 * 1.96)
        plot_horizontal_bar(
            names, vals, errs,
            output_dir / "plot_r8_exotic_ranking.png",
            title="Exotic Prompt Variants: Pilot Results\n(Qwen3-32B, n=100, 4 modes)",
            figsize=(10, 7),
            highlight_top=3,
        )

        # Per-mode for top 6 exotic
        exotic_order = sorted(exotic_stats.keys(),
                              key=lambda v: macro_avg(exotic_stats[v], exotic_modes)[0],
                              reverse=True)[:6]
        plot_by_mode(
            exotic_stats, exotic_order, exotic_modes,
            output_dir / "plot_r8_exotic_by_mode.png",
            title="Top 6 Exotic Variants: Per-Mode Compliance (Qwen3-32B, n=100)",
            figsize=(14, 6),
        )

    # ── 8. R9 combo results (if available) ──
    r9_combo_path = Path("results/prompt_ablation_r9_combo/prompt_ablation_qwen3-32b.jsonl")
    if r9_combo_path.exists():
        r9_stats = compute_stats(load_jsonl(r9_combo_path))
        r9_modes = ["lowercase_thinking", "word_suppression", "uppercase_thinking", "end_of_sentence"]
        r9_order = sorted(r9_stats.keys(),
                          key=lambda v: macro_avg(r9_stats[v], r9_modes)[0],
                          reverse=True)
        plot_aggregate(
            r9_stats, r9_order, r9_modes,
            output_dir / "plot_r9_combo_aggregate.png",
            title="R9 Bounty Combos: Aggregate Compliance (Qwen3-32B)\nmacro-avg across 4 modes, n=100",
            figsize=(14, 5),
        )
        plot_by_mode(
            r9_stats, r9_order, r9_modes,
            output_dir / "plot_r9_combo_by_mode.png",
            title="R9 Bounty Combos: Per-Mode Compliance (Qwen3-32B, n=100)",
            figsize=(18, 6),
        )

    # ── 9. R9 8-mode confirmation: final best variants ──
    r9_confirm_path = Path("results/prompt_ablation_r9_confirm/prompt_ablation_qwen3-32b.jsonl")
    if r9_confirm_path.exists():
        r9c_stats = compute_stats(load_jsonl(r9_confirm_path))

        r9c_order = [
            "baseline", "token_bounty",
            "combo_render_selfcheck", "combo_bounty_selfcheck",
        ]

        plot_aggregate(
            r9c_stats, r9c_order, MODE_ORDER_8,
            output_dir / "plot_r9_confirm_aggregate.png",
            title="R9 Confirmation: Aggregate Compliance (Qwen3-32B)\nmacro-avg across 8 modes, n=200/mode",
            figsize=(10, 5),
        )
        plot_by_mode(
            r9c_stats, r9c_order, MODE_ORDER_8,
            output_dir / "plot_r9_confirm_by_mode.png",
            title="R9 Confirmation: Per-Mode Compliance (Qwen3-32B, n=200)",
            figsize=(18, 6),
        )

        # Final overall comparison: best zero-shot + calibration + fewshot
        final_combined: dict[str, dict[str, dict]] = {}
        final_combined["baseline"] = r9c_stats["baseline"]
        final_combined["combo_bounty_selfcheck"] = r9c_stats["combo_bounty_selfcheck"]
        final_combined["combo_render_selfcheck"] = r9c_stats["combo_render_selfcheck"]

        if "cal_32b" in r7_data:
            for v in ["cal_2shot"]:
                if v in r7_data["cal_32b"]:
                    final_combined[v] = r7_data["cal_32b"][v]
        if "32b" in fewshot_15:
            final_combined["15-shot"] = fewshot_15["32b"]

        final_order = [
            "baseline",
            "combo_render_selfcheck", "combo_bounty_selfcheck",
            "cal_2shot", "15-shot",
        ]
        final_order = [l for l in final_order if l in final_combined]

        plot_aggregate(
            final_combined, final_order, MODE_ORDER_8,
            output_dir / "plot_final_comparison.png",
            title="Final: Best Zero-Shot vs Calibration/Few-Shot (Qwen3-32B)\nmacro-avg across 8 modes",
            figsize=(12, 5),
        )
        plot_by_mode(
            final_combined, final_order, MODE_ORDER_8,
            output_dir / "plot_final_comparison_by_mode.png",
            title="Final Comparison: Per-Mode (Qwen3-32B)",
            figsize=(18, 6),
        )

    # ── 10. R9 8B: top zero-shot variants + calibration + fewshot ──
    r9_8b_path = Path("results/prompt_ablation_r9_8b/prompt_ablation_qwen3-8b.jsonl")
    if r9_8b_path.exists():
        r9_8b_stats = compute_stats(load_jsonl(r9_8b_path))

        # Add calibration and fewshot references
        r9_8b_all: dict[str, dict[str, dict]] = {}
        r9_8b_all.update(r9_8b_stats)

        cal_8b_path = Path("results/prompt_ablation_r7_cal_8b/prompt_ablation_qwen3-8b.jsonl")
        if cal_8b_path.exists():
            cal_8b = compute_stats(load_jsonl(cal_8b_path))
            if "cal_2shot" in cal_8b:
                r9_8b_all["2-shot cal"] = cal_8b["cal_2shot"]

        fs_8b_path = fewshot_dir / "eval_qwen3-8b_15shot.jsonl"
        if fs_8b_path.exists():
            r9_8b_all["15-shot"] = compute_stats_single(load_jsonl(fs_8b_path))

        r9_8b_order = [
            "baseline", "self_check", "combo_render_selfcheck",
            "combo_bounty_selfcheck", "combo_all_three", "token_bounty",
            "2-shot cal", "15-shot",
        ]
        r9_8b_order = [l for l in r9_8b_order if l in r9_8b_all]

        plot_aggregate(
            r9_8b_all, r9_8b_order, MODE_ORDER_8,
            output_dir / "plot_r9_8b_aggregate.png",
            title="Zero-Shot vs Calibration vs Few-Shot (Qwen3-8B)\nmacro-avg across 8 modes",
            figsize=(14, 5.5),
        )
        plot_by_mode(
            r9_8b_all, r9_8b_order, MODE_ORDER_8,
            output_dir / "plot_r9_8b_by_mode.png",
            title="Per-Mode Compliance: Zero-Shot vs Calibration vs Few-Shot (Qwen3-8B)",
            figsize=(22, 6),
        )

        # Side-by-side 8B vs 32B for the best zero-shot variants
        r9_32b_path = Path("results/prompt_ablation_r9_confirm/prompt_ablation_qwen3-32b.jsonl")
        if r9_32b_path.exists():
            r9_32b_stats = compute_stats(load_jsonl(r9_32b_path))
            shared_variants = ["baseline", "combo_bounty_selfcheck",
                               "combo_render_selfcheck", "token_bounty", "self_check"]
            shared_variants = [v for v in shared_variants
                               if v in r9_8b_stats and v in r9_32b_stats]

            if shared_variants:
                fig, axes = plt.subplots(1, 2, figsize=(18, 5.5), sharey=True)
                bar_colors_side = [COLORS[i % len(COLORS)] for i in range(len(shared_variants))]

                for ax, stats, model_label in [
                    (axes[0], r9_32b_stats, "Qwen3-32B"),
                    (axes[1], r9_8b_stats, "Qwen3-8B"),
                ]:
                    vals, errs, labels = [], [], []
                    for v in shared_variants:
                        if v not in stats:
                            continue
                        avg, se = macro_avg(stats[v], MODE_ORDER_8)
                        vals.append(avg * 100)
                        errs.append(se * 100 * 1.96)
                        labels.append(v)

                    x = np.arange(len(labels))
                    bars = ax.bar(x, vals, yerr=errs, capsize=4,
                                  color=bar_colors_side[:len(labels)],
                                  edgecolor="white", linewidth=0.5)
                    for bar, err, val in zip(bars, errs, vals):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + err + 0.3,
                                f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

                    ax.set_xticks(x)
                    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
                    ax.set_title(model_label, fontsize=13)
                    ax.set_ylabel("Compliance (%)" if ax == axes[0] else "", fontsize=12)

                top = max(ax.get_ylim()[1] for ax in axes)
                for ax in axes:
                    ax.set_ylim(0, top * 1.15 + 2)

                fig.suptitle("Zero-Shot Prompt Engineering: 32B vs 8B\n(macro-avg across 8 modes)",
                             fontsize=14)
                plt.tight_layout()
                plt.savefig(output_dir / "plot_r9_model_comparison.png",
                            dpi=150, bbox_inches="tight")
                print(f"Saved: {output_dir / 'plot_r9_model_comparison.png'}")
                plt.close()

    # ── 11. Combined 2-panel plots: 32B vs 8B ──
    # Build unified data dicts for both models
    combo_data: dict[str, dict[str, dict[str, dict]]] = {}

    # 32B: zero-shot from R7 + combos from R9 confirm + cal + fewshot
    data_32b: dict[str, dict[str, dict]] = {}
    r7_zs_path_c = Path("results/prompt_ablation_r7_zs_32b/prompt_ablation_qwen3-32b.jsonl")
    if r7_zs_path_c.exists():
        r7_zs_c = compute_stats(load_jsonl(r7_zs_path_c))
        data_32b["baseline"] = r7_zs_c.get("baseline", {})
        if "self_check" in r7_zs_c:
            data_32b["self_check"] = r7_zs_c["self_check"]
    r9c_path = Path("results/prompt_ablation_r9_confirm/prompt_ablation_qwen3-32b.jsonl")
    if r9c_path.exists():
        r9c_d = compute_stats(load_jsonl(r9c_path))
        if "combo_bounty_selfcheck" in r9c_d:
            data_32b["bounty+selfcheck"] = r9c_d["combo_bounty_selfcheck"]
        if "combo_render_selfcheck" in r9c_d:
            data_32b["render+selfcheck"] = r9c_d["combo_render_selfcheck"]
    cal_32b_path = Path("results/prompt_ablation_r7_cal_32b/prompt_ablation_qwen3-32b.jsonl")
    if cal_32b_path.exists():
        cal_32b_d = compute_stats(load_jsonl(cal_32b_path))
        if "cal_2shot" in cal_32b_d:
            data_32b["2-shot cal"] = cal_32b_d["cal_2shot"]
    p15_32b = fewshot_dir / "eval_qwen3-32b_15shot.jsonl"
    if p15_32b.exists():
        data_32b["15-shot"] = compute_stats_single(load_jsonl(p15_32b))
    combo_data["32b"] = data_32b

    # 8B: zero-shot from R9 + cal + fewshot
    data_8b: dict[str, dict[str, dict]] = {}
    r9_8b_path_c = Path("results/prompt_ablation_r9_8b/prompt_ablation_qwen3-8b.jsonl")
    if r9_8b_path_c.exists():
        r9_8b_d = compute_stats(load_jsonl(r9_8b_path_c))
        data_8b["baseline"] = r9_8b_d.get("baseline", {})
        if "self_check" in r9_8b_d:
            data_8b["self_check"] = r9_8b_d["self_check"]
        if "combo_bounty_selfcheck" in r9_8b_d:
            data_8b["bounty+selfcheck"] = r9_8b_d["combo_bounty_selfcheck"]
        if "combo_render_selfcheck" in r9_8b_d:
            data_8b["render+selfcheck"] = r9_8b_d["combo_render_selfcheck"]
    cal_8b_path_c = Path("results/prompt_ablation_r7_cal_8b/prompt_ablation_qwen3-8b.jsonl")
    if cal_8b_path_c.exists():
        cal_8b_d = compute_stats(load_jsonl(cal_8b_path_c))
        if "cal_2shot" in cal_8b_d:
            data_8b["2-shot cal"] = cal_8b_d["cal_2shot"]
    p15_8b = fewshot_dir / "eval_qwen3-8b_15shot.jsonl"
    if p15_8b.exists():
        data_8b["15-shot"] = compute_stats_single(load_jsonl(p15_8b))
    combo_data["8b"] = data_8b

    shared_order = [
        "baseline", "self_check",
        "render+selfcheck", "bounty+selfcheck",
        "2-shot cal", "15-shot",
    ]

    if combo_data.get("32b") and combo_data.get("8b"):
        # --- Panel 1: Aggregate ---
        fig, axes = plt.subplots(1, 2, figsize=(16, 5.5))
        bar_colors_p = [COLORS[i % len(COLORS)] for i in range(len(shared_order))]

        for ax, tag, label in [
            (axes[0], "32b", "Qwen3-32B"),
            (axes[1], "8b", "Qwen3-8B"),
        ]:
            d = combo_data[tag]
            vals, errs, labels = [], [], []
            for v in shared_order:
                if v not in d:
                    continue
                avg, se = macro_avg(d[v], MODE_ORDER_8)
                vals.append(avg * 100)
                errs.append(se * 100 * 1.96)
                labels.append(v)

            x = np.arange(len(labels))
            bars = ax.bar(x, vals, yerr=errs, capsize=4,
                          color=bar_colors_p[:len(labels)],
                          edgecolor="white", linewidth=0.5)
            for bar, err, val in zip(bars, errs, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=9)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
            ax.set_title(label, fontsize=13)
            ax.set_ylabel("Compliance (%)", fontsize=12)

        fig.suptitle("Aggregate Compliance: Qwen3-32B vs Qwen3-8B\n(macro-avg across 8 CoTControl modes)",
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_combined_aggregate.png",
                    dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_combined_aggregate.png'}")
        plt.close()

        # --- Panel 2: By-mode ---
        fig, axes = plt.subplots(2, 1, figsize=(22, 13))

        for ax, tag, label in [
            (axes[0], "32b", "Qwen3-32B"),
            (axes[1], "8b", "Qwen3-8B"),
        ]:
            d = combo_data[tag]
            present = [v for v in shared_order if v in d]
            n_labels = len(present)
            width = 0.8 / n_labels
            x = np.arange(len(MODE_ORDER_8))

            for i, v in enumerate(present):
                mode_stats = d[v]
                vals = [mode_stats.get(m, {}).get("rate", 0.0) * 100 for m in MODE_ORDER_8]
                errs = [mode_stats.get(m, {}).get("se", 0.0) * 100 * 1.96 for m in MODE_ORDER_8]
                offset = (i - (n_labels - 1) / 2) * width
                bars = ax.bar(x + offset, vals, width, yerr=errs,
                              capsize=2, label=v, color=COLORS[i % len(COLORS)])
                for bar, err in zip(bars, errs):
                    h = bar.get_height()
                    if h > 1.0:
                        ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                                f"{h:.1f}%", ha="center", va="bottom", fontsize=6.5)

            ax.set_xticks(x)
            ax.set_xticklabels([m.replace("_", "\n") for m in MODE_ORDER_8], fontsize=9)
            ax.set_ylabel("Compliance (%)", fontsize=11)
            ax.set_title(label, fontsize=13)
            ax.legend(fontsize=8, loc="upper right", ncol=2)

        fig.suptitle("Per-Mode Compliance: Qwen3-32B vs Qwen3-8B", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_combined_by_mode.png",
                    dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_combined_by_mode.png'}")
        plt.close()

    # ── 12. Reasoning length analysis plots ──
    # Collect reasoning lengths per variant × mode from the same data sources
    def _get_reasoning_len(r: dict) -> int | None:
        if "reasoning_len" in r:
            return r["reasoning_len"]
        if "reasoning" in r and r["reasoning"]:
            return len(r["reasoning"])
        return None

    def _collect_lengths(rows: list[dict], variant_label: str | None = None) -> list[dict]:
        """Return list of {variant, mode, reasoning_len, compliant} dicts."""
        out = []
        for r in rows:
            rlen = _get_reasoning_len(r)
            if rlen is None:
                continue
            m = _get_mode(r)
            c = _get_compliant(r)
            if not m or c is None:
                continue
            v = variant_label or r.get("variant", "")
            out.append({"variant": v, "mode": m, "reasoning_len": rlen, "compliant": c})
        return out

    all_len_rows: list[dict] = []

    # Zero-shot from R7
    r7_zs_path_l = Path("results/prompt_ablation_r7_zs_32b/prompt_ablation_qwen3-32b.jsonl")
    if r7_zs_path_l.exists():
        r7_zs_rows = load_jsonl(r7_zs_path_l)
        for r in r7_zs_rows:
            v = r.get("variant", "")
            if v in ("baseline", "self_check"):
                all_len_rows.extend(_collect_lengths([r]))

    # Combos from R9 confirm
    r9c_path_l = Path("results/prompt_ablation_r9_confirm/prompt_ablation_qwen3-32b.jsonl")
    if r9c_path_l.exists():
        r9c_rows = load_jsonl(r9c_path_l)
        rename_l = {"combo_render_selfcheck": "render+selfcheck", "combo_bounty_selfcheck": "bounty+selfcheck"}
        for r in r9c_rows:
            v = r.get("variant", "")
            if v in rename_l:
                all_len_rows.extend(_collect_lengths([{**r, "variant": rename_l[v]}]))

    # Calibration 2-shot
    cal_32b_l = Path("results/prompt_ablation_r7_cal_32b/prompt_ablation_qwen3-32b.jsonl")
    if cal_32b_l.exists():
        cal_rows = load_jsonl(cal_32b_l)
        for r in cal_rows:
            if r.get("variant") == "cal_2shot":
                all_len_rows.extend(_collect_lengths([{**r, "variant": "2-shot cal"}]))

    # 15-shot fewshot
    p15_l = fewshot_dir / "eval_qwen3-32b_15shot.jsonl"
    if p15_l.exists():
        all_len_rows.extend(_collect_lengths(load_jsonl(p15_l), variant_label="15-shot"))

    if all_len_rows:
        import pandas as pd

        df = pd.DataFrame(all_len_rows)
        df["reasoning_len_k"] = df["reasoning_len"] / 1000

        len_variants = ["baseline", "self_check", "render+selfcheck",
                        "bounty+selfcheck", "2-shot cal", "15-shot"]
        len_variants = [v for v in len_variants if v in df["variant"].unique()]

        len_colors = {v: COLORS[i % len(COLORS)] for i, v in enumerate(len_variants)}

        # --- Plot A: By prompt variant (aggregate across modes) ---
        fig, ax = plt.subplots(figsize=(12, 5.5))

        vals, errs, sems, labels, ns = [], [], [], [], []
        for v in len_variants:
            sub = df[df["variant"] == v]
            mean = sub["reasoning_len_k"].mean()
            se = sub["reasoning_len_k"].std() / np.sqrt(len(sub))
            vals.append(mean)
            sems.append(se)
            errs.append(se * 1.96)
            labels.append(v)
            ns.append(len(sub))

        x = np.arange(len(labels))
        bars = ax.bar(x, vals, yerr=errs, capsize=5,
                      color=[len_colors[v] for v in labels],
                      edgecolor="white", linewidth=0.5)
        for bar, err, val, n in zip(bars, errs, vals, ns):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err + 0.05,
                    f"{val:.1f}k", ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + err + 0.35,
                    f"n={n}", ha="center", va="bottom", fontsize=8, color="gray")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=10)
        ax.set_ylabel("Mean reasoning length (k chars)", fontsize=12)
        ax.set_title("Reasoning Length by Prompt Variant (Qwen3-32B)\n"
                     "all samples, 95% CI", fontsize=13)
        ax.set_ylim(0, max(vals) * 1.25 + 0.5)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_reasoning_len_by_prompt.png",
                    dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_reasoning_len_by_prompt.png'}")
        plt.close()

        # --- Plot B: By mode (aggregate across variants) ---
        fig, ax = plt.subplots(figsize=(14, 5.5))

        modes_present = [m for m in MODE_ORDER_8 if m in df["mode"].unique()]
        n_vars = len(len_variants)
        width = 0.8 / n_vars
        x = np.arange(len(modes_present))

        for i, v in enumerate(len_variants):
            sub_v = df[df["variant"] == v]
            m_vals, m_errs, m_ns = [], [], []
            for m in modes_present:
                sub = sub_v[sub_v["mode"] == m]
                if len(sub) > 0:
                    mean = sub["reasoning_len_k"].mean()
                    se = sub["reasoning_len_k"].std() / np.sqrt(len(sub))
                    m_vals.append(mean)
                    m_errs.append(se * 1.96)
                    m_ns.append(len(sub))
                else:
                    m_vals.append(0)
                    m_errs.append(0)
                    m_ns.append(0)
            offset = (i - (n_vars - 1) / 2) * width
            bars = ax.bar(x + offset, m_vals, width, yerr=m_errs,
                          capsize=2, label=v, color=len_colors[v])
            for bar, err, val, n in zip(bars, m_errs, m_vals, m_ns):
                h = bar.get_height()
                if h > 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            h + err + 0.05,
                            f"{val:.1f}k", ha="center", va="bottom",
                            fontsize=6, fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in modes_present], fontsize=9)
        ax.set_ylabel("Mean reasoning length (k chars)", fontsize=12)
        ax.set_title("Reasoning Length by Mode × Prompt Variant (Qwen3-32B)\n"
                     "95% CI error bars", fontsize=13)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_reasoning_len_by_mode.png",
                    dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_reasoning_len_by_mode.png'}")
        plt.close()

        # --- Plot B2: By mode, compliant only ---
        df_comp = df[df["compliant"] == True]
        fig, ax = plt.subplots(figsize=(14, 5.5))

        modes_present_c = [m for m in MODE_ORDER_8 if m in df_comp["mode"].unique()]
        n_vars_c = len(len_variants)
        width_c = 0.8 / n_vars_c
        x_c = np.arange(len(modes_present_c))

        for i, v in enumerate(len_variants):
            sub_v = df_comp[df_comp["variant"] == v]
            m_vals, m_errs, m_ns = [], [], []
            for m in modes_present_c:
                sub = sub_v[sub_v["mode"] == m]
                if len(sub) >= 2:
                    mean = sub["reasoning_len_k"].mean()
                    se = sub["reasoning_len_k"].std() / np.sqrt(len(sub))
                    m_vals.append(mean)
                    m_errs.append(se * 1.96)
                    m_ns.append(len(sub))
                elif len(sub) == 1:
                    m_vals.append(sub["reasoning_len_k"].iloc[0])
                    m_errs.append(0)
                    m_ns.append(1)
                else:
                    m_vals.append(0)
                    m_errs.append(0)
                    m_ns.append(0)
            offset = (i - (n_vars_c - 1) / 2) * width_c
            bars = ax.bar(x_c + offset, m_vals, width_c, yerr=m_errs,
                          capsize=2, label=v, color=len_colors[v])
            for bar, err, val, n in zip(bars, m_errs, m_vals, m_ns):
                h = bar.get_height()
                if h > 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            h + err + 0.05,
                            f"{val:.1f}k\n(n={n})", ha="center", va="bottom",
                            fontsize=5.5)

        ax.set_xticks(x_c)
        ax.set_xticklabels([m.replace("_", "\n") for m in modes_present_c], fontsize=9)
        ax.set_ylabel("Mean reasoning length (k chars)", fontsize=12)
        ax.set_title("Reasoning Length by Mode × Prompt (Qwen3-32B, compliant only)\n"
                     "95% CI error bars, n shown per bar", fontsize=13)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_reasoning_len_by_mode_compliant.png",
                    dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_reasoning_len_by_mode_compliant.png'}")
        plt.close()

        # --- Plot C: Compliant vs non-compliant split, by prompt ---
        fig, axes = plt.subplots(1, 2, figsize=(18, 5.5))

        for ax, comp_val, comp_label in [
            (axes[0], True, "Compliant responses"),
            (axes[1], False, "Non-compliant responses"),
        ]:
            sub_comp = df[df["compliant"] == comp_val]
            vals_c, errs_c, labels_c, ns_c = [], [], [], []
            for v in len_variants:
                sub = sub_comp[sub_comp["variant"] == v]
                if len(sub) >= 2:
                    mean = sub["reasoning_len_k"].mean()
                    se = sub["reasoning_len_k"].std() / np.sqrt(len(sub))
                    vals_c.append(mean)
                    errs_c.append(se * 1.96)
                    ns_c.append(len(sub))
                    labels_c.append(v)
                elif len(sub) == 1:
                    vals_c.append(sub["reasoning_len_k"].iloc[0])
                    errs_c.append(0)
                    ns_c.append(1)
                    labels_c.append(v)

            if not vals_c:
                ax.set_title(f"{comp_label}\n(no data)")
                continue

            xc = np.arange(len(labels_c))
            bars = ax.bar(xc, vals_c, yerr=errs_c, capsize=4,
                          color=[len_colors[v] for v in labels_c],
                          edgecolor="white", linewidth=0.5)
            for bar, err, val, n in zip(bars, errs_c, vals_c, ns_c):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 0.05,
                        f"{val:.1f}k", ha="center", va="bottom",
                        fontsize=10, fontweight="bold")
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 0.35,
                        f"n={n}", ha="center", va="bottom",
                        fontsize=8, color="gray")

            ax.set_xticks(xc)
            ax.set_xticklabels(labels_c, rotation=20, ha="right", fontsize=9)
            ax.set_ylabel("Mean reasoning length (k chars)", fontsize=11)
            ax.set_title(comp_label, fontsize=13)

        top = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(0, top * 1.05)

        fig.suptitle("Reasoning Length: Compliant vs Non-Compliant (Qwen3-32B)\n"
                     "95% CI, n shown per bar", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_reasoning_len_compliant_split.png",
                    dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_reasoning_len_compliant_split.png'}")
        plt.close()

    # ── 13. GPT-OSS results ──
    # Load initial GPT-OSS runs (zero-shot + 2-shot cal)
    gptoss_data: dict[str, dict[str, dict[str, dict]]] = {}
    gptoss_rename = {
        "combo_render_selfcheck": "render+selfcheck",
        "combo_bounty_selfcheck": "bounty+selfcheck",
        "cal_2shot": "2-shot cal",
    }

    for size in ["20b", "120b"]:
        p = Path(f"results/gptoss_gpt-oss-{size}/prompt_ablation_gpt-oss-{size}.jsonl")
        if not p.exists():
            continue
        raw = compute_stats(load_jsonl(p))
        renamed: dict[str, dict[str, dict]] = {}
        for v, mstats in raw.items():
            label = gptoss_rename.get(v, v)
            renamed[label] = mstats
        gptoss_data[size] = renamed

    gptoss_order = [
        "baseline", "self_check", "render_mode",
        "render+selfcheck", "bounty+selfcheck", "2-shot cal",
    ]

    for size, data in gptoss_data.items():
        present = [v for v in gptoss_order if v in data]
        tag = f"GPT-OSS-{size.upper()}"

        plot_aggregate(
            data, present, MODE_ORDER_8,
            output_dir / f"plot_gptoss_{size}_aggregate.png",
            title=f"{tag}: Aggregate Compliance\nmacro-avg across 8 modes, n=200/mode",
            figsize=(12, 5),
        )
        plot_by_mode(
            data, present, MODE_ORDER_8,
            output_dir / f"plot_gptoss_{size}_by_mode.png",
            title=f"{tag}: Per-Mode Compliance (n=200)",
            figsize=(18, 6),
        )

    # Load R3 confirmation data (best calibration variants, 8 modes × 200)
    gptoss_r3: dict[str, dict[str, dict[str, dict]]] = {}
    gptoss_r3_rename = {
        "cal_2shot": "2-shot cal",
        "cal_3shot": "3-shot cal",
        "cal_3shot_strong": "3-shot cal (strong)",
        "cal_5shot_system": "5-shot cal (system)",
    }

    for size in ["20b", "120b"]:
        p = Path(f"results/gptoss_r3_confirm_{size}/prompt_ablation_gpt-oss-{size}.jsonl")
        if not p.exists():
            continue
        raw = compute_stats(load_jsonl(p))
        renamed: dict[str, dict[str, dict]] = {}
        for v, mstats in raw.items():
            label = gptoss_r3_rename.get(v, v)
            renamed[label] = mstats
        gptoss_r3[size] = renamed

    gptoss_r3_order = [
        "baseline", "2-shot cal", "3-shot cal",
        "3-shot cal (strong)", "5-shot cal (system)",
    ]

    for size, data in gptoss_r3.items():
        present = [v for v in gptoss_r3_order if v in data]
        tag = f"GPT-OSS-{size.upper()}"

        plot_aggregate(
            data, present, MODE_ORDER_8,
            output_dir / f"plot_gptoss_r3_{size}_aggregate.png",
            title=f"{tag}: Calibration Variants\nmacro-avg across 8 modes, n=200/mode",
            figsize=(12, 5),
        )
        plot_by_mode(
            data, present, MODE_ORDER_8,
            output_dir / f"plot_gptoss_r3_{size}_by_mode.png",
            title=f"{tag}: Calibration Variants Per-Mode (n=200)",
            figsize=(18, 6),
        )

    # Load R4 confirmation data (high-shot calibration, 8 modes × 200)
    gptoss_r4: dict[str, dict[str, dict[str, dict]]] = {}
    gptoss_r4_rename = {
        "cal_10shot_system": "10-shot cal (system)",
        "cal_15shot_system": "15-shot cal (system)",
    }

    for size in ["20b", "120b"]:
        p = Path(f"results/gptoss_r4_confirm_{size}/prompt_ablation_gpt-oss-{size}.jsonl")
        if not p.exists():
            continue
        raw = compute_stats(load_jsonl(p))
        renamed: dict[str, dict[str, dict]] = {}
        for v, mstats in raw.items():
            label = gptoss_r4_rename.get(v, v)
            renamed[label] = mstats
        gptoss_r4[size] = renamed

    gptoss_r4_order = [
        "baseline", "10-shot cal (system)", "15-shot cal (system)",
    ]

    for size, data in gptoss_r4.items():
        present = [v for v in gptoss_r4_order if v in data]
        tag = f"GPT-OSS-{size.upper()}"

        plot_aggregate(
            data, present, MODE_ORDER_8,
            output_dir / f"plot_gptoss_r4_{size}_aggregate.png",
            title=f"{tag}: High-Shot Calibration\nmacro-avg across 8 modes, n=200/mode",
            figsize=(10, 5),
        )
        plot_by_mode(
            data, present, MODE_ORDER_8,
            output_dir / f"plot_gptoss_r4_{size}_by_mode.png",
            title=f"{tag}: High-Shot Calibration Per-Mode (n=200)",
            figsize=(18, 6),
        )

    # --- GPT-OSS shot-scaling plot: baseline vs 5/10/15-shot across both sizes ---
    for size in ["20b", "120b"]:
        merged: dict[str, dict[str, dict]] = {}
        if size in gptoss_r3:
            for label in ["baseline", "5-shot cal (system)"]:
                if label in gptoss_r3[size]:
                    merged[label] = gptoss_r3[size][label]
        if size in gptoss_r4:
            for label in ["10-shot cal (system)", "15-shot cal (system)"]:
                if label in gptoss_r4[size]:
                    merged[label] = gptoss_r4[size][label]
            if "baseline" not in merged and "baseline" in gptoss_r4[size]:
                merged["baseline"] = gptoss_r4[size]["baseline"]
        if len(merged) >= 3:
            scaling_order = ["baseline", "5-shot cal (system)",
                             "10-shot cal (system)", "15-shot cal (system)"]
            present = [v for v in scaling_order if v in merged]
            tag = f"GPT-OSS-{size.upper()}"
            plot_aggregate(
                merged, present, MODE_ORDER_8,
                output_dir / f"plot_gptoss_shot_scaling_{size}_aggregate.png",
                title=f"{tag}: Shot-Scaling (5→10→15)\nmacro-avg across 8 modes, n=200/mode",
                figsize=(10, 5),
            )
            plot_by_mode(
                merged, present, MODE_ORDER_8,
                output_dir / f"plot_gptoss_shot_scaling_{size}_by_mode.png",
                title=f"{tag}: Shot-Scaling Per-Mode (n=200)",
                figsize=(18, 6),
            )

    # --- 4-model comparison: Qwen3-32B vs Qwen3-8B vs GPT-OSS-120B vs GPT-OSS-20B ---
    # Merge R3 + R4 data for GPT-OSS (best variants from each round)
    all_models: dict[str, dict[str, dict[str, dict]]] = {}

    if combo_data.get("32b"):
        all_models["Qwen3-32B"] = combo_data["32b"]
    if combo_data.get("8b"):
        all_models["Qwen3-8B"] = combo_data["8b"]
    for size in ["120b", "20b"]:
        merged_gptoss: dict[str, dict[str, dict]] = {}
        for src in [gptoss_data.get(size, {}), gptoss_r3.get(size, {}), gptoss_r4.get(size, {})]:
            merged_gptoss.update(src)
        if merged_gptoss:
            all_models[f"GPT-OSS-{size.upper()}"] = merged_gptoss

    cross_model_order = [
        "baseline", "self_check",
        "render+selfcheck", "bounty+selfcheck",
        "2-shot cal", "5-shot cal (system)",
        "10-shot cal (system)", "15-shot cal (system)",
    ]

    if len(all_models) >= 3:
        # --- Panel 1: Aggregate side-by-side ---
        n_models = len(all_models)
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 5.5))
        if n_models == 1:
            axes = [axes]
        bar_colors_m = [COLORS[i % len(COLORS)] for i in range(len(cross_model_order))]

        for ax, (model_name, mdata) in zip(axes, all_models.items()):
            present = [v for v in cross_model_order if v in mdata]
            vals, errs, labels = [], [], []
            for v in present:
                avg, se = macro_avg(mdata[v], MODE_ORDER_8)
                vals.append(avg * 100)
                errs.append(se * 100 * 1.96)
                labels.append(v)

            x = np.arange(len(labels))
            bars = ax.bar(x, vals, yerr=errs, capsize=4,
                          color=bar_colors_m[:len(labels)],
                          edgecolor="white", linewidth=0.5)
            for bar, err, val in zip(bars, errs, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
            ax.set_title(model_name, fontsize=13)
            ax.set_ylabel("Compliance (%)" if ax == axes[0] else "", fontsize=12)

        top = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(0, top * 1.15 + 2)

        fig.suptitle("Aggregate Compliance Across Models\n"
                     "(macro-avg across 8 CoTControl modes, 95% CI)",
                     fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_all_models_aggregate.png",
                    dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_all_models_aggregate.png'}")
        plt.close()

        # --- Panel 2: By-mode stacked ---
        fig, axes = plt.subplots(n_models, 1, figsize=(22, 4.5 * n_models))
        if n_models == 1:
            axes = [axes]

        for ax, (model_name, mdata) in zip(axes, all_models.items()):
            present = [v for v in cross_model_order if v in mdata]
            n_labels = len(present)
            width = 0.8 / n_labels
            x = np.arange(len(MODE_ORDER_8))

            for i, v in enumerate(present):
                mode_stats = mdata[v]
                vals = [mode_stats.get(m, {}).get("rate", 0.0) * 100 for m in MODE_ORDER_8]
                errs = [mode_stats.get(m, {}).get("se", 0.0) * 100 * 1.96 for m in MODE_ORDER_8]
                offset = (i - (n_labels - 1) / 2) * width
                bars = ax.bar(x + offset, vals, width, yerr=errs,
                              capsize=2, label=v, color=COLORS[i % len(COLORS)])
                for bar, err in zip(bars, errs):
                    h = bar.get_height()
                    if h > 1.0:
                        ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                                f"{h:.1f}%", ha="center", va="bottom", fontsize=6.5)

            ax.set_xticks(x)
            ax.set_xticklabels([m.replace("_", "\n") for m in MODE_ORDER_8], fontsize=9)
            ax.set_ylabel("Compliance (%)", fontsize=11)
            ax.set_title(model_name, fontsize=13)
            ax.legend(fontsize=8, loc="upper right", ncol=2)

        fig.suptitle("Per-Mode Compliance Across Models", fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / "plot_all_models_by_mode.png",
                    dpi=150, bbox_inches="tight")
        print(f"Saved: {output_dir / 'plot_all_models_by_mode.png'}")
        plt.close()

    print("\nDone.")
