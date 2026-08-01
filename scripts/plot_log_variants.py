"""Generate symlog versions of three cross-model plots for comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_final import (
    COLORS,
    MODE_ORDER_8,
    _hc_stats,
    load_model_data,
    load_fewshot_data,
    macro_avg,
    QWEN_RENAME,
    GPTOSS_RENAME,
)

RESULTS = Path("results")
out = RESULTS / "plots" / "main"

LINTHRESH = 1.0


def _symlog_setup(ax):
    ax.set_yscale("symlog", linthresh=LINTHRESH)
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.yaxis.set_minor_locator(mticker.NullLocator())


def load_all_data():
    q8b_data = load_model_data([
        RESULTS / "prompt_ablation_r7_zs_8b" / "prompt_ablation_qwen3-8b.jsonl",
        RESULTS / "prompt_ablation_r7_cal_8b" / "prompt_ablation_qwen3-8b.jsonl",
        RESULTS / "prompt_ablation_r9_8b" / "prompt_ablation_qwen3-8b.jsonl",
        RESULTS / "prompt_ablation_cal_qwen8b" / "prompt_ablation_qwen3-8b.jsonl",
    ], QWEN_RENAME)
    q8b_fewshot_15 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-8b_15shot.jsonl",
        RESULTS / "fewshot_tinker_newmodes" / "eval_qwen3-8b_15shot.jsonl",
    ], "fewshot 15-shot")
    q8b_data.update(q8b_fewshot_15)
    q8b_data["zs: repeat 15x"] = _hc_stats([0.10, 0.12, 0.06, 0.00, 0.00, 0.00, 0.00, 0.02])
    q8b_data["zs: metacognition"] = _hc_stats([0.02, 0.08, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00])

    q32b_data = load_model_data([
        RESULTS / "prompt_ablation_r7_zs_32b" / "prompt_ablation_qwen3-32b.jsonl",
        RESULTS / "prompt_ablation_r7_cal_32b" / "prompt_ablation_qwen3-32b.jsonl",
        RESULTS / "prompt_ablation_r8_confirm" / "prompt_ablation_qwen3-32b.jsonl",
        RESULTS / "prompt_ablation_r9_confirm" / "prompt_ablation_qwen3-32b.jsonl",
        RESULTS / "prompt_ablation_cal_qwen32b" / "prompt_ablation_qwen3-32b.jsonl",
    ], QWEN_RENAME)
    q32b_fewshot_15 = load_fewshot_data([
        RESULTS / "fewshot_tinker" / "eval_qwen3-32b_15shot.jsonl",
        RESULTS / "fewshot_tinker_newmodes" / "eval_qwen3-32b_15shot.jsonl",
    ], "fewshot 15-shot")
    q32b_data.update(q32b_fewshot_15)
    q32b_data["zs: repeat 15x"] = _hc_stats([0.16, 0.24, 0.06, 0.00, 0.00, 0.00, 0.00, 0.00])
    q32b_data["zs: metacognition"] = _hc_stats([0.26, 0.16, 0.08, 0.00, 0.00, 0.00, 0.00, 0.00])

    gptoss20b_data = load_model_data([
        RESULTS / "gptoss_gpt-oss-20b" / "prompt_ablation_gpt-oss-20b.jsonl",
        RESULTS / "gptoss_r3_confirm_20b" / "prompt_ablation_gpt-oss-20b.jsonl",
        RESULTS / "gptoss_r4_confirm_20b" / "prompt_ablation_gpt-oss-20b.jsonl",
    ], GPTOSS_RENAME)
    gptoss20b_fewshot_15 = load_fewshot_data([
        RESULTS / "fewshot_gptoss_20b" / "eval_gpt-oss-20b_15shot.jsonl",
    ], "fewshot 15-shot")
    gptoss20b_data.update(gptoss20b_fewshot_15)
    gptoss20b_data["zs: repeat 15x"] = _hc_stats([0.16, 0.38, 0.20, 0.34, 0.00, 0.02, 0.20, 0.02])
    gptoss20b_data["zs: metacognition"] = _hc_stats([0.16, 0.14, 0.12, 0.16, 0.00, 0.00, 0.40, 0.00])

    gptoss120b_data = load_model_data([
        RESULTS / "gptoss_gpt-oss-120b" / "prompt_ablation_gpt-oss-120b.jsonl",
        RESULTS / "gptoss_r3_confirm_120b" / "prompt_ablation_gpt-oss-120b.jsonl",
        RESULTS / "gptoss_r4_confirm_120b" / "prompt_ablation_gpt-oss-120b.jsonl",
    ], GPTOSS_RENAME)
    gptoss120b_fewshot_15 = load_fewshot_data([
        RESULTS / "fewshot_gptoss_120b" / "eval_gpt-oss-120b_15shot.jsonl",
    ], "fewshot 15-shot")
    gptoss120b_data.update(gptoss120b_fewshot_15)
    gptoss120b_data["zs: repeat 15x"] = _hc_stats([0.02, 0.32, 0.10, 0.06, 0.00, 0.02, 0.08, 0.00])
    gptoss120b_data["zs: metacognition"] = _hc_stats([0.18, 0.22, 0.06, 0.18, 0.06, 0.02, 0.82, 0.02])

    for mdata in [q8b_data, q32b_data, gptoss20b_data, gptoss120b_data]:
        if mdata and "fewshot 15-shot" in mdata:
            mdata["15-shot (from eval)"] = mdata["fewshot 15-shot"]

    all_models = {
        "Qwen3-8B": q8b_data,
        "Qwen3-32B": q32b_data,
        "GPT-OSS-20B": gptoss20b_data,
        "GPT-OSS-120B": gptoss120b_data,
    }
    return all_models


def plot_bestof_log(all_models):
    models_list = list(all_models.keys())
    modes = MODE_ORDER_8

    bl_vals, bl_errs = [], []
    zs_vals, zs_errs = [], []
    fs_vals, fs_errs = [], []

    for name in models_list:
        mdata = all_models[name]
        if "baseline" in mdata:
            avg, se = macro_avg(mdata["baseline"], modes)
            bl_vals.append(avg * 100); bl_errs.append(se * 100 * 1.96)
        else:
            bl_vals.append(0); bl_errs.append(0)

        best_zs_val, best_zs_err = 0, 0
        for v in ["zs: repeat 15x", "zs: metacognition"]:
            if v in mdata:
                avg, se = macro_avg(mdata[v], modes)
                if avg > best_zs_val / 100:
                    best_zs_val, best_zs_err = avg * 100, se * 100 * 1.96
        zs_vals.append(best_zs_val); zs_errs.append(best_zs_err)

        best_fs_val, best_fs_err = 0, 0
        for v, stats in mdata.items():
            if ("shot" in v or "cal" in v) and v not in ("baseline", "fewshot 0-shot"):
                avg, se = macro_avg(stats, modes)
                if avg > best_fs_val / 100:
                    best_fs_val, best_fs_err = avg * 100, se * 100 * 1.96
        fs_vals.append(best_fs_val); fs_errs.append(best_fs_err)

    n = len(models_list)
    x = np.arange(n)
    width = 0.25
    cat_colors = ["#8C8C8C", "#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars_bl = ax.bar(x - width, bl_vals, width, yerr=bl_errs, capsize=4,
                     color=cat_colors[0], edgecolor="white", linewidth=0.5, label="Baseline")
    bars_zs = ax.bar(x, zs_vals, width, yerr=zs_errs, capsize=4,
                     color=cat_colors[1], edgecolor="white", linewidth=0.5, label="Best zero-shot")
    bars_fs = ax.bar(x + width, fs_vals, width, yerr=fs_errs, capsize=4,
                     color=cat_colors[2], edgecolor="white", linewidth=0.5, label="Best few-shot")

    for bars, errs in [(bars_bl, bl_errs), (bars_zs, zs_errs), (bars_fs, fs_errs)]:
        for bar, err in zip(bars, errs):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models_list, fontsize=11)
    ax.set_ylabel("Compliance (%)", fontsize=12)
    ax.set_title("All Models: Baseline vs Best Zero-Shot vs Best Few-Shot\n(macro-avg across 8 modes, symlog scale)", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    _symlog_setup(ax)
    plt.tight_layout()
    plt.savefig(out / "all_models_bestof_summary_log.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'all_models_bestof_summary_log.png'}")
    plt.close()


def plot_cross_aggregate_log(all_models):
    variant_order = [
        "baseline",
        "zs: repeat 15x", "zs: metacognition",
        "15-shot cal (OOD)",
        "15-shot (from eval)",
    ]
    modes = MODE_ORDER_8
    n_models = len(all_models)
    bar_colors = [COLORS[i % len(COLORS)] for i in range(len(variant_order))]

    fig, axes = plt.subplots(1, n_models, figsize=(5.5 * n_models, 5.5))
    if n_models == 1:
        axes = [axes]

    global_top = 0
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
        _symlog_setup(ax)
        if vals:
            global_top = max(global_top, max(v + e for v, e in zip(vals, errs)))

    fig.suptitle("Aggregate Compliance: Best Variants Across Models\n(macro-avg across 8 modes, symlog scale)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "cross_model_highlights_aggregate_log.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'cross_model_highlights_aggregate_log.png'}")
    plt.close()


def plot_cross_by_mode_log(all_models):
    variant_order = [
        "baseline",
        "zs: repeat 15x", "zs: metacognition",
        "15-shot cal (OOD)",
        "15-shot (from eval)",
    ]
    modes = MODE_ORDER_8
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
                if h > 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                            f"{h:.1f}%", ha="center", va="bottom", fontsize=6.5)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=9)
        ax.set_ylabel("Compliance (%)", fontsize=11)
        ax.set_title(model_name, fontsize=13)
        ax.legend(fontsize=8, loc="upper right", ncol=2)
        _symlog_setup(ax)

    fig.suptitle("Per-Mode Compliance: Best Variants Across Models (symlog scale)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out / "cross_model_highlights_by_mode_log.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'cross_model_highlights_by_mode_log.png'}")
    plt.close()


def plot_cross_grouped_log(all_models):
    variant_order = [
        "baseline",
        "zs: repeat 15x", "zs: metacognition",
        "15-shot cal (OOD)",
        "15-shot (from eval)",
    ]
    variant_colors = {
        "baseline":           "#8C8C8C",
        "zs: repeat 15x":    "#7BA3D4",
        "zs: metacognition":  "#3B6EA5",
        "15-shot cal (OOD)":  "#88CC88",
        "15-shot (from eval)":"#2D8E2D",
    }
    modes = MODE_ORDER_8
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
    ax.set_title("Aggregate Compliance: Best Variants Across Models\n(macro-avg across 8 modes, symlog scale)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    _symlog_setup(ax)
    plt.tight_layout()
    plt.savefig(out / "cross_model_highlights_grouped_log.png", dpi=150, bbox_inches="tight")
    print(f"Saved: {out / 'cross_model_highlights_grouped_log.png'}")
    plt.close()


if __name__ == "__main__":
    all_models = load_all_data()
    plot_bestof_log(all_models)
    plot_cross_aggregate_log(all_models)
    plot_cross_by_mode_log(all_models)
    plot_cross_grouped_log(all_models)
