"""Generate final publication-quality plots for CoT controllability results.

Plots produced (all saved to results/plots/new/):
  1. bestof_summary.png         — baseline vs best ZS vs best FS (no prefill)
  2. highlights_grouped.png     — all non-prefill variants, grouped by model
  3. highlights_by_mode.png     — same, split by mode (one row per model)
  4. eval_fewshot_shot_scaling.png  — on-policy fewshot shot scaling
  5. ood_cal_shot_scaling.png       — OOD calibration shot scaling
  6. vs_metr_sft.png            — baseline/best-ZS/best-FS vs METR SFT
  7. accuracy_by_variant.png    — accuracy per variant
  8. reasoning_length_combined.png — reasoning length (all + compliant only)
"""

from __future__ import annotations

import json
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

R = Path("results")
OUT = R / "plots" / "new"
OUT.mkdir(parents=True, exist_ok=True)

MODE_ORDER = [
    "lowercase_thinking", "word_suppression", "uppercase_thinking",
    "alternating_case", "repeat_sentences", "end_of_sentence",
    "meow_between_words", "multiple_word_suppression",
]

MODEL_ORDER = ["Qwen3-8B", "Qwen3-32B", "GPT-OSS-20B", "GPT-OSS-120B"]

VARIANT_ORDER = [
    "baseline",
    "zs: repeat 15x", "zs: metacognition", "zs: all weapons (no PF)",
    "fs: OOD 5-shot", "fs: on-policy 10-shot",
]

VARIANT_COLORS = {
    "baseline":                 "#8C8C8C",
    "zs: repeat 15x":          "#7BA3D4",
    "zs: metacognition":       "#3B6EA5",
    "zs: all weapons (no PF)": "#08306B",
    "fs: OOD 5-shot":          "#88CC88",
    "fs: on-policy 10-shot":   "#2D8E2D",
}

# ── Data loading ──

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


def _extract_sid(r: dict) -> str:
    sid = r.get("sample_id", "")
    if not sid and "sample" in r and isinstance(r["sample"], dict):
        sid = r["sample"].get("id", "")
    return sid


def compute_mode_stats(rows: list[dict]) -> dict[str, dict]:
    """Compute per-mode compliance stats: {mode: {rate, se, n, k}}."""
    counts: dict[str, dict] = defaultdict(lambda: {"k": 0, "n": 0})
    for r in rows:
        m = r.get("mode") or r.get("control_mode", "")
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
    return stats


def macro_avg(mode_stats: dict[str, dict], modes: list[str]) -> tuple[float, float]:
    present = [mode_stats[m] for m in modes if m in mode_stats]
    if not present:
        return 0.0, 0.0
    avg = float(np.mean([s["rate"] for s in present]))
    se = float(np.sqrt(np.sum([s["se"] ** 2 for s in present])) / len(present))
    return avg, se


def load_hillclimb_variant(path: Path, variant: str) -> list[dict]:
    rows = load_jsonl(path)
    return [r for r in rows if r.get("variant") == variant]


def load_fewshot_dir(dirpath: Path) -> list[dict]:
    rows = []
    for f in sorted(dirpath.glob("*.jsonl")):
        rows.extend(load_jsonl(f))
    return rows


def compute_accuracy(rows: list[dict]) -> dict:
    k, n = 0, 0
    for r in rows:
        sid = _extract_sid(r)
        if r.get("correct") is not None and not sid.startswith("cotcontrol/mmlu_pro"):
            n += 1
            if r["correct"]:
                k += 1
    return {"k": k, "n": n}


def compute_reasoning_lengths(rows: list[dict]) -> dict[str, list[tuple[int, bool]]]:
    data: dict[str, list[tuple[int, bool]]] = {}
    for r in rows:
        mode = r.get("mode") or r.get("control_mode", "")
        if not mode:
            continue
        rl = r.get("reasoning_len")
        if rl is None:
            reasoning = r.get("reasoning", "")
            rl = len(reasoning) if reasoning else 0
        if not rl or rl <= 0:
            continue
        compliant = bool(r.get("compliant"))
        data.setdefault(mode, []).append((rl, compliant))
    return data


# ── Load all data ──

def load_all_data() -> dict[str, dict[str, dict]]:
    """Returns {model: {variant_label: {mode: {rate, se, n, k}}}}."""
    configs = {
        "Qwen3-8B": {
            "baseline":                 (R / "full_eval_all_qwen_8b/hillclimb_qwen3-8b.jsonl", "baseline"),
            "zs: metacognition":        (R / "full_eval_all_qwen_8b/hillclimb_qwen3-8b.jsonl", "metacognition"),
            "zs: repeat 15x":           (R / "full_eval_all_zs7_8b/hillclimb_qwen3-8b.jsonl", "repeat_15x"),
            "zs: all weapons (no PF)":  (R / "full_eval_all_zs8_8b/hillclimb_qwen3-8b.jsonl", "all_weapons_noprefill"),
            "fs: OOD 5-shot":           (R / "full_eval_all_ood5s_8b", None),
            "fs: on-policy 10-shot":    (R / "full_eval_all_1000c10s_8b", None),
        },
        "Qwen3-32B": {
            "baseline":                 (R / "full_eval_all_qwen/hillclimb_qwen3-32b.jsonl", "baseline"),
            "zs: metacognition":        (R / "full_eval_all_qwen/hillclimb_qwen3-32b.jsonl", "metacognition"),
            "zs: repeat 15x":           (R / "full_eval_all_zs7/hillclimb_qwen3-32b.jsonl", "repeat_15x"),
            "zs: all weapons (no PF)":  (R / "full_eval_all_zs8/hillclimb_qwen3-32b.jsonl", "all_weapons_noprefill"),
            "fs: OOD 5-shot":           (R / "full_eval_all_ood5s", None),
            "fs: on-policy 10-shot":    (R / "full_eval_all_1000c10s", None),
        },
        "GPT-OSS-20B": {
            "baseline":                 (R / "full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl", "baseline"),
            "zs: metacognition":        (R / "full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl", "metacognition"),
            "zs: repeat 15x":           (R / "full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl", "repeat_15x"),
            "zs: all weapons (no PF)":  (R / "full_eval_all_zs8_oss20b/hillclimb_gpt-oss-20b.jsonl", "all_weapons_noprefill"),
            "fs: OOD 5-shot":           (R / "full_eval_all_ood5s_oss20b", None),
            "fs: on-policy 10-shot":    (R / "full_eval_all_1000c10s_oss20b", None),
        },
        "GPT-OSS-120B": {
            "baseline":                 (R / "full_eval_all_qwen_oss120b/hillclimb_gpt-oss-120b.jsonl", "baseline"),
            "zs: metacognition":        (R / "full_eval_all_qwen_oss120b/hillclimb_gpt-oss-120b.jsonl", "metacognition"),
            "zs: repeat 15x":           (R / "full_eval_all_zs7_oss120b/hillclimb_gpt-oss-120b.jsonl", "repeat_15x"),
            "zs: all weapons (no PF)":  (R / "full_eval_all_zs8_oss120b/hillclimb_gpt-oss-120b.jsonl", "all_weapons_noprefill"),
            "fs: OOD 5-shot":           (R / "full_eval_all_ood5s_oss120b", None),
            "fs: on-policy 10-shot":    (R / "full_eval_all_1000c10s_oss120b", None),
        },
    }

    all_data: dict[str, dict[str, dict]] = {}
    for model, variants in configs.items():
        all_data[model] = {}
        for label, (path, variant_name) in variants.items():
            try:
                if variant_name is not None:
                    rows = load_hillclimb_variant(path, variant_name)
                else:
                    rows = load_fewshot_dir(path)
                if rows:
                    all_data[model][label] = compute_mode_stats(rows)
            except Exception as e:
                print(f"  Warning: {model}/{label}: {e}")
    return all_data


def load_all_raw_rows() -> dict[str, dict[str, list[dict]]]:
    """Returns {model: {variant_label: [raw_rows]}} for accuracy/rlen computation."""
    configs = {
        "Qwen3-8B": {
            "baseline":                 (R / "full_eval_all_qwen_8b/hillclimb_qwen3-8b.jsonl", "baseline"),
            "zs: metacognition":        (R / "full_eval_all_qwen_8b/hillclimb_qwen3-8b.jsonl", "metacognition"),
            "zs: repeat 15x":           (R / "full_eval_all_zs7_8b/hillclimb_qwen3-8b.jsonl", "repeat_15x"),
            "zs: all weapons (no PF)":  (R / "full_eval_all_zs8_8b/hillclimb_qwen3-8b.jsonl", "all_weapons_noprefill"),
            "fs: OOD 5-shot":           (R / "full_eval_all_ood5s_8b", None),
            "fs: on-policy 10-shot":    (R / "full_eval_all_1000c10s_8b", None),
        },
        "Qwen3-32B": {
            "baseline":                 (R / "full_eval_all_qwen/hillclimb_qwen3-32b.jsonl", "baseline"),
            "zs: metacognition":        (R / "full_eval_all_qwen/hillclimb_qwen3-32b.jsonl", "metacognition"),
            "zs: repeat 15x":           (R / "full_eval_all_zs7/hillclimb_qwen3-32b.jsonl", "repeat_15x"),
            "zs: all weapons (no PF)":  (R / "full_eval_all_zs8/hillclimb_qwen3-32b.jsonl", "all_weapons_noprefill"),
            "fs: OOD 5-shot":           (R / "full_eval_all_ood5s", None),
            "fs: on-policy 10-shot":    (R / "full_eval_all_1000c10s", None),
        },
        "GPT-OSS-20B": {
            "baseline":                 (R / "full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl", "baseline"),
            "zs: metacognition":        (R / "full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl", "metacognition"),
            "zs: repeat 15x":           (R / "full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl", "repeat_15x"),
            "zs: all weapons (no PF)":  (R / "full_eval_all_zs8_oss20b/hillclimb_gpt-oss-20b.jsonl", "all_weapons_noprefill"),
            "fs: OOD 5-shot":           (R / "full_eval_all_ood5s_oss20b", None),
            "fs: on-policy 10-shot":    (R / "full_eval_all_1000c10s_oss20b", None),
        },
        "GPT-OSS-120B": {
            "baseline":                 (R / "full_eval_all_qwen_oss120b/hillclimb_gpt-oss-120b.jsonl", "baseline"),
            "zs: metacognition":        (R / "full_eval_all_qwen_oss120b/hillclimb_gpt-oss-120b.jsonl", "metacognition"),
            "zs: repeat 15x":           (R / "full_eval_all_zs7_oss120b/hillclimb_gpt-oss-120b.jsonl", "repeat_15x"),
            "zs: all weapons (no PF)":  (R / "full_eval_all_zs8_oss120b/hillclimb_gpt-oss-120b.jsonl", "all_weapons_noprefill"),
            "fs: OOD 5-shot":           (R / "full_eval_all_ood5s_oss120b", None),
            "fs: on-policy 10-shot":    (R / "full_eval_all_1000c10s_oss120b", None),
        },
    }

    all_raw: dict[str, dict[str, list[dict]]] = {}
    for model, variants in configs.items():
        all_raw[model] = {}
        for label, (path, variant_name) in variants.items():
            try:
                if variant_name is not None:
                    rows = load_hillclimb_variant(path, variant_name)
                else:
                    rows = load_fewshot_dir(path)
                if rows:
                    all_raw[model][label] = rows
            except Exception:
                pass
    return all_raw


# ── Plot 1: Best-of summary ──

def plot_bestof_summary(all_data, output):
    """Baseline vs best ZS vs best FS per model (no prefill)."""
    zs_variants = [v for v in VARIANT_ORDER if v.startswith("zs:")]
    fs_variants = [v for v in VARIANT_ORDER if v.startswith("fs:")]

    models, bl_vals, bl_errs = [], [], []
    zs_vals, zs_errs, fs_vals, fs_errs = [], [], [], []

    for model in MODEL_ORDER:
        mdata = all_data.get(model, {})
        if not mdata:
            continue
        models.append(model)

        bl_avg, bl_se = macro_avg(mdata.get("baseline", {}), MODE_ORDER)
        bl_vals.append(bl_avg * 100)
        bl_errs.append(bl_se * 100 * 1.96)

        best_zs_val, best_zs_se = 0.0, 0.0
        for v in zs_variants:
            if v in mdata:
                avg, se = macro_avg(mdata[v], MODE_ORDER)
                if avg > best_zs_val:
                    best_zs_val, best_zs_se = avg, se
        zs_vals.append(best_zs_val * 100)
        zs_errs.append(best_zs_se * 100 * 1.96)

        best_fs_val, best_fs_se = 0.0, 0.0
        for v in fs_variants:
            if v in mdata:
                avg, se = macro_avg(mdata[v], MODE_ORDER)
                if avg > best_fs_val:
                    best_fs_val, best_fs_se = avg, se
        fs_vals.append(best_fs_val * 100)
        fs_errs.append(best_fs_se * 100 * 1.96)

    n = len(models)
    x = np.arange(n)
    width = 0.25
    colors = ["#8C8C8C", "#4C72B0", "#55A868"]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    bars_bl = ax.bar(x - width, bl_vals, width, yerr=bl_errs, capsize=4,
                     color=colors[0], edgecolor="white", linewidth=0.5, label="Baseline")
    bars_zs = ax.bar(x, zs_vals, width, yerr=zs_errs, capsize=4,
                     color=colors[1], edgecolor="white", linewidth=0.5, label="Best zero-shot")
    bars_fs = ax.bar(x + width, fs_vals, width, yerr=fs_errs, capsize=4,
                     color=colors[2], edgecolor="white", linewidth=0.5, label="Best few-shot")

    for bars, errs in [(bars_bl, bl_errs), (bars_zs, zs_errs), (bars_fs, fs_errs)]:
        for bar, err in zip(bars, errs):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Compliance (%)", fontsize=12)
    ax.set_title("All Models: Baseline vs Best Zero-Shot vs Best Few-Shot\n"
                 "(macro-avg across 8 modes, 95% CI, no prefill)", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")

    all_tops = [h + e for h, e in zip(bl_vals + zs_vals + fs_vals,
                                       bl_errs + zs_errs + fs_errs)]
    ax.set_ylim(0, max(all_tops) * 1.3 + 2)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


# ── Plot 2: Cross-model highlights grouped ──

def plot_highlights_grouped(all_data, output):
    """All non-prefill variants, one grouped bar chart."""
    model_names = [m for m in MODEL_ORDER if m in all_data]
    n_models = len(model_names)
    n_variants = len(VARIANT_ORDER)
    width = 0.8 / n_variants
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(14, 5.5))
    for i, v in enumerate(VARIANT_ORDER):
        vals, errs = [], []
        for name in model_names:
            mdata = all_data[name]
            if v in mdata:
                avg, se = macro_avg(mdata[v], MODE_ORDER)
                vals.append(avg * 100)
                errs.append(se * 100 * 1.96)
            else:
                vals.append(0)
                errs.append(0)
        offset = (i - (n_variants - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                      color=VARIANT_COLORS.get(v, "#333"),
                      edgecolor="white", linewidth=0.5, label=v)
        for bar, err, val in zip(bars, errs, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Compliance (%)", fontsize=12)
    ax.set_title("Aggregate Compliance: All Non-Prefill Variants Across Models\n"
                 "(macro-avg across 8 modes, 95% CI)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    all_tops = []
    for name in model_names:
        for v in VARIANT_ORDER:
            if v in all_data[name]:
                avg, se = macro_avg(all_data[name][v], MODE_ORDER)
                all_tops.append(avg * 100 + se * 100 * 1.96)
    ax.set_ylim(0, max(all_tops) * 1.3 + 2 if all_tops else 10)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


# ── Plot 3: Cross-model highlights by mode ──

def plot_highlights_by_mode(all_data, output):
    """One subplot row per model, modes on x-axis, bars per variant."""
    model_names = [m for m in MODEL_ORDER if m in all_data]
    n_models = len(model_names)
    fig, axes = plt.subplots(n_models, 1, figsize=(22, 4.5 * n_models))
    if n_models == 1:
        axes = [axes]

    for ax, model_name in zip(axes, model_names):
        mdata = all_data[model_name]
        present = [v for v in VARIANT_ORDER if v in mdata]
        n_labels = len(present)
        width = 0.8 / n_labels
        x = np.arange(len(MODE_ORDER))

        for i, v in enumerate(present):
            mode_data = mdata[v]
            vals, errs = [], []
            for m in MODE_ORDER:
                if m in mode_data:
                    rate = mode_data[m]["rate"] * 100
                    se = mode_data[m]["se"] * 100 * 1.96
                    vals.append(rate)
                    errs.append(se)
                else:
                    vals.append(0)
                    errs.append(0)
            offset = (i - (n_labels - 1) / 2) * width
            color = VARIANT_COLORS.get(v, "#333")
            bars = ax.bar(x + offset, vals, width, yerr=errs,
                          capsize=2, label=v, color=color,
                          edgecolor="white", linewidth=0.5)
            for bar, err in zip(bars, errs):
                h = bar.get_height()
                if h > 0.5:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                            f"{h:.1f}%", ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in MODE_ORDER], fontsize=9)
        ax.set_ylabel("Compliance (%)", fontsize=11)
        ax.set_title(model_name, fontsize=13)
        ax.legend(fontsize=8, loc="upper right", ncol=2)

    fig.suptitle("Per-Mode Compliance: All Non-Prefill Variants Across Models", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


# ── Plot 4: On-policy fewshot shot scaling ──

def _load_fewshot_data(paths, label):
    """Load fewshot data and compute per-mode stats, returning {label: {mode: stats}}."""
    rows = []
    for p in paths:
        if p.exists():
            rows.extend(load_jsonl(p))
    if not rows:
        return {}
    stats = {}
    counts: dict[str, dict] = defaultdict(lambda: {"k": 0, "n": 0})
    for r in rows:
        m = r.get("control_mode") or r.get("mode", "")
        c = r.get("compliant")
        if not m or c is None:
            continue
        counts[m]["n"] += 1
        if c:
            counts[m]["k"] += 1
    for m, ct in counts.items():
        n, k = ct["n"], ct["k"]
        p = k / n if n > 0 else 0.0
        se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        stats[m] = {"rate": p, "se": se, "n": n, "k": k}
    return {label: stats}


def _load_prompt_ablation_data(paths, variant_name, label):
    """Load prompt ablation data filtered by variant."""
    rows = []
    for p in paths:
        if p.exists():
            rows.extend(load_jsonl(p))
    filtered = [r for r in rows if r.get("variant") == variant_name]
    if not filtered:
        return {}
    return {label: compute_mode_stats(filtered)}


def plot_eval_fewshot_shot_scaling(output):
    """On-policy fewshot shot-scaling: baseline, 2, 3, 5, 10, 15-shot."""
    order = ["baseline", "2-shot", "3-shot", "5-shot", "10-shot", "15-shot"]
    colors = {
        "baseline": "#8C8C8C",
        "2-shot":   "#C7E9C0",
        "3-shot":   "#A1D99B",
        "5-shot":   "#74C476",
        "10-shot":  "#238B45",
        "15-shot":  "#00441B",
    }

    BASELINE_PATHS = {
        "Qwen3-8B":     (R / "full_eval_all_qwen_8b/hillclimb_qwen3-8b.jsonl", "baseline"),
        "Qwen3-32B":    (R / "full_eval_all_qwen/hillclimb_qwen3-32b.jsonl", "baseline"),
        "GPT-OSS-20B":  (R / "full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl", "baseline"),
        "GPT-OSS-120B": (R / "full_eval_all_qwen_oss120b/hillclimb_gpt-oss-120b.jsonl", "baseline"),
    }

    FS_PATHS = {
        "Qwen3-8B": [
            (R / "fewshot_scaling_onpolicy/eval_qwen3-8b_2shot.jsonl", "2-shot"),
            (R / "fewshot_scaling_onpolicy/eval_qwen3-8b_3shot.jsonl", "3-shot"),
            (R / "fewshot_scaling_onpolicy/eval_qwen3-8b_5shot.jsonl", "5-shot"),
            (R / "full_eval_all_1000c10s_8b/eval_qwen3-8b_10shot.jsonl", "10-shot"),
            (R / "fewshot_scaling_onpolicy/eval_qwen3-8b_15shot.jsonl", "15-shot"),
        ],
        "Qwen3-32B": [
            (R / "fewshot_scaling_onpolicy/eval_qwen3-32b_2shot.jsonl", "2-shot"),
            (R / "fewshot_scaling_onpolicy/eval_qwen3-32b_3shot.jsonl", "3-shot"),
            (R / "fewshot_scaling_onpolicy/eval_qwen3-32b_5shot.jsonl", "5-shot"),
            (R / "full_eval_all_1000c10s/eval_qwen3-32b_10shot.jsonl", "10-shot"),
            (R / "fewshot_scaling_onpolicy/eval_qwen3-32b_15shot.jsonl", "15-shot"),
        ],
        "GPT-OSS-20B": [
            (R / "fewshot_scaling_onpolicy/eval_gpt-oss-20b_2shot.jsonl", "2-shot"),
            (R / "fewshot_scaling_onpolicy/eval_gpt-oss-20b_3shot.jsonl", "3-shot"),
            (R / "fewshot_scaling_onpolicy/eval_gpt-oss-20b_5shot.jsonl", "5-shot"),
            (R / "full_eval_all_1000c10s_oss20b/eval_gpt-oss-20b_10shot.jsonl", "10-shot"),
            (R / "fewshot_scaling_onpolicy/eval_gpt-oss-20b_15shot.jsonl", "15-shot"),
        ],
        "GPT-OSS-120B": [
            (R / "fewshot_scaling_onpolicy/eval_gpt-oss-120b_2shot.jsonl", "2-shot"),
            (R / "fewshot_scaling_onpolicy/eval_gpt-oss-120b_3shot.jsonl", "3-shot"),
            (R / "fewshot_scaling_onpolicy/eval_gpt-oss-120b_5shot.jsonl", "5-shot"),
            (R / "full_eval_all_1000c10s_oss120b/eval_gpt-oss-120b_10shot.jsonl", "10-shot"),
            (R / "fewshot_scaling_onpolicy/eval_gpt-oss-120b_15shot.jsonl", "15-shot"),
        ],
    }

    all_models: dict[str, dict] = {}
    for model in BASELINE_PATHS:
        mdata: dict[str, dict] = {}
        bl_path, bl_variant = BASELINE_PATHS[model]
        bl_rows = load_hillclimb_variant(bl_path, bl_variant)
        if bl_rows:
            mdata["baseline"] = compute_mode_stats(bl_rows)
        for path, label in FS_PATHS[model]:
            fs = _load_fewshot_data([Path(path)], label)
            mdata.update(fs)
        if mdata:
            all_models[model] = mdata

    if not all_models:
        print("  Skipping eval_fewshot_shot_scaling (no data)")
        return

    _plot_grouped_bar(all_models, order, colors, output,
                      '"On-Policy" Fewshot Shot-Scaling\n(macro-avg across 8 modes, 95% CI)')


def plot_ood_cal_shot_scaling(output):
    """OOD fewshot shot-scaling: baseline, 2, 3, 5, 9-shot."""
    order = ["baseline", "2-shot", "3-shot", "5-shot", "9-shot"]
    colors = {
        "baseline": "#8C8C8C",
        "2-shot":   "#C6DBEF",
        "3-shot":   "#9ECAE1",
        "5-shot":   "#6BAED6",
        "9-shot":   "#08306B",
    }

    BASELINE_PATHS = {
        "Qwen3-8B":     (R / "full_eval_all_qwen_8b/hillclimb_qwen3-8b.jsonl", "baseline"),
        "Qwen3-32B":    (R / "full_eval_all_qwen/hillclimb_qwen3-32b.jsonl", "baseline"),
        "GPT-OSS-20B":  (R / "full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl", "baseline"),
        "GPT-OSS-120B": (R / "full_eval_all_qwen_oss120b/hillclimb_gpt-oss-120b.jsonl", "baseline"),
    }

    FS_PATHS = {
        "Qwen3-8B": [
            (R / "fewshot_scaling_ood/eval_qwen3-8b_2shot.jsonl", "2-shot"),
            (R / "fewshot_scaling_ood/eval_qwen3-8b_3shot.jsonl", "3-shot"),
            (R / "full_eval_all_ood5s_8b/eval_qwen3-8b_5shot.jsonl", "5-shot"),
            (R / "fewshot_scaling_ood/eval_qwen3-8b_9shot.jsonl", "9-shot"),
        ],
        "Qwen3-32B": [
            (R / "fewshot_scaling_ood/eval_qwen3-32b_2shot.jsonl", "2-shot"),
            (R / "fewshot_scaling_ood/eval_qwen3-32b_3shot.jsonl", "3-shot"),
            (R / "full_eval_all_ood5s/eval_qwen3-32b_5shot.jsonl", "5-shot"),
            (R / "fewshot_scaling_ood/eval_qwen3-32b_9shot.jsonl", "9-shot"),
        ],
        "GPT-OSS-20B": [
            (R / "fewshot_scaling_ood/eval_gpt-oss-20b_2shot.jsonl", "2-shot"),
            (R / "fewshot_scaling_ood/eval_gpt-oss-20b_3shot.jsonl", "3-shot"),
            (R / "full_eval_all_ood5s_oss20b/eval_gpt-oss-20b_5shot.jsonl", "5-shot"),
            (R / "fewshot_scaling_ood/eval_gpt-oss-20b_9shot.jsonl", "9-shot"),
        ],
        "GPT-OSS-120B": [
            (R / "fewshot_scaling_ood/eval_gpt-oss-120b_2shot.jsonl", "2-shot"),
            (R / "fewshot_scaling_ood/eval_gpt-oss-120b_3shot.jsonl", "3-shot"),
            (R / "full_eval_all_ood5s_oss120b/eval_gpt-oss-120b_5shot.jsonl", "5-shot"),
            (R / "fewshot_scaling_ood/eval_gpt-oss-120b_9shot.jsonl", "9-shot"),
        ],
    }

    all_models: dict[str, dict] = {}
    for model in BASELINE_PATHS:
        mdata: dict[str, dict] = {}
        bl_path, bl_variant = BASELINE_PATHS[model]
        bl_rows = load_hillclimb_variant(bl_path, bl_variant)
        if bl_rows:
            mdata["baseline"] = compute_mode_stats(bl_rows)
        for path, label in FS_PATHS[model]:
            fs = _load_fewshot_data([Path(path)], label)
            mdata.update(fs)
        if mdata:
            all_models[model] = mdata

    if not all_models:
        print("  Skipping ood_cal_shot_scaling (no data)")
        return

    _plot_grouped_bar(all_models, order, colors, output,
                      "OOD Fewshot Shot-Scaling\n(macro-avg across 8 modes, 95% CI)")


def _plot_grouped_bar(all_models, variant_order, variant_colors, output, title):
    """Generic grouped bar chart."""
    model_names = [m for m in MODEL_ORDER if m in all_models]
    n_models = len(model_names)
    present = [v for v in variant_order
               if any(v in all_models.get(m, {}) for m in model_names)]
    n_variants = len(present)
    width = 0.8 / n_variants
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, v in enumerate(present):
        vals, errs = [], []
        for name in model_names:
            mdata = all_models.get(name, {})
            if v in mdata:
                avg, se = macro_avg(mdata[v], MODE_ORDER)
                vals.append(avg * 100)
                errs.append(se * 100 * 1.96)
            else:
                vals.append(0)
                errs.append(0)
        offset = (i - (n_variants - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                      color=variant_colors.get(v, "#333"),
                      edgecolor="white", linewidth=0.5, label=v)
        for bar, err, val in zip(bars, errs, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + err + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Compliance (%)", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    all_tops = []
    for name in model_names:
        for v in present:
            if v in all_models.get(name, {}):
                avg, se = macro_avg(all_models[name][v], MODE_ORDER)
                all_tops.append(avg * 100 + se * 100 * 1.96)
    ax.set_ylim(0, max(all_tops) * 1.3 + 2 if all_tops else 10)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


# ── Plot 6: vs METR SFT ──

def plot_vs_metr_sft(all_data, output):
    """Baseline, best ZS, best FS vs METR SFT (ReasonIF)."""
    metr_sft = {
        "Qwen3-8B":     5.6,
        "Qwen3-32B":    9.8,
        "GPT-OSS-20B":  7.3,
        "GPT-OSS-120B": 13.1,
    }
    zs_variants = [v for v in VARIANT_ORDER if v.startswith("zs:")]
    fs_variants = [v for v in VARIANT_ORDER if v.startswith("fs:")]

    models = []
    bl_vals, bl_errs = [], []
    zs_vals, zs_errs = [], []
    fs_vals, fs_errs = [], []
    metr_vals = []

    for model in MODEL_ORDER:
        mdata = all_data.get(model, {})
        if not mdata:
            continue
        models.append(model)

        bl_avg, bl_se = macro_avg(mdata.get("baseline", {}), MODE_ORDER)
        bl_vals.append(bl_avg * 100)
        bl_errs.append(bl_se * 100 * 1.96)

        best_zs_val, best_zs_se = 0.0, 0.0
        for v in zs_variants:
            if v in mdata:
                avg, se = macro_avg(mdata[v], MODE_ORDER)
                if avg > best_zs_val:
                    best_zs_val, best_zs_se = avg, se
        zs_vals.append(best_zs_val * 100)
        zs_errs.append(best_zs_se * 100 * 1.96)

        best_fs_val, best_fs_se = 0.0, 0.0
        for v in fs_variants:
            if v in mdata:
                avg, se = macro_avg(mdata[v], MODE_ORDER)
                if avg > best_fs_val:
                    best_fs_val, best_fs_se = avg, se
        fs_vals.append(best_fs_val * 100)
        fs_errs.append(best_fs_se * 100 * 1.96)

        metr_vals.append(metr_sft.get(model, 0.0))

    n = len(models)
    x = np.arange(n)
    width = 0.19
    colors = ["#8C8C8C", "#4C72B0", "#55A868", "#E07B39"]

    fig, ax = plt.subplots(figsize=(12, 5.5))

    bars_bl = ax.bar(x - 1.5 * width, bl_vals, width, yerr=bl_errs, capsize=3,
                     color=colors[0], edgecolor="white", linewidth=0.5, label="Baseline")
    bars_zs = ax.bar(x - 0.5 * width, zs_vals, width, yerr=zs_errs, capsize=3,
                     color=colors[1], edgecolor="white", linewidth=0.5, label="Best zero-shot")
    bars_fs = ax.bar(x + 0.5 * width, fs_vals, width, yerr=fs_errs, capsize=3,
                     color=colors[2], edgecolor="white", linewidth=0.5, label="Best few-shot")
    bars_metr = ax.bar(x + 1.5 * width, metr_vals, width, capsize=3,
                       color=colors[3], edgecolor="white", linewidth=0.5,
                       label="METR SFT (ReasonIF)")

    for bars, errs in [(bars_bl, bl_errs), (bars_zs, zs_errs), (bars_fs, fs_errs),
                       (bars_metr, [0] * n)]:
        for bar, err in zip(bars, errs):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylabel("Compliance (%)", fontsize=12)
    ax.set_title("Prompt Engineering vs METR SFT — ReasonIF-only\n"
                 "(macro-avg across 8 modes, 95% CI)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")

    all_tops = (bl_vals + zs_vals + fs_vals + metr_vals +
                [h + e for h, e in zip(bl_vals + zs_vals + fs_vals,
                                       bl_errs + zs_errs + fs_errs)])
    ax.set_ylim(0, max(all_tops) * 1.3 + 2)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


# ── Plot 7: Accuracy by variant ──

def plot_accuracy_by_variant(all_raw, output):
    """Accuracy bars per variant across models."""
    model_names = [m for m in MODEL_ORDER if m in all_raw]
    present = [v for v in VARIANT_ORDER
               if any(v in all_raw.get(m, {}) for m in model_names)]
    n_variants = len(present)
    width = 0.8 / n_variants
    x = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    for i, v in enumerate(present):
        vals, errs = [], []
        for name in model_names:
            rows = all_raw.get(name, {}).get(v, [])
            acc = compute_accuracy(rows)
            if acc["n"] > 0:
                rate = acc["k"] / acc["n"]
                se = np.sqrt(rate * (1 - rate) / acc["n"])
                vals.append(rate * 100)
                errs.append(se * 100 * 1.96)
            else:
                vals.append(0)
                errs.append(0)
        offset = (i - (n_variants - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                      color=VARIANT_COLORS.get(v, "#333"),
                      edgecolor="white", linewidth=0.5, label=v)
        for bar, err, val in zip(bars, errs, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + err + 0.3,
                        f"{val:.1f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy by Variant — GPQA + HLE\n(95% CI)", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylim(0, 50)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


# ── Plot 8: Reasoning length combined ──

def _rlen_stats(lengths: list[int]) -> tuple[float, float]:
    if not lengths:
        return 0.0, 0.0
    arr = np.array(lengths) / 1000.0
    mean = float(np.mean(arr))
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, se


def plot_reasoning_length_combined(all_raw, output):
    """Two subplots: all responses (left) and compliant-only (right)."""
    model_names = [m for m in MODEL_ORDER if m in all_raw]
    n_models = len(model_names)
    present = [v for v in VARIANT_ORDER
               if any(v in all_raw.get(m, {}) for m in model_names)]
    n_variants = len(present)
    width = 0.8 / n_variants
    x = np.arange(n_models)

    fig, (ax_all, ax_comp) = plt.subplots(1, 2, figsize=(20, 5.5))

    for ax, compliant_only, subtitle in [
        (ax_all, False, "All Responses"),
        (ax_comp, True, "Compliant Only"),
    ]:
        top_val = 0
        for i, v in enumerate(present):
            vals, errs = [], []
            for name in model_names:
                rows = all_raw.get(name, {}).get(v, [])
                rlen_data = compute_reasoning_lengths(rows)
                all_lens = []
                for m in MODE_ORDER:
                    entries = rlen_data.get(m, [])
                    for rl, c in entries:
                        if compliant_only and not c:
                            continue
                        all_lens.append(rl)
                mean, se = _rlen_stats(all_lens)
                vals.append(mean)
                errs.append(se * 1.96)
                top_val = max(top_val, mean + se * 1.96)
            offset = (i - (n_variants - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, yerr=errs, capsize=3,
                          color=VARIANT_COLORS.get(v, "#333"),
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
    fig.suptitle("Mean Reasoning Length by Variant (95% CI)", fontsize=14, y=1.02)
    fig.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")
    plt.close()


# ── Main ──

def main():
    print("Loading data...")
    all_data = load_all_data()
    all_raw = load_all_raw_rows()

    for model in MODEL_ORDER:
        if model in all_data:
            for v, mstats in all_data[model].items():
                avg, se = macro_avg(mstats, MODE_ORDER)
                print(f"  {model} / {v}: {avg*100:.1f}% +/- {se*100*1.96:.1f}%")

    print("\nGenerating plots...")

    plot_bestof_summary(all_data, OUT / "bestof_summary.png")
    plot_highlights_grouped(all_data, OUT / "highlights_grouped.png")
    plot_highlights_by_mode(all_data, OUT / "highlights_by_mode.png")
    plot_eval_fewshot_shot_scaling(OUT / "eval_fewshot_shot_scaling.png")
    plot_ood_cal_shot_scaling(OUT / "ood_cal_shot_scaling.png")
    plot_vs_metr_sft(all_data, OUT / "vs_metr_sft.png")
    plot_accuracy_by_variant(all_raw, OUT / "accuracy_by_variant.png")
    plot_reasoning_length_combined(all_raw, OUT / "reasoning_length_combined.png")

    print(f"\nDone. {len(list(OUT.glob('*.png')))} plots in {OUT}/")


if __name__ == "__main__":
    main()
