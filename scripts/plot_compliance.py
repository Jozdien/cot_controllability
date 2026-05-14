"""Plot compliance rates: 0-shot baseline vs METR reference, and few-shot comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0] / ".." / "src"))


def load_rollouts(path: Path) -> list[dict]:
    rollouts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rollouts.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rollouts


def compute_compliance_by_mode(rollouts: list[dict]) -> dict[str, dict]:
    mode_counts: dict[str, dict] = {}
    for r in rollouts:
        mode = r.get("control_mode", "")
        if not mode:
            continue
        if mode not in mode_counts:
            mode_counts[mode] = {"compliant": 0, "total": 0}
        compliant = r.get("compliant")
        if compliant is not None:
            mode_counts[mode]["total"] += 1
            if compliant:
                mode_counts[mode]["compliant"] += 1
    result = {}
    for mode, c in mode_counts.items():
        n = c["total"]
        k = c["compliant"]
        p = k / n if n > 0 else 0.0
        se = np.sqrt(p * (1 - p) / n) if n > 0 else 0.0
        result[mode] = {"rate": p, "se": se, "n": n, "k": k}
    return result


def plot_0shot_vs_metr(result_files: dict[str, Path], output_path: Path):
    metr = {
        "qwen3-8b": {
            "lowercase_thinking": 0.07,
            "word_suppression": 0.02,
            "ignore_question": 0.0,
        },
        "qwen3-32b": {
            "lowercase_thinking": 0.12,
            "word_suppression": 0.09,
            "ignore_question": 0.10,
        },
    }

    modes = ["lowercase_thinking", "word_suppression", "ignore_question"]
    models = list(result_files.keys())

    fig, axes = plt.subplots(1, len(models), figsize=(12, 5), sharey=True)
    if len(models) == 1:
        axes = [axes]

    for ax, model_short in zip(axes, models):
        rollouts = load_rollouts(result_files[model_short])
        ours = compute_compliance_by_mode(rollouts)

        x = np.arange(len(modes))
        width = 0.35

        ours_vals = [ours.get(m, {}).get("rate", 0.0) * 100 for m in modes]
        ours_errs = [ours.get(m, {}).get("se", 0.0) * 100 * 1.96 for m in modes]
        metr_vals = [metr.get(model_short, {}).get(m, 0.0) * 100 for m in modes]

        bars1 = ax.bar(x - width / 2, ours_vals, width, yerr=ours_errs,
                       capsize=3, label="Ours (Tinker)", color="#4C72B0")
        bars2 = ax.bar(x + width / 2, metr_vals, width, label="METR", color="#DD8452")

        ax.set_title(f"Qwen3-{model_short.split('-')[-1].upper()}", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=9)
        ax.set_ylabel("Compliance (%)" if ax == axes[0] else "")
        ax.legend(fontsize=9)

        for bar, err in zip(bars1, ours_errs):
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                        f"{h:.1f}%", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            if bar.get_height() > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f"{bar.get_height():.1f}%", ha="center", va="bottom", fontsize=8)

    fig.suptitle("0-Shot CoT Compliance: Ours (Tinker) vs METR", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def plot_fewshot_comparison(
    condition_files: dict[str, dict[str, Path]],
    modes: list[str],
    output_path: Path,
    figsize: tuple[float, float] | None = None,
):
    condition_names = list(condition_files.keys())
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3"]
    models = list(next(iter(condition_files.values())).keys())

    if figsize is None:
        figsize = (max(14, len(modes) * 1.8), 5)
    fig, axes = plt.subplots(1, len(models), figsize=figsize, sharey=True)
    if len(models) == 1:
        axes = [axes]

    n_cond = len(condition_names)
    width = 0.8 / n_cond

    for ax, model_short in zip(axes, models):
        x = np.arange(len(modes))
        all_bars = []

        for i, cond_name in enumerate(condition_names):
            path = condition_files[cond_name].get(model_short)
            if not path or not path.exists():
                continue
            stats = compute_compliance_by_mode(load_rollouts(path))
            vals = [stats.get(m, {}).get("rate", 0.0) * 100 for m in modes]
            errs = [stats.get(m, {}).get("se", 0.0) * 100 * 1.96 for m in modes]
            offset = (i - (n_cond - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width, yerr=errs,
                          capsize=3, label=cond_name, color=colors[i % len(colors)])
            all_bars.append((bars, errs))

        ax.set_title(f"Qwen3-{model_short.split('-')[-1].upper()}", fontsize=13)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=9)
        ax.set_ylabel("Compliance (%)" if ax == axes[0] else "")
        ax.legend(fontsize=9)

        for bars, errs in all_bars:
            for bar, err in zip(bars, errs):
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.3,
                            f"{h:.1f}%", ha="center", va="bottom", fontsize=7)

    shot_labels = ", ".join(condition_names)
    fig.suptitle(f"CoT Compliance: {shot_labels}", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


def compute_overall_compliance(rollouts: list[dict]) -> dict:
    """Macro-average compliance: average of per-mode rates."""
    by_mode = compute_compliance_by_mode(rollouts)
    if not by_mode:
        return {"rate": 0.0, "se": 0.0, "n": 0, "k": 0}
    rates = [m["rate"] for m in by_mode.values()]
    ses = [m["se"] for m in by_mode.values()]
    p = np.mean(rates)
    se = np.sqrt(np.sum(np.array(ses) ** 2)) / len(ses)
    total_n = sum(m["n"] for m in by_mode.values())
    total_k = sum(m["k"] for m in by_mode.values())
    return {"rate": p, "se": se, "n": total_n, "k": total_k}


def plot_aggregate_comparison(
    condition_files: dict[str, dict[str, Path]],
    output_path: Path,
):
    condition_names = list(condition_files.keys())
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B3"]
    models = list(next(iter(condition_files.values())).keys())

    n_cond = len(condition_names)
    x = np.arange(len(models))
    width = 0.8 / n_cond

    fig, ax = plt.subplots(figsize=(7, 5))
    all_bars = []

    for i, cond_name in enumerate(condition_names):
        vals, errs = [], []
        for model_short in models:
            path = condition_files[cond_name].get(model_short)
            if path and path.exists():
                s = compute_overall_compliance(load_rollouts(path))
                vals.append(s["rate"] * 100)
                errs.append(s["se"] * 100 * 1.96)
            else:
                vals.append(0.0)
                errs.append(0.0)
        offset = (i - (n_cond - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width, yerr=errs,
                      capsize=4, label=cond_name, color=colors[i % len(colors)])
        all_bars.append((bars, errs))

    ax.set_xticks(x)
    ax.set_xticklabels([f"Qwen3-{m.split('-')[-1].upper()}" for m in models], fontsize=12)
    ax.set_ylabel("Compliance (%)", fontsize=12)
    ax.legend(fontsize=11)

    for bars, errs in all_bars:
        for bar, err in zip(bars, errs):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + err + 0.2,
                    f"{h:.1f}%", ha="center", va="bottom", fontsize=10)

    ax.set_title("Overall CoT Compliance (averaged across modes)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")
    plt.close()


if __name__ == "__main__":
    results_dir = Path("results/rollouts")
    fewshot_dir = Path("results/fewshot_tinker")

    # Find the 0-shot baseline result files
    eight_b = list(results_dir.glob("qwen_qwen3-8b_cotcontrol_all_*.jsonl"))
    thirty_two_b = list(results_dir.glob("qwen_qwen3-32b_cotcontrol_all_*.jsonl"))

    if eight_b and thirty_two_b:
        result_files = {
            "qwen3-8b": eight_b[0],
            "qwen3-32b": thirty_two_b[0],
        }
        plot_0shot_vs_metr(result_files, Path("results/plot_0shot_vs_metr.png"))

    # Build multi-condition file maps
    original_modes = ["lowercase_thinking", "word_suppression", "ignore_question"]
    all_modes = [
        "lowercase_thinking", "word_suppression", "ignore_question",
        "uppercase_thinking", "alternating_case", "repeat_sentences",
        "end_of_sentence", "meow_between_words", "multiple_word_suppression",
    ]
    model_shorts = ["qwen3-8b", "qwen3-32b"]
    shot_counts = [0, 3, 5, 15]

    condition_files: dict[str, dict[str, Path]] = {}
    for n in shot_counts:
        label = f"{n}-shot"
        files = {}
        for model_short in model_shorts:
            p = fewshot_dir / f"eval_{model_short}_{n}shot.jsonl"
            if p.exists():
                files[model_short] = p
        if files:
            condition_files[label] = files

    if condition_files:
        # Original 3-mode comparison (0/3/5/15-shot)
        plot_fewshot_comparison(
            condition_files, original_modes,
            Path("results/plot_fewshot_comparison.png"),
        )
        plot_aggregate_comparison(
            condition_files,
            Path("results/plot_aggregate.png"),
        )

        # All 9 modes — only conditions that have them (15-shot)
        all_mode_conditions = {
            k: v for k, v in condition_files.items() if k == "15-shot"
        }
        if all_mode_conditions:
            plot_fewshot_comparison(
                all_mode_conditions, all_modes,
                Path("results/plot_all_modes_15shot.png"),
            )

        # 0-shot vs 15-shot across all 9 modes
        zero_vs_fifteen = {
            k: v for k, v in condition_files.items() if k in ("0-shot", "15-shot")
        }
        if len(zero_vs_fifteen) == 2:
            plot_fewshot_comparison(
                zero_vs_fifteen, all_modes,
                Path("results/plot_0shot_vs_15shot_all_modes.png"),
            )

    # 15-shot original vs truncated variants
    trunc_variants = [
        ("15-shot", fewshot_dir),
        ("15-shot (450-tok)", Path("results/fewshot_tinker_450tok")),
        ("15-shot (300-tok)", Path("results/fewshot_tinker_300tok")),
    ]
    trunc_conditions: dict[str, dict[str, Path]] = {}
    for label, d in trunc_variants:
        files = {}
        for model_short in model_shorts:
            p = d / f"eval_{model_short}_15shot.jsonl"
            if p.exists():
                files[model_short] = p
        if files:
            trunc_conditions[label] = files

    if len(trunc_conditions) >= 2:
        plot_fewshot_comparison(
            trunc_conditions, original_modes,
            Path("results/plot_15shot_truncation.png"),
        )
