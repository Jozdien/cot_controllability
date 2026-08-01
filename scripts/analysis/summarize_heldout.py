"""Summarize + plot the held-out instruction generalization eval.

Reads results/heldout_instructions/heldout_<model>.jsonl and produces:
  - stdout tables: per-model condition x instruction compliance, macro-avgs,
    accuracy by condition, error rates
  - results/heldout_instructions/summary.csv (long format, one row per
    model x condition x instruction)
  - plots in results/plots/heldout/:
      heldout_macro_by_condition.png   (grouped bars, 4 models)
      heldout_by_instruction.png       (per-instruction breakdown, best conditions)
      heldout_vs_original_modes.png    (macro-avg: held-out vs original 8 modes)

Usage:
    uv run scripts/analysis/summarize_heldout.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results/heldout_instructions")
PLOTS_DIR = Path("results/plots/heldout")

MODELS = ["qwen3-8b", "qwen3-32b", "gpt-oss-20b", "gpt-oss-120b"]
MODEL_LABELS = {
    "qwen3-8b": "Qwen3-8B", "qwen3-32b": "Qwen3-32B",
    "gpt-oss-20b": "GPT-OSS-20B", "gpt-oss-120b": "GPT-OSS-120B",
}

CONDITIONS = [
    "baseline", "metacognition", "repeat_15x", "all_weapons_noprefill",
    "prefill_only", "prefill_repeat", "all_weapons",
    "fs_ood_5shot", "fs_ood5s_inline_prefill", "fs_onpolicy_10shot",
]
COND_LABELS = {
    "baseline": "Baseline",
    "metacognition": "ZS: Metacognition",
    "repeat_15x": "ZS: Repeat 15x",
    "all_weapons_noprefill": "ZS: All Weapons (no PF)",
    "prefill_only": "ZS: Prefill only",
    "prefill_repeat": "ZS: Prefill + Repeat",
    "all_weapons": "ZS: All Weapons",
    "fs_ood_5shot": "FS: OOD 5-shot",
    "fs_ood5s_inline_prefill": "FS: OOD 5s + Inline + PF",
    "fs_onpolicy_10shot": "FS: On-policy 10-shot",
}

INSTRUCTIONS = [
    "pirate_speak", "four_languages", "conclusion_first", "answer_hiding",
    "channel_inversion", "double_solve", "slip_and_repair", "emoji",
    "confidence_tags", "quote_before_use",
]

# For the comparison plot: macro-avg compliance on the ORIGINAL 8 CoTControl
# modes under matching conditions, read from the hillclimb/fewshot output dirs.
ORIGINAL_SOURCES = {
    # model_short -> condition -> (path, variant-name-in-file)
    "qwen3-8b": {
        "baseline": ("results/full_eval_all_qwen_8b/hillclimb_qwen3-8b.jsonl", "baseline"),
        "metacognition": ("results/full_eval_all_qwen_8b/hillclimb_qwen3-8b.jsonl", "metacognition"),
        "repeat_15x": ("results/full_eval_all_qwen_8b/hillclimb_qwen3-8b.jsonl", "repeat_15x"),
        "all_weapons_noprefill": ("results/full_eval_all_zs8_8b/hillclimb_qwen3-8b.jsonl", "all_weapons_noprefill"),
        "prefill_only": ("results/full_eval_all_zs7_8b/hillclimb_qwen3-8b.jsonl", "prefill_only"),
        "all_weapons": ("results/full_eval_all_zs7_8b/hillclimb_qwen3-8b.jsonl", "all_weapons"),
    },
    "qwen3-32b": {
        "baseline": ("results/full_eval_all_qwen/hillclimb_qwen3-32b.jsonl", "baseline"),
        "metacognition": ("results/full_eval_all_qwen/hillclimb_qwen3-32b.jsonl", "metacognition"),
        "repeat_15x": ("results/full_eval_all_qwen/hillclimb_qwen3-32b.jsonl", "repeat_15x"),
        "all_weapons_noprefill": ("results/full_eval_all_zs8/hillclimb_qwen3-32b.jsonl", "all_weapons_noprefill"),
        "prefill_only": ("results/full_eval_all_zs7/hillclimb_qwen3-32b.jsonl", "prefill_only"),
        "all_weapons": ("results/full_eval_all_zs7/hillclimb_qwen3-32b.jsonl", "all_weapons"),
    },
    "gpt-oss-20b": {
        "baseline": ("results/full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl", "baseline"),
        "metacognition": ("results/full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl", "metacognition"),
        "repeat_15x": ("results/full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl", "repeat_15x"),
        "all_weapons_noprefill": ("results/full_eval_all_zs8_oss20b/hillclimb_gpt-oss-20b.jsonl", "all_weapons_noprefill"),
        "prefill_only": ("results/full_eval_all_zs7_oss20b/hillclimb_gpt-oss-20b.jsonl", "prefill_only"),
        "all_weapons": ("results/full_eval_all_zs7_oss20b/hillclimb_gpt-oss-20b.jsonl", "all_weapons"),
    },
    "gpt-oss-120b": {
        "baseline": ("results/full_eval_all_qwen_oss120b/hillclimb_gpt-oss-120b.jsonl", "baseline"),
        "metacognition": ("results/full_eval_all_qwen_oss120b/hillclimb_gpt-oss-120b.jsonl", "metacognition"),
        "repeat_15x": ("results/full_eval_all_qwen_oss120b/hillclimb_gpt-oss-120b.jsonl", "repeat_15x"),
        "all_weapons_noprefill": ("results/full_eval_all_zs8_oss120b/hillclimb_gpt-oss-120b.jsonl", "all_weapons_noprefill"),
        "prefill_only": ("results/full_eval_all_zs7_oss120b/hillclimb_gpt-oss-120b.jsonl", "prefill_only"),
        "all_weapons": ("results/full_eval_all_zs7_oss120b/hillclimb_gpt-oss-120b.jsonl", "all_weapons"),
    },
}

COLORS = {
    "baseline": "#9e9e9e",
    "metacognition": "#5b8db8",
    "repeat_15x": "#4a7ba6",
    "all_weapons_noprefill": "#2e5f8a",
    "prefill_only": "#e8a04c",
    "prefill_repeat": "#d98e3a",
    "all_weapons": "#c77d28",
    "fs_ood_5shot": "#6aa56a",
    "fs_ood5s_inline_prefill": "#4d8b4d",
    "fs_onpolicy_10shot": "#2f6f2f",
}


def load_heldout(model_short: str) -> list[dict]:
    path = RESULTS_DIR / f"heldout_{model_short}.jsonl"
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    # Deduplicate on (variant, mode, sample_id): keep the last graded row
    dedup: dict[tuple, dict] = {}
    for r in rows:
        key = (r["variant"], r["mode"], r["sample_id"])
        prev = dedup.get(key)
        if prev is None or (prev.get("compliant") is None and r.get("compliant") is not None):
            dedup[key] = r
    return list(dedup.values())


def stats(rows: list[dict]) -> dict:
    """Nested {condition: {instruction: (k, n)}} over graded rows."""
    agg: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        if r.get("compliant") is None:
            continue
        d = agg[r["variant"]][r["mode"]]
        d[1] += 1
        if r["compliant"]:
            d[0] += 1
    return agg


def macro_avg(agg: dict, cond: str, instructions=INSTRUCTIONS) -> float | None:
    rates = []
    for instr in instructions:
        k, n = agg.get(cond, {}).get(instr, [0, 0])
        if n == 0:
            return None
        rates.append(k / n)
    return float(np.mean(rates)) if rates else None


def macro_se(agg: dict, cond: str, instructions=INSTRUCTIONS) -> float | None:
    """SE of the macro average via per-instruction binomial SE propagation."""
    variances = []
    for instr in instructions:
        k, n = agg.get(cond, {}).get(instr, [0, 0])
        if n == 0:
            return None
        p = k / n
        variances.append(p * (1 - p) / n)
    return float(np.sqrt(np.sum(variances)) / len(instructions))


def load_original_macro(model_short: str, cond: str) -> float | None:
    src = ORIGINAL_SOURCES.get(model_short, {}).get(cond)
    if src is None:
        return None
    path, variant = src
    p = Path(path)
    if not p.exists():
        return None
    agg: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("variant") != variant or r.get("compliant") is None:
                continue
            d = agg[r["mode"]]
            d[1] += 1
            if r["compliant"]:
                d[0] += 1
    if not agg:
        return None
    return float(np.mean([k / n for k, n in agg.values() if n > 0]))


def print_model_table(model_short: str, agg: dict) -> None:
    print(f"\n{'=' * 118}")
    print(f"{MODEL_LABELS.get(model_short, model_short)} — held-out instruction compliance")
    short = [i[:11] for i in INSTRUCTIONS]
    header = f"{'Condition':<26s}" + "".join(f"{s:>11s}" for s in short) + f"{'MACRO':>9s}"
    print(header)
    print("-" * len(header))
    for cond in CONDITIONS:
        if cond not in agg:
            continue
        cells = []
        for instr in INSTRUCTIONS:
            k, n = agg[cond].get(instr, [0, 0])
            cells.append(f"{k / n:>10.1%} " if n else f"{'—':>11s}")
        m = macro_avg(agg, cond)
        mstr = f"{m:>8.1%}" if m is not None else f"{'—':>8s}"
        print(f"{cond:<26s}" + "".join(cells) + mstr)


def plot_macro_by_condition(all_aggs: dict[str, dict]) -> None:
    fig, ax = plt.subplots(figsize=(13, 5.5))
    models = [m for m in MODELS if all_aggs.get(m)]
    x = np.arange(len(models), dtype=float)
    conds = [c for c in CONDITIONS
             if any(macro_avg(all_aggs[m], c) is not None for m in models)]
    width = 0.85 / len(conds)
    for i, cond in enumerate(conds):
        vals, errs = [], []
        for m in models:
            v = macro_avg(all_aggs[m], cond)
            e = macro_se(all_aggs[m], cond)
            vals.append(v if v is not None else 0.0)
            errs.append(e if e is not None else 0.0)
        ax.bar(x + i * width, vals, width * 0.92, yerr=errs, capsize=1.5,
               label=COND_LABELS[cond], color=COLORS[cond],
               error_kw={"lw": 0.8, "alpha": 0.6})
    ax.set_xticks(x + 0.425 - width / 2)
    ax.set_xticklabels([MODEL_LABELS[m] for m in models])
    ax.set_ylabel("Macro-avg compliance (10 held-out instructions)")
    ax.set_title("Held-out CoT instructions: compliance by prompt strategy")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    fig.tight_layout()
    out = PLOTS_DIR / "heldout_macro_by_condition.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def plot_by_instruction(all_aggs: dict[str, dict]) -> None:
    show_conds = ["baseline", "all_weapons_noprefill", "all_weapons",
                  "fs_ood_5shot", "fs_onpolicy_10shot"]
    models = [m for m in MODELS if all_aggs.get(m)]
    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5), sharey=True)
    for ax, m in zip(axes.flat, models):
        agg = all_aggs[m]
        x = np.arange(len(INSTRUCTIONS), dtype=float)
        conds = [c for c in show_conds if c in agg]
        width = 0.8 / max(1, len(conds))
        for i, cond in enumerate(conds):
            vals = []
            for instr in INSTRUCTIONS:
                k, n = agg[cond].get(instr, [0, 0])
                vals.append(k / n if n else 0.0)
            ax.bar(x + i * width, vals, width * 0.9,
                   label=COND_LABELS[cond], color=COLORS[cond])
        ax.set_title(MODEL_LABELS[m], fontsize=11)
        ax.set_xticks(x + 0.4 - width / 2)
        ax.set_xticklabels([i.replace("_", "\n") for i in INSTRUCTIONS],
                           fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes.flat[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Held-out instructions: per-instruction compliance", y=0.995)
    fig.tight_layout()
    out = PLOTS_DIR / "heldout_by_instruction.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def plot_vs_original(all_aggs: dict[str, dict]) -> None:
    """Generalization gap: original 8 modes vs 10 held-out, per condition."""
    conds = ["baseline", "metacognition", "repeat_15x", "all_weapons_noprefill",
             "prefill_only", "all_weapons"]
    models = [m for m in MODELS if all_aggs.get(m)]
    fig, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4.6),
                             sharey=True)
    if len(models) == 1:
        axes = [axes]
    for ax, m in zip(axes, models):
        xs, ys, labels = [], [], []
        for cond in conds:
            orig = load_original_macro(m, cond)
            held = macro_avg(all_aggs[m], cond)
            if orig is None or held is None:
                continue
            xs.append(orig)
            ys.append(held)
            labels.append(cond)
        lim = max([0.02] + xs + ys) * 1.15
        ax.plot([0, lim], [0, lim], ls="--", lw=0.8, color="#bbbbbb", zorder=0)
        for x, y, lab in zip(xs, ys, labels):
            ax.scatter([x], [y], s=42, color=COLORS[lab], zorder=3)
            ax.annotate(COND_LABELS[lab].replace("ZS: ", ""), (x, y),
                        fontsize=6.5, xytext=(4, 3), textcoords="offset points")
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_title(MODEL_LABELS[m], fontsize=10)
        ax.set_xlabel("Macro compliance, original 8 modes")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    axes[0].set_ylabel("Macro compliance, 10 held-out instructions")
    fig.suptitle("Do prompt strategies generalize to held-out instructions?", y=0.99)
    fig.tight_layout()
    out = PLOTS_DIR / "heldout_vs_original_modes.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


def write_csv(all_aggs: dict[str, dict], all_rows: dict[str, list[dict]]) -> None:
    out = RESULTS_DIR / "summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "condition", "instruction", "compliant", "graded",
                    "compliance_rate", "accuracy", "n_correct_graded",
                    "mean_reasoning_len", "n_errors"])
        for m, agg in all_aggs.items():
            rows = all_rows[m]
            for cond in CONDITIONS:
                for instr in INSTRUCTIONS:
                    k, n = agg.get(cond, {}).get(instr, [0, 0])
                    sub = [r for r in rows
                           if r["variant"] == cond and r["mode"] == instr]
                    corr = [r for r in sub if r.get("correct") is not None]
                    acc = (sum(1 for r in corr if r["correct"]) / len(corr)
                           if corr else None)
                    rlens = [r["reasoning_len"] for r in sub if r["reasoning_len"]]
                    errs = sum(1 for r in sub if r.get("error"))
                    if not sub:
                        continue
                    w.writerow([
                        m, cond, instr, k, n,
                        round(k / n, 4) if n else None,
                        round(acc, 4) if acc is not None else None,
                        len(corr),
                        int(np.mean(rlens)) if rlens else 0,
                        errs,
                    ])
    print(f"Saved {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    all_aggs: dict[str, dict] = {}
    all_rows: dict[str, list[dict]] = {}
    for m in MODELS:
        rows = load_heldout(m)
        if not rows:
            print(f"[skip] no data for {m}")
            continue
        all_rows[m] = rows
        all_aggs[m] = stats(rows)
        print_model_table(m, all_aggs[m])
        n_err = sum(1 for r in rows if r.get("error"))
        n_judge_null = sum(
            1 for r in rows
            if r.get("compliant") is None and not r.get("error")
        )
        print(f"  rows={len(rows)}  errors={n_err}  ungraded={n_judge_null}")

    if not all_aggs:
        print("No data found.")
        return

    write_csv(all_aggs, all_rows)
    plot_macro_by_condition(all_aggs)
    plot_by_instruction(all_aggs)
    plot_vs_original(all_aggs)


if __name__ == "__main__":
    main()
