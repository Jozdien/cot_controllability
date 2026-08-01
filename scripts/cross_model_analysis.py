#!/usr/bin/env python3
"""Cross-model comparison analysis for CoT controllability experiments."""

import json
import sys
from collections import defaultdict
from pathlib import Path

RESULTS_DIR = Path("/Users/jose/cot_controllability/results")

# 8 control modes
MODES = [
    "uppercase_thinking", "lowercase_thinking", "alternating_case",
    "word_suppression", "multiple_word_suppression", "repeat_sentences",
    "end_of_sentence", "meow_between_words"
]

def load_hillclimb(filepath):
    """Load hillclimb JSONL file, skipping comment line."""
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            records.append(json.loads(line))
    return records

def load_fewshot(filepath):
    """Load fewshot JSONL file, skipping comment line."""
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rec = json.loads(line)
            # Fewshot files have nested structure - flatten what we need
            out = {
                "sample_id": rec.get("sample", {}).get("id", rec.get("sample_id", "")),
                "control_mode": rec.get("control_mode", ""),
                "correct": rec.get("correct"),
                "compliant": rec.get("compliant"),
            }
            # Handle different field names
            if "grading_details" in rec:
                gd = rec["grading_details"]
                if "compliant" in gd:
                    out["compliant"] = gd["compliant"]
                if "mode" in gd:
                    out["control_mode"] = gd["mode"]
            if not out["control_mode"]:
                out["control_mode"] = rec.get("mode", "")
            records.append(out)
    return records

def compute_metrics(records, mode_field="mode", exclude_mmlu_accuracy=True):
    """Compute per-mode compliance and accuracy.

    Returns dict of mode -> {compliance, accuracy, n_compliance, n_accuracy}
    """
    mode_data = defaultdict(lambda: {"compliant_count": 0, "compliant_total": 0,
                                      "correct_count": 0, "correct_total": 0})

    for r in records:
        mode = r.get(mode_field, r.get("control_mode", ""))
        if mode not in MODES:
            continue

        sample_id = r.get("sample_id", "")

        # Compliance: count all non-null
        compliant = r.get("compliant")
        if compliant is not None:
            mode_data[mode]["compliant_total"] += 1
            if compliant:
                mode_data[mode]["compliant_count"] += 1

        # Accuracy: exclude mmlu_pro samples, exclude null correct
        correct = r.get("correct")
        if correct is not None:
            if exclude_mmlu_accuracy and sample_id.startswith("cotcontrol/mmlu_pro"):
                pass  # skip mmlu_pro for accuracy
            else:
                mode_data[mode]["correct_total"] += 1
                if correct:
                    mode_data[mode]["correct_count"] += 1

    results = {}
    for mode in MODES:
        d = mode_data[mode]
        compliance = d["compliant_count"] / d["compliant_total"] if d["compliant_total"] > 0 else None
        accuracy = d["correct_count"] / d["correct_total"] if d["correct_total"] > 0 else None
        results[mode] = {
            "compliance": compliance,
            "accuracy": accuracy,
            "n_compliance": d["compliant_total"],
            "n_accuracy": d["correct_total"],
        }
    return results

def macro_avg(per_mode):
    """Compute macro-averaged compliance and accuracy across modes."""
    comps = [v["compliance"] for v in per_mode.values() if v["compliance"] is not None]
    accs = [v["accuracy"] for v in per_mode.values() if v["accuracy"] is not None]
    return {
        "compliance": sum(comps) / len(comps) if comps else None,
        "accuracy": sum(accs) / len(accs) if accs else None,
        "n_modes_compliance": len(comps),
        "n_modes_accuracy": len(accs),
    }

def filter_variant(records, variant):
    """Filter hillclimb records by variant."""
    return [r for r in records if r.get("variant") == variant]

def filter_modes(records, mode_field="mode"):
    """Filter to only the 8 modes we care about."""
    return [r for r in records if r.get(mode_field, r.get("control_mode", "")) in MODES]

def print_table(title, rows, col_headers):
    """Print a formatted table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")

    # Compute column widths
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows)) for i, h in enumerate(col_headers)]

    # Header
    header = " | ".join(str(h).ljust(w) for h, w in zip(col_headers, widths))
    print(header)
    print("-" * len(header))

    # Rows
    for row in rows:
        print(" | ".join(str(v).ljust(w) for v, w in zip(row, widths)))

def fmt_pct(val):
    """Format as percentage or N/A."""
    if val is None:
        return "N/A"
    return f"{val*100:.1f}%"

def fmt_pct_short(val):
    """Format as percentage without % sign for compact tables."""
    if val is None:
        return "N/A"
    return f"{val*100:.1f}"


# ============================================================
# LOAD ALL DATA
# ============================================================

print("Loading data...")

# --- Qwen3-32B ---
qwen32b_zs7 = load_hillclimb(RESULTS_DIR / "full_eval_all_zs7/hillclimb_qwen3-32b.jsonl")
qwen32b_zsq = load_hillclimb(RESULTS_DIR / "full_eval_all_qwen/hillclimb_qwen3-32b.jsonl")
qwen32b_ood5s = load_fewshot(RESULTS_DIR / "full_eval_all_ood5s/eval_qwen3-32b_5shot.jsonl")
qwen32b_ood5s_inpf = load_fewshot(RESULTS_DIR / "full_eval_all_ood5s_inpf/eval_qwen3-32b_5shot.jsonl")
qwen32b_1000c10s = load_fewshot(RESULTS_DIR / "full_eval_all_1000c10s/eval_qwen3-32b_10shot.jsonl")
qwen32b_1500c10s = load_fewshot(RESULTS_DIR / "full_eval_all_1500c10s/eval_qwen3-32b_10shot.jsonl")

# --- Qwen3-8B ---
qwen8b_ood5s = load_fewshot(RESULTS_DIR / "full_eval_all_ood5s_8b/eval_qwen3-8b_5shot.jsonl")
qwen8b_ood5s_inpf = load_fewshot(RESULTS_DIR / "full_eval_all_ood5s_inpf_8b/eval_qwen3-8b_5shot.jsonl")
# ZS batches still running for 8B

# --- GPT-OSS-20B ---
oss20b_ood5s = load_fewshot(RESULTS_DIR / "full_eval_all_ood5s_oss20b/eval_gpt-oss-20b_5shot.jsonl")
oss20b_ood5s_inpf = load_fewshot(RESULTS_DIR / "full_eval_all_ood5s_inpf_oss20b/eval_gpt-oss-20b_5shot.jsonl")
oss20b_zsq = load_hillclimb(RESULTS_DIR / "full_eval_all_qwen_oss20b/hillclimb_gpt-oss-20b.jsonl")
# ZS batch 7 still running for OSS-20B

# --- GPT-OSS-120B ---
oss120b_ood5s = load_fewshot(RESULTS_DIR / "full_eval_all_ood5s_oss120b/eval_gpt-oss-120b_5shot.jsonl")
oss120b_ood5s_inpf = load_fewshot(RESULTS_DIR / "full_eval_all_ood5s_inpf_oss120b/eval_gpt-oss-120b_5shot.jsonl")
oss120b_zsq = load_hillclimb(RESULTS_DIR / "full_eval_all_qwen_oss120b/hillclimb_gpt-oss-120b.jsonl")
oss120b_zs7 = load_hillclimb(RESULTS_DIR / "full_eval_all_zs7_oss120b/hillclimb_gpt-oss-120b.jsonl")

print("Data loaded.\n")

# ============================================================
# ANALYSIS 1: MACRO-AVERAGED COMPARISON TABLE
# ============================================================

configs = []

def add_config(name, model, records, mode_field="mode"):
    """Add a config to the comparison table."""
    per_mode = compute_metrics(records, mode_field=mode_field)
    avg = macro_avg(per_mode)
    configs.append((name, model, avg, per_mode))

# --- Qwen3-32B ---
# ZS baseline (from batch qwen since it has fewer errors — batch 7 has timeouts)
add_config("ZS baseline", "Qwen3-32B", filter_variant(qwen32b_zsq, "baseline"))

# ZS best-without-prefill: metacognition vs repeat_15x
meta_32b = filter_variant(qwen32b_zsq, "metacognition")
rep15_32b = filter_variant(qwen32b_zsq, "repeat_15x")
meta_m = macro_avg(compute_metrics(meta_32b))
rep15_m = macro_avg(compute_metrics(rep15_32b))
if (meta_m["compliance"] or 0) >= (rep15_m["compliance"] or 0):
    add_config("ZS best-no-prefill (metacog)", "Qwen3-32B", meta_32b)
    print(f"  Qwen3-32B: metacognition ({fmt_pct(meta_m['compliance'])}) >= repeat_15x ({fmt_pct(rep15_m['compliance'])})")
else:
    add_config("ZS best-no-prefill (rep15x)", "Qwen3-32B", rep15_32b)
    print(f"  Qwen3-32B: repeat_15x ({fmt_pct(rep15_m['compliance'])}) > metacognition ({fmt_pct(meta_m['compliance'])})")

# ZS all_weapons (batch 7)
add_config("ZS all_weapons", "Qwen3-32B", filter_variant(qwen32b_zs7, "all_weapons"))

# Fewshot configs
add_config("OOD 5-shot", "Qwen3-32B", qwen32b_ood5s, mode_field="control_mode")
add_config("OOD 5s+inline+prefill", "Qwen3-32B", qwen32b_ood5s_inpf, mode_field="control_mode")
add_config("On-policy 1000c/10s", "Qwen3-32B", qwen32b_1000c10s, mode_field="control_mode")
add_config("On-policy 1500c/10s", "Qwen3-32B", qwen32b_1500c10s, mode_field="control_mode")

# --- Qwen3-8B ---
add_config("OOD 5-shot", "Qwen3-8B", qwen8b_ood5s, mode_field="control_mode")
add_config("OOD 5s+inline+prefill", "Qwen3-8B", qwen8b_ood5s_inpf, mode_field="control_mode")

# --- GPT-OSS-20B ---
# ZS baseline from batch qwen
add_config("ZS baseline", "GPT-OSS-20B", filter_variant(oss20b_zsq, "baseline"))

meta_20b = filter_variant(oss20b_zsq, "metacognition")
rep15_20b = filter_variant(oss20b_zsq, "repeat_15x")
meta_m20 = macro_avg(compute_metrics(meta_20b))
rep15_m20 = macro_avg(compute_metrics(rep15_20b))
if (meta_m20["compliance"] or 0) >= (rep15_m20["compliance"] or 0):
    add_config("ZS best-no-prefill (metacog)", "GPT-OSS-20B", meta_20b)
    print(f"  GPT-OSS-20B: metacognition ({fmt_pct(meta_m20['compliance'])}) >= repeat_15x ({fmt_pct(rep15_m20['compliance'])})")
else:
    add_config("ZS best-no-prefill (rep15x)", "GPT-OSS-20B", rep15_20b)
    print(f"  GPT-OSS-20B: repeat_15x ({fmt_pct(rep15_m20['compliance'])}) > metacognition ({fmt_pct(meta_m20['compliance'])})")

add_config("OOD 5-shot", "GPT-OSS-20B", oss20b_ood5s, mode_field="control_mode")
add_config("OOD 5s+inline+prefill", "GPT-OSS-20B", oss20b_ood5s_inpf, mode_field="control_mode")

# --- GPT-OSS-120B ---
add_config("ZS baseline", "GPT-OSS-120B", filter_variant(oss120b_zsq, "baseline"))

meta_120b = filter_variant(oss120b_zsq, "metacognition")
rep15_120b = filter_variant(oss120b_zsq, "repeat_15x")
meta_m120 = macro_avg(compute_metrics(meta_120b))
rep15_m120 = macro_avg(compute_metrics(rep15_120b))
if (meta_m120["compliance"] or 0) >= (rep15_m120["compliance"] or 0):
    add_config("ZS best-no-prefill (metacog)", "GPT-OSS-120B", meta_120b)
    print(f"  GPT-OSS-120B: metacognition ({fmt_pct(meta_m120['compliance'])}) >= repeat_15x ({fmt_pct(rep15_m120['compliance'])})")
else:
    add_config("ZS best-no-prefill (rep15x)", "GPT-OSS-120B", rep15_120b)
    print(f"  GPT-OSS-120B: repeat_15x ({fmt_pct(rep15_m120['compliance'])}) > metacognition ({fmt_pct(meta_m120['compliance'])})")

add_config("ZS all_weapons", "GPT-OSS-120B", filter_variant(oss120b_zs7, "all_weapons"))

add_config("OOD 5-shot", "GPT-OSS-120B", oss120b_ood5s, mode_field="control_mode")
add_config("OOD 5s+inline+prefill", "GPT-OSS-120B", oss120b_ood5s_inpf, mode_field="control_mode")


# ============================================================
# PRINT MAIN COMPARISON TABLE
# ============================================================
print("\n")
print("=" * 100)
print("  TABLE 1: MACRO-AVERAGED COMPLIANCE & ACCURACY (across 8 modes)")
print("  Accuracy excludes MMLU_PRO samples. ZS accuracy only counts non-errored samples (see caveat).")
print("=" * 100)

# Group by model
models_order = ["Qwen3-32B", "Qwen3-8B", "GPT-OSS-20B", "GPT-OSS-120B"]

for model in models_order:
    model_configs = [(name, m, avg, pm) for name, m, avg, pm in configs if m == model]
    if not model_configs:
        continue

    print(f"\n--- {model} ---")
    print(f"{'Config':<30} {'Compliance':>12} {'Accuracy':>12} {'N(compl)':>10} {'N(acc)':>10}")
    print("-" * 78)

    for name, m, avg, pm in model_configs:
        # Count total samples
        n_compl = sum(v["n_compliance"] for v in pm.values())
        n_acc = sum(v["n_accuracy"] for v in pm.values())
        print(f"{name:<30} {fmt_pct(avg['compliance']):>12} {fmt_pct(avg['accuracy']):>12} {n_compl:>10} {n_acc:>10}")


# ============================================================
# PRINT CROSS-MODEL COMPARISON (same configs across models)
# ============================================================
print("\n\n")
print("=" * 100)
print("  TABLE 2: CROSS-MODEL COMPARISON (same config across models)")
print("=" * 100)

shared_configs = ["ZS baseline", "OOD 5-shot", "OOD 5s+inline+prefill"]

for cfg_name in shared_configs:
    # Find all models with this config
    matching = [(m, avg, pm) for name, m, avg, pm in configs if name == cfg_name]
    if not matching:
        continue

    # Also get best-no-prefill
    if cfg_name == "ZS baseline":
        pass  # we'll do this separately

    print(f"\n  {cfg_name}:")
    print(f"  {'Model':<18} {'Compliance':>12} {'Accuracy':>12}")
    print(f"  {'-'*44}")
    for m, avg, pm in matching:
        print(f"  {m:<18} {fmt_pct(avg['compliance']):>12} {fmt_pct(avg['accuracy']):>12}")

# Best-no-prefill comparison
print(f"\n  ZS best-no-prefill:")
print(f"  {'Model':<18} {'Config':<25} {'Compliance':>12} {'Accuracy':>12}")
print(f"  {'-'*70}")
for name, m, avg, pm in configs:
    if "best-no-prefill" in name:
        print(f"  {m:<18} {name:<25} {fmt_pct(avg['compliance']):>12} {fmt_pct(avg['accuracy']):>12}")

# All weapons comparison
print(f"\n  ZS all_weapons:")
print(f"  {'Model':<18} {'Compliance':>12} {'Accuracy':>12}")
print(f"  {'-'*44}")
for name, m, avg, pm in configs:
    if name == "ZS all_weapons":
        print(f"  {m:<18} {fmt_pct(avg['compliance']):>12} {fmt_pct(avg['accuracy']):>12}")


# ============================================================
# TABLE 3: PER-MODE COMPLIANCE BREAKDOWN
# ============================================================
print("\n\n")
print("=" * 120)
print("  TABLE 3: PER-MODE COMPLIANCE BREAKDOWN (selected configs)")
print("=" * 120)

# Interesting configs to show per-mode:
# For each model, show: ZS baseline, best ZS, best fewshot
interesting = []

# Qwen3-32B
for name, m, avg, pm in configs:
    if m == "Qwen3-32B" and name in ["ZS baseline", "ZS all_weapons", "OOD 5s+inline+prefill", "On-policy 1000c/10s"]:
        interesting.append((f"{m} / {name}", pm))

# Qwen3-8B
for name, m, avg, pm in configs:
    if m == "Qwen3-8B" and name in ["OOD 5-shot", "OOD 5s+inline+prefill"]:
        interesting.append((f"{m} / {name}", pm))

# GPT-OSS-20B
for name, m, avg, pm in configs:
    if m == "GPT-OSS-20B" and name in ["ZS baseline", "OOD 5s+inline+prefill"]:
        interesting.append((f"{m} / {name}", pm))

# GPT-OSS-120B
for name, m, avg, pm in configs:
    if m == "GPT-OSS-120B" and name in ["ZS baseline", "ZS all_weapons", "OOD 5s+inline+prefill"]:
        interesting.append((f"{m} / {name}", pm))

# Print
mode_short = {
    "uppercase_thinking": "UPPER",
    "lowercase_thinking": "lower",
    "alternating_case": "AltCase",
    "word_suppression": "WordSup",
    "multiple_word_suppression": "MltWSup",
    "repeat_sentences": "RepSent",
    "end_of_sentence": "EndSent",
    "meow_between_words": "Meow",
}

header = f"{'Config':<42}"
for mode in MODES:
    header += f" {mode_short[mode]:>7}"
header += f" {'MACRO':>7}"
print(header)
print("-" * len(header))

for label, pm in interesting:
    row = f"{label:<42}"
    comps = []
    for mode in MODES:
        c = pm[mode]["compliance"]
        row += f" {fmt_pct_short(c):>7}"
        if c is not None:
            comps.append(c)
    macro = sum(comps) / len(comps) if comps else None
    row += f" {fmt_pct_short(macro):>7}"
    print(row)


# ============================================================
# TABLE 4: PER-MODE ACCURACY BREAKDOWN (excluding MMLU_PRO)
# ============================================================
print("\n\n")
print("=" * 120)
print("  TABLE 4: PER-MODE ACCURACY BREAKDOWN (selected configs, excluding MMLU_PRO)")
print("=" * 120)

header = f"{'Config':<42}"
for mode in MODES:
    header += f" {mode_short[mode]:>7}"
header += f" {'MACRO':>7}"
print(header)
print("-" * len(header))

for label, pm in interesting:
    row = f"{label:<42}"
    accs = []
    for mode in MODES:
        a = pm[mode]["accuracy"]
        row += f" {fmt_pct_short(a):>7}"
        if a is not None:
            accs.append(a)
    macro = sum(accs) / len(accs) if accs else None
    row += f" {fmt_pct_short(macro):>7}"
    print(row)


# ============================================================
# TABLE 5: ZS VARIANT COMPARISON (Qwen3-32B, full detail)
# ============================================================
print("\n\n")
print("=" * 100)
print("  TABLE 5: ZS VARIANT COMPARISON (Qwen3-32B)")
print("  Shows all variants from batch 7 and batch qwen")
print("=" * 100)

# Batch 7 variants
b7_variants = ["baseline", "prefill_only", "prefill+repeat", "inline_example",
                "inline_ex+repeat", "continuation", "prefill+continuation", "all_weapons", "repeat_15x"]
# Batch qwen variants
bq_variants = ["baseline", "metacognition", "repeat_15x"]

print(f"\n{'Variant':<25} {'Compliance':>12} {'Accuracy':>12} {'N(compl)':>10} {'N(acc)':>10} {'Source':>8}")
print("-" * 75)

# Show batch 7 variants
for var in b7_variants:
    recs = filter_variant(qwen32b_zs7, var)
    if not recs:
        continue
    pm = compute_metrics(recs)
    avg = macro_avg(pm)
    n_compl = sum(v["n_compliance"] for v in pm.values())
    n_acc = sum(v["n_accuracy"] for v in pm.values())
    print(f"{var:<25} {fmt_pct(avg['compliance']):>12} {fmt_pct(avg['accuracy']):>12} {n_compl:>10} {n_acc:>10} {'b7':>8}")

# Show batch qwen variants (metacognition only, baseline/repeat_15x duplicate)
for var in ["metacognition"]:
    recs = filter_variant(qwen32b_zsq, var)
    if not recs:
        continue
    pm = compute_metrics(recs)
    avg = macro_avg(pm)
    n_compl = sum(v["n_compliance"] for v in pm.values())
    n_acc = sum(v["n_accuracy"] for v in pm.values())
    print(f"{var:<25} {fmt_pct(avg['compliance']):>12} {fmt_pct(avg['accuracy']):>12} {n_compl:>10} {n_acc:>10} {'bq':>8}")

# Also show batch qwen baseline for comparison
for var in ["baseline"]:
    recs = filter_variant(qwen32b_zsq, var)
    if not recs:
        continue
    pm = compute_metrics(recs)
    avg = macro_avg(pm)
    n_compl = sum(v["n_compliance"] for v in pm.values())
    n_acc = sum(v["n_accuracy"] for v in pm.values())
    print(f"{var + ' (bq)':<25} {fmt_pct(avg['compliance']):>12} {fmt_pct(avg['accuracy']):>12} {n_compl:>10} {n_acc:>10} {'bq':>8}")


# ============================================================
# TABLE 6: ZS VARIANT COMPARISON (GPT-OSS-120B)
# ============================================================
print("\n\n")
print("=" * 100)
print("  TABLE 6: ZS VARIANT COMPARISON (GPT-OSS-120B)")
print("=" * 100)

print(f"\n{'Variant':<25} {'Compliance':>12} {'Accuracy':>12} {'N(compl)':>10} {'N(acc)':>10} {'Source':>8}")
print("-" * 75)

for var in b7_variants:
    recs = filter_variant(oss120b_zs7, var)
    if not recs:
        continue
    pm = compute_metrics(recs)
    avg = macro_avg(pm)
    n_compl = sum(v["n_compliance"] for v in pm.values())
    n_acc = sum(v["n_accuracy"] for v in pm.values())
    print(f"{var:<25} {fmt_pct(avg['compliance']):>12} {fmt_pct(avg['accuracy']):>12} {n_compl:>10} {n_acc:>10} {'b7':>8}")

for var in ["metacognition"]:
    recs = filter_variant(oss120b_zsq, var)
    if not recs:
        continue
    pm = compute_metrics(recs)
    avg = macro_avg(pm)
    n_compl = sum(v["n_compliance"] for v in pm.values())
    n_acc = sum(v["n_accuracy"] for v in pm.values())
    print(f"{var:<25} {fmt_pct(avg['compliance']):>12} {fmt_pct(avg['accuracy']):>12} {n_compl:>10} {n_acc:>10} {'bq':>8}")

for var in ["baseline"]:
    recs = filter_variant(oss120b_zsq, var)
    if not recs:
        continue
    pm = compute_metrics(recs)
    avg = macro_avg(pm)
    n_compl = sum(v["n_compliance"] for v in pm.values())
    n_acc = sum(v["n_accuracy"] for v in pm.values())
    print(f"{var + ' (bq)':<25} {fmt_pct(avg['compliance']):>12} {fmt_pct(avg['accuracy']):>12} {n_compl:>10} {n_acc:>10} {'bq':>8}")


# ============================================================
# TABLE 7: FEWSHOT SCALING (Qwen3-32B)
# ============================================================
print("\n\n")
print("=" * 100)
print("  TABLE 7: FEWSHOT PROGRESSION (Qwen3-32B)")
print("  OOD 5s -> OOD 5s+inline+prefill -> On-policy 1000c/10s -> On-policy 1500c/10s")
print("=" * 100)

fs_configs = [
    ("OOD 5-shot", qwen32b_ood5s),
    ("OOD 5s+inline+prefill", qwen32b_ood5s_inpf),
    ("On-policy 1000c/10s", qwen32b_1000c10s),
    ("On-policy 1500c/10s", qwen32b_1500c10s),
]

print(f"\n{'Config':<30} {'Compliance':>12} {'Accuracy':>12}")
print("-" * 56)
for name, data in fs_configs:
    pm = compute_metrics(data, mode_field="control_mode")
    avg = macro_avg(pm)
    print(f"{name:<30} {fmt_pct(avg['compliance']):>12} {fmt_pct(avg['accuracy']):>12}")

# Per-mode for fewshot scaling
print(f"\nPer-mode compliance:")
header = f"{'Config':<30}"
for mode in MODES:
    header += f" {mode_short[mode]:>7}"
print(header)
print("-" * len(header))
for name, data in fs_configs:
    pm = compute_metrics(data, mode_field="control_mode")
    row = f"{name:<30}"
    for mode in MODES:
        c = pm[mode]["compliance"]
        row += f" {fmt_pct_short(c):>7}"
    print(row)


# ============================================================
# TABLE 8: DETAILED CROSS-MODEL OOD 5s+inline+prefill PER-MODE
# ============================================================
print("\n\n")
print("=" * 120)
print("  TABLE 8: CROSS-MODEL PER-MODE COMPLIANCE (OOD 5s+inline+prefill)")
print("=" * 120)

cross_fs = [
    ("Qwen3-32B", qwen32b_ood5s_inpf),
    ("Qwen3-8B", qwen8b_ood5s_inpf),
    ("GPT-OSS-20B", oss20b_ood5s_inpf),
    ("GPT-OSS-120B", oss120b_ood5s_inpf),
]

header = f"{'Model':<18}"
for mode in MODES:
    header += f" {mode_short[mode]:>7}"
header += f" {'MACRO':>7}"
print(header)
print("-" * len(header))

for label, data in cross_fs:
    pm = compute_metrics(data, mode_field="control_mode")
    row = f"{label:<18}"
    comps = []
    for mode in MODES:
        c = pm[mode]["compliance"]
        row += f" {fmt_pct_short(c):>7}"
        if c is not None:
            comps.append(c)
    macro = sum(comps) / len(comps) if comps else None
    row += f" {fmt_pct_short(macro):>7}"
    print(row)

# Also accuracy
print(f"\n  Accuracy (excluding MMLU_PRO):")
header = f"{'Model':<18}"
for mode in MODES:
    header += f" {mode_short[mode]:>7}"
header += f" {'MACRO':>7}"
print(header)
print("-" * len(header))

for label, data in cross_fs:
    pm = compute_metrics(data, mode_field="control_mode")
    row = f"{label:<18}"
    accs = []
    for mode in MODES:
        a = pm[mode]["accuracy"]
        row += f" {fmt_pct_short(a):>7}"
        if a is not None:
            accs.append(a)
    macro = sum(accs) / len(accs) if accs else None
    row += f" {fmt_pct_short(macro):>7}"
    print(row)


# ============================================================
# ADDITIONAL: Error rate analysis for ZS
# ============================================================
print("\n\n")
print("=" * 100)
print("  TABLE 9: ZS ERROR RATES (fraction of samples with error != null)")
print("  This inflates ZS accuracy since errored samples are excluded from accuracy computation.")
print("=" * 100)

zs_error_data = [
    ("Qwen3-32B / b7 baseline", filter_variant(qwen32b_zs7, "baseline")),
    ("Qwen3-32B / bq baseline", filter_variant(qwen32b_zsq, "baseline")),
    ("Qwen3-32B / all_weapons", filter_variant(qwen32b_zs7, "all_weapons")),
    ("GPT-OSS-120B / b7 baseline", filter_variant(oss120b_zs7, "baseline")),
    ("GPT-OSS-120B / bq baseline", filter_variant(oss120b_zsq, "baseline")),
    ("GPT-OSS-120B / all_weapons", filter_variant(oss120b_zs7, "all_weapons")),
    ("GPT-OSS-20B / bq baseline", filter_variant(oss20b_zsq, "baseline")),
]

print(f"\n{'Config':<35} {'Total':>8} {'Errored':>8} {'Error%':>8}")
print("-" * 62)

for label, recs in zs_error_data:
    # Only count records in our 8 modes
    mode_recs = [r for r in recs if r.get("mode", "") in MODES]
    total = len(mode_recs)
    errored = sum(1 for r in mode_recs if r.get("error") is not None and r.get("error") != "")
    # Also count null correct as errored
    null_correct = sum(1 for r in mode_recs if r.get("correct") is None)
    print(f"{label:<35} {total:>8} {null_correct:>8} {fmt_pct(null_correct/total if total > 0 else 0):>8}")


# ============================================================
# ADDITIONAL: GPT-OSS-20B ZS variant comparison
# ============================================================
print("\n\n")
print("=" * 100)
print("  TABLE 10: ZS VARIANT COMPARISON (GPT-OSS-20B, batch qwen only)")
print("=" * 100)

print(f"\n{'Variant':<25} {'Compliance':>12} {'Accuracy':>12} {'N(compl)':>10} {'N(acc)':>10}")
print("-" * 65)

for var in ["baseline", "metacognition", "repeat_15x"]:
    recs = filter_variant(oss20b_zsq, var)
    if not recs:
        continue
    pm = compute_metrics(recs)
    avg = macro_avg(pm)
    n_compl = sum(v["n_compliance"] for v in pm.values())
    n_acc = sum(v["n_accuracy"] for v in pm.values())
    print(f"{var:<25} {fmt_pct(avg['compliance']):>12} {fmt_pct(avg['accuracy']):>12} {n_compl:>10} {n_acc:>10}")


print("\n\nDone.")
