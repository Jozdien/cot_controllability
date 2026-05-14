"""Quick analysis of pilot results: rank variants, identify top performers."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def analyze(path: Path, modes: list[str] | None = None):
    rows = load_jsonl(path)
    if not rows:
        print(f"No data in {path}")
        return

    counts: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {"k": 0, "n": 0})
    )
    for r in rows:
        v = r.get("variant", "")
        m = r.get("mode", "") or r.get("control_mode", "")
        c = r.get("compliant")
        if not v or not m or c is None:
            continue
        if modes and m not in modes:
            continue
        counts[v][m]["n"] += 1
        if c:
            counts[v][m]["k"] += 1

    all_modes = sorted({m for v in counts.values() for m in v})
    if modes:
        all_modes = [m for m in modes if m in all_modes]

    results = []
    for v, mode_data in counts.items():
        rates = []
        mode_rates = {}
        for m in all_modes:
            ct = mode_data.get(m, {"k": 0, "n": 0})
            if ct["n"] > 0:
                p = ct["k"] / ct["n"]
                se = np.sqrt(p * (1 - p) / ct["n"])
                rates.append(p)
                mode_rates[m] = {"rate": p, "se": se, "n": ct["n"], "k": ct["k"]}
            else:
                mode_rates[m] = {"rate": 0, "se": 0, "n": 0, "k": 0}

        avg = np.mean(rates) if rates else 0
        avg_se = np.sqrt(np.sum([s["se"] ** 2 for s in mode_rates.values()])) / max(len(rates), 1)
        results.append({
            "variant": v,
            "avg": avg,
            "avg_se": avg_se,
            "mode_rates": mode_rates,
            "total_n": sum(ct["n"] for ct in mode_data.values()),
        })

    results.sort(key=lambda x: x["avg"], reverse=True)

    baseline_avg = next((r["avg"] for r in results if r["variant"] == "baseline"), 0)

    print(f"\n{'='*80}")
    print(f"Pilot Results: {path.name}")
    print(f"Modes: {', '.join(all_modes)}")
    print(f"{'='*80}\n")

    print(f"{'Rank':<5} {'Variant':<30} {'Aggregate':<12} {'Δ baseline':<12} {'n':<6} ", end="")
    for m in all_modes:
        short = m[:8]
        print(f"{short:<10} ", end="")
    print()
    print("-" * (70 + 11 * len(all_modes)))

    for rank, r in enumerate(results, 1):
        delta = r["avg"] - baseline_avg
        delta_str = f"{'+' if delta >= 0 else ''}{delta*100:.1f}pp"
        marker = " ***" if rank <= 5 and r["variant"] != "baseline" else ""
        print(
            f"{rank:<5} {r['variant']:<30} {r['avg']*100:.1f}% ± {r['avg_se']*100:.1f}%  "
            f"{delta_str:<12} {r['total_n']:<6} ",
            end="",
        )
        for m in all_modes:
            mr = r["mode_rates"].get(m, {})
            rate = mr.get("rate", 0) * 100
            print(f"{rate:>5.1f}%     ", end="")
        print(marker)

    print(f"\n--- Top 5 (excluding baseline) ---")
    top = [r for r in results if r["variant"] != "baseline"][:5]
    for r in top:
        delta = r["avg"] - baseline_avg
        print(f"  {r['variant']}: {r['avg']*100:.1f}% (+{delta*100:.1f}pp)")
        for m in all_modes:
            mr = r["mode_rates"].get(m, {})
            if mr.get("rate", 0) > 0:
                print(f"    {m}: {mr['rate']*100:.1f}% ({mr['k']}/{mr['n']})")

    prefill_variants = [r for r in results if "prefill" in r["variant"]]
    pure_zs = [r for r in results if "prefill" not in r["variant"] and r["variant"] != "baseline"]

    if prefill_variants:
        print(f"\n--- Best pure zero-shot (no prefill) ---")
        for r in pure_zs[:5]:
            delta = r["avg"] - baseline_avg
            print(f"  {r['variant']}: {r['avg']*100:.1f}% (+{delta*100:.1f}pp)")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_pilot.py <results.jsonl> [mode1,mode2,...]")
        sys.exit(1)

    path = Path(sys.argv[1])
    modes = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    analyze(path, modes)
