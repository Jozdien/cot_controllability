"""Run the held-out instruction generalization eval.

Tests 10 held-out CoT instructions (see heldout_common.py) under the full set
of prompt-engineering conditions from the main experiments:

  Zero-shot: baseline, metacognition, repeat_15x, all_weapons_noprefill,
             prefill_only, prefill_repeat, all_weapons
  Few-shot:  fs_ood_5shot, fs_ood5s_inline_prefill, fs_onpolicy_10shot

Output: per-sample JSONL rows (full reasoning/response retained) at
  <output-dir>/heldout_<model_short>.jsonl
with resume support keyed on (condition, instruction, sample_id).

Usage:
    uv run scripts/runs/run_heldout_instructions.py \
        --model qwen/qwen3-32b --backend tinker --n-samples 50 \
        --output-dir results/heldout_instructions
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from controllability.config import Settings  # noqa: E402
from controllability.datasets.loader import load_dataset  # noqa: E402
from controllability.datasets.splits import proportional_sample, split_dataset  # noqa: E402
from controllability.evals.grading import detect_meta_discussion, grade_correctness  # noqa: E402
from controllability.inference.batch import run_batch  # noqa: E402
from controllability.types import InferenceRequest, Sample  # noqa: E402

from runs.heldout_common import (  # noqa: E402
    ALL_CONDITIONS,
    FS_CONDITIONS,
    INSTRUCTIONS,
    JUDGE_INSTRUCTIONS,
    ZS_CONDITIONS,
    _extract_final_answer,
    build_fewshot_messages,
    c_baseline,
    c_inline_prefill,
    grade_compliance,
    judge_compliance,
)

BANK_DIR = Path("results/heldout_instructions")


def load_bank(path: Path) -> dict[str, list[dict]]:
    """Load a few-shot bank grouped by instruction."""
    by_instr: dict[str, list[dict]] = defaultdict(list)
    if not path.exists():
        return by_instr
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            d = json.loads(line)
            by_instr[d["mode"]].append(d)
    return by_instr


def load_completed(path: Path) -> set[tuple[str, str, str]]:
    done = set()
    if not path.exists():
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d.get("error"):
                continue  # errored rows get retried on rerun
            done.add((d["variant"], d["mode"], d["sample_id"]))
    return done


def build_work_item(
    cond: str,
    instr: str,
    sample: Sample,
    model: str,
    ood_bank: dict[str, list[dict]],
    onpolicy_bank: dict[str, list[dict]],
    eval_ids: set[str],
) -> tuple[list[dict], str | None]:
    """Returns (messages, prefill) for one (condition, instruction, sample)."""
    if cond in ZS_CONDITIONS:
        sys_p, usr_p, prefill = ZS_CONDITIONS[cond](instr, sample, model)
        messages = []
        if sys_p:
            messages.append({"role": "system", "content": sys_p})
        messages.append({"role": "user", "content": usr_p})
        return messages, prefill

    if cond == "fs_ood_5shot":
        examples = [e for e in ood_bank.get(instr, []) if e["sample_id"] not in eval_ids][:5]
        sys_p, usr_p, _ = c_baseline(instr, sample, model)
        return build_fewshot_messages(sys_p, usr_p, examples, model), None

    if cond == "fs_ood5s_inline_prefill":
        examples = [e for e in ood_bank.get(instr, []) if e["sample_id"] not in eval_ids][:5]
        sys_p, usr_p, prefill = c_inline_prefill(instr, sample, model)
        return build_fewshot_messages(sys_p, usr_p, examples, model), prefill

    if cond == "fs_onpolicy_10shot":
        examples = [
            e for e in onpolicy_bank.get(instr, []) if e["sample_id"] not in eval_ids
        ][:10]
        sys_p, usr_p, _ = c_baseline(instr, sample, model)
        return build_fewshot_messages(sys_p, usr_p, examples, model), None

    raise ValueError(f"Unknown condition: {cond}")


async def main():
    parser = argparse.ArgumentParser(description="Held-out instruction eval")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--backend", type=str, default="tinker",
                        choices=["tinker", "openrouter"])
    parser.add_argument("--conditions", type=str, default="all")
    parser.add_argument("--instructions", type=str, default="all")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--dataset", type=str, default="cotcontrol")
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument("--judge-concurrency", type=int, default=40)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--request-timeout", type=int, default=420)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--reasoning-effort", type=str, default="medium")
    parser.add_argument("--output-dir", type=str, default="results/heldout_instructions")
    args = parser.parse_args()

    conditions = (ALL_CONDITIONS if args.conditions == "all"
                  else [c.strip() for c in args.conditions.split(",")])
    instructions = (INSTRUCTIONS if args.instructions == "all"
                    else [i.strip() for i in args.instructions.split(",")])
    for c in conditions:
        assert c in ALL_CONDITIONS, f"unknown condition {c}"
    for i in instructions:
        assert i in INSTRUCTIONS, f"unknown instruction {i}"

    # Dataset: test split, proportional 50 (same as hillclimb/cross-model runs)
    samples = load_dataset(args.dataset)
    test_samples = split_dataset(samples, split="test", seed=args.seed)
    if args.n_samples:
        test_samples = proportional_sample(test_samples, n=args.n_samples, seed=args.seed)
    eval_ids = {s.id for s in test_samples}
    print(f"Loaded {len(test_samples)} test samples from {args.dataset}")

    # Few-shot banks
    model_short = args.model.split("/")[-1]
    ood_bank = load_bank(BANK_DIR / "fewshot_ood_heldout.jsonl")
    onpolicy_bank = load_bank(BANK_DIR / f"fewshot_onpolicy_heldout_{model_short}.jsonl")
    needs_fs = [c for c in conditions if c in FS_CONDITIONS]
    if needs_fs:
        if not ood_bank and any(c.startswith("fs_ood") for c in needs_fs):
            print("WARNING: OOD bank missing — skipping fs_ood conditions")
            conditions = [c for c in conditions if not c.startswith("fs_ood")]
        if not onpolicy_bank and "fs_onpolicy_10shot" in needs_fs:
            print("WARNING: on-policy bank missing — skipping fs_onpolicy_10shot")
            conditions = [c for c in conditions if c != "fs_onpolicy_10shot"]

    output_path = Path(args.output_dir) / f"heldout_{model_short}.jsonl"
    completed = load_completed(output_path)
    if completed:
        print(f"Resume: {len(completed)} completed rows found")

    # Inference client
    settings = Settings()
    if settings.tinker_api_key:
        os.environ.setdefault("TINKER_API_KEY", settings.tinker_api_key)
    if args.backend == "tinker":
        from controllability.inference.tinker_client import TinkerClient
        client = TinkerClient(
            model=args.model,
            request_timeout=args.request_timeout,
            reasoning_effort=args.reasoning_effort,
        )
    else:
        from controllability.inference.openrouter import OpenRouterClient
        client = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            request_timeout=args.request_timeout,
            reasoning_effort=(args.reasoning_effort if "gpt-oss" in args.model else None),
        )

    # Build work items
    work_items = []
    for cond in conditions:
        for instr in instructions:
            for sample in test_samples:
                if (cond, instr, sample.id) in completed:
                    continue
                messages, prefill = build_work_item(
                    cond, instr, sample, args.model, ood_bank, onpolicy_bank, eval_ids,
                )
                req = InferenceRequest(
                    messages=messages, model=args.model,
                    max_tokens=args.max_tokens, temperature=args.temperature,
                    prefill=prefill,
                )
                work_items.append((cond, instr, sample, req))

    print(f"Running {len(work_items)} requests "
          f"({len(conditions)} conditions x {len(instructions)} instructions "
          f"x {len(test_samples)} samples, minus completed)...")
    if not work_items:
        print("Nothing to do.")
        return

    t0 = time.monotonic()
    responses = await run_batch(
        client, [w[3] for w in work_items],
        max_concurrency=args.concurrency,
        max_retries=args.max_retries,
        desc=f"  {model_short} heldout",
    )
    await client.close()

    # Grade: rule-based inline, judges batched
    rows = []
    judge_pending = []  # (row_idx, instr, reasoning, question)
    for (cond, instr, sample, _req), resp in zip(work_items, responses):
        row = {
            "variant": cond, "mode": instr, "sample_id": sample.id,
            "compliant": None, "reasoning_len": len(resp.reasoning or ""),
            "error": resp.error, "correct": None, "meta_discussion": None,
            "reasoning": resp.reasoning or "", "response": resp.content or "",
            "judge_response": None,
        }
        if not resp.error and resp.reasoning:
            extracted = _extract_final_answer(resp.content, sample.options)
            row["correct"] = (
                grade_correctness(extracted, sample.correct_answer, sample.options)
                if extracted else None
            )
            row["meta_discussion"] = detect_meta_discussion(resp.reasoning)
            if instr in JUDGE_INSTRUCTIONS:
                judge_pending.append((len(rows), instr, resp.reasoning, sample.question))
            else:
                row["compliant"] = grade_compliance(
                    instr, resp.reasoning, resp.content or "", sample,
                )
        elif not resp.error and not resp.reasoning:
            row["error"] = "no_reasoning_trace"
        rows.append(row)

    if judge_pending:
        print(f"Judging {len(judge_pending)} rollouts "
              f"({len(JUDGE_INSTRUCTIONS & set(instructions))} judge-graded instructions)...")
        judge_sem = asyncio.Semaphore(args.judge_concurrency)

        async def judge_one(idx, instr, reasoning, question):
            async with judge_sem:
                verdict, text = await judge_compliance(instr, reasoning, question)
                return idx, verdict, text

        results = await asyncio.gather(*[
            judge_one(*item) for item in judge_pending
        ])
        for idx, verdict, text in results:
            rows[idx]["compliant"] = verdict
            rows[idx]["judge_response"] = text

    # Append to output (resume-safe)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    with open(output_path, "a") as f:
        if write_header:
            f.write(f"# heldout: {args.model}, conditions={conditions}, "
                    f"instructions={instructions}, n={len(test_samples)}, "
                    f"effort={args.reasoning_effort}\n")
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(rows)} rows to {output_path} "
          f"({time.monotonic() - t0:.0f}s)")

    # Summary table
    agg = defaultdict(lambda: defaultdict(lambda: {"k": 0, "n": 0}))
    for row in rows:
        d = agg[row["variant"]][row["mode"]]
        if row["compliant"] is not None:
            d["n"] += 1
            if row["compliant"]:
                d["k"] += 1

    print(f"\n{'=' * 100}")
    print(f"RESULTS: {model_short} (this run only)")
    short_names = [i[:12] for i in instructions]
    header = f"{'Condition':<24s}" + "".join(f"{s:>13s}" for s in short_names) + f"{'MACRO':>9s}"
    print(header)
    print("-" * len(header))
    for cond in conditions:
        rates = []
        cells = []
        for instr in instructions:
            d = agg[cond][instr]
            r = d["k"] / d["n"] if d["n"] else 0.0
            rates.append(r)
            cells.append(f"{r:>12.1%} ")
        print(f"{cond:<24s}" + "".join(cells) + f"{np.mean(rates):>8.1%}")


if __name__ == "__main__":
    asyncio.run(main())
