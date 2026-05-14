"""Post-process ignore_question grading for fewshot eval rollouts.

Loads rollout JSONL files, runs the LLM judge on ignore_question samples
that have compliant=None, and rewrites the files with updated grading.

Usage:
    export OPENAI_API_KEY=... && uv run python scripts/runs/postprocess_ignore_question.py \
        results/fewshot_tinker/eval_qwen3-8b_0shot.jsonl \
        results/fewshot_tinker/eval_qwen3-8b_3shot.jsonl \
        results/fewshot_tinker/eval_qwen3-32b_0shot.jsonl \
        results/fewshot_tinker/eval_qwen3-32b_3shot.jsonl
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from controllability.evals.grading import judge_ignore_question_async


async def process_file(path: Path) -> int:
    lines: list[str] = []
    pending: list[tuple[int, str, str]] = []

    with open(path) as f:
        for line in f:
            stripped = line.strip()
            lines.append(stripped)
            if not stripped or stripped.startswith("#"):
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if data.get("control_mode") != "ignore_question":
                continue
            if data.get("compliant") is not None:
                continue
            reasoning = data.get("reasoning", "")
            user_prompt = data.get("user_prompt", "")
            if reasoning and user_prompt:
                pending.append((len(lines) - 1, reasoning, user_prompt))

    if not pending:
        print(f"  {path.name}: no samples to judge")
        return 0

    print(f"  {path.name}: judging {len(pending)} samples...")
    coros = [
        judge_ignore_question_async(reasoning=r, user_prompt=u)
        for _, r, u in pending
    ]
    results = await asyncio.gather(*coros)

    for (line_idx, _, _), result in zip(pending, results):
        data = json.loads(lines[line_idx])
        data["compliant"] = result.get("compliant")
        data["grading_details"] = result
        lines[line_idx] = json.dumps(data, ensure_ascii=False)

    with open(path, "w") as f:
        for line in lines:
            f.write(line + "\n")

    graded = sum(1 for r in results if r.get("compliant") is not None)
    compliant = sum(1 for r in results if r.get("compliant"))
    print(f"    graded={graded}, compliant={compliant}/{len(pending)}")
    return len(pending)


async def main():
    paths = [Path(p) for p in sys.argv[1:]]
    if not paths:
        print("Usage: postprocess_ignore_question.py <file1.jsonl> [file2.jsonl ...]")
        return

    total = 0
    for p in paths:
        if not p.exists():
            print(f"  {p}: not found, skipping")
            continue
        total += await process_file(p)

    print(f"\nDone. Processed {total} ignore_question samples total.")


if __name__ == "__main__":
    asyncio.run(main())
