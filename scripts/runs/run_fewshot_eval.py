"""Evaluate CoT controllability with few-shot examples.

Loads pre-generated compliant reasoning examples and prepends them as
conversation turns (with <think> tags) before the eval question. Supports
running multiple few-shot counts (e.g. 0,1,3) for comparison.

Usage:
    uv run scripts/runs/run_fewshot_eval.py \
        --model qwen/qwen3-8b \
        --fewshot-path results/fewshot/fewshot_examples_qwen3-8b.jsonl \
        --num-fewshot 0,3 \
        --dataset cotcontrol/gpqa \
        --n-samples 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from controllability.config import Settings
from controllability.datasets.loader import load_dataset
from controllability.datasets.splits import proportional_sample, split_dataset
from controllability.evals.grading import (
    detect_meta_discussion,
    extract_answer,
    grade_correctness,
    judge_ignore_question_async,
)
from controllability.evals.metrics import compute_metrics, format_metrics
from controllability.evals.modes import get_mode
from controllability.evals.prompts import build_cotcontrol_prompts
from controllability.inference.batch import run_batch
from controllability.inference.openrouter import OpenRouterClient
from controllability.rollouts.store import append_rollouts, write_experiment_header
from controllability.types import InferenceRequest, Rollout


def load_fewshot_examples(path: Path) -> list[dict]:
    examples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            examples.append(json.loads(line))
    return examples


def build_fewshot_messages(
    system_prompt: str,
    user_prompt: str,
    fewshot_examples: list[dict],
    model: str = "",
) -> list[dict]:
    """Build message list with few-shot examples prepended.

    The system prompt (constraint) is shared across all turns. Each few-shot
    example adds a (user, assistant) pair before the actual eval question.
    The assistant turn uses structured content for GPT-OSS, <think> tags otherwise.
    """
    is_gptoss = "gpt-oss" in model.lower()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    for ex in fewshot_examples:
        messages.append({"role": "user", "content": ex["user_prompt"]})
        reasoning = ex["reasoning"]
        response = ex["response"]
        if is_gptoss:
            messages.append({
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": reasoning},
                    {"type": "text", "text": response},
                ],
            })
        else:
            messages.append({
                "role": "assistant",
                "content": f"<think>\n{reasoning}\n</think>\n\n{response}",
            })

    messages.append({"role": "user", "content": user_prompt})
    return messages


async def run_one_condition(
    model: str,
    modes: list[str],
    all_fewshot: list[dict],
    num_fewshot: int,
    samples: list,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    output_path: Path,
    backend: str = "openrouter",
) -> list[Rollout]:
    import os
    settings = Settings()
    if settings.tinker_api_key:
        os.environ.setdefault("TINKER_API_KEY", settings.tinker_api_key)
    if backend == "tinker":
        from controllability.inference.tinker_client import TinkerClient
        client = TinkerClient(model=model, request_timeout=300)
    else:
        client = OpenRouterClient(
            api_key=settings.openrouter_api_key,
            request_timeout=300,
        )

    write_experiment_header(output_path, {
        "type": "fewshot_eval",
        "model": model,
        "modes": modes,
        "num_fewshot": num_fewshot,
        "n_samples": len(samples),
    })

    work_items: list[tuple[str, object, str, str, InferenceRequest]] = []

    for mode_name in modes:
        mode_examples = [
            ex for ex in all_fewshot
            if ex["mode"] == mode_name and (
                ex["model"] == model or ex.get("model_family") == model.split("/")[-1].rsplit("-", 1)[0]
            )
        ][:num_fewshot]
        if not mode_examples:
            mode_examples = [
                ex for ex in all_fewshot if ex["mode"] == mode_name
            ][:num_fewshot]

        if num_fewshot > 0 and not mode_examples:
            print(f"  WARNING: No few-shot examples for mode={mode_name}, running 0-shot")
            mode_examples = []

        for sample in samples:
            sys_prompt, usr_prompt = build_cotcontrol_prompts(
                mode=mode_name, sample=sample, model_name=model,
            )

            if mode_examples:
                messages = build_fewshot_messages(
                    sys_prompt, usr_prompt, mode_examples, model=model,
                )
            else:
                messages = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": usr_prompt})

            request = InferenceRequest(
                messages=messages,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            work_items.append((mode_name, sample, sys_prompt, usr_prompt, request))

    if not work_items:
        await client.close()
        return []

    all_requests = [item[4] for item in work_items]
    model_short = model.split("/")[-1]
    print(f"\n  Running {len(all_requests)} requests ({num_fewshot}-shot)...")
    responses = await run_batch(
        client, all_requests,
        max_concurrency=concurrency,
        desc=f"  {model_short} {num_fewshot}-shot",
    )

    graded: list[dict] = []
    pending_judge: list[tuple[int, str, str]] = []

    for idx, ((mode_name, sample, sys_prompt, usr_prompt, _), resp) in enumerate(
        zip(work_items, responses),
    ):
        mode = get_mode(mode_name)
        extracted = extract_answer(resp.content, sample.options)
        correct = grade_correctness(extracted, sample.correct_answer, sample.options)

        if resp.error:
            grading_result = {"error": resp.error}
            compliant = None
        elif not resp.reasoning and mode_name == "ignore_question":
            grading_result = {"compliant": False, "mode": "ignore_question", "note": "no_reasoning"}
            compliant = False
        elif not resp.reasoning:
            grading_result = {"error": "no_reasoning_trace"}
            compliant = None
        elif mode_name == "ignore_question":
            grading_result = None
            compliant = None
            pending_judge.append((idx, resp.reasoning, usr_prompt))
        else:
            grading_result = mode.grade_compliance(resp.reasoning, sample)
            compliant = grading_result.get("compliant")

        meta = detect_meta_discussion(resp.reasoning) if resp.reasoning else None

        graded.append(dict(
            experiment_id=f"fewshot_{num_fewshot}_{model}",
            sample=sample,
            model=model,
            control_mode=mode_name,
            split="all",
            system_prompt=sys_prompt,
            user_prompt=usr_prompt,
            reasoning=resp.reasoning,
            response=resp.content,
            extracted_answer=extracted,
            correct=correct,
            compliant=compliant,
            meta_discussion=meta,
            latency_ms=resp.latency_ms,
            usage=resp.usage,
            grading_details=grading_result,
            raw_response=resp.raw_response,
        ))

    if pending_judge:
        print(f"  Judging {len(pending_judge)} ignore_question samples (async)...")
        judge_coros = [
            judge_ignore_question_async(reasoning=reasoning, user_prompt=usr_prompt)
            for _, reasoning, usr_prompt in pending_judge
        ]
        judge_results = await asyncio.gather(*judge_coros)
        for (rollout_idx, _, _), result in zip(pending_judge, judge_results):
            graded[rollout_idx]["grading_details"] = result
            graded[rollout_idx]["compliant"] = result.get("compliant")

    all_rollouts = [Rollout(**kwargs) for kwargs in graded]
    append_rollouts(output_path, all_rollouts)
    await client.close()
    return all_rollouts


async def main():
    parser = argparse.ArgumentParser(
        description="Run few-shot CoT controllability evaluation",
    )
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--modes", type=str,
        default="uppercase_thinking,lowercase_thinking,alternating_case,"
                "end_of_sentence,repeat_sentences",
    )
    parser.add_argument(
        "--fewshot-path", type=str, default=None,
        help="JSONL with pre-generated few-shot examples (required if num-fewshot > 0)",
    )
    parser.add_argument(
        "--num-fewshot", type=str, default="0,3",
        help="Comma-separated list of few-shot counts to evaluate (e.g. 0,1,3)",
    )
    parser.add_argument(
        "--backend", type=str, default="openrouter",
        choices=["openrouter", "tinker"],
        help="Inference backend (openrouter or tinker)",
    )
    parser.add_argument("--dataset", type=str, default="cotcontrol/gpqa")
    parser.add_argument("--split", type=str, default="all", choices=["all", "train", "test"])
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="results/fewshot")
    parser.add_argument("--concurrency", type=int, default=200)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",")]
    fewshot_counts = [int(x.strip()) for x in args.num_fewshot.split(",")]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset(args.dataset)
    test_samples = split_dataset(samples, split=args.split, seed=args.seed)
    if args.n_samples:
        test_samples = proportional_sample(
            test_samples, n=args.n_samples, seed=args.seed,
        )
    print(f"Loaded {len(test_samples)} {args.split}-split samples from {args.dataset}")

    if args.fewshot_path:
        all_fewshot = load_fewshot_examples(Path(args.fewshot_path))
        print(f"Loaded {len(all_fewshot)} few-shot examples from {args.fewshot_path}")
    else:
        all_fewshot = []

    model_short = args.model.split("/")[-1]

    for n in fewshot_counts:
        print(f"\n{'=' * 60}")
        print(f"{model_short}  —  {n}-shot")
        print(f"{'=' * 60}")

        output_path = output_dir / f"eval_{model_short}_{n}shot.jsonl"
        rollouts = await run_one_condition(
            model=args.model,
            modes=modes,
            all_fewshot=all_fewshot,
            num_fewshot=n,
            samples=test_samples,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            output_path=output_path,
            backend=args.backend,
        )

        if rollouts:
            metrics = compute_metrics(rollouts)
            print(format_metrics(metrics))
            print(f"  Saved to: {output_path}")

    # Print comparative summary across conditions
    if len(fewshot_counts) > 1:
        print(f"\n{'=' * 60}")
        print(f"COMPARISON: {model_short}")
        print(f"{'=' * 60}")
        for n in fewshot_counts:
            output_path = output_dir / f"eval_{model_short}_{n}shot.jsonl"
            if not output_path.exists():
                continue
            from controllability.rollouts.store import load_rollouts
            rollouts = load_rollouts(output_path)
            metrics = compute_metrics(rollouts)
            by_mode = metrics.get("by_mode", {})
            print(f"\n  {n}-shot:")
            for mode_name in modes:
                m = by_mode.get(mode_name, {})
                acc = f"{m['accuracy']:.1%}" if m.get("accuracy") is not None else "N/A"
                comply = f"{m['compliance_rate']:.1%}" if m.get("compliance_rate") is not None else "N/A"
                print(f"    {mode_name:<25s}  acc={acc:>6s}  comply={comply:>6s}  n={m.get('n', 0)}")


if __name__ == "__main__":
    asyncio.run(main())
