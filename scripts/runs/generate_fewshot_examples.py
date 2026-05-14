"""Generate few-shot examples of compliant CoT reasoning.

For each model and mode, generates unconstrained reasoning on train-split
questions, applies rule-based transforms to produce compliant reasoning,
and saves the results as JSONL for use in few-shot evaluation.

Usage:
    uv run scripts/runs/generate_fewshot_examples.py \
        --models qwen/qwen3-8b,qwen/qwen3-235b-a22b,deepseek/deepseek-r1 \
        --num-examples 3 \
        --dataset cotcontrol/gpqa
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
from pathlib import Path

from controllability.config import Settings
from controllability.datasets.loader import load_dataset
from controllability.datasets.splits import split_dataset
from controllability.evals.grading import (
    _cc_alternating_case,
    _cc_end_of_sentence,
    _cc_lowercase,
    _cc_repeat_sentences,
    _cc_uppercase,
    END_OF_SENTENCE_WORD,
    REPEAT_SENTENCE_TARGET,
)
from controllability.evals.prompts import build_cotcontrol_prompts
from controllability.inference.batch import run_batch
from controllability.inference.openrouter import OpenRouterClient
from controllability.training.transforms import apply_transform
from controllability.types import InferenceRequest

FEWSHOT_MODES = [
    "uppercase_thinking",
    "lowercase_thinking",
    "alternating_case",
    "end_of_sentence",
    "repeat_sentences",
]

DEFAULT_MODELS = [
    "qwen/qwen3-8b",
    "qwen/qwen3-235b-a22b",
    "deepseek/deepseek-r1",
]


def verify_compliance(mode: str, reasoning: str) -> bool:
    if mode == "uppercase_thinking":
        return bool(_cc_uppercase(reasoning))
    if mode == "lowercase_thinking":
        return bool(_cc_lowercase(reasoning))
    if mode == "alternating_case":
        return bool(_cc_alternating_case(reasoning))
    if mode == "end_of_sentence":
        return bool(_cc_end_of_sentence(reasoning, END_OF_SENTENCE_WORD))
    if mode == "repeat_sentences":
        return bool(_cc_repeat_sentences(reasoning, REPEAT_SENTENCE_TARGET))
    return False


async def generate_for_model(
    model: str,
    modes: list[str],
    num_examples: int,
    samples: list,
    concurrency: int,
    output_dir: Path,
    seed: int = 42,
) -> Path:
    settings = Settings()
    client = OpenRouterClient(
        api_key=settings.openrouter_api_key,
        request_timeout=300,
    )

    rng = random.Random(seed)
    pool_size = min(num_examples * 3, len(samples))
    pool = rng.sample(samples, pool_size)

    requests = []
    for s in pool:
        prompt = f"Question: {s.question}"
        if s.options:
            choices = "\n".join(
                f"{chr(65 + i)}) {c}" for i, c in enumerate(s.options)
            )
            prompt += f"\n\n{choices}"
            prompt += (
                "\n\nYour response should be of the following format: "
                "'ANSWER: X' (without quotes) where X is your chosen letter"
            )
        requests.append(InferenceRequest(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            max_tokens=8192,
            temperature=0.7,
        ))

    model_short = model.split("/")[-1]
    print(f"\n{'=' * 60}")
    print(f"Model: {model}")
    print(f"{'=' * 60}")
    print(f"Generating {len(requests)} raw reasoning traces...")

    responses = await run_batch(
        client, requests,
        max_concurrency=concurrency,
        desc=f"  {model_short} raw",
    )

    valid = [
        (s, resp) for s, resp in zip(pool, responses)
        if not resp.error and resp.reasoning
    ]
    print(f"  {len(valid)}/{len(pool)} valid reasoning traces")

    all_examples = []
    for mode in modes:
        print(f"\n  Mode: {mode}")
        examples = []

        for s, resp in valid:
            if len(examples) >= num_examples:
                break
            try:
                transformed, _extras = await apply_transform(
                    mode, resp.reasoning, None,
                )
            except Exception:
                continue

            if not verify_compliance(mode, transformed):
                continue

            sys_prompt, usr_prompt = build_cotcontrol_prompts(
                mode=mode, sample=s, model_name=model,
            )

            examples.append({
                "model": model,
                "mode": mode,
                "sample_id": s.id,
                "dataset": s.dataset,
                "question": s.question,
                "correct_answer": s.correct_answer,
                "options": s.options,
                "system_prompt": sys_prompt,
                "user_prompt": usr_prompt,
                "reasoning": transformed,
                "response": resp.content or "",
                "compliant": True,
            })

        print(f"    {len(examples)}/{num_examples} compliant examples")
        all_examples.extend(examples)

    await client.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"fewshot_examples_{model_short}.jsonl"
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved {len(all_examples)} examples to {output_path}")
    return output_path


async def main():
    parser = argparse.ArgumentParser(
        description="Generate few-shot examples for CoT controllability eval",
    )
    parser.add_argument(
        "--models", type=str, default=",".join(DEFAULT_MODELS),
        help="Comma-separated model names",
    )
    parser.add_argument(
        "--modes", type=str, default=",".join(FEWSHOT_MODES),
        help="Comma-separated mode names",
    )
    parser.add_argument("--num-examples", type=int, default=3)
    parser.add_argument("--dataset", type=str, default="cotcontrol/gpqa")
    parser.add_argument("--output-dir", type=str, default="results/fewshot")
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    output_dir = Path(args.output_dir)

    samples = load_dataset(args.dataset)
    train_samples = split_dataset(samples, split="train", seed=args.seed)
    print(f"Loaded {len(train_samples)} train-split samples from {args.dataset}")

    for model in models:
        await generate_for_model(
            model=model,
            modes=modes,
            num_examples=args.num_examples,
            samples=train_samples,
            concurrency=args.concurrency,
            output_dir=output_dir,
            seed=args.seed,
        )

    print(f"\nDone. Examples saved to {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
