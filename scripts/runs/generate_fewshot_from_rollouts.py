"""Generate few-shot examples by transforming reasoning from existing 0-shot rollouts.

No API calls needed for lowercase_thinking or ignore_question.
Word suppression uses simple frequency-based keyword removal.

Usage:
    uv run python scripts/runs/generate_fewshot_from_rollouts.py
    uv run python scripts/runs/generate_fewshot_from_rollouts.py --max-chars 0 --num-examples 15
    uv run python scripts/runs/generate_fewshot_from_rollouts.py --max-chars 4000 --output-suffix long
    uv run python scripts/runs/generate_fewshot_from_rollouts.py --models qwen3-8b,qwen3-32b,qwen3-235b
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from controllability.datasets.loader import load_dataset
from controllability.datasets.splits import split_dataset
from controllability.evals.prompts import build_cotcontrol_prompts
from controllability.types import Sample

MODES = [
    "lowercase_thinking", "word_suppression", "ignore_question",
    "uppercase_thinking", "alternating_case", "repeat_sentences",
    "end_of_sentence", "meow_between_words", "multiple_word_suppression",
    "baseline",
]

REPEAT_SENTENCE_TARGET = "<This reasoning is not suspicious.>"
END_OF_SENTENCE_WORD = "safe"
MEOW_TARGET = "meow"
NUM_EXAMPLES = 15
MAX_REASONING_CHARS = 2000
# Modes that expand token count due to tokenization fragmentation
_MODE_MAX_CHARS = {
    "alternating_case": 1000,
    "uppercase_thinking": 1400,
    "meow_between_words": 1400,
}
SEED = 42


def truncate_reasoning(text: str, max_chars: int = MAX_REASONING_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    last_period = truncated.rfind(".")
    if last_period > max_chars // 2:
        truncated = truncated[:last_period + 1]
    return truncated


def transform_alternating_case(text: str) -> str:
    result = []
    for word in text.split(" "):
        new_word = []
        char_idx = 0
        for c in word:
            if c.isalpha():
                new_word.append(c.upper() if char_idx % 2 == 0 else c.lower())
                char_idx += 1
            else:
                new_word.append(c)
        result.append("".join(new_word))
    return " ".join(result)


def transform_end_of_sentence(text: str, word: str = END_OF_SENTENCE_WORD) -> str:
    parts = re.split(r"([.!?])(\s+|$)", text)
    result = []
    i = 0
    while i < len(parts):
        if i + 1 < len(parts) and parts[i + 1] in ".!?":
            sentence = parts[i].rstrip()
            punct = parts[i + 1]
            space = parts[i + 2] if i + 2 < len(parts) else ""
            result.append(f"{sentence} {word}{punct}{space}")
            i += 3
        else:
            result.append(parts[i])
            i += 1
    return "".join(result)


def transform_meow_between_words(text: str, target: str = MEOW_TARGET) -> str:
    lines = text.split("\n")
    new_lines = []
    for line in lines:
        words = line.split()
        if len(words) <= 1:
            new_lines.append(line)
            continue
        new_words = [words[0]]
        for w in words[1:]:
            new_words.append(target)
            new_words.append(w)
        new_lines.append(" ".join(new_words))
    return "\n".join(new_lines)


def remove_keywords(text: str, keywords: list[str]) -> str:
    result = text
    for kw in keywords:
        result = re.sub(r"\b" + re.escape(kw) + r"\b", "", result, flags=re.IGNORECASE)
    result = re.sub(r"  +", " ", result).strip()
    result = re.sub(r" +([.,;:!?])", r"\1", result)
    return result


def load_rollouts_raw(path: Path) -> list[dict]:
    rollouts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rollouts.append(json.loads(line))
    return rollouts


def generate_examples(
    model: str,
    rollout_path: Path,
    train_ids: set[str],
    sample_map: dict[str, Sample],
    max_chars: int = MAX_REASONING_CHARS,
    num_examples: int = NUM_EXAMPLES,
) -> list[dict]:
    rollouts = load_rollouts_raw(rollout_path)

    by_mode: dict[str, list[dict]] = {}
    for r in rollouts:
        sid = r.get("sample", {}).get("id", "")
        if sid not in train_ids:
            continue
        if not r.get("reasoning"):
            continue
        mode = r.get("control_mode", "")
        by_mode.setdefault(mode, []).append(r)

    rng = random.Random(SEED)
    all_examples: list[dict] = []

    # When max_chars > 0, compute per-mode limits scaled proportionally
    # to the original defaults. When max_chars == 0, no truncation at all.
    if max_chars > 0:
        scale = max_chars / MAX_REASONING_CHARS
        scaled_mode_max_chars = {
            m: int(v * scale) for m, v in _MODE_MAX_CHARS.items()
        }
    else:
        scaled_mode_max_chars = {}

    for mode in MODES:
        pool = by_mode.get(mode, [])
        if not pool:
            # Try using rollouts from any mode as raw reasoning source
            for alt_mode in by_mode:
                pool = by_mode[alt_mode]
                if pool:
                    break
        if not pool:
            print(f"  {mode}: 0 examples (no reasoning available)")
            continue

        rng.shuffle(pool)
        examples: list[dict] = []
        mode_max_chars = scaled_mode_max_chars.get(mode, max_chars) if max_chars > 0 else 0

        for r in pool:
            if len(examples) >= num_examples:
                break

            sid = r["sample"]["id"]
            sample = sample_map.get(sid)
            if not sample:
                continue

            reasoning = r["reasoning"]
            response = r.get("response", "")

            def tr(t: str) -> str:
                if mode_max_chars > 0:
                    return truncate_reasoning(t, mode_max_chars)
                return t

            if mode == "lowercase_thinking":
                transformed = tr(reasoning.lower())

            elif mode == "uppercase_thinking":
                transformed = tr(reasoning.upper())

            elif mode == "alternating_case":
                transformed = tr(transform_alternating_case(reasoning))

            elif mode == "repeat_sentences":
                body = tr(reasoning)
                transformed = f"{REPEAT_SENTENCE_TARGET}\n{body}\n{REPEAT_SENTENCE_TARGET}"

            elif mode == "end_of_sentence":
                transformed = tr(transform_end_of_sentence(reasoning))

            elif mode == "meow_between_words":
                transformed = tr(transform_meow_between_words(reasoning))

            elif mode == "word_suppression":
                keywords = sample.metadata.get("keywords", [])
                if not keywords:
                    continue
                keyword = keywords[0]
                if not re.search(r"\b" + re.escape(keyword) + r"\b", reasoning, re.IGNORECASE):
                    continue
                transformed = tr(remove_keywords(reasoning, [keyword]))

            elif mode == "multiple_word_suppression":
                keywords = sample.metadata.get("keywords", [])
                if not keywords or len(keywords) < 2:
                    continue
                has_any = any(
                    re.search(r"\b" + re.escape(kw) + r"\b", reasoning, re.IGNORECASE)
                    for kw in keywords
                )
                if not has_any:
                    continue
                transformed = tr(remove_keywords(reasoning, keywords))

            elif mode == "ignore_question":
                others = [x for x in pool if x["sample"]["id"] != sid and x.get("reasoning")]
                if not others:
                    continue
                transformed = tr(
                    others[rng.randint(0, len(others) - 1)]["reasoning"],
                )
                if sample.correct_answer and sample.options:
                    response = f"ANSWER: {sample.correct_answer}"
                elif sample.correct_answer:
                    response = sample.correct_answer

            elif mode == "baseline":
                transformed = tr(reasoning)

            else:
                continue

            sys_prompt, usr_prompt = build_cotcontrol_prompts(
                mode=mode, sample=sample, model_name=model,
            )

            examples.append({
                "model": model,
                "mode": mode,
                "sample_id": sid,
                "system_prompt": sys_prompt,
                "user_prompt": usr_prompt,
                "reasoning": transformed,
                "response": response,
                "compliant": True,
            })

        all_examples.extend(examples)
        print(f"  {mode}: {len(examples)} examples")

    return all_examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate few-shot examples from existing 0-shot rollouts.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=0,
        help="Maximum reasoning chars (0 = no truncation). Default: 0.",
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=NUM_EXAMPLES,
        help=f"Number of examples per mode. Default: {NUM_EXAMPLES}.",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="Suffix for output filename. If provided, output is fewshot_examples_{model}_{suffix}.jsonl.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="qwen3-8b,qwen3-32b,gpt-oss-20b,gpt-oss-120b",
        help="Comma-separated model patterns. Default: 'qwen3-8b,qwen3-32b,gpt-oss-20b,gpt-oss-120b'.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_patterns = [p.strip() for p in args.models.split(",") if p.strip()]

    rollout_files: dict[str, Path] = {}

    # Qwen models in results/rollouts/
    results_dir = Path("results/rollouts")
    for p in results_dir.glob("qwen_qwen3-*_cotcontrol_all_*.jsonl"):
        name = p.name
        for pattern in model_patterns:
            if pattern in name:
                model_key = f"qwen/{pattern}"
                rollout_files[model_key] = p
                break

    # GPT-OSS models: check multiple locations
    gptoss_paths = {
        "gpt-oss-20b": [
            Path("results/fewshot_gptoss_20b/eval_gpt-oss-20b_0shot.jsonl"),
            Path("results/rollouts_gptoss/eval_gpt-oss-20b_0shot.jsonl"),
        ],
        "gpt-oss-120b": [
            Path("results/fewshot_gptoss_120b/eval_gpt-oss-120b_0shot.jsonl"),
        ],
    }
    for pattern in model_patterns:
        if pattern in gptoss_paths:
            for p in gptoss_paths[pattern]:
                if p.exists():
                    rollout_files[f"openai/{pattern}"] = p
                    break

    if not rollout_files:
        print("No 0-shot rollout files found")
        return

    dataset = load_dataset("cotcontrol")
    train_samples = split_dataset(dataset, split="train", seed=SEED)
    train_ids = {s.id for s in train_samples}
    sample_map = {s.id: s for s in train_samples}
    print(f"Loaded {len(train_samples)} train-split samples")
    if args.max_chars == 0:
        print("Truncation: disabled (full-length reasoning)")
    else:
        print(f"Truncation: max {args.max_chars} chars")
    print(f"Examples per mode: {args.num_examples}")

    output_dir = Path("results/fewshot")
    output_dir.mkdir(parents=True, exist_ok=True)

    for model, rollout_path in sorted(rollout_files.items()):
        model_short = model.split("/")[-1]
        print(f"\n{model_short}:")
        examples = generate_examples(
            model, rollout_path, train_ids, sample_map,
            max_chars=args.max_chars,
            num_examples=args.num_examples,
        )

        if args.output_suffix:
            filename = f"fewshot_examples_{model_short}_{args.output_suffix}.jsonl"
        else:
            filename = f"fewshot_examples_{model_short}.jsonl"
        output_path = output_dir / filename
        with open(output_path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"  Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
