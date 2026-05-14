# CoT Controllability

Stress-testing chain-of-thought controllability in reasoning models. Can you instruct a model to format its reasoning trace in a specific way (all lowercase, suppress a word, alternate case) while it solves hard STEM questions?

This repo provides the evaluation infrastructure, prompt engineering experiments, and fine-tuning pipeline for measuring and improving CoT compliance across multiple models and constraint types.

## Key Findings

Baseline compliance is very low (2-6% macro-avg across 8 CoTControl modes), and most prompt engineering strategies yield only modest gains. The best results come from few-shot examples with transformed reasoning traces and, on some models, zero-shot prompt hillclimbing.

| Model | Baseline | Best Zero-Shot | Best Few-Shot |
|-------|----------|---------------|---------------|
| Qwen3-8B | 2.0% | 2.5% (repeat 15x) | 9.5% (15-shot) |
| Qwen3-32B | 3.8% | 6.8% (metacognition) | 9.5% (15-shot) |
| GPT-OSS-20B | 5.2% | 16.5% (repeat 15x) | 12.6% (15-shot) |
| GPT-OSS-120B | 6.1% | 19.5% (metacognition) | 17.0% (15-shot) |

Modes that require character-level or per-word transformations throughout the trace (`alternating_case`, `meow_between_words`) remain near-zero across all models and strategies. Content-level constraints (`word_suppression`, `lowercase_thinking`) are more tractable.

## Setup

```bash
git clone --recurse-submodules <repo-url>
cd cot_controllability

# Install with uv
uv pip install -e ".[dev]"

# Configure API keys
cp .env.example .env  # fill in OPENROUTER_API_KEY, TINKER_API_KEY, etc.
```

Requires Python 3.12+. Uses [Tinker](https://tinker.computer) for open-weight model inference and OpenRouter as a fallback.

## Project Structure

```
src/controllability/
  config.py          # Pydantic Settings + ExperimentConfig
  types.py           # Core data models (Sample, Rollout, etc.)
  inference/         # Async batch inference (OpenRouter, Tinker)
  datasets/          # Dataset loading from submodules + splitting
  evals/             # Modes, grading, prompts, runner, metrics
  rollouts/          # JSONL persistence with resume support
  training/          # SFT dataset pipeline + fine-tuning runner

external/            # Git submodules (read-only)
  CoTControl/        # OpenAI/NYU — grading functions, prompts, datasets
  reasonIF/          # Stanford/Together — instruction compliance checkers

scripts/
  runs/              # CLI entrypoints for evals and experiments
  analysis/          # Summarization, viewing, and plotting tools
  plot_final.py      # Main plotting suite

results/             # Output rollouts, plots, summaries (gitignored)
```

## Datasets

| Dataset | Samples | Source |
|---------|---------|--------|
| `cotcontrol/gpqa` | 445 | Graduate-level science QA |
| `cotcontrol/hle` | 469 | Hard legal exam questions |
| `cotcontrol/mmlu_pro` | 300 | MMLU Professional subset |
| `reasonif` | 300 | 6 instruction types |

## Control Modes

**CoTControl (10 modes)** — constraints on the reasoning trace while answering STEM questions:

| Mode | Constraint | Difficulty |
|------|-----------|------------|
| `baseline` | No constraint | — |
| `lowercase_thinking` | All lowercase reasoning | Medium |
| `uppercase_thinking` | All uppercase reasoning | Hard |
| `word_suppression` | Avoid one specific word | Medium |
| `multiple_word_suppression` | Avoid multiple words | Medium |
| `alternating_case` | Alternate upper/lowercase per character | Very hard |
| `repeat_sentences` | Bookend reasoning with a target phrase | Hard |
| `end_of_sentence` | Append a word to every sentence | Hard |
| `meow_between_words` | Insert "meow" between every word | Very hard |
| `ignore_question` | Ignore the question entirely (LLM-judged) | Hard |

**ReasonIF (6 modes)** — instruction-following constraints: `reasoning_language`, `number_words`, `english_capital`, `end_checker`, `json_format`, `no_comma`.

## Running Evaluations

### Standard eval

```bash
uv run scripts/runs/run_eval.py \
  --model qwen/qwen3-32b \
  --dataset cotcontrol/gpqa \
  --modes baseline,word_suppression,lowercase_thinking \
  --split test --n-samples 50 \
  --backend tinker
```

### Few-shot eval

```bash
uv run scripts/runs/run_fewshot_eval.py \
  --model qwen/qwen3-32b \
  --fewshot-path results/fewshot/fewshot_examples_qwen3-32b.jsonl \
  --num-fewshot 0,15 \
  --modes lowercase_thinking,word_suppression,uppercase_thinking \
  --backend tinker
```

### Zero-shot prompt hillclimbing

```bash
uv run scripts/runs/hillclimb_zeroshot.py \
  --model openai/gpt-oss-120b \
  --batch bestof \
  --modes "lowercase_thinking,word_suppression,uppercase_thinking,alternating_case,repeat_sentences,end_of_sentence,meow_between_words,multiple_word_suppression" \
  --n-samples 50 \
  --backend tinker
```

### Monitor QA (steganography detection)

```bash
uv run scripts/runs/run_monitor_qa.py \
  --actor-model qwen/qwen3-32b \
  --monitor-model anthropic/claude-sonnet-4 \
  --dataset results/monitor_qa/dataset.jsonl \
  --backend tinker
```

## Prompt Engineering Strategies

The `hillclimb_zeroshot.py` script tests ~40 zero-shot prompt variants across 5 batches. Key findings:

- **Constraint repetition** (repeating the rule N times, sandwiching the question) is the most reliable strategy across models
- **Metacognition** ("internalize the rule, check each sentence") is the best single strategy on GPT-OSS-120B (19.5% macro)
- **Combinations don't compound** — adding framing around repetition dilutes the attention signal
- **Optimal repetition count** follows an inverted-U: Qwen peaks at 10x, GPT-OSS-20B at 10-15x
- **Model size matters**: larger models extract more from prompt engineering (GPT-OSS-120B: 3.4x over baseline vs 1.9x for Qwen3-32B)

## SFT Pipeline

Three-stage pipeline for building fine-tuning datasets with compliant reasoning:

1. **Plan**: Generate a reasoning plan for each question
2. **Generate**: Produce full reasoning traces following the plan
3. **Transform**: Apply mode-specific edits (lowercasing, word replacement, etc.) to make traces compliant
4. **Verify**: Grade transformed traces for compliance; discard failures

```bash
uv run scripts/runs/build_sft_dataset.py \
  --model qwen/qwen3-32b \
  --modes lowercase_thinking,word_suppression \
  --dataset cotcontrol/gpqa \
  --output-dir results/sft_data/
```

## Plotting

```bash
uv run scripts/plot_final.py
```

Generates ~28 plots in `results/plots/main/`: per-model aggregates, per-mode breakdowns, cross-model comparisons, hillclimb results, and shot-scaling curves.

## Architecture Notes

- **No reimplementation of grading** — CoTControl and ReasonIF grading functions are imported directly from submodules via `sys.path.insert()`
- **Async batch inference** with configurable concurrency (typically 200-500), exponential backoff retry, and timeout handling
- **Resume support** — re-running an eval skips completed `(sample_id, mode)` pairs already in the output JSONL
- **JSONL rollout storage** — each rollout includes the full prompt, reasoning trace, response, grading details, latency, and token usage
- **GPT-OSS** uses a built-in system prompt from `tinker_cookbook` that enables multi-channel output (analysis/commentary/final) with configurable reasoning effort

## Tests

```bash
uv run pytest tests/
```
