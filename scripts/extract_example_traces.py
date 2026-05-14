"""Extract example reasoning traces — just self_check, full output."""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from controllability.config import Settings
from controllability.datasets.loader import load_dataset
from controllability.datasets.splits import proportional_sample, split_dataset
from controllability.inference.tinker_client import TinkerClient
from controllability.types import InferenceRequest

sys.path.insert(0, str(Path(__file__).resolve().parent / "runs"))
from run_prompt_ablation import ALL_VARIANTS


async def main():
    settings = Settings()
    os.environ.setdefault("TINKER_API_KEY", settings.tinker_api_key)

    client = TinkerClient(model="qwen/qwen3-32b", request_timeout=180)

    samples = load_dataset("cotcontrol/gpqa")
    test_samples = split_dataset(samples, split="all", seed=42)
    test_samples = proportional_sample(test_samples, n=5, seed=42)

    sample = test_samples[0]
    vfunc = ALL_VARIANTS["self_check"]
    sys_prompt, usr_prompt = vfunc("lowercase_thinking", sample)
    messages = []
    if sys_prompt:
        messages.append({"role": "system", "content": sys_prompt})
    messages.append({"role": "user", "content": usr_prompt})

    request = InferenceRequest(
        messages=messages, model="qwen/qwen3-32b", max_tokens=4096, temperature=1.0,
    )

    resp = await asyncio.wait_for(client.complete(request), timeout=180)
    print("=== REASONING ===", flush=True)
    print(repr(resp.reasoning[:1500]), flush=True)
    print("\n=== CONTENT (first 1500) ===", flush=True)
    print(resp.content[:1500], flush=True)

    await client.close()

if __name__ == "__main__":
    asyncio.run(main())
