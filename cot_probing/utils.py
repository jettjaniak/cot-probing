import json
import os
import random
import time

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)
from wandb.apis.public.api import Api
from wandb.apis.public.runs import Run

from cot_probing import DATA_DIR
from cot_probing.typing import *


def load_task_data(task_name: str) -> tuple[str, str, list[dict]]:
    with open(DATA_DIR / f"{task_name}/few_shot_prompts.json", "r") as f:
        few_shot_prompts_dict = json.load(f)
        baseline_fsp = few_shot_prompts_dict["baseline_few_shot_prompt"]
        alla_fsp = few_shot_prompts_dict["all_a_few_shot_prompt"]
    with open(DATA_DIR / f"{task_name}/val_data.json", "r") as f:
        # the other key is "canary"
        question_dicts = json.load(f)["data"]

    return baseline_fsp, alla_fsp, question_dicts


def load_model_and_tokenizer(
    model_size: int,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    assert model_size in [8, 70]
    model_id = f"hugging-quants/Meta-Llama-3.1-{model_size}B-BNB-NF4-BF16"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    return model, tokenizer


def get_train_test_split(
    idxs: list[int], train_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    random.seed(seed)
    shuffled_idxs = idxs.copy()
    random.shuffle(shuffled_idxs)
    split_idx = int(len(idxs) * train_frac)
    return shuffled_idxs[:split_idx], shuffled_idxs[split_idx:]


def to_str_tokens(
    input: str | list[int], tokenizer: PreTrainedTokenizerBase
) -> list[str]:
    if isinstance(input, str):
        input_ids = tokenizer.encode(input)
    else:
        input_ids = input

    return [
        str_token.replace("Ġ", " ").replace("Ċ", "\\n")
        for str_token in tokenizer.convert_ids_to_tokens(input_ids)
    ]


def find_sublist(haystack: list[int], needle: list[int]) -> int | None:
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i : i + len(needle)] == needle:
            return i
    return None


def setup_determinism(seed: int):
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def get_git_commit_hash() -> str:
    """Returns the current git commit hash."""
    try:
        import subprocess

        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("ascii")
            .strip()
        )
    except:
        return "git_hash_not_found"


def safe_torch_save(obj, filepath, timeout=10):
    """
    Safely save a PyTorch object and verify it exists before proceeding.

    Args:
        obj: The PyTorch object to save
        filepath: Path where the file should be saved
        timeout: Maximum time in seconds to wait for file to appear

    Returns:
        bool: True if save was successful and verified, False otherwise
    """
    # Save the object
    torch.save(obj, filepath)

    # Force sync to filesystem
    os.sync()

    # Wait for file to appear and be non-empty
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
            # Verify we can load it
            try:
                torch.load(filepath, weights_only=True)
                return True
            except Exception:
                time.sleep(0.1)
                continue
        time.sleep(0.1)

    return False


def fetch_runs(
    api: Api,
    probe_class: str,
    min_layer: int,
    max_layer: int,
    min_seed: int,
    max_seed: int,
    context: Literal["biased-fsp", "unbiased-fsp", "no-fsp"],
    entity: str = "cot-probing",
    project: str = "attn-probes",
) -> dict[int, dict[int, Run]]:
    """Fetch all runs matching criteria in a single query."""
    # Construct query filters
    filters = {
        "$and": [
            {"config.args_probe_class": probe_class},
            {
                "$and": [
                    {"config.args_data_seed": {"$gte": min_seed}},
                    {"config.args_data_seed": {"$lte": max_seed}},
                ]
            },
            {
                "$and": [
                    {"config.args_weight_init_seed": {"$gte": min_seed}},
                    {"config.args_weight_init_seed": {"$lte": max_seed}},
                ]
            },
            {
                "$and": [
                    {"config.layer": {"$gte": min_layer}},
                    {"config.layer": {"$lte": max_layer}},
                ]
            },
            {"config.dataset_context": context},
        ]
    }

    # Fetch all matching runs at once
    runs = list(api.runs(f"{entity}/{project}", filters))
    print(f"Fetched {len(runs)} runs")

    # Organize runs by layer and seed
    layers = range(min_layer, max_layer + 1)
    run_by_seed_by_layer = {layer: {} for layer in layers}
    for run in runs:
        layer = run.config["layer"]
        seed = run.config["args_data_seed"]
        run_by_seed_by_layer[layer][seed] = run

    # Verify we got all expected runs
    expected_count = (max_layer - min_layer + 1) * (max_seed - min_seed + 1)
    actual_count = sum(len(seeds_dict) for seeds_dict in run_by_seed_by_layer.values())
    if actual_count != expected_count:
        print(f"Warning: Expected {expected_count} runs, got {actual_count}")

    return run_by_seed_by_layer
