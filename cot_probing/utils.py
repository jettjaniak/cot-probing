import json
import random

from transformers import PreTrainedTokenizerBase

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
