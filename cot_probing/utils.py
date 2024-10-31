import json
import random
from typing import Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

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
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
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
