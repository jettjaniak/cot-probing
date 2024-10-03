import json
from pprint import pprint

from cot_probing import DATA_DIR


def load_task_data(task_name: str) -> tuple[str, str, list[dict]]:
    with open(DATA_DIR / f"{task_name}/few_shot_prompts.json", "r") as f:
        few_shot_prompts_dict = json.load(f)
        baseline_fsp = few_shot_prompts_dict["baseline_few_shot_prompt"]
        alla_fsp = few_shot_prompts_dict["all_a_few_shot_prompt"]
    with open(DATA_DIR / f"{task_name}/val_data.json", "r") as f:
        # the other key is "canary"
        question_dicts = json.load(f)["data"]

    return baseline_fsp, alla_fsp, question_dicts
