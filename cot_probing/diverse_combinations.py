import random

from cot_probing import DATA_DIR
from cot_probing.typing import *


def load_and_split_file(file_path: Path) -> list[str]:
    """
    Loads a text file and splits it by double line break.

    Args:
        file_path (str): The path to the text file.

    Returns:
        list[str]: A list of strings, each representing a section split by double line breaks.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    return content.split("\n\n")


def generate_combinations(
    seed: int, q_idx: int = 0
) -> tuple[dict[str, str], tuple[str, str, str]]:
    """
    Generates combinations of few-shot prompts and questions based on the given seed and question index.

    Args:
        seed (int): The random seed to use for shuffling.
        q_idx (int): The index of the question to use. Defaults to 0.

    Returns:
        tuple[dict[str, str], tuple[str, str, str]]: A tuple containing:
            - A dictionary of combined prompts
            - A tuple of (unbiased, yes-biased, no-biased) few-shot prompts
    """
    random.seed(seed)

    qs_unb = load_and_split_file(DATA_DIR / "diverse_unbiased.txt")
    qs_yes = load_and_split_file(DATA_DIR / "diverse_yes.txt")
    qs_no = load_and_split_file(DATA_DIR / "diverse_no.txt")

    shuffled_indices = random.sample(range(len(qs_unb)), len(qs_unb))
    qs_unb = [qs_unb[i] for i in shuffled_indices]
    qs_yes = [qs_yes[i] for i in shuffled_indices]
    qs_no = [qs_no[i] for i in shuffled_indices]

    unb_fsps = "\n\n".join(qs_unb[:q_idx] + qs_unb[q_idx + 1 :])
    yes_fsps = "\n\n".join(qs_yes[:q_idx] + qs_yes[q_idx + 1 :])
    no_fsps = "\n\n".join(qs_no[:q_idx] + qs_no[q_idx + 1 :])

    split_string = "Let's think step by step:\n-"
    q_yes = qs_yes[q_idx].split(split_string)[0] + split_string
    q_no = qs_no[q_idx].split(split_string)[0] + split_string

    combinations = [
        (unb_fsps, q_yes, "unb_yes"),
        (no_fsps, q_yes, "no_yes"),
        (unb_fsps, q_no, "unb_no"),
        (yes_fsps, q_no, "yes_no"),
    ]

    combined_prompts = {}

    for fsps, question, key in combinations:
        combined_prompts[key] = f"{fsps}\n\n{question}"

    return combined_prompts, (unb_fsps, yes_fsps, no_fsps)
