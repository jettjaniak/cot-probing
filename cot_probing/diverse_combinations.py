import random

from cot_probing import DATA_DIR
from cot_probing.typing import *


def load_and_process_file(file_path: Path) -> list[str]:
    """
    Loads a text file, splits it by double line break, and processes it.

    Args:
        file_path (Path): The path to the text file.
        q_idx (int): The index of the question to use.

    Returns:
        tuple[str, list[str]]: A tuple containing:
            - The question at the given index
            - A list of all other questions
    """
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read().split("\n\n")
    return content


def generate_combinations(
    q_yes: str, qs_yes: list[str], q_no: str, qs_no: list[str], seed: int
) -> dict[str, str]:
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
    shuffled_indices = random.sample(range(len(qs_yes)), len(qs_yes))
    qs_yes = [qs_yes[i] for i in shuffled_indices]
    qs_no = [qs_no[i] for i in shuffled_indices]

    unb_yes_indices = random.sample(range(len(qs_yes)), len(qs_yes) // 2)
    qs_unb = [
        qs_yes[i] if i in unb_yes_indices else qs_no[i] for i in range(len(qs_yes))
    ]

    unb_fsps = "\n\n".join(qs_unb)
    yes_fsps = "\n\n".join(qs_yes)
    no_fsps = "\n\n".join(qs_no)

    combinations = [
        (unb_fsps, q_yes, "unb_yes"),
        (no_fsps, q_yes, "no_yes"),
        (unb_fsps, q_no, "unb_no"),
        (yes_fsps, q_no, "yes_no"),
    ]

    combined_prompts = {}

    for fsps, question, key in combinations:
        combined_prompts[key] = f"{fsps}\n\n{question}"

    return combined_prompts


def split_questions(qs: list[str], q_idx: int) -> tuple[str, list[str]]:
    split_string = "Let's think step by step:\n-"
    q = qs[q_idx].split(split_string)[0] + split_string
    qs = qs[:q_idx] + qs[q_idx + 1 :]
    return q, qs


def generate_all_combinations(seed: int) -> list[dict[str, str]]:
    all_qs_yes = load_and_process_file(DATA_DIR / "diverse_yes.txt")
    all_qs_no = load_and_process_file(DATA_DIR / "diverse_no.txt")

    combinations = []
    for q_idx in range(len(all_qs_yes)):
        q_yes, qs_yes = split_questions(all_qs_yes, q_idx)
        q_no, qs_no = split_questions(all_qs_no, q_idx)
        combinations.append(generate_combinations(q_yes, qs_yes, q_no, qs_no, seed))
    return combinations
