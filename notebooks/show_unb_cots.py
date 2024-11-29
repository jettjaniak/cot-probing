#!/usr/bin/env python3
import pickle
from pathlib import Path

from cot_probing import DATA_DIR
from cot_probing.generation import UnbiasedCotGeneration
from cot_probing.qs_generation import Question
from transformers import PreTrainedTokenizerBase, AutoTokenizer


def load_unb_cots_files() -> (
    dict[str, tuple[UnbiasedCotGeneration, PreTrainedTokenizerBase]]
):
    """Load all unbiased CoT files from the unb-cots directory and their tokenizers."""
    unb_cots_dir = DATA_DIR / "unb-cots"
    results = {}

    for file_path in unb_cots_dir.glob("*.pkl"):
        with open(file_path, "rb") as f:
            unb_cots = pickle.load(f)
            tokenizer = AutoTokenizer.from_pretrained(unb_cots.model)
            results[file_path.stem] = (unb_cots, tokenizer)

    return results


def load_questions(dataset_id: str) -> dict[str, Question]:
    """Load questions for a specific dataset."""
    questions_path = DATA_DIR / "questions" / f"{dataset_id}.pkl"
    with open(questions_path, "rb") as f:
        return pickle.load(f)


def get_common_question_ids(
    unb_cots_dict: dict[str, tuple[UnbiasedCotGeneration, PreTrainedTokenizerBase]]
) -> set[str]:
    """Find question IDs that appear in all files."""
    if not unb_cots_dict:
        return set()

    # Get sets of question IDs from each file
    qid_sets = [set(cots.cots_by_qid.keys()) for cots, _ in unb_cots_dict.values()]

    # Return intersection of all sets
    return set.intersection(*qid_sets)


def main():
    # Load all unbiased CoT files and their tokenizers
    unb_cots_dict = load_unb_cots_files()
    if not unb_cots_dict:
        print("No unbiased CoT files found!")
        return

    # Get common question IDs
    common_qids = get_common_question_ids(unb_cots_dict)
    if not common_qids:
        print("No questions found in common across all files!")
        return

    # Extract dataset_id from first file name (assumes format model_dataset.pkl)
    first_file = next(iter(unb_cots_dict.keys()))
    dataset_id = first_file.split("_")[-1]

    # Load questions
    questions = load_questions(dataset_id)

    # Print CoTs for each common question
    for qid in sorted(common_qids):
        q = questions[qid]
        print(f"\nQuestion ID: {qid}")
        print(f"Question: {q.question}")
        print(f"Answer: {q.expected_answer}")
        print("\nCoTs from different models:")
        print("-" * 80)

        for model_name, (unb_cots, tokenizer) in unb_cots_dict.items():
            print(f"\n{model_name}:")
            for i, cot in enumerate(unb_cots.cots_by_qid[qid], 1):
                print(f"\nGeneration {i}:")
                print(tokenizer.decode(cot))
            print("-" * 40)


if __name__ == "__main__":
    main()
