#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from cot_probing import DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delete questions from the dataset by their indices"
    )
    parser.add_argument(
        "indices", nargs="+", type=int, help="Indices of questions to delete"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def delete_questions(questions, indices_to_delete):
    indices_to_delete = set(indices_to_delete)
    return [q for i, q in enumerate(questions) if i not in indices_to_delete]


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    questions_dataset_path = DATA_DIR / "generated_questions_dataset.json"

    if questions_dataset_path.exists():
        with open(questions_dataset_path, "r") as f:
            question_dataset = json.load(f)
    else:
        raise FileNotFoundError(
            f"Questions dataset file not found at {questions_dataset_path}"
        )

    original_count = len(question_dataset)
    updated_dataset = delete_questions(question_dataset, args.indices)
    deleted_count = original_count - len(updated_dataset)

    # Save the updated dataset
    with open(questions_dataset_path, "w") as f:
        json.dump(updated_dataset, f, indent=2)

    print(f"Deleted {deleted_count} questions from the dataset.")
    print(f"Updated dataset saved to {questions_dataset_path}")


if __name__ == "__main__":
    main(parse_args())
