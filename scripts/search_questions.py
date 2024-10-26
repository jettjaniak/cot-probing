#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from cot_probing import DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Search for questions containing specified keywords"
    )
    parser.add_argument(
        "keywords", nargs="+", help="Keywords to search for in questions"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def search_questions(questions, keywords):
    results = []
    for idx, item in enumerate(questions):
        question = item["question"]
        if all(keyword.lower() in question.lower() for keyword in keywords):
            results.append((idx, question))
    return results


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

    matching_questions = search_questions(question_dataset, args.keywords)

    if matching_questions:
        print(f"Found {len(matching_questions)} questions containing all keywords:")
        for idx, question in matching_questions:
            print(f"\nIndex: {idx}")
            print(f"Question: {question}")
    else:
        print("No questions found containing all specified keywords.")


if __name__ == "__main__":
    main(parse_args())
