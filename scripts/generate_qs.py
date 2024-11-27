#!/usr/bin/env python3
import argparse
import logging
import pickle

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing import DATA_DIR
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.questions_generation import generate_questions_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate questions dataset")
    parser.add_argument("-s", "--size", type=int, default=8, help="Model size")
    parser.add_argument(
        "-o", "--openai-model", type=str, default="gpt-4o", help="OpenAI model"
    )
    parser.add_argument(
        "-n",
        "--num-questions",
        type=int,
        default=1,
        help="Number of questions to generate",
    )
    parser.add_argument(
        "-e",
        "--expected-answers",
        type=str,
        default="mixed",
        choices=["yes", "no", "mixed"],
        help="Expected answers for the questions",
    )
    parser.add_argument(
        "-a", "--max-attempts", type=int, default=100, help="Maximum number of attempts"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--fsp-size",
        type=int,
        default=7,
        help="Size of the few-shot prompt.",
    )
    parser.add_argument(
        "--unb-n-gen",
        type=int,
        default=10,
        help="Number of unbiased responses to generate.",
    )
    parser.add_argument(
        "--unb-temp",
        type=float,
        default=0.7,
        help="Temperature for sampling unbiased responses.",
    )
    parser.add_argument(
        "--expected-min-completion-accuracy-in-unbiased-context",
        type=float,
        default=0.7,
        help="Expected min accuracy in unbiased context.",
    )
    parser.add_argument(
        "--expected-max-completion-accuracy-in-unbiased-context",
        type=float,
        default=0.9,
        help="Expected max accuracy in unbiased context.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    questions_dataset_path = DATA_DIR / "generated_qs.pkl"

    question_dataset = []
    if questions_dataset_path.exists():
        with open(questions_dataset_path, "rb") as f:
            question_dataset = pickle.load(f)

    all_qs_yes = load_and_process_file(
        DATA_DIR / "diverse_qs_expected_yes_with_cot.txt"
    )
    all_qs_no = load_and_process_file(DATA_DIR / "diverse_qs_expected_no_with_cot.txt")
    assert len(all_qs_yes) == len(all_qs_no)

    # Add questions to all_qs_yes and all_qs_no so that we don't repeat them
    for row in question_dataset:
        if row["expected_answer"] == "yes":
            all_qs_yes.append(row["question"])
        else:
            all_qs_no.append(row["question"])

    # Generate the dataset
    generate_questions_dataset(
        openai_model=args.openai_model,
        num_questions=args.num_questions,
        expected_answers=args.expected_answers,
        max_attempts=args.max_attempts,
        all_qs_yes=all_qs_yes,
        all_qs_no=all_qs_no,
        questions_dataset_path=questions_dataset_path,
        fsp_size=args.fsp_size,
    )

    if questions_dataset_path.exists():
        with open(questions_dataset_path, "rb") as f:
            question_dataset = pickle.load(f)
        for question in question_dataset:
            print(question["question"])
            print()


if __name__ == "__main__":
    main(parse_args())
