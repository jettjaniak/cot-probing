#!/usr/bin/env python3
import argparse
import logging
import pickle

from cot_probing import DATA_DIR
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.qs_generation import generate_questions_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate questions dataset")
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
        "--save-every",
        type=int,
        help="Save every N questions",
        default=50,
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    questions_dataset_path = DATA_DIR / "questions" / "generated_qs.pkl"

    all_qs_yes = load_and_process_file(
        DATA_DIR / "diverse_qs_expected_yes_with_cot.txt"
    )
    all_qs_no = load_and_process_file(DATA_DIR / "diverse_qs_expected_no_with_cot.txt")
    assert len(all_qs_yes) == len(all_qs_no)

    # Add existing questions to all_qs_yes and all_qs_no so that we have less chance of repeating them
    question_dataset = {}
    if questions_dataset_path.exists():
        with open(questions_dataset_path, "rb") as f:
            question_dataset = pickle.load(f)

    for q in question_dataset.values():
        if "openai_generation" in q.extra_data:
            full_question = q.extra_data["openai_generation"]
            if q.expected_answer == "yes":
                all_qs_yes.append(full_question)
            else:
                all_qs_no.append(full_question)

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
        save_every=args.save_every,
    )

    if questions_dataset_path.exists():
        with open(questions_dataset_path, "rb") as f:
            question_dataset = pickle.load(f)

        if args.verbose:
            for question in question_dataset.values():
                print(question.question)
                print()


if __name__ == "__main__":
    main(parse_args())
