#!/usr/bin/env python3
import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Literal

from openai import OpenAI
from tqdm import tqdm

from cot_probing import DATA_DIR
from cot_probing.qs_evaluation import get_openai_expected_answer
from cot_probing.qs_generation import Question


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add open ai inferred expected answer to questions"
    )
    parser.add_argument("-d", "--dataset-id", type=str, default="strategyqa")
    parser.add_argument(
        "-o",
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model used to evaluate unbiased CoTs",
    )
    parser.add_argument(
        "--answer-mode",
        type=str,
        default="with-cot",
        choices=["with-cot", "answer-only"],
        help="Mode to use to get the expected answer",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save every N questions",
        default=50,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def process_qs(
    questions_dataset: dict[str, Question],
    openai_model: str,
    answer_mode: Literal["with-cot", "answer-only"],
    output_path: Path,
    verbose: bool = False,
    save_every: int = 50,
):
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    processed_count = 0
    for q_id, q in tqdm(questions_dataset.items(), desc="Processing questions"):
        extra_data_key = f"{openai_model}_expected-answer-{answer_mode}"
        if extra_data_key in q.extra_data:
            continue

        openai_expected_answer, raw_openai_answer = get_openai_expected_answer(
            q=q,
            q_id=q_id,
            openai_client=openai_client,
            openai_model=openai_model,
            answer_mode=answer_mode,
            verbose=verbose,
        )
        q.extra_data[extra_data_key] = openai_expected_answer
        q.extra_data[f"{extra_data_key}-raw"] = raw_openai_answer

        processed_count += 1

        if processed_count % save_every == 0:
            with open(output_path, "wb") as f:
                pickle.dump(questions_dataset, f)

    with open(output_path, "wb") as f:
        pickle.dump(questions_dataset, f)


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    questions_dir = DATA_DIR / "questions"
    questions_path = questions_dir / f"{args.dataset_id}.pkl"
    with open(questions_path, "rb") as f:
        questions_dataset: dict[str, Question] = pickle.load(f)

    process_qs(
        questions_dataset=questions_dataset,
        openai_model=args.openai_model,
        answer_mode=args.answer_mode,
        output_path=questions_path,
        verbose=args.verbose,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main(parse_args())
