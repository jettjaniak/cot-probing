#!/usr/bin/env python3
import argparse
import logging
import pickle
from pathlib import Path

from cot_probing import DATA_DIR
from cot_probing.data.qs_evaluation import evaluate_no_cot_accuracy
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.qs_generation import generate_unbiased_few_shot_prompt
from cot_probing.utils import load_model_and_tokenizer, setup_determinism


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate no-CoT accuracy")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the dataset of questions",
    )
    parser.add_argument(
        "-m",
        "--model-size",
        type=int,
        default=8,
        help="Model size in billions of parameters",
    )
    parser.add_argument("-s", "--seed", type=int, help="Random seed", default=0)
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--fsp-size",
        type=int,
        default=7,
        help="Size of the few-shot prompt.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    dataset_path = Path(args.file)
    assert dataset_path.exists()
    with open(dataset_path, "rb") as f:
        question_dataset = pickle.load(f)

    all_qs_yes = load_and_process_file(
        DATA_DIR / "diverse_qs_expected_yes_without_cot.txt"
    )
    all_qs_no = load_and_process_file(
        DATA_DIR / "diverse_qs_expected_no_without_cot.txt"
    )
    assert len(all_qs_yes) == len(all_qs_no)

    setup_determinism(args.seed)
    unbiased_fsp_without_cot = generate_unbiased_few_shot_prompt(
        all_qs_yes, all_qs_no, args.fsp_size
    )

    model, tokenizer = load_model_and_tokenizer(args.model_size)

    # Generate the dataset
    results = evaluate_no_cot_accuracy(
        model=model,
        tokenizer=tokenizer,
        question_dataset=question_dataset,
        unbiased_fsp_without_cot=unbiased_fsp_without_cot,
        fsp_size=args.fsp_size,
        seed=args.seed,
    )

    file_identifier = dataset_path.stem.split("_")[-1]
    with open(DATA_DIR / f"no-cot-accuracy_{file_identifier}.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main(parse_args())
