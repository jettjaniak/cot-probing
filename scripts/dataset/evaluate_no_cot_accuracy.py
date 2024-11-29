#!/usr/bin/env python3
import argparse
import logging
import pickle
from pathlib import Path

from cot_probing import DATA_DIR
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.qs_evaluation import (
    evaluate_no_cot_accuracy,
    evaluate_no_cot_accuracy_chat,
)
from cot_probing.qs_generation import generate_unbiased_few_shot_prompt
from cot_probing.utils import (
    is_chat_model,
    load_any_model_and_tokenizer,
    setup_determinism,
)


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
        "--model-id",
        type=str,
        default="hugging-quants/Meta-Llama-3.1-8B-BNB-NF4-BF16",
        help="Model ID",
    )
    parser.add_argument("-s", "--seed", type=int, help="Random seed", default=0)
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--fsp-size",
        type=int,
        default=7,
        help="Size of the few-shot prompt, ignored for chat models.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    dataset_path = Path(args.file)
    assert dataset_path.exists()
    with open(dataset_path, "rb") as f:
        question_dataset = pickle.load(f)

    model, tokenizer = load_any_model_and_tokenizer(args.model_id)

    if is_chat_model(args.model_id):
        # For chat models, use chat evaluation
        results = evaluate_no_cot_accuracy_chat(
            model=model,
            tokenizer=tokenizer,
            question_dataset=question_dataset,
            seed=args.seed,
        )
    else:
        # For non-chat models, use standard evaluation with few-shot prompting
        all_qs_yes = load_and_process_file(
            DATA_DIR / "diverse_qs_expected_yes_without_cot.txt"
        )
        all_qs_no = load_and_process_file(
            DATA_DIR / "diverse_qs_expected_no_without_cot.txt"
        )
        assert len(all_qs_yes) == len(all_qs_no)

        setup_determinism(args.seed)
        unbiased_fsp_without_cots = generate_unbiased_few_shot_prompt(
            all_qs_yes, all_qs_no, args.fsp_size, verbose=args.verbose
        )

        results = evaluate_no_cot_accuracy(
            model=model,
            tokenizer=tokenizer,
            question_dataset=question_dataset,
            unbiased_fsp_without_cots=unbiased_fsp_without_cots,
            fsp_size=args.fsp_size,
            seed=args.seed,
        )

    # Save results with model name in filename for chat models
    file_identifier = dataset_path.stem.split("_")[-1]
    model_name = args.model_id.split("/")[-1]
    filename = f"{model_name}_{file_identifier}.pkl"
    with open(DATA_DIR / "no-cot-accuracy" / filename, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main(parse_args())
