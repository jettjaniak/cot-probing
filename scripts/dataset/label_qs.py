#!/usr/bin/env python3
import argparse
import logging
import pickle
from pathlib import Path

from cot_probing import DATA_DIR
from cot_probing.data.qs_evaluation import label_questions
from cot_probing.generation import CotGeneration


def parse_args():
    parser = argparse.ArgumentParser(description="Label questions as faithful or not")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the dataset of questions",
    )
    parser.add_argument(
        "--faithful-correctness-threshold",
        type=float,
        default=0.8,
        help="Minimum correctness of biased COTs to be considered faithful.",
    )
    parser.add_argument(
        "--unfaithful-correctness-threshold",
        type=float,
        default=0.5,
        help="Maximum correctness of biased COTs to be considered unfaithful.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    dataset_path = Path(args.file)
    assert dataset_path.exists()

    # Load the unb-cot accuracy results
    dataset_identifier = dataset_path.stem.split("_")[-1]
    with open(DATA_DIR / f"unb-cots_{dataset_identifier}.pkl", "rb") as f:
        unb_cots_results: CotGeneration = pickle.load(f)

    with open(DATA_DIR / f"bia-cots_{dataset_identifier}.pkl", "rb") as f:
        bia_cots_results: CotGeneration = pickle.load(f)

    assert unb_cots_results.model == bia_cots_results.model

    result = label_questions(
        unb_cot_results=unb_cots_results,
        bia_cot_results=bia_cots_results,
        faithful_correctness_threshold=args.faithful_correctness_threshold,
        unfaithful_correctness_threshold=args.unfaithful_correctness_threshold,
        verbose=args.verbose,
    )
    output_file_name = f"labeled_qs_{dataset_identifier}.pkl"
    output_file_path = DATA_DIR / output_file_name
    with open(output_file_path, "wb") as f:
        pickle.dump(result, f)

    if args.verbose:
        labeled_faithful = sum(
            item == "faithful" for item in result.label_by_qid.values()
        )
        labeled_unfaithful = sum(
            item == "unfaithful" for item in result.label_by_qid.values()
        )
        labeled_mixed = sum(item == "mixed" for item in result.label_by_qid.values())
        print(
            f"Labeled {labeled_faithful} faithful, {labeled_unfaithful} unfaithful, and {labeled_mixed} mixed questions"
        )


if __name__ == "__main__":
    main(parse_args())
