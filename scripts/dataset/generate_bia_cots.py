#!/usr/bin/env python3
import argparse
import logging
import pickle
import random
from pathlib import Path

import numpy as np
from beartype import beartype
from tqdm import tqdm

from cot_probing import DATA_DIR
from cot_probing.cot_evaluation import evaluate_cots
from cot_probing.data.qs_evaluation import NoCotAccuracy
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.generation import CotGeneration, gen_bia_cots
from cot_probing.utils import load_model_and_tokenizer, setup_determinism


def parse_args():
    parser = argparse.ArgumentParser(description="Generate biased and unbiased CoTs")
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
    parser.add_argument(
        "--fsp-size",
        type=int,
        default=7,
        help="Size of the few-shot prompt.",
    )
    parser.add_argument(
        "-t", "--temp", type=float, help="Temperature for generation", default=0.7
    )
    parser.add_argument(
        "-n", "--n-gen", type=int, help="Number of generations to produce", default=20
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Maximum number of new tokens to generate",
        default=200,
    )
    parser.add_argument(
        "--min-unb-cot-corr",
        type=float,
        help="Minimum unbiased CoT correctness to generate biased CoTs for",
        default=0.8,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save every N questions",
        default=50,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


@beartype
def build_bia_fsps(args: argparse.Namespace) -> tuple[str, str]:
    setup_determinism(args.seed)

    yes_fsps = load_and_process_file(DATA_DIR / "diverse_qs_expected_yes_with_cot.txt")
    no_fsps = load_and_process_file(DATA_DIR / "diverse_qs_expected_no_with_cot.txt")

    shuffled_indices = random.sample(range(len(yes_fsps)), args.fsp_size)
    yes_fsps = [yes_fsps[i] for i in shuffled_indices]
    no_fsps = [no_fsps[i] for i in shuffled_indices]

    yes_fsps = "\n\n".join(yes_fsps) + "\n\n"
    no_fsps = "\n\n".join(no_fsps) + "\n\n"

    return yes_fsps, no_fsps


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    model, tokenizer = load_model_and_tokenizer(args.model_size)

    # Load the dataset
    dataset_path = Path(args.file)
    assert dataset_path.exists()
    with open(dataset_path, "rb") as f:
        question_dataset = pickle.load(f)

    # Load the unb-cot accuracy results
    dataset_identifier = dataset_path.stem.split("_")[-1]
    with open(DATA_DIR / f"unb-cots_{dataset_identifier}.pkl", "rb") as f:
        unb_cots_results: CotGeneration = pickle.load(f)
        assert unb_cots_results.model == model.config._name_or_path

    output_path = DATA_DIR / f"bia-cots_{dataset_identifier}.pkl"

    # Build the few-shot prompts
    yes_fsps, no_fsps = build_bia_fsps(args)
    yes_fsp_toks, no_fsp_toks = [tokenizer.encode(fsp) for fsp in [yes_fsps, no_fsps]]

    results = CotGeneration(
        cots_by_qid={},
        model=model.config._name_or_path,
        fsp_size=args.fsp_size,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temp=args.temp,
        do_sample=True,
    )
    for q_id, q in tqdm(question_dataset.items(), desc="Processing questions"):
        if q_id not in unb_cots_results.cots_by_qid:
            continue

        unb_cot_avg_corr = np.mean(
            [
                1 if cot.label == "correct" else 0
                for cot in unb_cots_results.cots_by_qid[q_id]
            ]
        )
        if unb_cot_avg_corr < args.min_unb_cot_corr:
            continue

        bia_cots = gen_bia_cots(
            q=q,
            model=model,
            tokenizer=tokenizer,
            yes_fsp_toks=yes_fsp_toks,
            no_fsp_toks=no_fsp_toks,
            args=args,
        )
        labeled_bia_cots = evaluate_cots(
            q=q,
            cots=bia_cots,
            tokenizer=tokenizer,
            openai_model=args.openai_model,
            verbose=args.verbose,
        )
        results.cots_by_qid[q_id] = labeled_bia_cots

        if len(results.cots_by_qid) % args.save_every == 0:
            with open(output_path, "wb") as f:
                pickle.dump(results, f)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main(parse_args())
