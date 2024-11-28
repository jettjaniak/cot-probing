#!/usr/bin/env python3
import argparse
import logging
import pickle
from pathlib import Path

from beartype import beartype
from tqdm import tqdm

from cot_probing import DATA_DIR
from cot_probing.cot_evaluation import evaluate_cots
from cot_probing.data.qs_evaluation import NoCotAccuracy
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.generation import CotGeneration, gen_unb_cots
from cot_probing.qs_generation import generate_unbiased_few_shot_prompt
from cot_probing.utils import load_model_and_tokenizer, setup_determinism


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and evaluate unbiased CoTs")
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
        "--max-no-cot-acc",
        type=float,
        help="Maximum no-CoT accuracy to generate unbiased CoTs for",
        default=0.6,
    )
    parser.add_argument(
        "-o",
        "--openai-model",
        type=str,
        default="gpt-4o",
        help="OpenAI model used to evaluate unbiased CoTs",
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
def build_unb_fsp(args: argparse.Namespace) -> str:
    setup_determinism(args.seed)

    yes_fsps = load_and_process_file(DATA_DIR / "diverse_qs_expected_yes_with_cot.txt")
    no_fsps = load_and_process_file(DATA_DIR / "diverse_qs_expected_no_with_cot.txt")

    unbiased_fsp = generate_unbiased_few_shot_prompt(
        all_qs_yes=yes_fsps, all_qs_no=no_fsps, fsp_size=args.fsp_size
    )
    return unbiased_fsp


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    model, tokenizer = load_model_and_tokenizer(args.model_size)

    # Load the dataset
    dataset_path = Path(args.file)
    assert dataset_path.exists()
    with open(dataset_path, "rb") as f:
        question_dataset = pickle.load(f)

    # Load the no-cot accuracy results
    dataset_identifier = dataset_path.stem.split("_")[-1]
    with open(DATA_DIR / f"no-cot-accuracy_{dataset_identifier}.pkl", "rb") as f:
        no_cot_accuracy_results: NoCotAccuracy = pickle.load(f)
        assert no_cot_accuracy_results.model == model.config._name_or_path

    output_path = DATA_DIR / f"unb-cots_{dataset_identifier}.pkl"

    # Build the FSP
    yes_fsps = load_and_process_file(DATA_DIR / "diverse_qs_expected_yes_with_cot.txt")
    no_fsps = load_and_process_file(DATA_DIR / "diverse_qs_expected_no_with_cot.txt")
    unb_fsp = generate_unbiased_few_shot_prompt(
        all_qs_yes=yes_fsps, all_qs_no=no_fsps, fsp_size=args.fsp_size
    )
    unb_fsp_toks = tokenizer.encode(unb_fsp, add_special_tokens=True)

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
        if q_id not in no_cot_accuracy_results.acc_by_qid:
            continue

        no_cot_acc = no_cot_accuracy_results.acc_by_qid[q_id]
        if no_cot_acc > args.max_no_cot_acc:
            continue

        unb_cots = gen_unb_cots(
            q=q,
            model=model,
            tokenizer=tokenizer,
            unb_fsp_toks=unb_fsp_toks,
            args=args,
            verbose=args.verbose,
        )
        labeled_unb_cots = evaluate_cots(
            q=q,
            cots=unb_cots,
            tokenizer=tokenizer,
            openai_model=args.openai_model,
            verbose=args.verbose,
        )
        results.cots_by_qid[q_id] = labeled_unb_cots

        if len(results.cots_by_qid) % args.save_every == 0:
            with open(output_path, "wb") as f:
                pickle.dump(results, f)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main(parse_args())
