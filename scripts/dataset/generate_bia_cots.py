#!/usr/bin/env python3
import argparse
import logging
import pickle
import random
from pathlib import Path

from beartype import beartype
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing import DATA_DIR
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.generation import BiasedCotGeneration, gen_bia_cots
from cot_probing.qs_generation import Question
from cot_probing.utils import (
    is_chat_model,
    load_any_model_and_tokenizer,
    setup_determinism,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate biased and unbiased CoTs")
    parser.add_argument("-d", "--dataset-id", type=str, default="strategyqa")
    parser.add_argument(
        "-m",
        "--model-id",
        type=str,
        default="hugging-quants/Meta-Llama-3.1-8B-BNB-NF4-BF16",
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

    if args.verbose:
        logging.info(f"Yes FSP:\n{yes_fsps}\n\n")
        logging.info(f"No FSP:\n{no_fsps}\n\n")

    return yes_fsps, no_fsps


def generate_bia_cots(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions_dataset: dict[str, Question],
    bia_yes_fsp_toks: list[int],
    bia_no_fsp_toks: list[int],
    args: argparse.Namespace,
    output_path: Path,
    verbose: bool = False,
):

    results = BiasedCotGeneration(
        cots_by_qid={},
        model=model.config._name_or_path,
        fsp_size=args.fsp_size,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temp=args.temp,
        do_sample=True,
        bia_yes_fsp=bia_yes_fsp_toks,
        bia_no_fsp=bia_no_fsp_toks,
    )
    for q_id, q in tqdm(questions_dataset.items(), desc="Processing questions"):
        results.cots_by_qid[q_id] = gen_bia_cots(
            q=q,
            model=model,
            tokenizer=tokenizer,
            bia_fsp_toks=(
                bia_yes_fsp_toks if q.expected_answer == "no" else bia_no_fsp_toks
            ),
            args=args,
            verbose=verbose,
        )

        if len(results.cots_by_qid) % args.save_every == 0:
            with open(output_path, "wb") as f:
                pickle.dump(results, f)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)


def main(args: argparse.Namespace):
    assert not is_chat_model(args.model_id)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    model, tokenizer = load_any_model_and_tokenizer(args.model_id)

    questions_dir = DATA_DIR / "questions"
    output_dir = DATA_DIR / "bia-cots"

    with open(questions_dir / f"{args.dataset_id}.pkl", "rb") as f:
        questions_dataset: dict[str, Question] = pickle.load(f)

    yes_fsps, no_fsps = build_bia_fsps(args)
    yes_fsp_toks, no_fsp_toks = [tokenizer.encode(fsp) for fsp in [yes_fsps, no_fsps]]

    model_name = args.model_id.split("/")[-1]
    output_path = output_dir / f"{model_name}_{args.dataset_id}.pkl"

    generate_bia_cots(
        model=model,
        tokenizer=tokenizer,
        questions_dataset=questions_dataset,
        bia_yes_fsp_toks=yes_fsp_toks,
        bia_no_fsp_toks=no_fsp_toks,
        args=args,
        output_path=output_path,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main(parse_args())
