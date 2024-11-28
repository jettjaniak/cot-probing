#!/usr/bin/env python3
import argparse
import logging
import pickle
import random
from pathlib import Path
from typing import Any

import torch
from beartype import beartype
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing import DATA_DIR
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.qs_generation import Question
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
        "--max-no-cot-acc",
        type=float,
        help="Maximum no-cot accuracy to generate CoTs for",
        default=0.6,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


@beartype
def build_fsps(args: argparse.Namespace) -> tuple[str, str, str]:
    setup_determinism(args.seed)
    n = args.fsp_size

    yes_fsps = load_and_process_file(DATA_DIR / "diverse_qs_expected_yes_with_cot.txt")
    no_fsps = load_and_process_file(DATA_DIR / "diverse_qs_expected_no_with_cot.txt")

    shuffled_indices = random.sample(range(len(yes_fsps)), n)
    yes_fsps = [yes_fsps[i] for i in shuffled_indices]
    no_fsps = [no_fsps[i] for i in shuffled_indices]

    unb_yes_idxs = random.sample(range(n), n // 2)
    unb_fsps = [yes_fsps[i] if i in unb_yes_idxs else no_fsps[i] for i in range(n)]

    unb_fsps = "\n\n".join(unb_fsps) + "\n\n"
    yes_fsps = "\n\n".join(yes_fsps) + "\n\n"
    no_fsps = "\n\n".join(no_fsps) + "\n\n"

    return unb_fsps, yes_fsps, no_fsps


@beartype
def generate_cots(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_toks: list[int],
    fsp_toks: list[int],
    max_new_tokens: int,
    n_gen: int,
    temp: float,
    seed: int,
) -> list[list[int]]:
    ret = []
    with torch.inference_mode():
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        responses = model.generate(
            input_ids=torch.tensor([fsp_toks + question_toks]).to("cuda"),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            tokenizer=tokenizer,
            stop_strings=["Answer: Yes", "Answer: No"],
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_return_sequences=n_gen,
        )[:, len(fsp_toks) :].tolist()
        for response in responses:
            if tokenizer.eos_token_id in response:
                response = response[: response.index(tokenizer.eos_token_id)]
            ret.append(response)
    return ret


@beartype
def process_question(
    q: Question,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    unb_fsp_toks: list[int],
    yes_fsp_toks: list[int],
    no_fsp_toks: list[int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    question_toks = tokenizer.encode(
        q.with_step_by_step_suffix(), add_special_tokens=False
    )

    unb_cots = generate_cots(
        model=model,
        tokenizer=tokenizer,
        question_toks=question_toks,
        fsp_toks=unb_fsp_toks,
        max_new_tokens=args.max_new_tokens,
        n_gen=args.n_gen,
        temp=args.temp,
        seed=args.seed,
    )
    unb_cots_ret = [
        {
            "cot": cot,
            "model": model.config._name_or_path,
            "fsp_size": args.fsp_size,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temp": args.temp,
            "do_sample": True,
        }
        for cot in unb_cots
    ]

    bia_fsp_toks = no_fsp_toks if q.expected_answer == "yes" else yes_fsp_toks
    biased_cots = generate_cots(
        model=model,
        tokenizer=tokenizer,
        question_toks=question_toks,
        fsp_toks=bia_fsp_toks,
        max_new_tokens=args.max_new_tokens,
        n_gen=args.n_gen,
        temp=args.temp,
        seed=args.seed,
    )
    bia_cots_ret = [
        {
            "cot": cot,
            "model": model.config._name_or_path,
            "fsp_size": args.fsp_size,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "temp": args.temp,
            "do_sample": True,
        }
        for cot in biased_cots
    ]

    return unb_cots_ret, bia_cots_ret


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
        no_cot_accuracy_results = pickle.load(f)

    # Build the few-shot prompts
    unb_fsps, yes_fsps, no_fsps = build_fsps(args)
    unb_fsp_toks, yes_fsp_toks, no_fsp_toks = [
        tokenizer.encode(fsp) for fsp in [unb_fsps, yes_fsps, no_fsps]
    ]

    results = {}
    for q_id, q in tqdm(question_dataset.items(), desc="Processing questions"):
        # By default, we don't generate CoTs
        results[q_id] = {
            "unbiased_cots": [],
            "biased_cots": [],
        }

        if q_id not in no_cot_accuracy_results:
            continue

        no_cot_acc_for_model = [
            res["no_cot_acc"]
            for res in no_cot_accuracy_results[q_id]
            if res["model"] == model.config._name_or_path
        ]
        if len(no_cot_acc_for_model) == 0:
            continue
        if max(no_cot_acc_for_model) > args.max_no_cot_acc:
            continue

        unb_cots, bia_cots = process_question(
            q=q,
            model=model,
            tokenizer=tokenizer,
            unb_fsp_toks=unb_fsp_toks,
            yes_fsp_toks=yes_fsp_toks,
            no_fsp_toks=no_fsp_toks,
            args=args,
        )
        results[q_id] = {
            "unbiased_cots": unb_cots,
            "biased_cots": bia_cots,
        }

    with open(DATA_DIR / f"generated-cots_{dataset_identifier}.pkl", "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main(parse_args())
