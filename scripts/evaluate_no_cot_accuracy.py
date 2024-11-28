#!/usr/bin/env python3
import argparse
import logging
import math
import pickle
from pathlib import Path
from typing import Literal

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing import DATA_DIR
from cot_probing.activations import build_fsp_cache
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.questions_generation import generate_unbiased_few_shot_prompt
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


def get_no_cot_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_without_cot: str,
    expected_answer: Literal["yes", "no"],
    unbiased_no_cot_cache: tuple,
):
    assert question_without_cot.endswith("Answer:")
    assert "Let's think step by step:" not in question_without_cot
    assert question_without_cot.startswith("Question: ")

    yes_tok_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_tok_id = tokenizer.encode(" No", add_special_tokens=False)[0]

    prompt = tokenizer.encode(question_without_cot, add_special_tokens=False)
    logits = model(
        torch.tensor([prompt]).cuda(),
        past_key_values=unbiased_no_cot_cache,
    ).logits[0, -1]
    yes_logit = logits[yes_tok_id].item()
    no_logit = logits[no_tok_id].item()

    exp_yes = math.exp(yes_logit)
    exp_no = math.exp(no_logit)
    denom = exp_yes + exp_no
    prob_yes = exp_yes / denom
    prob_no = exp_no / denom
    if expected_answer == "yes":
        accuracy = prob_yes
    else:
        accuracy = prob_no

    return accuracy


def evaluate_no_cot_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_dataset: list[dict],
    unbiased_fsp_without_cot: str,
    fsp_size: int,
    seed: int,
):
    unbiased_no_cot_cache = build_fsp_cache(model, tokenizer, unbiased_fsp_without_cot)

    results = []
    for q in tqdm(question_dataset, desc="Evaluating no-CoT accuracy"):
        question = q["question"]
        expected_answer = q["expected_answer"]
        assert question.endswith("?")

        no_cot_acc = get_no_cot_accuracy(
            model=model,
            tokenizer=tokenizer,
            question_without_cot=f"{question}\nAnswer:",
            expected_answer=expected_answer,
            unbiased_no_cot_cache=unbiased_no_cot_cache,
        )

        if "no_cot_accuracy" not in q:
            q["no_cot_accuracy"] = []

        q["no_cot_accuracy"].append(
            {
                "acc": no_cot_acc,
                "model": model.config._name_or_path,
                "fsp_size": fsp_size,
                "seed": seed,
            }
        )
        results.append(q)

    return results


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
