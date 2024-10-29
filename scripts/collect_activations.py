#!/usr/bin/env python3
import argparse
import copy
import json
import logging
import os
import pickle
import random
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing import DATA_DIR
from cot_probing.activations import collect_resid_acts_with_pastkv
from cot_probing.typing import *
from cot_probing.utils import load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Collect activations from a model")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the dataset of labeled questions",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=str,
        default=None,
        help="List of comma separated layers to cache activations for. Defaults to all layers.",
    )
    parser.add_argument(
        "-b",
        "--biased-cots-collection-mode",
        type=str,
        choices=["none", "one", "all"],
        default="none",
        help="Mode for collecting biased COTs. If one or all is selected, we filter first the biased COTs by the biased_cot_label.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def build_fsp_cache(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    fsp: str,
):
    fsp_input_ids = torch.tensor([tokenizer.encode(fsp)]).to("cuda")
    with torch.inference_mode():
        return model(fsp_input_ids).past_key_values


def get_last_q_toks_to_cache(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    expected_answer: Literal["yes", "no"],
    biased_cots: list[str],
    biased_cot_label: Literal["faithful", "unfaithful"],
    biased_cots_collection_mode: Literal["none", "one", "all"],
):
    """May or may not include the CoT tokens"""
    question_toks = tokenizer.encode(question, add_special_tokens=False)
    if biased_cots_collection_mode == "none":
        # Don't collect activations for biased COTs
        return [question_toks]

    biased_cot_answer = None
    if biased_cot_label == "faithful":
        if expected_answer == "yes":
            biased_cot_answer = "yes"
        else:
            biased_cot_answer = "no"
    elif biased_cot_label == "unfaithful":
        if expected_answer == "yes":
            biased_cot_answer = "no"
        else:
            biased_cot_answer = "yes"

    # Keep only the COTs that have the answer we expect based on the biased cot label
    biased_cots = [cot for cot in biased_cots if cot["answer"] == biased_cot_answer]

    assert (
        len(biased_cots) > 0
    ), f"No biased COTs found that match the biased CoT label {biased_cot_label}"

    # Assert all biased COTs have the question. This is due to a bug in the measure_qs script.
    for cot in biased_cots:
        assert cot["cot"].startswith(
            question
        ), f"Biased COT {cot['cot']} does not start with question {question}"

    # Decide the answer token to cache based on the biased cot answer
    yes_tok = tokenizer.encode(" Yes", add_special_tokens=False)
    no_tok = tokenizer.encode(" No", add_special_tokens=False)
    if biased_cot_answer == "yes":
        answer_tok = yes_tok
    else:
        answer_tok = no_tok

    biased_cot_indexes_to_cache = []
    if biased_cots_collection_mode == "one":
        # Pick one random biased COT
        biased_cot_indexes_to_cache = [random.randint(0, len(biased_cots) - 1)]
    elif biased_cots_collection_mode == "all":
        # Collect activations for all biased COTs
        biased_cot_indexes_to_cache = list(range(len(biased_cots)))

    input_ids_to_cache = [
        # Don't include the question tokens since they are already in the biased CoT due to a bug in the measure_qs script
        # question_toks +
        tokenizer.encode(biased_cots[i]["cot"], add_special_tokens=False) + answer_tok
        for i in biased_cot_indexes_to_cache
    ]

    return input_ids_to_cache


def collect_activations_for_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    q_data: dict,
    layers: list[int],
    biased_no_fsp_cache: tuple,
    biased_yes_fsp_cache: tuple,
    biased_cots_collection_mode: Literal["none", "one", "all"],
):
    question = q_data["question"]
    assert question.endswith("by step:\n-"), question
    expected_answer = q_data["expected_answer"]
    biased_cots = q_data["biased_cots"]
    biased_cot_label = q_data["biased_cot_label"]

    last_q_toks_to_cache = get_last_q_toks_to_cache(
        tokenizer=tokenizer,
        question=question,
        expected_answer=expected_answer,
        biased_cots=biased_cots,
        biased_cot_label=biased_cot_label,
        biased_cots_collection_mode=biased_cots_collection_mode,
    )

    # Choose the biased FSP based on the expected answer
    if expected_answer == "yes":
        biased_fsp_cache = biased_no_fsp_cache
    else:
        biased_fsp_cache = biased_yes_fsp_cache

    resid_acts_by_layer_by_cot = []
    for last_q_toks in last_q_toks_to_cache:
        resid_acts_by_layer = collect_resid_acts_with_pastkv(
            model=model,
            last_q_toks=last_q_toks,
            layers=layers,
            past_key_values=biased_fsp_cache,
        )
        resid_acts_by_layer_by_cot.append(resid_acts_by_layer)
    if biased_cots_collection_mode == "none":
        assert len(resid_acts_by_layer_by_cot) == 1
        resid_acts = resid_acts_by_layer_by_cot[0]
    else:
        resid_acts = resid_acts_by_layer_by_cot

    return {
        "question": question,
        "expected_answer": expected_answer,
        "biased_cot_label": biased_cot_label,
        "biased_cots_tokens_to_cache": last_q_toks_to_cache,
        "cached_acts": resid_acts,
    }


def collect_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: dict,
    layers: list[int],
    biased_cots_collection_mode: Literal["none", "one", "all"],
):
    biased_no_fsp = dataset["biased_no_fsp"] + "\n\n"
    biased_yes_fsp = dataset["biased_yes_fsp"] + "\n\n"

    # Pre-cache FSP activations
    biased_no_fsp_cache = build_fsp_cache(model, tokenizer, biased_no_fsp)
    biased_yes_fsp_cache = build_fsp_cache(model, tokenizer, biased_yes_fsp)

    result = []
    for q_data in tqdm(dataset["qs"], "Processing questions"):
        if "biased_cots" not in q_data:
            print("Warning: No biased COTs found for question")
            continue

        biased_cot_label = q_data["biased_cot_label"]
        if biased_cot_label not in ["faithful", "unfaithful"]:
            # Skip questions that are labeled as "mixed"
            continue

        res = collect_activations_for_question(
            model=model,
            tokenizer=tokenizer,
            q_data=q_data,
            layers=layers,
            biased_no_fsp_cache=biased_no_fsp_cache,
            biased_yes_fsp_cache=biased_yes_fsp_cache,
            biased_cots_collection_mode=biased_cots_collection_mode,
        )
        result.append(res)

    return result


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    input_file_path = Path(args.file)
    if not input_file_path.exists():
        raise FileNotFoundError(f"File not found at {input_file_path}")

    if not input_file_path.name.startswith("labeled_qs_"):
        raise ValueError(
            f"Input file must start with 'labeled_qs_', got {input_file_path}"
        )

    with open(input_file_path, "r") as f:
        labeled_questions_dataset = json.load(f)

    model_size = labeled_questions_dataset["arg_model_size"]
    model, tokenizer = load_model_and_tokenizer(model_size)

    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
    else:
        layers = list(range(model.config.num_hidden_layers + 1))

    acts_results = collect_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=labeled_questions_dataset,
        layers=layers,
        biased_cots_collection_mode=args.biased_cots_collection_mode,
    )

    skip_args = ["verbose", "file"]
    ret = dict(
        unbiased_fsp=labeled_questions_dataset["unbiased_fsp"],
        biased_no_fsp=labeled_questions_dataset["biased_no_fsp"],
        biased_yes_fsp=labeled_questions_dataset["biased_yes_fsp"],
        qs=acts_results,
        arg_model_size=model_size,
        **{f"arg_{k}": v for k, v in vars(args).items() if k not in skip_args},
    )

    output_file_stem = input_file_path.stem.replace("labeled_qs_", "acts_")
    output_file_path = f"activations/{output_file_stem}.pkl"

    # Dump the result as a pickle file
    with open(output_file_path, "wb") as f:
        pickle.dump(ret, f)


if __name__ == "__main__":
    main(parse_args())
