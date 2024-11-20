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

from cot_probing.activations import (
    build_fsp_cache,
    collect_resid_acts_no_pastkv,
    collect_resid_acts_with_pastkv,
)
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
        default="all",
        help="Mode for collecting biased COTs. If one or all is selected, we filter first the biased COTs by the biased_cot_label.",
    )
    parser.add_argument(
        "-c",
        "--context",
        type=str,
        choices=["biased-fsp", "unbiased-fsp", "no-fsp"],
        default="biased-fsp",
        help="FSP context for the activations to collect.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=50,
        help="Save results to disk every N questions.",
    )
    return parser.parse_args()


def get_last_q_toks_to_cache(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    expected_answer: Literal["yes", "no"],
    biased_cots: list[dict],
    biased_cot_label: Literal["faithful", "unfaithful"],
    biased_cots_collection_mode: Literal["none", "one", "all"],
    fsp_context: Literal["biased-fsp", "unbiased-fsp", "no-fsp"],
):
    """May or may not include the CoT tokens"""
    if biased_cots_collection_mode == "none":
        # Don't collect activations for biased COTs
        question_toks = tokenizer.encode(
            question, add_special_tokens=fsp_context == "no-fsp"
        )
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
        tokenizer.encode(
            biased_cots[i]["cot"], add_special_tokens=fsp_context == "no-fsp"
        )
        + answer_tok
        for i in biased_cot_indexes_to_cache
    ]

    return input_ids_to_cache


def collect_activations_for_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    q_data: dict,
    layers: list[int],
    unbiased_fsp_cache: tuple,
    biased_no_fsp_cache: tuple,
    biased_yes_fsp_cache: tuple,
    biased_cots_collection_mode: Literal["none", "one", "all"],
    fsp_context: Literal["biased-fsp", "unbiased-fsp", "no-fsp"],
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
        fsp_context=fsp_context,
    )

    if fsp_context == "biased-fsp":
        # Choose the biased FSP based on the expected answer
        if expected_answer == "yes":
            fsp_cache = biased_no_fsp_cache
        else:
            fsp_cache = biased_yes_fsp_cache
    elif fsp_context == "unbiased-fsp":
        fsp_cache = unbiased_fsp_cache
    elif fsp_context == "no-fsp":
        fsp_cache = None
    else:
        raise ValueError(f"Invalid FSP context: {fsp_context}")

    resid_acts_by_layer_by_cot = []
    for last_q_toks in last_q_toks_to_cache:
        if fsp_cache is None:
            resid_acts_by_layer = collect_resid_acts_no_pastkv(
                model=model,
                all_input_ids=last_q_toks,
                layers=layers,
            )
        else:
            resid_acts_by_layer = collect_resid_acts_with_pastkv(
                model=model,
                last_q_toks=last_q_toks,
                layers=layers,
                past_key_values=fsp_cache,
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


def get_layer_file_path(output_file_stem: str, layer: int) -> str:
    return f"activations/acts_L{layer:02d}_{output_file_stem}.pkl"


def load_existing_results(output_file_stem: str, layers: list[int]) -> tuple[dict, int]:
    """
    Loads existing results for all layers and returns them along with the last consistent index.
    Returns (layer_results, last_processed_index)
    """
    layer_results = {}
    last_indices = []

    for layer in layers:
        layer_file = get_layer_file_path(output_file_stem, layer)
        print(f"Looking to load layer {layer} from {layer_file}")
        if os.path.exists(layer_file):
            try:
                with open(layer_file, "rb") as f:
                    layer_data = pickle.load(f)
                    layer_results[layer] = layer_data["qs"]
                    last_indices.append(len(layer_data["qs"]))
                    print(f"Loaded layer {layer} with {len(layer_data['qs'])} elements")
            except (EOFError, pickle.UnpicklingError):
                print(
                    f"Error loading layer {layer} from {layer_file}. Creating empty list."
                )
                layer_results[layer] = []
                last_indices.append(0)
        else:
            print(f"Layer {layer} file does not exist. Creating empty list.")
            layer_results[layer] = []
            last_indices.append(0)

    # Get minimum index across all layers
    start_idx = min(last_indices) if len(last_indices) > 0 else 0

    # Remove layer results on all layers after the start index
    for layer in layer_results:
        layer_results[layer] = layer_results[layer][:start_idx]

    assert len(layer_results) == len(layers)
    for layer in layers:
        assert (
            len(layer_results[layer]) == start_idx
        ), f"Layer {layer} has {len(layer_results[layer])} elements but should have {start_idx}"

    return layer_results, start_idx


def collect_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: dict,
    layers: list[int],
    output_file_stem: str,
    biased_cots_collection_mode: Literal["none", "one", "all"],
    args: argparse.Namespace,
):
    unbiased_fsp = dataset["unbiased_fsp"] + "\n\n"
    biased_no_fsp = dataset["biased_no_fsp"] + "\n\n"
    biased_yes_fsp = dataset["biased_yes_fsp"] + "\n\n"

    # Pre-cache FSP activations
    biased_no_fsp_cache = build_fsp_cache(model, tokenizer, biased_no_fsp)
    biased_yes_fsp_cache = build_fsp_cache(model, tokenizer, biased_yes_fsp)
    unbiased_fsp_cache = build_fsp_cache(model, tokenizer, unbiased_fsp)

    # Create output directory if it doesn't exist
    os.makedirs("activations", exist_ok=True)

    # Load existing results and get last processed index in one pass
    layer_results, start_idx = load_existing_results(output_file_stem, layers)

    questions = [
        q
        for q in dataset["qs"]
        if "biased_cots" in q and q["biased_cot_label"] in ["faithful", "unfaithful"]
    ]

    # Create progress bar for remaining questions
    remaining_qs = questions[start_idx:]
    if start_idx > 0:
        print(f"Resuming from question {start_idx}")

    last_question_fully_processed = True

    try:
        for idx, q_data in enumerate(tqdm(remaining_qs, "Processing questions")):
            res = collect_activations_for_question(
                model=model,
                tokenizer=tokenizer,
                q_data=q_data,
                layers=layers,
                unbiased_fsp_cache=unbiased_fsp_cache,
                biased_no_fsp_cache=biased_no_fsp_cache,
                biased_yes_fsp_cache=biased_yes_fsp_cache,
                biased_cots_collection_mode=biased_cots_collection_mode,
                fsp_context=args.context,
            )

            # Create temporary dict for new results
            new_layer_results = {}
            for layer in layers:
                layer_res = copy.deepcopy(res)
                layer_res["cached_acts"] = (
                    res["cached_acts"][layer]
                    if isinstance(res["cached_acts"], dict)
                    else [r[layer] for r in res["cached_acts"]]
                )
                new_layer_results[layer] = layer_res

            # Atomic update of all layers at once.
            # Getting interrupted before this block is not a problem: we lose one question but we save everything else.
            # Getting interrupted after (i.e., while saving the layer results) this block is not a problem: we are going to save everything again anyway.
            # Getting interrupted in the middle of this block is not a problem: we take the minimum length across layers and remove the extra element.
            last_question_fully_processed = False
            for layer in layers:
                layer_results[layer].append(new_layer_results[layer])
            last_question_fully_processed = True

            # Save if we hit the frequency or it's the last item
            if (idx + 1) % args.save_frequency == 0 or idx == len(remaining_qs) - 1:
                save_layer_results(
                    layer_results, layers, dataset, args, output_file_stem
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")

        if not last_question_fully_processed:
            # Get minimum number of elements across all layers and remove the extra elements
            min_len = min([len(layer_results[layer]) for layer in layers])
            for layer in layers:
                layer_results[layer] = layer_results[layer][:min_len]

        # Save one final time regardless of save_frequency
        save_layer_results(layer_results, layers, dataset, args, output_file_stem)
        raise  # Re-raise the interrupt to exit the program


def save_layer_results(layer_results, layers, dataset, args, output_file_stem):
    """Helper function to save results for all layers"""
    for layer in layers:
        skip_args = ["verbose", "file", "layers"]
        layer_output = {
            "unbiased_fsp": dataset["unbiased_fsp"],
            "biased_no_fsp": dataset["biased_no_fsp"],
            "biased_yes_fsp": dataset["biased_yes_fsp"],
            "qs": layer_results[layer],
            "arg_model_size": dataset["arg_model_size"],
            **{f"arg_{k}": v for k, v in vars(args).items() if k not in skip_args},
        }

        layer_file = get_layer_file_path(output_file_stem, layer)
        with open(layer_file, "wb") as f:
            pickle.dump(layer_output, f)


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

    output_file_stem = input_file_path.stem.replace("labeled_qs_", "").replace(
        "with-unbiased-cots-", ""
    )
    if args.context == "no-fsp":
        output_file_stem = "no-fsp-" + output_file_stem
    elif args.context == "unbiased-fsp":
        output_file_stem = "unbiased-fsp-" + output_file_stem
    elif args.context == "biased-fsp":
        output_file_stem = "biased-fsp-" + output_file_stem

    collect_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=labeled_questions_dataset,
        layers=layers,
        output_file_stem=output_file_stem,
        biased_cots_collection_mode=args.biased_cots_collection_mode,
        args=args,
    )


if __name__ == "__main__":
    main(parse_args())
