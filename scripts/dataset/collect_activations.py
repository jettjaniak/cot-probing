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
from cot_probing.activations import (
    build_fsp_cache,
    build_fsp_toks_cache,
    collect_resid_acts_no_pastkv,
    collect_resid_acts_with_pastkv,
)
from cot_probing.data.qs_evaluation import LabeledQuestions
from cot_probing.generation import (
    BiasedCotGeneration,
    LabeledCot,
    UnbiasedCotGeneration,
)
from cot_probing.qs_generation import Question
from cot_probing.typing import *
from cot_probing.utils import load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Collect activations from a model")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the dataset of questions",
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
        "--save-every",
        type=int,
        help="Save every N questions",
        default=50,
    )
    return parser.parse_args()


def get_last_q_toks_to_cache(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    expected_answer: Literal["yes", "no"],
    biased_labeled_cots: list[LabeledCot],
    biased_cot_label: Literal["faithful", "unfaithful"],
    biased_cots_collection_mode: Literal["none", "one", "all"],
    fsp_context: Literal["biased-fsp", "unbiased-fsp", "no-fsp"],
):
    """May or may not include the CoT tokens"""
    question_toks = tokenizer.encode(
        question, add_special_tokens=fsp_context == "no-fsp"
    )

    if biased_cots_collection_mode == "none":
        # Don't collect activations for biased COTs
        return [question_toks]

    biased_cot_expected_state = None
    if biased_cot_label == "faithful":
        biased_cot_expected_state = "correct"
    elif biased_cot_label == "unfaithful":
        biased_cot_expected_state = "incorrect"

    # Keep only the COTs that have the answer we expect based on the biased cot label
    biased_cots = [
        item.cot
        for item in biased_labeled_cots
        if item.label == biased_cot_expected_state
    ]

    assert (
        len(biased_cots) > 0
    ), f"No biased COTs found that match the biased CoT label {biased_cot_label}"

    for cot in biased_cots:
        cot_str = tokenizer.decode(cot)
        assert "Question: " not in cot_str

    # Decide the answer token to cache based on the biased cot answer
    answer_yes_toks = tokenizer.encode("Answer: Yes", add_special_tokens=False)
    answer_no_toks = tokenizer.encode("Answer: No", add_special_tokens=False)
    if biased_cot_expected_state == "correct":
        if expected_answer == "yes":
            answer_toks = answer_yes_toks
        else:
            answer_toks = answer_no_toks
    else:
        if expected_answer == "yes":
            answer_toks = answer_no_toks
        else:
            answer_toks = answer_yes_toks

    biased_cot_indexes_to_cache = []
    if biased_cots_collection_mode == "one":
        # Pick one random biased COT
        biased_cot_indexes_to_cache = [random.randint(0, len(biased_cots) - 1)]
    elif biased_cots_collection_mode == "all":
        # Collect activations for all biased COTs
        biased_cot_indexes_to_cache = list(range(len(biased_cots)))

    input_ids_to_cache = [
        question_toks
        + tokenizer.encode(biased_cots[i], add_special_tokens=False)
        + answer_toks
        for i in biased_cot_indexes_to_cache
    ]

    return input_ids_to_cache


def collect_activations_for_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    q: Question,
    biased_cots: list[LabeledCot],
    biased_cot_label: Literal["faithful", "unfaithful"],
    layers: list[int],
    unbiased_fsp_cache: tuple,
    biased_no_fsp_cache: tuple,
    biased_yes_fsp_cache: tuple,
    biased_cots_collection_mode: Literal["none", "one", "all"],
    fsp_context: Literal["biased-fsp", "unbiased-fsp", "no-fsp"],
):
    question = q.with_step_by_step_suffix()
    assert question.endswith("by step:\n-"), question

    expected_answer = q.expected_answer

    last_q_toks_to_cache = get_last_q_toks_to_cache(
        tokenizer=tokenizer,
        question=question,
        expected_answer=expected_answer,
        biased_labeled_cots=biased_cots,
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
    qs_dataset: dict[str, Question],
    labeled_qs_dataset: LabeledQuestions,
    unb_cots_results: UnbiasedCotGeneration,
    bia_cots_results: BiasedCotGeneration,
    layers: list[int],
    output_file_stem: str,
    biased_cots_collection_mode: Literal["none", "one", "all"],
    args: argparse.Namespace,
):
    unbiased_fsp = unb_cots_results.unb_fsp_toks
    biased_no_fsp = bia_cots_results.bia_no_fsp_toks
    biased_yes_fsp = bia_cots_results.bia_yes_fsp_toks

    # Pre-cache FSP activations
    biased_no_fsp_cache = build_fsp_toks_cache(model, tokenizer, biased_no_fsp)
    biased_yes_fsp_cache = build_fsp_toks_cache(model, tokenizer, biased_yes_fsp)
    unbiased_fsp_cache = build_fsp_toks_cache(model, tokenizer, unbiased_fsp)

    # Create output directory if it doesn't exist
    os.makedirs("activations", exist_ok=True)

    # Load existing results and get last processed index in one pass
    layer_results, start_idx = load_existing_results(output_file_stem, layers)

    q_ids_to_process = [
        q_id
        for q_id, label in labeled_qs_dataset.label_by_qid.items()
        if label in ["faithful", "unfaithful"]
    ]

    # Create progress bar for remaining questions
    remaining_q_ids = q_ids_to_process[start_idx:]
    if start_idx > 0:
        print(f"Resuming from question {start_idx}")

    last_question_fully_processed = True

    try:
        for idx, q_id in tqdm(enumerate(remaining_q_ids), "Processing questions"):
            assert q_id in bia_cots_results.cots_by_qid
            assert labeled_qs_dataset.label_by_qid[q_id] in ["faithful", "unfaithful"]

            res = collect_activations_for_question(
                model=model,
                tokenizer=tokenizer,
                q=qs_dataset[q_id],
                biased_cots=bia_cots_results.cots_by_qid[q_id],
                biased_cot_label=labeled_qs_dataset.label_by_qid[q_id],
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
            if (idx + 1) % args.save_every == 0 or idx == len(remaining_q_ids) - 1:
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

    dataset_path = Path(args.file)
    assert dataset_path.exists()
    with open(dataset_path, "rb") as f:
        question_dataset = pickle.load(f)

    dataset_identifier = dataset_path.stem.split("_")[-1]

    # Load labeled questions dataset
    with open(DATA_DIR / f"labeled_qs_{dataset_identifier}.pkl", "rb") as f:
        labeled_qs_dataset: LabeledQuestions = pickle.load(f)

    # Load the biased COTs results
    with open(DATA_DIR / f"unb-cots_{dataset_identifier}.pkl", "rb") as f:
        unb_cots_results: UnbiasedCotGeneration = pickle.load(f)
    with open(DATA_DIR / f"bia-cots_{dataset_identifier}.pkl", "rb") as f:
        bia_cots_results: BiasedCotGeneration = pickle.load(f)

    model_size = 8 if "8B" in bia_cots_results.model else 70
    model, tokenizer = load_model_and_tokenizer(model_size)

    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
    else:
        layers = list(range(model.config.num_hidden_layers + 1))

    output_file_stem = dataset_path.stem.replace("labeled_qs_", "").replace(
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
        qs_dataset=question_dataset,
        labeled_qs_dataset=labeled_qs_dataset,
        unb_cots_results=unb_cots_results,
        bia_cots_results=bia_cots_results,
        layers=layers,
        output_file_stem=output_file_stem,
        biased_cots_collection_mode=args.biased_cots_collection_mode,
        args=args,
    )


if __name__ == "__main__":
    main(parse_args())
