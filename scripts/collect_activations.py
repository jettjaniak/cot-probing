#!/usr/bin/env python3
import argparse
import copy
import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import torch
import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from cot_probing import DATA_DIR
from cot_probing.activations import clean_run_with_cache


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
        "-e",
        "--collect-embeddings",
        action="store_true",
        help="Collect also embeddings",
    )
    parser.add_argument(
        "-b",
        "--biased-cots-collection-mode",
        type=str,
        choices=["none", "one", "all"],
        default="one",
        help="Mode for collecting biased COTs. If one or all is selected, we filter first the biased COTs by the biased_cot_label.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def load_model_and_tokenizer(
    model_size: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    assert model_size in [8, 70]
    model_id = f"hugging-quants/Meta-Llama-3.1-{model_size}B-BNB-NF4-BF16"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    return model, tokenizer


def build_fsp_cache(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    fsp: str,
):
    fsp_input_ids = torch.tensor([tokenizer.encode(fsp)]).to("cuda")
    with torch.inference_mode():
        return model(fsp_input_ids).past_key_values


def get_input_ids_to_cache(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    expected_answer: Literal["yes", "no"],
    biased_cots: List[str],
    biased_cot_label: Literal["faithful", "unfaithful"],
    biased_cots_collection_mode: Literal["none", "one", "all"],
):
    question_toks = tokenizer.encode(question)
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
        question_toks
        + tokenizer.encode(biased_cots[i]["cot"], add_special_tokens=False)
        + answer_tok
        for i in biased_cot_indexes_to_cache
    ]

    return input_ids_to_cache


def collect_activations_for_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    q_data: Dict,
    locs_to_cache: Dict,
    layers_to_cache: List[int],
    biased_no_fsp_cache: Dict,
    biased_yes_fsp_cache: Dict,
    biased_cots_collection_mode: Literal["none", "one", "all"],
    collect_embeddings: bool,
):
    question = q_data["question"]
    expected_answer = q_data["expected_answer"]
    biased_cots = q_data["biased_cots"]
    biased_cot_label = q_data["biased_cot_label"]

    input_ids_to_cache = get_input_ids_to_cache(
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

    biased_cot_acts = []
    question_token = tokenizer.encode("Question", add_special_tokens=False)[0]
    for input_ids in input_ids_to_cache:
        # Figure out where the last question starts
        if "last_question_tokens" in locs_to_cache:
            last_question_token_position = [
                pos for pos, t in enumerate(input_ids) if t == question_token
            ][-1]
            locs_to_cache["last_question_tokens"] = (
                last_question_token_position,
                None,
            )

        fsp_past_key_values = copy.deepcopy(biased_fsp_cache)

        resid_acts_by_layer_by_locs = clean_run_with_cache(
            model=model,
            input_ids=input_ids,
            locs_to_cache=locs_to_cache,
            collect_embeddings=collect_embeddings,
            past_key_values=fsp_past_key_values,
        )
        biased_cot_acts.append(resid_acts_by_layer_by_locs)

    return {
        "question": question,
        "expected_answer": expected_answer,
        "biased_cot_label": biased_cot_label,
        "biased_cots_tokens_to_cache": input_ids_to_cache,
        "cached_acts": biased_cot_acts,
    }


def collect_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dict,
    locs_to_cache: Dict,
    layers_to_cache: List[int],
    biased_cots_collection_mode: Literal["none", "one", "all"],
    collect_embeddings: bool,
):
    biased_no_fsp = dataset["biased_no_fsp"] + "\n\n"
    biased_yes_fsp = dataset["biased_yes_fsp"] + "\n\n"

    # Pre-cache FSP activations
    biased_no_fsp_cache = build_fsp_cache(model, tokenizer, biased_no_fsp)
    biased_yes_fsp_cache = build_fsp_cache(model, tokenizer, biased_yes_fsp)

    result = []
    for q_data in tqdm.tqdm(dataset["qs"]):
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
            locs_to_cache=locs_to_cache,
            layers_to_cache=layers_to_cache,
            biased_no_fsp_cache=biased_no_fsp_cache,
            biased_yes_fsp_cache=biased_yes_fsp_cache,
            biased_cots_collection_mode=biased_cots_collection_mode,
            collect_embeddings=collect_embeddings,
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
        layers_to_cache = args.layers.split(",")
    else:
        layers_to_cache = list(range(model.config.num_hidden_layers))

    collect_embeddings = args.collect_embeddings

    locs_to_cache = {
        "last_question_tokens": (None, None),
        # "first_cot_dash": (-1, None),  # last token before CoT
        # "last_new_line": (-2, -1),  # newline before first dash in CoT
        # "step_by_step_colon": (-3, -2),  # colon before last new line.
    }

    acts_results = collect_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=labeled_questions_dataset,
        locs_to_cache=locs_to_cache,
        layers_to_cache=layers_to_cache,
        biased_cots_collection_mode=args.biased_cots_collection_mode,
        collect_embeddings=collect_embeddings,
    )

    skip_args = ["verbose", "file"]
    ret = dict(
        biased_no_fsp=labeled_questions_dataset["biased_no_fsp"],
        biased_yes_fsp=labeled_questions_dataset["biased_yes_fsp"],
        qs=acts_results,
        arg_model_size=model_size,
        **{f"arg_{k}": v for k, v in vars(args).items() if k not in skip_args},
    )

    output_file_name = input_file_path.name.replace("labeled_qs_", "acts_").replace(
        ".json", ".pkl"
    )
    output_file_path = DATA_DIR / output_file_name

    # Dump the result as a pickle file
    with open(output_file_path, "wb") as f:
        pickle.dump(ret, f)


if __name__ == "__main__":
    main(parse_args())
