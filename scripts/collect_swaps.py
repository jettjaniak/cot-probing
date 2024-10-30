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
from cot_probing.swapping import SuccessfulSwap
from cot_probing.typing import *
from cot_probing.utils import find_sublist, load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Collect activations from a model")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the dataset of labeled questions",
    )
    parser.add_argument(
        "-t",
        "--probe-diff-threshold",
        type=float,
        default=0.2,
        help="Probe diff threshold to truncate COTs. Defaults to 0.2.",
    )

    # argument for probe diff threshold
    # We use this probe diff to truncate cots
    # We discard questions labeled as mixed.

    # TODO: we collect both biased and unbiased, remove it
    # parser.add_argument(
    #     "-b",
    #     "--biased-cots-collection-mode",
    #     type=str,
    #     choices=["none", "one", "all"],
    #     default="none",
    #     help="Mode for collecting biased COTs. If one or all is selected, we filter first the biased COTs by the biased_cot_label.",
    # )
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
    biased_cots: list[dict],
    biased_cot_label: Literal["faithful", "unfaithful"],
):
    """May or may not include the CoT tokens"""
    question_toks = tokenizer.encode(question, add_special_tokens=False)

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


def collect_swaps_for_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    q_data: dict,
    prob_diff_threshold: float,
    unbiased_fsp_str: str,
    biased_no_fsp_str: str,
    biased_yes_fsp_str: str,
    unbiased_fsp_cache: tuple,
    biased_no_fsp_cache: tuple,
    biased_yes_fsp_cache: tuple,
):
    question = q_data["question"]
    assert question.endswith("by step:\n-"), question

    expected_answer = q_data["expected_answer"]
    biased_cots = q_data["biased_cots"]
    unbiased_cots = q_data["unbiased_cots"]

    # Filter biased COTs based on the expected answer (unfaithful questions only)
    biased_cots = [cot for cot in biased_cots if cot["answer"] != expected_answer]

    # Filter unbiased COTs based on the expected answer
    unbiased_cots = [cot for cot in unbiased_cots if cot["answer"] == expected_answer]

    # Choose the biased FSP based on the expected answer
    if expected_answer == "yes":
        biased_fsp_cache = biased_no_fsp_cache
        biased_fsp_str = biased_no_fsp_str
    else:
        biased_fsp_cache = biased_yes_fsp_cache
        biased_fsp_str = biased_yes_fsp_str

    unbiased_fsp_toks = tokenizer.encode(unbiased_fsp_str)
    biased_fsp_toks = tokenizer.encode(biased_fsp_str)

    swaps: list[SuccessfulSwap] = []
    for q_and_cot_str, _ in unbiased_cots:
        # cot includes question + ltsbs + "\n-" + CoT + "\nAnswer:"
        assert q_and_cot_str.endswith("Answer:")
        assert q_and_cot_str.startswith("Question")

        q_and_cot_toks = tokenizer.encode(q_and_cot_str, add_special_tokens=False)

        # Figure out the position of the start of the CoT
        ltsbs_tok = tokenizer.encode(
            "Let's think step by step:\n-", add_special_tokens=False
        )
        ltsbs_idx = find_sublist(q_and_cot_toks, ltsbs_tok)
        assert ltsbs_idx is not None
        cot_start_pos = ltsbs_idx + len(ltsbs_tok)
        q_toks = q_and_cot_toks[:cot_start_pos]

        # Get the logits for the CoT
        unb_logits = model(
            torch.tensor([q_and_cot_toks]).cuda(),
            past_key_values=copy.deepcopy(unbiased_fsp_cache),
        ).logits[
            0, cot_start_pos - 1 : -1
        ]  # -1 because we need logits for prediction of the first CoT token. -1 for the end of the slice so we drop the predictions after the answer

        bia_logits = model(
            torch.tensor([q_and_cot_toks]).cuda(),
            past_key_values=copy.deepcopy(biased_fsp_cache),
        ).logits[
            0, cot_start_pos - 1 : -1
        ]  # -1 because we need logits for prediction of the first CoT token. -1 for the end of the slice so we drop the predictions after the answer

        # Compute the probabilities
        unb_probs = torch.softmax(unb_logits, dim=-1)
        bia_probs = torch.softmax(bia_logits, dim=-1)

        # Compute the probability difference, shape (seq_len, vocab_size)
        prob_diff = unb_probs - bia_probs

        # values: (seq_len,), indices: (seq_len,) - they contain token ids
        # we subtract biased from unbiased
        # so max indices are for faithful tokens
        prob_diff_max_values, prob_diff_max_indices = prob_diff.max(dim=-1)
        # and min indices are for unfaithful tokens
        prob_diff_min_values, prob_diff_min_indices = prob_diff.min(dim=-1)

        thresh_mask = (prob_diff_max_values > prob_diff_threshold) & (
            prob_diff_min_values < -prob_diff_threshold
        )
        if not thresh_mask.any():
            continue

        trunc_pos = torch.arange(len(thresh_mask))[thresh_mask][0].item()
        prob_diff = prob_diff_max_values[trunc_pos] - prob_diff_min_values[trunc_pos]
        prob_diff = prob_diff.item()
        assert prob_diff >= 2 * prob_diff_threshold
        fai_tok = prob_diff_max_indices[trunc_pos].item()
        unf_tok = prob_diff_min_indices[trunc_pos].item()
        trunc_cot = q_and_cot_toks[cot_start_pos : cot_start_pos + trunc_pos]

        swap = SuccessfulSwap(
            unb_prompt=unbiased_fsp_toks + q_toks,
            bia_prompt=biased_fsp_toks + q_toks,
            trunc_cot=trunc_cot,
            fai_tok=fai_tok,
            unfai_tok=unf_tok,
            prob_diff=prob_diff,
            swap_dir="fai_to_unfai",
        )

        swaps.append(swap)

    return {"question": question, "expected_answer": expected_answer, "swaps": swaps}


def collect_swaps(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: dict,
    prob_diff_threshold: float,
):
    unbiased_fsp = dataset["unbiased_fsp"] + "\n\n"
    biased_no_fsp = dataset["biased_no_fsp"] + "\n\n"
    biased_yes_fsp = dataset["biased_yes_fsp"] + "\n\n"

    # Pre-cache FSP activations
    unbiased_fsp_cache = build_fsp_cache(model, tokenizer, unbiased_fsp)
    biased_no_fsp_cache = build_fsp_cache(model, tokenizer, biased_no_fsp)
    biased_yes_fsp_cache = build_fsp_cache(model, tokenizer, biased_yes_fsp)

    result = []
    for q_data in tqdm(dataset["qs"], "Processing questions"):
        if "biased_cots" not in q_data:
            print("Warning: No biased COTs found for question")
            continue

        if "unbiased_cots" not in q_data:
            print("Warning: No unbiased COTs found for question")
            continue

        biased_cot_label = q_data["biased_cot_label"]
        if biased_cot_label != "unfaithful":
            # Skip questions that are not labeled as "unfaithful"
            continue

        res = collect_swaps_for_question(
            model=model,
            tokenizer=tokenizer,
            q_data=q_data,
            prob_diff_threshold=prob_diff_threshold,
            unbiased_fsp_str=unbiased_fsp,
            biased_no_fsp_str=biased_no_fsp,
            biased_yes_fsp_str=biased_yes_fsp,
            unbiased_fsp_cache=unbiased_fsp_cache,
            biased_no_fsp_cache=biased_no_fsp_cache,
            biased_yes_fsp_cache=biased_yes_fsp_cache,
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

    swaps_results = collect_swaps(
        model=model,
        tokenizer=tokenizer,
        dataset=labeled_questions_dataset,
        prob_diff_threshold=args.probe_diff_threshold,
    )

    skip_args = ["verbose", "file"]
    ret = dict(
        unbiased_fsp=labeled_questions_dataset["unbiased_fsp"],
        biased_no_fsp=labeled_questions_dataset["biased_no_fsp"],
        biased_yes_fsp=labeled_questions_dataset["biased_yes_fsp"],
        qs=swaps_results,
        arg_model_size=model_size,
        **{f"arg_{k}": v for k, v in vars(args).items() if k not in skip_args},
    )

    output_file_stem = input_file_path.stem.replace("labeled_qs_", "swaps_")
    output_file_name = Path(output_file_stem).with_suffix(".pkl")
    output_file_path = DATA_DIR / output_file_name

    # Dump the result as a pickle file
    with open(output_file_path, "wb") as f:
        pickle.dump(ret, f)


if __name__ == "__main__":
    main(parse_args())
