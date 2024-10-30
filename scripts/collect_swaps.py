#!/usr/bin/env python3
import argparse
import copy
import json
import logging
import pickle
from functools import partial
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing import DATA_DIR
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
        default=0.1,
        help="Probe diff threshold to truncate COTs.",
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


def process_single_cot(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    q_and_cot_str: str,
    unbiased_fsp_toks: list[int],
    biased_fsp_toks: list[int],
    unbiased_fsp_cache: tuple,
    biased_fsp_cache: tuple,
    prob_diff_threshold: float,
    swap_dir: Literal["fai_to_unfai", "unfai_to_fai"],
) -> SuccessfulSwap | None:
    # cot includes question + ltsbs + "\n-" + CoT + "\nAnswer:"
    assert q_and_cot_str.endswith("Answer:"), q_and_cot_str
    assert q_and_cot_str.startswith("Question"), q_and_cot_str

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
    unb_logits = (
        model(
            torch.tensor([q_and_cot_toks]).cuda(),
            past_key_values=copy.deepcopy(unbiased_fsp_cache),
        )
        .logits[0, cot_start_pos - 1 : -1]
        .cpu()
    )

    bia_logits = (
        model(
            torch.tensor([q_and_cot_toks]).cuda(),
            past_key_values=copy.deepcopy(biased_fsp_cache),
        )
        .logits[0, cot_start_pos - 1 : -1]
        .cpu()
    )

    # Compute the probabilities
    unb_probs = torch.softmax(unb_logits, dim=-1)
    bia_probs = torch.softmax(bia_logits, dim=-1)

    # Compute the probability difference, shape (seq_len, vocab_size)
    prob_diff = unb_probs - bia_probs

    prob_diff_max_values, prob_diff_max_indices = prob_diff.max(dim=-1)
    prob_diff_min_values, prob_diff_min_indices = prob_diff.min(dim=-1)

    thresh_mask = (prob_diff_max_values > prob_diff_threshold) & (
        prob_diff_min_values < -prob_diff_threshold
    )
    if not thresh_mask.any():
        return None

    trunc_pos = int(torch.arange(len(thresh_mask))[thresh_mask][0].item())
    prob_diff = float(prob_diff_max_values[trunc_pos] - prob_diff_min_values[trunc_pos])
    assert prob_diff >= 2 * prob_diff_threshold
    fai_tok = int(prob_diff_max_indices[trunc_pos].item())
    unfai_tok = int(prob_diff_min_indices[trunc_pos].item())
    trunc_cot = q_and_cot_toks[cot_start_pos : cot_start_pos + trunc_pos]

    return SuccessfulSwap(
        unb_prompt=unbiased_fsp_toks + q_toks,
        bia_prompt=biased_fsp_toks + q_toks,
        trunc_cot=trunc_cot,
        fai_tok=fai_tok,
        unfai_tok=unfai_tok,
        prob_diff=prob_diff,
        swap_dir=swap_dir,
    )


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
    biased_cots_dict_list = q_data["biased_cots"]
    unbiased_cots_dict_list = q_data["unbiased_cots"]

    # Filter biased COTs based on the expected answer (unfaithful questions only)
    biased_cots_dict_list = [
        cot
        for cot in biased_cots_dict_list
        if cot["answer"] != expected_answer and cot["answer"] != "other"
    ]

    # Filter unbiased COTs based on the expected answer
    unbiased_cots_dict_list = [
        cot
        for cot in unbiased_cots_dict_list
        if cot["answer"] == expected_answer and cot["answer"] != "other"
    ]

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
    partial_process_single_cot = partial(
        process_single_cot,
        model=model,
        tokenizer=tokenizer,
        unbiased_fsp_toks=unbiased_fsp_toks,
        biased_fsp_toks=biased_fsp_toks,
        unbiased_fsp_cache=unbiased_fsp_cache,
        biased_fsp_cache=biased_fsp_cache,
        prob_diff_threshold=prob_diff_threshold,
    )
    for unbiased_cot_dict in unbiased_cots_dict_list:
        swap = partial_process_single_cot(
            q_and_cot_str=unbiased_cot_dict["cot"],
            swap_dir="fai_to_unfai",
        )
        if swap is not None:
            swaps.append(swap)
    for biased_cot_dict in biased_cots_dict_list:
        swap = partial_process_single_cot(
            q_and_cot_str=biased_cot_dict["cot"],
            swap_dir="unfai_to_fai",
        )
        if swap is not None:
            swaps.append(swap)

    return {"question": question, "expected_answer": expected_answer, "swaps": swaps}


def collect_swaps(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: dict,
    prob_diff_threshold: float,
):
    unbiased_fsp = dataset["unbiased_fsp"]
    biased_no_fsp = dataset["biased_no_fsp"]
    biased_yes_fsp = dataset["biased_yes_fsp"]

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
