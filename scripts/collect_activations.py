#!/usr/bin/env python3

import argparse
import os
import pickle
from typing import Literal

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing.activations import ActivationsInLayer, clean_run_with_cache
from cot_probing.eval import EvalQuestion
from cot_probing.task import INSTRUCTION_STR
from cot_probing.typing import *

LocTypeToCache = Literal["resp", "instr", "quest", "fsp"]


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--eval-questions",
        "-f",
        type=str,
        default=None,
        help="Absolute path to eval_questions.pkl file",
    )
    return parser.parse_args()


def process_questions(
    output_folder: str,
    biased: bool,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_questions: list[EvalQuestion],
    loc_types_to_cache: set[LocTypeToCache] = {"resp"},
):
    if biased:
        output_folder = os.path.join(output_folder, "biased_context")
    else:
        output_folder = os.path.join(output_folder, "unbiased_context")

    # Load few-shot prompts
    fsps_path = os.path.join(output_folder, "tokenized_fsps.pkl")
    with open(fsps_path, "rb") as f:
        fsps = pickle.load(f)

    # Load tokenized responses
    tokenized_responses_path = os.path.join(output_folder, "tokenized_responses.pkl")
    with open(tokenized_responses_path, "rb") as f:
        tokenized_responses = pickle.load(f)

    layers_to_cache = list(range(model.config.num_hidden_layers))
    activations_by_layer: dict[int, list[ActivationsInLayer]] = {
        layer: [] for layer in layers_to_cache
    }

    tokenized_instruction = tokenizer.encode(
        f"\n\n{INSTRUCTION_STR}\n", add_special_tokens=False, return_tensors="pt"
    )[0]

    for q_idx, question in tqdm(enumerate(eval_questions)):
        tokenized_fsp = torch.tensor(fsps[q_idx])
        tokenized_question = torch.tensor(question.tokenized_question)
        tokenized_response = torch.tensor(tokenized_responses[q_idx])

        # Build input tokens
        input_ids = torch.cat(
            (
                tokenized_fsp,
                tokenized_question,
                tokenized_instruction,
                tokenized_response,
            ),
            dim=0,
        )

        locs_to_cache = []
        for loc_type in loc_types_to_cache:
            if loc_type == "resp":
                locs_to_cache.extend(
                    range(
                        len(tokenized_fsp)
                        + len(tokenized_question)
                        + len(tokenized_instruction),
                        len(input_ids),
                    )
                )
            elif loc_type == "instr":
                locs_to_cache.extend(
                    range(
                        len(tokenized_fsp) + len(tokenized_question),
                        len(tokenized_fsp)
                        + len(tokenized_question)
                        + len(tokenized_instruction),
                    )
                )
            elif loc_type == "quest":
                locs_to_cache.extend(
                    range(
                        len(tokenized_fsp), len(tokenized_fsp) + len(tokenized_question)
                    )
                )
            elif loc_type == "fsp":
                locs_to_cache.extend(range(len(tokenized_fsp)))

        resid_acts = clean_run_with_cache(
            model, input_ids, layers_to_cache, locs_to_cache
        )

        for layer_idx, layer in enumerate(layers_to_cache):
            activations_by_layer[layer].append(resid_acts[layer_idx].cpu())

    # Create dir for activations
    acts_type = "+".join(list(loc_types_to_cache))
    acts_name = f"acts_{acts_type}"
    acts_folder_path = os.path.join(output_folder, acts_name)
    os.makedirs(acts_folder_path, exist_ok=True)

    # Dump to disk
    for layer, activations in activations_by_layer.items():
        acts_path = os.path.join(acts_folder_path, f"L{layer}.pkl")
        with open(acts_path, "wb") as f:
            pickle.dump(activations, f)


def main():
    args = parse_arguments()

    # Check that the eval_questions file was provided and exists
    if args.eval_questions is None or not os.path.exists(args.eval_questions):
        raise ValueError(
            "Please provide a valid eval_questions file with the flag --eval-questions or -f"
        )

    with open(args.eval_questions, "rb") as f:
        eval_questions = pickle.load(f)

    print(f"Processing {len(eval_questions)} questions from {args.eval_questions}...")

    # Split the path to get details
    parts = args.eval_questions.split("/")
    misc_details = parts[-2]
    print(f"Misc details: {misc_details}")
    task = parts[-3]
    print(f"Task: {task}")
    model_name = parts[-4].replace("--", "/")
    print(f"Model: {model_name}")

    # Output folder is parent of eval_questions file
    output_folder = "/".join(parts[:-1])
    print(f"Output folder: {output_folder}")

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for biased in [True, False]:
        process_questions(output_folder, biased, model, tokenizer, eval_questions)


if __name__ == "__main__":
    main()
