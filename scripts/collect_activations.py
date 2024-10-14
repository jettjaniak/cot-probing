#!/usr/bin/env python3
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import torch
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing.activations import clean_run_with_cache
from cot_probing.eval import TokenizedQuestion
from cot_probing.typing import *


def parse_arguments():
    # Example usage:
    # python scripts/collect_activations.py /workspace/cot-probing-hf/google--gemma-2-2b/movie_recommendation/bias-A_seed-0_total-10/tokenized_questions.pkl

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tokenized_questions",
        type=str,
        help="path to tokenized_questions.pkl file",
    )
    return parser.parse_args()


def process_questions(
    output_folder: Path,
    biased: bool,
    model: PreTrainedModel,
    tokenized_questions: list[TokenizedQuestion],
):
    if biased:
        output_folder = output_folder / "biased_context"
    else:
        output_folder = output_folder / "unbiased_context"

    # Load few-shot prompts
    fsp_path = output_folder / "tokenized_fsp.pkl"
    with open(fsp_path, "rb") as f:
        tokenized_fsp = pickle.load(f)

    # Load tokenized responses
    tokenized_responses_path = output_folder / "tokenized_responses.pkl"
    with open(tokenized_responses_path, "rb") as f:
        tokenized_responses = pickle.load(f)

    n_layers = model.config.num_hidden_layers

    act_by_q_by_layer_by_type: dict[
        str, list[list[Float[torch.Tensor, " pos d_model"]]]
    ] = defaultdict(lambda: [[] for _ in range(n_layers)])

    for q_idx in trange(len(tokenized_questions)):
        tokenized_question = tokenized_questions[q_idx].tokenized_question
        tokenized_response = tokenized_responses[q_idx]

        # Build input tokens
        input_ids = tokenized_fsp + tokenized_question + tokenized_response
        q_start = len(tokenized_fsp)
        resp_start = q_start + len(tokenized_question)
        locs_to_cache = {
            "q+instr": (q_start, resp_start),
            "resp": (resp_start, len(input_ids)),
        }

        resid_acts_by_layer_by_locs = clean_run_with_cache(
            model, input_ids, locs_to_cache
        )
        for loc_type, resid_acts_by_layer in resid_acts_by_layer_by_locs.items():
            for layer, resid_acts in enumerate(resid_acts_by_layer):
                act_by_q_by_layer_by_type[loc_type][layer].append(resid_acts)

    # Dump activations to disk
    for loc_type, activations_by_layer in act_by_q_by_layer_by_type.items():
        acts_name = f"acts_{loc_type}"
        acts_folder_path = output_folder / acts_name
        acts_folder_path.mkdir(parents=True, exist_ok=True)

        for layer, activations in enumerate(activations_by_layer):
            acts_path = acts_folder_path / f"L{layer:02}.pkl"
            with open(acts_path, "wb") as f:
                pickle.dump(activations, f)


def main():
    args = parse_arguments()
    tokenized_questions_path = Path(args.tokenized_questions)
    with open(tokenized_questions_path, "rb") as f:
        tokenized_questions = pickle.load(f)

    print(
        f"Processing {len(tokenized_questions)} questions from {tokenized_questions_path}..."
    )

    *_, model_name_dir, task, misc_details, __ = tokenized_questions_path.parts
    model_name = model_name_dir.replace("--", "/")
    print(f"{model_name=}, {task=}, {misc_details=}")
    output_folder = tokenized_questions_path.parent
    print(f"Output folder: {output_folder}")

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # get rid of the warning early
    model(torch.tensor([[tokenizer.bos_token_id]]).cuda())

    print("Collecting activations for unbiased context...")
    process_questions(output_folder, False, model, tokenized_questions)
    print("Collecting activations for biased context...")
    process_questions(output_folder, True, model, tokenized_questions)


if __name__ == "__main__":
    main()
