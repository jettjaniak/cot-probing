#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing.activations import clean_run_with_cache
from cot_probing.eval import TokenizedQuestion
from cot_probing.task import INSTRUCTION_STR
from cot_probing.typing import *

ActivationsInLayer = Float[torch.Tensor, " pos d_model"]


def parse_arguments():
    # Example usage:
    # python scripts/collect_activations.py /workspace/cot-probing-hf/google--gemma-2-2b/movie_recommendation/bias-A_seed-0_total-10/eval_questions.pkl

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "eval_questions",
        type=str,
        help="path to eval_questions.pkl file",
    )
    return parser.parse_args()


def process_questions(
    output_folder: Path,
    biased: bool,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    eval_questions: list[TokenizedQuestion],
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

    layers_to_cache = list(range(model.config.num_hidden_layers))
    q_and_instr_activations_by_layer: dict[int, list[ActivationsInLayer]] = {
        layer: [] for layer in layers_to_cache
    }
    resp_activations_by_layer: dict[int, list[ActivationsInLayer]] = {
        layer: [] for layer in layers_to_cache
    }

    tokenized_instruction = tokenizer.encode(
        f"\n\n{INSTRUCTION_STR}\n", add_special_tokens=False, return_tensors="pt"
    )[0]

    for q_idx, question in tqdm(enumerate(eval_questions)):
        tokenized_fsp = tokenized_fsp
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

        # We cache activations for question+instruction
        q_instr_locs = range(
            len(tokenized_fsp) + len(tokenized_question),
            len(tokenized_fsp) + len(tokenized_question) + len(tokenized_instruction),
        )
        locs_to_cache.extend(q_instr_locs)

        # and for response
        resp_locs = range(
            len(tokenized_fsp) + len(tokenized_question) + len(tokenized_instruction),
            len(input_ids),
        )
        locs_to_cache.extend(resp_locs)

        resid_acts = clean_run_with_cache(
            model, input_ids, layers_to_cache, locs_to_cache
        )

        for layer_idx, layer in enumerate(layers_to_cache):
            resid_acts_layer = resid_acts[layer_idx].cpu()
            q_instr_acts_layer = resid_acts_layer[: len(q_instr_locs)]
            resp_acts_layer = resid_acts_layer[len(q_instr_locs) :]

            q_and_instr_activations_by_layer[layer].append(q_instr_acts_layer)
            resp_activations_by_layer[layer].append(resp_acts_layer)

    # Dump activations to disk
    acts_name = f"acts_q_instr"
    acts_folder_path = output_folder / acts_name
    acts_folder_path.mkdir(parents=True, exist_ok=True)

    for layer, activations in q_and_instr_activations_by_layer.items():
        acts_path = acts_folder_path / f"L{layer:02}.pkl"
        with open(acts_path, "wb") as f:
            pickle.dump(activations, f)

    acts_name = f"acts_resp"
    acts_folder_path = output_folder / acts_name
    acts_folder_path.mkdir(parents=True, exist_ok=True)

    for layer, activations in resp_activations_by_layer.items():
        acts_path = acts_folder_path / f"L{layer:02}.pkl"
        with open(acts_path, "wb") as f:
            pickle.dump(activations, f)


def main():
    args = parse_arguments()
    eval_questions_path = Path(args.eval_questions)
    with open(eval_questions_path, "rb") as f:
        eval_questions = pickle.load(f)

    print(f"Processing {len(eval_questions)} questions from {eval_questions_path}...")

    *_, model_name_dir, task, misc_details, __ = eval_questions_path.parts
    model_name = model_name_dir.replace("--", "/")
    print(f"{model_name=}, {task=}, {misc_details=}")
    output_folder = eval_questions_path.parent
    print(f"Output folder: {output_folder}")

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for biased in [True, False]:
        process_questions(output_folder, biased, model, tokenizer, eval_questions)


if __name__ == "__main__":
    main()
