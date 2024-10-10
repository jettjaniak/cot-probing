#!/usr/bin/env python3

import argparse
import pickle

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM

from cot_probing.activations import (
    Activations,
    QuestionActivations,
    clean_run_with_cache,
)
from cot_probing.typing import *


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "eval_results_path",
        type=str,
        help="Path to the evaluation results pickle file",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    eval_results_path = args.eval_results_path

    with open(eval_results_path, "rb") as f:
        eval_results = pickle.load(f)

    model_name = eval_results.model_name
    task_name = eval_results.task_name
    seed = eval_results.seed
    num_samples = eval_results.num_samples
    questions = eval_results.questions

    print(
        f"Processing {model_name} on {task_name} with seed {seed} and {num_samples} samples"
    )

    model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

    layers_to_cache = list(range(model.config.num_hidden_layers))
    activations_by_question: list[QuestionActivations] = []
    for question in tqdm(questions):
        locs_to_cache = set()
        for key, locs in question.locs.items():
            locs_to_cache.update(locs)
        locs_to_cache = sorted(locs_to_cache)

        # hack to add "Let's think step by step"
        min_loc = locs_to_cache[0]
        # hack to not add anything else
        locs_to_cache = list(range(min_loc - 10, min_loc))  # + locs_to_cache
        # hack to have smaller prompt and faster inference
        input_ids = torch.tensor(question.tokens[:min_loc])
        resid_acts = clean_run_with_cache(
            model, input_ids, layers_to_cache, locs_to_cache
        )

        activations_by_question.append(QuestionActivations(resid_acts, locs_to_cache))

    model_name = model_name.replace("/", "--")
    resid_acts_path = (
        f"results/activations_{model_name}_{task_name}_S{seed}_N{num_samples}.pkl"
    )

    with open(resid_acts_path, "wb") as f:
        pickle.dump(
            Activations(eval_results, activations_by_question, layers_to_cache), f
        )
