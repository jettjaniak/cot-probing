#!/usr/bin/env python3
import argparse
import logging
import pickle
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing import DATA_DIR
from cot_probing.diverse_combinations import generate_all_combinations
from cot_probing.swapping import process_question
from cot_probing.typing import *


def main(args):
    logging.basicConfig(level=logging.INFO)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    model_id = f"hugging-quants/Meta-Llama-3.1-{args.model_size}B-BNB-NF4-BF16"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )

    with open(args.responses_pkl_path, "rb") as f:
        responses_by_q_by_seed = pickle.load(f)

    responses_by_q_by_seed_items = list(responses_by_q_by_seed.items())
    if args.debug:
        responses_by_q_by_seed_items = responses_by_q_by_seed_items[:2]
    for seed_i, (seed, responses_by_q) in enumerate(responses_by_q_by_seed_items):
        n_seeds = len(responses_by_q_by_seed_items)
        print(f"Processing seed {seed_i + 1} / {n_seeds}")
        swap_results_by_q = []
        all_combinations = generate_all_combinations(seed=seed)
        if args.debug:
            responses_by_q = responses_by_q[:2]
        for q_idx, responses_by_answer_by_ctx in enumerate(
            tqdm(
                responses_by_q,
                desc=f"Processing questions for seed {seed_i + 1} / {n_seeds}",
            )
        ):
            combined_prompts = all_combinations[q_idx]
            unbiased_prompt = combined_prompts["unb_yes"]
            bias_no_prompt = combined_prompts["no_yes"]

            question_swap_results = process_question(
                model,
                tokenizer,
                unbiased_prompt,
                bias_no_prompt,
                responses_by_answer_by_ctx,
                args.topk_pos,
                args.topk_tok,
                args.prob_diff_threshold,
                args.debug,
            )
            swap_results_by_q.append(question_swap_results)

            output_path = DATA_DIR / f"swap_results_by_q_seed_i_{seed_i}.pkl"
            with open(output_path, "wb") as f:
                pickle.dump(swap_results_by_q, f)
            print(f"Results up to {q_idx=} saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token swapping script")
    parser.add_argument(
        "--model-size", "-m", type=int, required=True, help="size of the model"
    )
    parser.add_argument(
        "--responses-pkl-path",
        "-r",
        type=str,
        required=True,
        help="Path to the pickle file with responses",
    )
    parser.add_argument(
        "--topk-pos",
        type=int,
        default=5,
        help="Number of top positions to try swapping",
    )
    parser.add_argument(
        "--topk-tok",
        type=int,
        default=5,
        help="Number of top tokens to try swapping with (if above threshold)",
    )
    parser.add_argument(
        "--prob-diff-threshold",
        "-p",
        type=float,
        default=0.01,
        help="Probability difference threshold",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)
