import argparse
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
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )

    with open(args.responses_pkl_path, "rb") as f:
        responses_by_q_by_seed = pickle.load(f)

    swap_results_by_q_by_seed = {}

    for seed_i, (seed, responses_by_q) in enumerate(responses_by_q_by_seed.items()):
        print(f"Processing seed {seed_i + 1} / {len(responses_by_q_by_seed)}")
        swap_results_by_q = swap_results_by_q_by_seed[seed] = {}
        all_combinations = generate_all_combinations(seed=seed)

        for q_idx, responses in enumerate(
            tqdm(responses_by_q, desc="Processing questions")
        ):
            combined_prompts = all_combinations[q_idx]
            unbiased_prompt = combined_prompts["unb_yes"]
            bias_no_prompt = combined_prompts["no_yes"]

            question_swap_results = process_question(
                model,
                tokenizer,
                unbiased_prompt,
                bias_no_prompt,
                responses,
                args.topk,
                args.prob_diff_threshold,
            )
            swap_results_by_q[q_idx] = question_swap_results

    output_path = DATA_DIR / f"swap_results_by_q_by_seed.pkl"
    with open(output_path, "wb") as f:
        pickle.dump(swap_results_by_q_by_seed, f)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Token swapping script")
    parser.add_argument(
        "--model", "-m", type=str, required=True, help="Hugging Face model ID"
    )
    parser.add_argument(
        "--responses-pkl-path",
        "-r",
        type=str,
        required=True,
        help="Path to the pickle file with responses",
    )
    parser.add_argument(
        "--topk",
        "-k",
        type=int,
        default=5,
        help="Number of top positions to try swapping",
    )
    parser.add_argument(
        "--prob-diff-threshold",
        "-p",
        type=float,
        default=0.01,
        help="Probability difference threshold",
    )
    args = parser.parse_args()

    main(args)
