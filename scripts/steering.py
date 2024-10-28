#!/usr/bin/env python3
import argparse
import json
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing import DATA_DIR
from cot_probing.probing import get_locs_to_probe, get_probe_data, split_dataset
from cot_probing.utils import load_model_and_tokenizer




def parse_args():
    parser = argparse.ArgumentParser(description="Train logistic regression probes")
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Path to the probing results",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=str,
        default=None,
        help="List of comma separated layers to steer. Defaults to all layers.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


answer_yes_tok = tokenizer.encode("Answer: Yes", add_special_tokens=False)
assert len(answer_yes_tok) == 3
answer_no_tok = tokenizer.encode("Answer: No", add_special_tokens=False)
assert len(answer_no_tok) == 3
end_of_text_tok = tokenizer.eos_token_id


def categorize_responses(responses):
    yes_responses = []
    no_responses = []
    other_responses = []
    for response in responses:
        response = response.tolist()

        # right strip as many end_of_text_tok as possible from each response
        # This is necessary because model.generate adds this when using cache
        while response[-1] == end_of_text_tok:
            response = response[:-1]

        if response[-3:] == answer_yes_tok:
            yes_responses.append(response)
        elif response[-3:] == answer_no_tok:
            no_responses.append(response)
        else:
            other_responses.append(response)

    return {
        "yes": yes_responses,
        "no": no_responses,
        "other": other_responses,
    }


def run_steering_experiment(
    probing_df_results: pd.DataFrame,
    locs_to_steer: Dict,
    layers_to_steer: List[int],
    seed: int = 42,
    verbose: bool = False,
):
    results = []
    for test_prompt_idx in tqdm.tqdm(range(len(probe_test_data))):
        # print(f"Running steering on test prompt index: {test_prompt_idx}")
        data_point = probe_test_data[test_prompt_idx]

        question_to_answer = data_point["question_to_answer"]
        expected_answer = data_point["expected_answer"]
        # print(f"Question to answer: {question_to_answer}")
        # print(f"Expected answer: {expected_answer}")

        unbiased_fsp = data_point["unbiased_fsp"]
        biased_fsp = data_point["biased_fsp"]
        prompt = biased_fsp + "\n\n" + question_to_answer

        # Build the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # Find the position of the last "Question" token
        question_token_id = tokenizer.encode("Question", add_special_tokens=False)[0]
        last_question_first_token_pos = [
            i for i, t in enumerate(input_ids[0]) if t == question_token_id
        ][-1]
        # print(f"Last question position first token pos: {last_question_first_token_pos}")

        n_gen = 10

        # print("\nUnsteered generation:")
        unsteered_responses = steer_generation(
            input_ids, [], n_layers, last_question_first_token_pos, 0, n_gen=n_gen
        )
        # for response in unsteered_responses:
        #     print(f"Response: {tokenizer.decode(response)}")
        #     print()

        loc_probe_keys = list(locs_to_probe.keys())
        loc_keys_to_steer = [
            # loc_probe_keys[0],
            loc_probe_keys[1],
            loc_probe_keys[2],
            # loc_probe_keys[5],
            # loc_probe_keys[8]
        ]
        # print(f"Location keys to steer: {loc_keys_to_steer}")

        layers_to_steer = list(range(13, 26))
        # print(f"Layers to steer: {layers_to_steer}")

        pos_steer_magnitude = 0.4
        # print(f"\nPositive steered generation: {pos_steer_magnitude}")
        positive_steered_responses = steer_generation(
            input_ids,
            loc_keys_to_steer,
            layers_to_steer,
            last_question_first_token_pos,
            pos_steer_magnitude,
            n_gen=n_gen,
        )
        # for i, response in enumerate(positive_steered_responses):
        #     print(f"Response {i}: {tokenizer.decode(response)}")
        #     print()

        neg_steer_magnitude = -0.4
        # print(f"\nNegative steered generation: {neg_steer_magnitude}")
        negative_steered_responses = steer_generation(
            input_ids,
            loc_keys_to_steer,
            layers_to_steer,
            last_question_first_token_pos,
            neg_steer_magnitude,
            n_gen=n_gen,
        )
        # for i, response in enumerate(negative_steered_responses):
        #     print(f"Response {i}: {tokenizer.decode(response)}")
        #     print()

        # Measure unbiased accuracy of the CoT's produced

        unbiased_fsp_with_question = f"{unbiased_fsp}\n\n{question_to_answer}"
        tokenized_unbiased_fsp_with_question = tokenizer.encode(unbiased_fsp_with_question)

        unsteered_unbiased_answers = {
            "yes": [],
            "no": [],
            "other": [],
        }
        for cot in unsteered_responses:
            cot_without_answer = cot.tolist()[:-1]
            answer = categorize_response_unbiased(
                model=model,
                tokenizer=tokenizer,
                unbiased_context_toks=tokenized_unbiased_fsp_with_question,
                response=cot_without_answer,
            )
            unsteered_unbiased_answers[answer].append(cot)

        unsteered_accuracy = len(unsteered_unbiased_answers[expected_answer]) / len(
            unsteered_responses
        )
        # print(f"Unsteered accuracy: {unsteered_accuracy:.4f}")

        pos_steering_unbiased_answers = {
            "yes": [],
            "no": [],
            "other": [],
        }
        for cot in positive_steered_responses:
            cot_without_answer = cot.tolist()[:-1]
            answer = categorize_response_unbiased(
                model=model,
                tokenizer=tokenizer,
                unbiased_context_toks=tokenized_unbiased_fsp_with_question,
                response=cot_without_answer,
            )
            pos_steering_unbiased_answers[answer].append(cot)

        pos_steering_accuracy = len(pos_steering_unbiased_answers[expected_answer]) / len(
            positive_steered_responses
        )
        # print(f"Positive steering accuracy: {pos_steering_accuracy:.4f}")

        neg_steering_unbiased_answers = {
            "yes": [],
            "no": [],
            "other": [],
        }
        for cot in negative_steered_responses:
            cot_without_answer = cot.tolist()[:-1]
            answer = categorize_response_unbiased(
                model=model,
                tokenizer=tokenizer,
                unbiased_context_toks=tokenized_unbiased_fsp_with_question,
                response=cot_without_answer,
            )
            neg_steering_unbiased_answers[answer].append(cot)

        neg_steering_accuracy = len(neg_steering_unbiased_answers[expected_answer]) / len(
            negative_steered_responses
        )
        # print(f"Negative steering accuracy: {neg_steering_accuracy:.4f}")

        res = {
            "unsteered": unsteered_unbiased_answers,
            "pos_steer": pos_steering_unbiased_answers,
            "neg_steer": neg_steering_unbiased_answers,
            "unsteered_accuracy": unsteered_accuracy,
            "pos_steering_accuracy": pos_steering_accuracy,
            "neg_steering_accuracy": neg_steering_accuracy,
        }

        # for variant in res.keys():
        #     print(f"{variant=}")
        #     for key in ["yes", "no", "other"]:
        #         print(f"- {key} {len(res[variant][key])}")

        results.append(res)


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_file_path = Path(args.file)
    if not input_file_path.exists():
        raise FileNotFoundError(f"File not found at {input_file_path}")

    with open(input_file_path, "rb") as f:
        probing_results = pickle.load(f)

    model_size = probing_results["arg_model_size"]
    model, tokenizer = load_model_and_tokenizer(model_size)

    if args.layers:
        layers_to_steer = args.layers.split(",")
    else:
        layers_to_steer = list(range(model.config.num_hidden_layers))

    locs_to_steer = get_locs_to_probe(tokenizer)

    probing_df_results = pd.DataFrame(probing_results["probing_results"])

    results = run_steering_experiment(
        probing_df_results=probing_df_results,
        locs_to_steer=locs_to_steer,
        layers_to_steer=layers_to_steer,
        seed=args.seed,
        verbose=args.verbose,
    )
    ret = dict(
        steering_results=results,
        **{f"arg_{k}": v for k, v in vars(args).items() if k != "file"},
    )

    # Save the results
    output_file_name = input_file_path.name.replace("_results_", "steering_results_")
    output_file_path = DATA_DIR / output_file_name
    with open(output_file_path, "wb") as f:
        pickle.dump(ret, f)


if __name__ == "__main__":
    main(parse_args())`
