#!/usr/bin/env python3
import argparse
import logging
import pickle
import random
from typing import Literal

import numpy as np
from tqdm.auto import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing import DATA_DIR
from cot_probing.attn_probes import AbstractAttnProbeModel
from cot_probing.attn_probes_utils import load_median_probe_test_data
from cot_probing.generation import categorize_response as categorize_response_unbiased
from cot_probing.steering import steer_generation_with_attn_probe
from cot_probing.utils import load_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Steer attn probes")
    parser.add_argument(
        "-l",
        "--layer",
        type=int,
        default=15,
        help="Layer on which to steer (and which the probe was trained on)",
    )
    parser.add_argument(
        "--data-size",
        "-d",
        type=int,
        default=None,
        help="Number of data points to run steering on. If None, all data points are run.",
    )
    parser.add_argument(
        "--seeds",
        "-s",
        type=str,
        default="1-10",
        help="Seed range for trained probes (inclusive)",
    )
    parser.add_argument(
        "--probe-class",
        type=str,
        choices=["V", "QV"],
        default="V",
        help="Type of attention probe to use",
    )
    parser.add_argument(
        "--metric",
        "-m",
        type=str,
        default="test_accuracy",
        help="Metric to select the median probe",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="attn-probes",
        help="Wandb project name",
    )
    parser.add_argument(
        "-c",
        "--context",
        type=str,
        choices=["biased-fsp", "unbiased-fsp", "no-fsp"],
        default="biased-fsp",
        help="FSP context for the steered generation.",
    )
    parser.add_argument(
        "-n",
        "--n-gen",
        type=int,
        default=10,
        help="Number of generations to run for each question",
    )
    parser.add_argument(
        "-ps",
        "--pos-steer-magnitude",
        type=float,
        default=0.4,
        help="Magnitude of the positive steering",
    )
    parser.add_argument(
        "-ns",
        "--neg-steer-magnitude",
        type=float,
        default=-0.4,
        help="Magnitude of the negative steering",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def run_steering_experiment(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    attn_probe_model: AbstractAttnProbeModel,
    test_acts_dataset: dict,
    layer_to_steer: int,
    fsp_context: Literal["biased-fsp", "unbiased-fsp", "no-fsp"],
    data_size: int | None = None,
    n_gen: int = 10,
    pos_steer_magnitude: float = 0.4,
    neg_steer_magnitude: float = -0.4,
    verbose: bool = False,
):
    # Pre-cache FSP activations
    unbiased_fsp = test_acts_dataset["unbiased_fsp"] + "\n\n"
    biased_no_fsp = test_acts_dataset["biased_no_fsp"] + "\n\n"
    biased_yes_fsp = test_acts_dataset["biased_yes_fsp"] + "\n\n"

    questions = test_acts_dataset["qs"]
    if data_size is not None:
        # Randomly sample data points
        questions = random.sample(questions, data_size)

    results = []
    for test_prompt_idx in tqdm(range(len(questions))):
        if verbose:
            print(f"Running steering on test prompt index: {test_prompt_idx}")

        data_point = test_acts_dataset["qs"][test_prompt_idx]

        question_to_answer = data_point["question"]
        expected_answer = data_point["expected_answer"]

        if verbose:
            print(f"Question to answer: {question_to_answer}")
            print(f"Expected answer: {expected_answer}")

        if fsp_context == "biased-fsp":
            # Choose the biased FSP based on the expected answer
            if expected_answer == "yes":
                fsp = biased_no_fsp
            else:
                fsp = biased_yes_fsp
        elif fsp_context == "unbiased-fsp":
            fsp = unbiased_fsp
        elif fsp_context == "no-fsp":
            fsp = None

        if fsp is not None:
            prompt = f"{fsp}\n\n{question_to_answer}"
        else:
            prompt = question_to_answer

        # Build the prompt
        input_ids = tokenizer.encode(prompt)

        steered_responses = []
        steering_magnitudes: list[float] = [
            0.0,
            pos_steer_magnitude,
            neg_steer_magnitude,
        ]
        for steer_magnitude in steering_magnitudes:
            if verbose:
                print(
                    f"\nGenerating responses with steering magnitude: {steer_magnitude}"
                )
            responses = steer_generation_with_attn_probe(
                model=model,
                tokenizer=tokenizer,
                attn_probe_model=attn_probe_model,
                input_ids=input_ids,
                layer_to_steer=layer_to_steer,
                steer_magnitude=steer_magnitude,
                n_gen=n_gen,
            )
            steered_responses.append(responses)
            if verbose:
                for response in responses:
                    print(f"Response: {tokenizer.decode(response)}")
                    print()

        # Measure unbiased accuracy of the CoT's produced
        unbiased_fsp_with_question = f"{unbiased_fsp}\n\n{question_to_answer}"
        tokenized_unbiased_fsp_with_question = tokenizer.encode(
            unbiased_fsp_with_question
        )

        accuracies = []
        answers = []
        for responses, steer_magnitude in zip(steered_responses, steering_magnitudes):
            unbiased_answers = {
                "yes": [],
                "no": [],
                "other": [],
            }
            for cot in responses:
                cot_without_answer = cot[:-1]
                answer = categorize_response_unbiased(
                    model=model,
                    tokenizer=tokenizer,
                    unbiased_context_toks=tokenized_unbiased_fsp_with_question,
                    response=cot_without_answer,
                )
                unbiased_answers[answer].append(cot)
            accuracy = len(unbiased_answers[expected_answer]) / len(responses)
            accuracies.append(accuracy)
            answers.append(unbiased_answers)

            if verbose:
                print(
                    f"Steering magnitude: {steer_magnitude}, accuracy: {accuracy:.4f}"
                )
                for key in unbiased_answers.keys():
                    print(f"- {key}: {len(unbiased_answers[key])}")

        res = {
            "question": question_to_answer,
            "expected_answer": expected_answer,
            "unsteered_cots": steered_responses[0],
            "unsteered_answers": answers[0],
            "unsteered_accuracy": accuracies[0],
            "pos_steering_cots": steered_responses[1],
            "pos_steering_answers": answers[1],
            "pos_steering_accuracy": accuracies[1],
            "neg_steering_cots": steered_responses[2],
            "neg_steering_answers": answers[2],
            "neg_steering_accuracy": accuracies[2],
        }

        results.append(res)

    return results


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    layer = args.layer
    min_seed, max_seed = map(int, args.seeds.split("-"))
    n_seeds = max_seed - min_seed + 1
    probe_class = args.probe_class
    fsp_context = args.context
    metric = args.metric
    trainer, test_acts_dataset = load_median_probe_test_data(
        probe_class=probe_class,
        fsp_context=fsp_context,
        layer=layer,
        min_seed=min_seed,
        max_seed=max_seed,
        metric=metric,
    )
    trainer.model.eval()
    trainer.model.requires_grad_(False)

    model, tokenizer = load_model_and_tokenizer(8)
    model.eval()
    model.requires_grad_(False)

    results = run_steering_experiment(
        model=model,
        tokenizer=tokenizer,
        attn_probe_model=trainer.model,
        test_acts_dataset=test_acts_dataset,
        layer_to_steer=layer,
        fsp_context=fsp_context,
        data_size=args.data_size,
        n_gen=args.n_gen,
        pos_steer_magnitude=args.pos_steer_magnitude,
        neg_steer_magnitude=args.neg_steer_magnitude,
        verbose=args.verbose,
    )
    if args.verbose:
        # Print averaged accuracy over all data points
        for key in [
            "unsteered_accuracy",
            "pos_steering_accuracy",
            "neg_steering_accuracy",
        ]:
            accuracies = [res[key] for res in results]
            print(f"Averaged {key}: {np.mean(accuracies):.4f}")
            print(f"Standard deviation for {key}: {np.std(accuracies):.4f}")

    ret = dict(
        steering_results=results,
        **{f"arg_{k}": v for k, v in vars(args).items()},
    )

    # Save the results
    output_file_name = f"steering_results_layer-{layer}_probe-{probe_class}_context-{args.context}_seeds-{args.seeds}.pkl"
    output_file_path = DATA_DIR / output_file_name
    with open(output_file_path, "wb") as f:
        pickle.dump(ret, f)


if __name__ == "__main__":
    main(parse_args())
