#!/usr/bin/env python3

import argparse
import pickle

import torch

from cot_probing.activations import Activations


def main():
    parser = argparse.ArgumentParser(
        description="Process activations and compute difference direction."
    )
    parser.add_argument(
        "activations_results_path",
        type=str,
        help="Path to the activations results file",
    )
    args = parser.parse_args()

    with open(args.activations_results_path, "rb") as f:
        activations: Activations = pickle.load(f)
    eval_results = activations.eval_results

    positive_activations = []
    negative_activations = []
    for question_idx, question in enumerate(eval_results.questions):
        acts = activations.activations_by_question[
            question_idx
        ].activations  # Shape: [n_layers locs d_model]

        # Take mean across locs
        mean_acts = acts.mean(1)  # Shape: [n_layers d_model]

        if question.is_correct:
            positive_activations.append(mean_acts)
        else:
            negative_activations.append(mean_acts)

    # Stack them up
    positive_activations = torch.stack(
        positive_activations
    )  # Shape: [n_positive n_layers d_model]
    negative_activations = torch.stack(
        negative_activations
    )  # Shape: [n_negative n_layers d_model]

    # Mean across positive and negative examples
    positive_activations = positive_activations.mean(0)  # Shape: [n_layers d_model]
    negative_activations = negative_activations.mean(0)  # Shape: [n_layers d_model]

    # Compute the difference
    diff = positive_activations - negative_activations  # Shape: [n_layers d_model]

    # Build the output file path from the data in activations
    model_name = eval_results.model_name.replace("/", "--")
    task_name = eval_results.task_name
    seed = eval_results.seed
    num_samples = eval_results.num_samples
    diff_direction_results_path = (
        f"results/diff_direction_{model_name}_{task_name}_S{seed}_N{num_samples}.pt"
    )

    # Dump the difference
    torch.save(diff, diff_direction_results_path)


if __name__ == "__main__":
    main()
