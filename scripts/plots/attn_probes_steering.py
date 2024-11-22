#!/usr/bin/env python3
import argparse
import logging
import pickle
import random
from typing import Literal

import matplotlib.pyplot as plt
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
        "-f",
        "--file",
        type=str,
        required=True,
        help="Path to pickle file containing the steeting results",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def plot_accuracy_differences(steering_results: dict):
    """Plot the accuracy differences for the steered and unsteered generations."""
    # Extract accuracies and steering magnitudes
    unsteered_accs = [
        res["unsteered_accuracy"] for res in steering_results["steering_results"]
    ]
    pos_steering_accs = [
        res["pos_steering_accuracy"] for res in steering_results["steering_results"]
    ]
    neg_steering_accs = [
        res["neg_steering_accuracy"] for res in steering_results["steering_results"]
    ]
    pos_magnitude = steering_results["arg_pos_steer_magnitude"]
    neg_magnitude = steering_results["arg_neg_steer_magnitude"]

    # Calculate differences
    pos_diffs = [
        unsteered - pos for pos, unsteered in zip(pos_steering_accs, unsteered_accs)
    ]
    neg_diffs = [
        unsteered - neg for neg, unsteered in zip(neg_steering_accs, unsteered_accs)
    ]

    # Create boxplot with colors
    colors = ["#2ecc71", "#e74c3c"]  # green and red
    box_plot = plt.boxplot([pos_diffs, neg_diffs], patch_artist=True)

    # Color the boxes
    for patch, color in zip(box_plot["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add horizontal line at y=0
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # Add xtick labels with steering magnitudes
    plt.xticks(
        [1, 2],
        [
            f"Positive Steering ({pos_magnitude})",
            f"Negative Steering ({neg_magnitude})",
        ],
    )

    # Customize plot
    plt.title("Accuracy Differences: Steered vs Unsteered")
    plt.ylabel("Accuracy Difference")

    # Save the plot
    output_file = f"steering_accuracy_diffs_layer-{steering_results['arg_layer']}_probe-{steering_results['arg_probe_class']}_context-{steering_results['arg_context']}.png"
    plt.savefig(output_file)
    plt.close()


def plot_histogram_of_answers(steering_results: dict):
    """Plot the histogram of answers for the steered and unsteered generations."""
    # Aggregate counts across all results
    unsteered_counts = {"yes": 0, "no": 0, "other": 0}
    pos_counts = {"yes": 0, "no": 0, "other": 0}
    neg_counts = {"yes": 0, "no": 0, "other": 0}

    for res in steering_results["steering_results"]:
        for answer_type in ["yes", "no", "other"]:
            unsteered_counts[answer_type] += len(res["unsteered_answers"][answer_type])
            pos_counts[answer_type] += len(res["pos_steering_answers"][answer_type])
            neg_counts[answer_type] += len(res["neg_steering_answers"][answer_type])

    # Set up the plot
    labels = ["Yes", "No", "Other"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(
        x - width,
        [unsteered_counts[k.lower()] for k in labels],
        width,
        label="Unsteered",
        color="gray",
        alpha=0.7,
    )
    rects2 = ax.bar(
        x,
        [pos_counts[k.lower()] for k in labels],
        width,
        label=f'Positive Steering ({steering_results["arg_pos_steer_magnitude"]})',
        color="#2ecc71",
        alpha=0.7,
    )
    rects3 = ax.bar(
        x + width,
        [neg_counts[k.lower()] for k in labels],
        width,
        label=f'Negative Steering ({steering_results["arg_neg_steer_magnitude"]})',
        color="#e74c3c",
        alpha=0.7,
    )

    # Customize plot
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Answers by Steering Condition")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Save the plot
    output_file = f"steering_answer_dist_layer-{steering_results['arg_layer']}_probe-{steering_results['arg_probe_class']}_context-{steering_results['arg_context']}.png"
    plt.savefig(output_file)
    plt.close()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    with open(args.file, "rb") as f:
        steering_results = pickle.load(f)

    plot_accuracy_differences(steering_results)
    plot_histogram_of_answers(steering_results)


if __name__ == "__main__":
    main(parse_args())
