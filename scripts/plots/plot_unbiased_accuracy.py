#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from cot_probing import DATA_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="Label questions as faithful or not")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the dataset of questions with measured accuracy of biased and unbiased CoTs",
    )
    parser.add_argument(
        "-o",
        "--output-images-dir",
        type=str,
        default="images",
        help="Path to the directory to save the images",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def plot_unbiased_accuracy_distribution(
    data: list[dict],
    biased: bool,
    images_dir: Path,
    verbose: bool = False,
):
    # Extract accuracies from the data
    if biased:
        accuracies = [
            item["n_correct_biased"] / item["n_gen"]
            for item in data
            if "n_correct_biased" in item
        ]
    else:
        accuracies = [
            item["n_correct_unbiased"] / item["n_gen"]
            for item in data
            if "n_correct_unbiased" in item
        ]

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=20, edgecolor="black")
    plt.title(
        "Distribution of Unbiased Accuracies for "
        + ("Biased" if biased else "Unbiased")
        + " COTs"
    )
    plt.xlabel("Accuracy")
    plt.ylabel("Frequency")

    # Add mean line
    mean_accuracy = np.mean(accuracies)
    plt.axvline(mean_accuracy, color="r", linestyle="dashed", linewidth=2)
    plt.text(
        mean_accuracy * 1.02,
        plt.ylim()[1] * 0.9,
        f"Mean: {mean_accuracy:.2f}",
        color="r",
    )

    plt.tight_layout()
    biased_str = "biased" if biased else "unbiased"
    plt.savefig(images_dir / f"unbiased_accuracy_distribution_{biased_str}_cots.png")
    plt.close()

    # Print some statistics
    if verbose:
        print(f"Mean accuracy: {mean_accuracy:.2f}")
        print(f"Median accuracy: {np.median(accuracies):.2f}")
        print(f"Min accuracy: {min(accuracies):.2f}")
        print(f"Max accuracy: {max(accuracies):.2f}")


def plot_combined_unbiased_accuracy_distribution(
    data: list[dict],
    images_dir: Path,
    verbose: bool = False,
):
    # Create the heatmap
    plt.figure(figsize=(12, 10))

    # Extract the accuracies
    unbiased_cot_acc = [
        item["n_correct_unbiased"] / item["n_gen"]
        for item in data
        if "n_correct_unbiased" in item
    ]
    biased_cot_acc = [
        item["n_correct_biased"] / item["n_gen"]
        for item in data
        if "n_correct_biased" in item
    ]

    # Create the heatmap using seaborn
    sns.histplot(
        x=unbiased_cot_acc,
        y=biased_cot_acc,
        bins=20,
        cmap="YlGnBu",
        cbar=True,
    )

    # Add diagonal line
    plt.plot([0, 1], [0, 1], "r--", label="y=x")

    plt.xlabel("Unbiased Accuracy for Unbiased COTs")
    plt.ylabel("Unbiased Accuracy for Biased COTs")
    plt.title("Heatmap: Unbiased Accuracy for Unbiased vs Biased COTs")
    plt.legend()

    # Set both axes to start at 0 and end at 1
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(images_dir / "unbiased_accuracy_for_unbiased_vs_biased_cots.png")
    plt.close()

    if verbose:
        all_diff = [
            item["unbiased_accuracy_for_unbiased_cots"]
            - item["unbiased_accuracy_for_biased_cots"]
            for item in all_data
        ]

        print(f"All data:")
        print(f"  Mean difference: {np.mean(all_diff):.4f}")
        print(f"  Median difference: {np.median(all_diff):.4f}")
        print(f"  Std deviation of difference: {np.std(all_diff):.4f}")


def label_questions(
    questions: list[dict],
    faithful_accuracy_threshold: float,
    unfaithful_accuracy_threshold: float,
    verbose: bool = False,
):
    """
    Label each question as faithful or unfaithful depending on the accuracy of biased and unbiased COTs

    Args:
        questions: list of questions with measured accuracy of biased and unbiased COTs
        faithful_accuracy_threshold: Minimum accuracy of biased COTs to be considered faithful
        unfaithful_accuracy_threshold: Maximum accuracy of biased COTs to be considered unfaithful
        verbose: Whether to print verbose output
    """
    for item in questions:
        if "n_correct_biased" not in item or "n_correct_unbiased" not in item:
            print("Warning: n_correct_biased or n_correct_unbiased not in item")
            continue

        biased_cots_accuracy = item["n_correct_biased"] / item["n_gen"]
        if verbose:
            print(f"Biased COTs accuracy: {biased_cots_accuracy}")

        unbiased_cots_accuracy = item["n_correct_unbiased"] / item["n_gen"]
        if verbose:
            print(f"Unbiased COTs accuracy: {unbiased_cots_accuracy}")

        if biased_cots_accuracy >= faithful_accuracy_threshold * unbiased_cots_accuracy:
            item["biased_cot_label"] = "faithful"
            if verbose:
                print("Labeled as faithful")
        elif (
            biased_cots_accuracy
            <= unfaithful_accuracy_threshold * unbiased_cots_accuracy
        ):
            item["biased_cot_label"] = "unfaithful"
            if verbose:
                print("Labeled as unfaithful")
        else:
            item["biased_cot_label"] = "mixed"
            if verbose:
                print("Labeled as mixed")


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    input_file_path = Path(args.file)
    if not input_file_path.exists():
        raise FileNotFoundError(f"File not found at {input_file_path}")

    input_file_name = input_file_path.name
    if not input_file_name.startswith("measured_qs_"):
        raise ValueError(
            f"Input file name must start with 'measured_qs_', got {input_file_name}"
        )

    with open(input_file_path, "r") as f:
        dataset = json.load(f)

    dataset_id = input_file_path.stem.split("_")[-1]

    if "qs" not in dataset:
        raise ValueError("Dataset must contain 'qs' key")

    images_dir = Path(args.output_images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    plot_unbiased_accuracy_distribution(
        data=dataset["qs"],
        biased=False,
        images_dir=images_dir,
        verbose=args.verbose,
    )
    plot_unbiased_accuracy_distribution(
        data=dataset["qs"],
        biased=True,
        images_dir=images_dir,
        verbose=args.verbose,
    )
    plot_combined_unbiased_accuracy_distribution(
        data=dataset["qs"],
        images_dir=images_dir,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main(parse_args())
