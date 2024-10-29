#!/usr/bin/env python3
import argparse
import json
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
        "-o",
        "--output-images-dir",
        type=str,
        default="images",
        help="Path to the directory to save the images",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def plot_accuracy_by_layer_and_loc_type(df_results: pd.DataFrame, images_dir: Path):
    # Define the order of columns based on locs_to_probe
    ordered_columns = [
        "loc_q_tok",
        "loc_colon_after_q_tok",
        "loc_question",
        "loc_question_mark_new_line_tok",
        "loc_let_tok",
        "loc_'s_tok",
        "loc_think_tok",
        "loc_first_step_tok",
        "loc_by_tok",
        "loc_second_step_tok",
        "loc_colon_new_line_tok",
    ]
    
    # Add the step locations in order
    for i in range(8):  # max_steps = 8
        ordered_columns.extend([
            f"loc_cot_step_{i}_dash",
            f"loc_cot_step_{i}_reasoning",
            f"loc_cot_step_{i}_newline_tok",
        ])
    
    # Add the final answer locations
    ordered_columns.extend([
        "loc_answer_tok",
        "loc_answer_colon_tok",
        "loc_actual_answer_tok",
    ])

    # Pivot the DataFrame to create a 2D matrix of accuracy_test values
    pivot_df = df_results.pivot(
        index="layer", columns="loc_type", values="accuracy_test"
    )
    
    # Reorder the columns according to our defined order
    # Only include columns that exist in the pivot_df
    ordered_columns = [col for col in ordered_columns if col in pivot_df.columns]
    pivot_df = pivot_df[ordered_columns]

    # Create the heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(
        pivot_df,
        cmap="viridis_r",
        annot=True,
        fmt=".4f",
        cbar_kws={"label": "Accuracy Test"},
    )

    plt.title(
        "Accuracy Test by Layer and Location Type (Probing for faithful/unfaithful answer)"
    )
    plt.xlabel("Location Type")
    plt.ylabel("Layer")
    plt.tight_layout()
    plt.savefig(images_dir / "accuracy_test_by_layer_and_loc_type.png")
    plt.close()


def plot_probe_performance_by_layer_and_loc_type(
    df_results: pd.DataFrame,
    images_dir: Path,
):
    # Plot the results for each loc_type
    for loc_type in locs_to_probe.keys():
        df_loc = df_results[df_results["loc_type"] == loc_type]

        plt.figure(figsize=(12, 6))
        plt.plot(df_loc["layer"], df_loc["accuracy_train"], label="Accuracy (train)")
        plt.plot(df_loc["layer"], df_loc["accuracy_test"], label="Accuracy (test)")
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.title(f"Logistic Probe Performance by Layer for {loc_type}")
        plt.legend()
        plt.savefig(images_dir / f"accuracy_comparison_{loc_type}.png")
        plt.close()


def plot_logistic_regression_results(
    df_results: pd.DataFrame,
    loc_type: str,
    layer: int,
    images_dir: Path,
):
    df_filtered = df_results[
        (df_results["loc_type"] == loc_type) & (df_results["layer"] == layer)
    ]

    if len(df_filtered) == 0:
        print(f"No data found for loc_type '{loc_type}' and layer {layer}")
        return

    y_test = df_filtered["y_test"].iloc[0]
    y_pred_test = df_filtered["y_pred_test"].iloc[0]
    probe = df_filtered["probe"].iloc[0]

    # Encode categorical labels
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    y_pred_test_encoded = le.transform(y_pred_test)

    # 1. Confusion Matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix\nLocation: {loc_type}, Layer: {layer}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(images_dir / f"confusion_matrix_{loc_type}.png")
    plt.close()

    # 2. ROC Curve
    plt.figure(figsize=(10, 8))
    y_pred_proba = probe.predict_proba(X_test[loc_type][layer])[:, 1]
    fpr, tpr, _ = roc_curve(y_test_encoded, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"Receiver Operating Characteristic (ROC) Curve\nLocation: {loc_type}, Layer: {layer}"
    )
    plt.legend(loc="lower right")
    plt.savefig(images_dir / f"roc_curve_{loc_type}.png")
    plt.close()

    # 3. Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    precision, recall, _ = precision_recall_curve(y_test_encoded, y_pred_proba)
    plt.plot(recall, precision, color="blue", lw=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve\nLocation: {loc_type}, Layer: {layer}")
    plt.savefig(images_dir / f"precision_recall_curve_{loc_type}.png")
    plt.close()

    # 4. Bar plot of class probabilities
    # plt.figure(figsize=(10, 8))
    # sns.histplot(y_pred_proba, bins=20, kde=True)
    # plt.xlabel('Predicted Probability of Positive Class')
    # plt.ylabel('Count')
    # plt.title(f'Distribution of Predicted Probabilities\nLocation: {loc_type}, Layer: {layer}')
    # plt.savefig(images_dir / f"class_probabilities_{loc_type}.png")
    # plt.close()

    # Print accuracy
    accuracy = df_filtered["accuracy_test"].iloc[0]
    print(f"Accuracy: {accuracy:.4f}")


def plot_faithful_vs_unfaithful_accuracy(
    df_results: pd.DataFrame,
    images_dir: Path,
):
    plt.figure(figsize=(12, 6))

    loc_types_to_plot = list(locs_to_probe.keys())[:3]
    for loc_type in loc_types_to_plot:
        plt.plot(
            range(n_layers),
            accuracies[loc_type]["faithful"],
            label=f"{loc_type} Faithful",
            marker="o",
        )
        plt.plot(
            range(n_layers),
            accuracies[loc_type]["unfaithful"],
            label=f"{loc_type} Unfaithful",
            marker="s",
        )

    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title("Faithful and Unfaithful Accuracy by Layer and Location Type")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # Add x-axis ticks for every 5th layer
    plt.xticks(range(0, n_layers, 5))

    plt.tight_layout()
    plt.savefig(images_dir / "faithful_unfaithful_accuracy.png")
    plt.close()


def print_accuracies(df_results: pd.DataFrame):
    accuracies = {
        loc_type: {"faithful": [], "unfaithful": []}
        for loc_type in locs_to_probe.keys()
    }

    print("\nAccuracy for each probe:")
    for loc_type in locs_to_probe.keys():
        df_loc = df_results[df_results["loc_type"] == loc_type]

        for layer in range(n_layers):
            row = df_loc[df_loc["layer"] == layer]
            if len(row) == 0:
                accuracies[loc_type]["faithful"].append(None)
                accuracies[loc_type]["unfaithful"].append(None)
                continue

            y_test = row["y_test"].iloc[0]
            y_pred_test = row["y_pred_test"].iloc[0]

            # Accuracy for "faithful" (y_test is "faithful")
            faithful_mask = np.array(y_test) == "faithful"
            faithful_pred = np.array(y_pred_test)[faithful_mask] == "faithful"
            faithful_accuracy = np.mean(faithful_pred)
            accuracies[loc_type]["faithful"].append(faithful_accuracy)

            # Accuracy for "unfaithful" (y_test is "unfaithful")
            unfaithful_mask = np.array(y_test) == "unfaithful"
            unfaithful_pred = np.array(y_pred_test)[unfaithful_mask] == "unfaithful"
            unfaithful_accuracy = np.mean(unfaithful_pred)
            accuracies[loc_type]["unfaithful"].append(unfaithful_accuracy)

            print(f"Location: {loc_type}, Layer {layer}:")
            print(f"  Faithful Accuracy: {faithful_accuracy:.4f}")
            print(f"  Unfaithful Accuracy: {unfaithful_accuracy:.4f}")


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    input_file_path = Path(args.file)
    if not input_file_path.exists():
        raise FileNotFoundError(f"File not found at {input_file_path}")

    with open(input_file_path, "rb") as f:
        probing_results = pickle.load(f)

    # model_size = probing_results["arg_model_size"]
    # model, tokenizer = load_model_and_tokenizer(model_size)

    # if args.layers:
    #     layers_to_steer = args.layers.split(",")
    # else:
    #     layers_to_steer = list(range(model.config.num_hidden_layers))

    # locs_to_steer = get_locs_to_probe(tokenizer)

    df_results = pd.DataFrame(probing_results["probing_results"])

    images_dir = Path(args.output_images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    plot_accuracy_by_layer_and_loc_type(df_results, images_dir)

    # k = 1
    # for loc_type in locs_to_probe.keys():
    #     df_loc = df_results[df_results["loc_type"] == loc_type]
    #     df_loc = df_loc.sort_values(by="accuracy_test", ascending=False)
    #     top_k_layers = df_loc["layer"].iloc[:k].tolist()
    #     for layer in top_k_layers:
    #         plot_logistic_regression_results(loc_type, layer)


if __name__ == "__main__":
    main(parse_args())
