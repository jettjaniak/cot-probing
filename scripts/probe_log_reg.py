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
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from cot_probing import DATA_DIR
from cot_probing.utils import to_str_tokens


def parse_args():
    parser = argparse.ArgumentParser(description="Train logistic regression probes")
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Path to the dataset of activations",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=str,
        default=None,
        help="List of comma separated layers to train probes for. Defaults to all layers.",
    )
    parser.add_argument(
        "-t",
        "--test-ratio",
        type=float,
        default=0.2,
        help="Ratio of the dataset to use as test set. Defaults to 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def load_model_and_tokenizer(
    model_size: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    assert model_size in [8, 70]
    model_id = f"hugging-quants/Meta-Llama-3.1-{model_size}B-BNB-NF4-BF16"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    return model, tokenizer


def get_locs_to_probe(tokenizer: PreTrainedTokenizerBase) -> Dict[str, int]:
    last_part_of_last_question = "?\nLet's think step by step:\n-"
    last_part_of_last_question_tokens = tokenizer.encode(
        last_part_of_last_question, add_special_tokens=False
    )
    str_tokens = to_str_tokens(last_part_of_last_question_tokens, tokenizer)

    locs_to_probe = {}
    loc = -1
    for str_token in reversed(str_tokens):
        loc_key = f"loc_{loc}_{str_token}"
        locs_to_probe[loc_key] = loc
        loc -= 1

    return locs_to_probe


def split_dataset(
    acts_dataset: List[Dict],
    test_ratio: float = 0.2,
    verbose: bool = False,
):
    # Split the dataset into faithful and unfaithful
    faithful_data = [
        item for item in acts_dataset if item["biased_cot_label"] == "faithful"
    ]
    unfaithful_data = [
        item for item in acts_dataset if item["biased_cot_label"] == "unfaithful"
    ]
    assert len(faithful_data) > 0, "No faithful data found"
    assert len(unfaithful_data) > 0, "No unfaithful data found"

    if verbose:
        print(f"Faithful data size: {len(faithful_data)}")
        print(f"Unfaithful data size: {len(unfaithful_data)}")

    # Discard data to have balanced train and test sets
    min_num_data = min(len(faithful_data), len(unfaithful_data))
    faithful_data = faithful_data[:min_num_data]
    unfaithful_data = unfaithful_data[:min_num_data]

    if verbose:
        print(f"Faithful data size after discarding: {len(faithful_data)}")
        print(f"Unfaithful data size after discarding: {len(unfaithful_data)}")

    # Split the data into train and test sets
    train_data = (
        faithful_data[: int(len(faithful_data) * (1 - test_ratio))]
        + unfaithful_data[: int(len(unfaithful_data) * (1 - test_ratio))]
    )
    test_data = (
        faithful_data[int(len(faithful_data) * (1 - test_ratio)) :]
        + unfaithful_data[int(len(unfaithful_data) * (1 - test_ratio)) :]
    )

    if verbose:
        print(f"Train data size: {len(train_data)}")
        print(f"Test data size: {len(test_data)}")

    return train_data, test_data


def get_probe_data(
    data_list: List[Dict],
    loc_pos: int,
    layer_idx: int,
    embeddings_in_acts: bool = False,
):
    X = []
    y = []

    for data in data_list:
        for biased_cot_acts in data["cached_acts"]:  # One per each biased_cot
            cached_loc_keys = ["last_question_tokens"]
            for cached_loc_key in cached_loc_keys:
                acts = biased_cot_acts[cached_loc_key][
                    layer_idx
                ]  # List of size biased_cots. Each item is a tensor of shape [seq len, d_model].

                X.append(np.array(acts[loc_pos].float().numpy()))
                y.append(data["biased_cot_label"])

    X = np.array(X)
    y = np.array(y)

    return X, y


def train_logistic_regression_probes(
    acts_dataset: Dict,
    locs_to_probe: Dict[str, int],
    layers_to_probe: List[int],
    embeddings_in_acts: bool = False,
    test_ratio: float = 0.2,
    seed: int = 42,
    verbose: bool = False,
):
    train_data, test_data = split_dataset(
        acts_dataset=acts_dataset["qs"], test_ratio=test_ratio, verbose=verbose
    )

    # Shuffle the data
    random.shuffle(train_data)
    random.shuffle(test_data)

    results = {
        "loc_type": [],
        "layer": [],
        "accuracy_train": [],
        "accuracy_test": [],
        "y_test": [],
        "y_pred_test": [],
        "probe": [],
    }

    # Train and evaluate logistic regression probes for each layer and loc_type
    for loc_type in locs_to_probe.keys():
        for layer_idx in layers_to_probe:

            X_train, y_train = get_probe_data(
                train_data, locs_to_probe[loc_type], layer_idx, embeddings_in_acts
            )
            X_test, y_test = get_probe_data(
                test_data, locs_to_probe[loc_type], layer_idx, embeddings_in_acts
            )

            # Train the logistic regression probe
            probe = LogisticRegression(random_state=seed, max_iter=1000)
            probe.fit(X_train, y_train)

            # Make predictions
            y_pred_train = probe.predict(X_train)
            y_pred_test = probe.predict(X_test)

            # Calculate metrics
            accuracy_train = accuracy_score(y_train, y_pred_train)
            accuracy_test = accuracy_score(y_test, y_pred_test)

            # Store results
            results["loc_type"].append(loc_type)
            results["layer"].append(layer_idx)
            results["accuracy_train"].append(accuracy_train)
            results["accuracy_test"].append(accuracy_test)
            results["y_test"].append(y_test)
            results["y_pred_test"].append(y_pred_test)
            results["probe"].append(probe)

            if verbose:
                print(f"Location: {loc_type}, Layer {layer_idx}:")
                print(f"  Accuracy (train): {accuracy_train:.4f}")
                print(f"  Accuracy (test): {accuracy_test:.4f}")
                print()

    df_results = pd.DataFrame(results)
    return df_results


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_file_path = Path(args.file)
    if not input_file_path.exists():
        raise FileNotFoundError(f"File not found at {input_file_path}")

    if not input_file_path.name.startswith("acts_"):
        raise ValueError(f"Input file must start with 'acts_', got {input_file_path}")

    with open(input_file_path, "rb") as f:
        acts_dataset = pickle.load(f)

    model_size = acts_dataset["arg_model_size"]
    model, tokenizer = load_model_and_tokenizer(model_size)

    if args.layers:
        layers_to_probe = args.layers.split(",")
    else:
        layers_to_probe = list(range(model.config.num_hidden_layers))

    locs_to_probe = get_locs_to_probe(tokenizer)

    df_results = train_logistic_regression_probes(
        acts_dataset=acts_dataset,
        locs_to_probe=locs_to_probe,
        layers_to_probe=layers_to_probe,
        test_ratio=args.test_ratio,
        embeddings_in_acts=acts_dataset["arg_collect_embeddings"],
        seed=args.seed,
        verbose=args.verbose,
    )

    # Save the results
    output_file_name = input_file_path.name.replace("acts_", "log_reg_probe_results_")
    output_file_path = DATA_DIR / output_file_name
    df_results.to_pickle(output_file_path)

    if args.verbose:
        for loc_type in locs_to_probe.keys():
            # Sort and print layers by lowest mse_test for this loc_type
            df_loc = df_results[df_results["loc_type"] == loc_type]
            df_loc = df_loc.sort_values(by="accuracy_test", ascending=False)
            print(f"Layers sorted by lowest accuracy_test for {loc_type}")
            with pd.option_context("display.max_rows", 2000):
                # Print the layer and accuracy_test, no index
                print(df_loc[["layer", "accuracy_test"]].to_string(index=False))

            top_5_layers_lowest_accuracy_test = df_loc["layer"].iloc[:5].tolist()
            print(f"Top 5 layers: {top_5_layers_lowest_accuracy_test}")


if __name__ == "__main__":
    main(parse_args())
