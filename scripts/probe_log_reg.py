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


def train_logistic_regression_probe(
    loc_type: str,
    layer_idx: int,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    results: Dict,
    seed: int = 42,
    verbose: bool = False,
):
    probe = LogisticRegression(random_state=seed, max_iter=10_000)
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


def train_logistic_regression_probes(
    acts_dataset: Dict,
    tokenizer: PreTrainedTokenizerBase,
    layers_to_probe: List[int],
    test_ratio: float = 0.2,
    seed: int = 42,
    verbose: bool = False,
):
    train_data, test_data = split_dataset(
        acts_dataset=acts_dataset["qs"], test_ratio=test_ratio, verbose=verbose
    )

    dash_tok = tokenizer.encode("-", add_special_tokens=False)[0]
    max_steps = max(
        len(
            [
                tok
                for i, tok in enumerate(cached_tokens)
                if tok == dash_tok
                and tokenizer.decode(cached_tokens[i - 1]).endswith("\n")
            ]
        )
        for q_data in acts_dataset["qs"]
        for cached_tokens in q_data["biased_cots_tokens_to_cache"]
    )

    # Shuffle the data
    random.shuffle(train_data)
    random.shuffle(test_data)

    # We need to build the probe locations after shuffling the data
    train_data_locs_to_probe = get_locs_to_probe(
        tokenizer=tokenizer,
        qs_data=train_data,
        max_steps=max_steps,
        biased_cots_collection_mode=acts_dataset["arg_biased_cots_collection_mode"],
    )
    test_data_locs_to_probe = get_locs_to_probe(
        tokenizer=tokenizer,
        qs_data=test_data,
        max_steps=max_steps,
        biased_cots_collection_mode=acts_dataset["arg_biased_cots_collection_mode"],
    )
    assert train_data_locs_to_probe.keys() == test_data_locs_to_probe.keys()

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
    for loc_type in train_data_locs_to_probe.keys():
        for layer_idx in layers_to_probe:
            X_train, y_train = get_probe_data(
                data_list=train_data,
                loc_pos=train_data_locs_to_probe[loc_type],
                layer_idx=layer_idx,
                biased_cots_collection_mode=acts_dataset[
                    "arg_biased_cots_collection_mode"
                ],
            )
            X_test, y_test = get_probe_data(
                data_list=test_data,
                loc_pos=test_data_locs_to_probe[loc_type],
                layer_idx=layer_idx,
                biased_cots_collection_mode=acts_dataset[
                    "arg_biased_cots_collection_mode"
                ],
            )

            # Check if y_train and y_test have data from both classes
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                # Skip this loc_type and layer_idx if there is no data from both classes for train and test
                continue

            train_logistic_regression_probe(
                loc_type=loc_type,
                layer_idx=layer_idx,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                results=results,
                seed=seed,
                verbose=verbose,
            )

    return results


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

    probing_results = train_logistic_regression_probes(
        acts_dataset=acts_dataset,
        tokenizer=tokenizer,
        layers_to_probe=layers_to_probe,
        test_ratio=args.test_ratio,
        seed=args.seed,
        verbose=args.verbose,
    )
    ret = dict(
        arg_model_size=model_size,
        probing_results=probing_results,
    )

    # Save the results
    output_file_name = input_file_path.name.replace("acts_", "log_reg_probe_results_")
    output_file_path = DATA_DIR / output_file_name
    with open(output_file_path, "wb") as f:
        pickle.dump(ret, f)

    if args.verbose:
        df_results = pd.DataFrame(probing_results)
        for loc_type in df_results["loc_type"].unique():
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
