#!/usr/bin/env python3
from typing import Dict, List

import numpy as np
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing.utils import to_str_tokens


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
