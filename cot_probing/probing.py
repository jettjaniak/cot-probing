#!/usr/bin/env python3
from typing import Dict, List, Literal

import numpy as np
import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing.utils import to_str_tokens


def get_locs_to_probe(
    tokenizer: PreTrainedTokenizerBase,
    qs_data: List[Dict],
    max_steps: int,
    biased_cots_collection_mode: Literal["none", "one", "all"],
) -> Dict[
    str, int | List[None | int | tuple[int | None, int | None]]
]:  # If we return a list, it means there is a different loc to probe for each question in that loc_type
    last_part_of_last_question = "?\nLet's think step by step:\n-"
    last_part_of_last_question_tokens = tokenizer.encode(
        last_part_of_last_question, add_special_tokens=False
    )
    str_tokens = to_str_tokens(last_part_of_last_question_tokens, tokenizer)

    locs_to_probe = {}
    if biased_cots_collection_mode == "none":
        # We can only probe for tokens in the last part of the last question.
        # We use negative indices since the activations end there.
        loc = -1
        for str_token in reversed(str_tokens):
            loc_key = f"loc_{loc}_{str_token}"
            locs_to_probe[loc_key] = loc
            loc -= 1

    else:
        question_mark_newline_tok = tokenizer.encode("?\n", add_special_tokens=False)[0]
        dash_tok = tokenizer.encode("-", add_special_tokens=False)[0]
        newline_tok = tokenizer.encode("\n", add_special_tokens=False)[0]
        answer_tok = tokenizer.encode("Answer", add_special_tokens=False)[0]

        # Example question (varying part is the actual question, and the actual reasoning and answer):
        # Question: Did The Truman Show recieve more Oscar nominations than Fargo?
        # Let's think step by step:
        # - The Truman Show received 3 Oscar nominations
        # - Fargo received 7 Oscar nominations
        # - 3 is less than 7
        # Answer: No

        locs_to_probe["loc_q_tok"] = []  # "Question"
        locs_to_probe["loc_colon_after_q_tok"] = []  # ":"
        locs_to_probe["loc_question"] = (
            []
        )  # "Did The Truman Show recieve more Oscar nominations than Fargo"
        locs_to_probe["loc_question_mark_new_line_tok"] = []  # "?\n"
        locs_to_probe["loc_let_tok"] = []  # "Let"
        locs_to_probe["loc_'s_tok"] = []  # "'s"
        locs_to_probe["loc_think_tok"] = []  # "think"
        locs_to_probe["loc_first_step_tok"] = []  # "step"
        locs_to_probe["loc_by_tok"] = []  # "by"
        locs_to_probe["loc_second_step_tok"] = []  # "step"
        locs_to_probe["loc_colon_new_line_tok"] = []  # ":\n"

        # Initialize probe locations for all possible steps
        for i in range(max_steps):
            locs_to_probe[f"loc_cot_step_{i}_dash"] = []
            locs_to_probe[f"loc_cot_step_{i}_reasoning"] = []
            locs_to_probe[f"loc_cot_step_{i}_newline_tok"] = []

        locs_to_probe["loc_answer_tok"] = []  # "Answer"
        locs_to_probe["loc_answer_colon_tok"] = []  # ":"
        locs_to_probe["loc_actual_answer_tok"] = []  # "No" or "Yes"

        for q_data in qs_data:
            for cached_tokens in q_data["biased_cots_tokens_to_cache"]:
                locs_to_probe["loc_q_tok"].append(0)
                assert tokenizer.decode(cached_tokens[0]) == "Question"

                locs_to_probe["loc_colon_after_q_tok"].append(1)
                assert tokenizer.decode(cached_tokens[1]) == ":"

                loc_question_mark = cached_tokens.index(question_mark_newline_tok)
                assert tokenizer.decode(cached_tokens[loc_question_mark]) == "?\n"

                locs_to_probe["loc_question"].append((2, loc_question_mark))
                locs_to_probe["loc_question_mark_new_line_tok"].append(
                    loc_question_mark
                )

                locs_to_probe["loc_let_tok"].append(loc_question_mark + 1)
                assert tokenizer.decode(cached_tokens[loc_question_mark + 1]) == "Let"

                locs_to_probe["loc_'s_tok"].append(loc_question_mark + 2)
                assert tokenizer.decode(cached_tokens[loc_question_mark + 2]) == "'s"

                locs_to_probe["loc_think_tok"].append(loc_question_mark + 3)
                assert (
                    tokenizer.decode(cached_tokens[loc_question_mark + 3]) == " think"
                )

                locs_to_probe["loc_first_step_tok"].append(loc_question_mark + 4)
                assert tokenizer.decode(cached_tokens[loc_question_mark + 4]) == " step"

                locs_to_probe["loc_by_tok"].append(loc_question_mark + 5)
                assert tokenizer.decode(cached_tokens[loc_question_mark + 5]) == " by"

                locs_to_probe["loc_second_step_tok"].append(loc_question_mark + 6)
                assert tokenizer.decode(cached_tokens[loc_question_mark + 6]) == " step"

                locs_to_probe["loc_colon_new_line_tok"].append(loc_question_mark + 7)
                assert tokenizer.decode(cached_tokens[loc_question_mark + 7]) == ":\n"

                # Get positions for answer section
                answer_tok_pos = len(cached_tokens) - 3
                locs_to_probe["loc_answer_tok"].append(answer_tok_pos)
                assert tokenizer.decode(cached_tokens[answer_tok_pos]) == "Answer"

                locs_to_probe["loc_answer_colon_tok"].append(answer_tok_pos + 1)
                assert tokenizer.decode(cached_tokens[answer_tok_pos + 1]) == ":"

                locs_to_probe["loc_actual_answer_tok"].append(answer_tok_pos + 2)
                assert tokenizer.decode(cached_tokens[answer_tok_pos + 2]) in [
                    " Yes",
                    " No",
                ]

                # Find positions for CoT steps
                dash_positions = [
                    i
                    for i, tok in enumerate(cached_tokens)
                    if tok == dash_tok  # A dash token
                    and tokenizer.decode(cached_tokens[i - 1]).endswith(
                        "\n"
                    )  # Preceded by something that ends with a newline
                ]

                # Require at least one reasoning step
                assert (
                    len(dash_positions) >= 1
                ), "Each question must have at least one reasoning step"

                # Replace the hardcoded step processing with dynamic steps
                for dash_idx, dash_pos in enumerate(dash_positions):
                    # Add dash position
                    locs_to_probe[f"loc_cot_step_{dash_idx}_dash"].append(dash_pos)
                    assert tokenizer.decode(cached_tokens[dash_pos]) == "-"

                    # Add reasoning and newline positions
                    next_pos = (
                        dash_positions[dash_idx + 1]
                        if dash_idx + 1 < len(dash_positions)
                        else answer_tok_pos
                    )
                    locs_to_probe[f"loc_cot_step_{dash_idx}_reasoning"].append(
                        (dash_pos + 1, next_pos - 1)
                    )
                    locs_to_probe[f"loc_cot_step_{dash_idx}_newline_tok"].append(
                        next_pos - 1
                    )
                    assert tokenizer.decode(cached_tokens[next_pos - 1]).endswith(
                        "\n"
                    )  # Sometimes just "\n", but it might also be something like ")\n"

                # Fill None for missing steps
                for i in range(len(dash_positions), max_steps):
                    locs_to_probe[f"loc_cot_step_{i}_dash"].append(None)
                    locs_to_probe[f"loc_cot_step_{i}_reasoning"].append(None)
                    locs_to_probe[f"loc_cot_step_{i}_newline_tok"].append(None)

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
    loc_pos: int | List[None | int | tuple[int | None, int | None]],
    layer_idx: int,
    biased_cots_collection_mode: Literal["none", "one", "all"],
):
    X = []
    y = []
    item_counter = 0

    for data in data_list:
        cached_acts_biased_cot_by_layer = data["cached_acts"]
        if biased_cots_collection_mode == "none":
            # We add an extra dimension to the cached_acts tensor to account for the fact that we didn't cache multiple biased COTs, just the question
            cached_acts_biased_cot_by_layer = [
                cached_acts_biased_cot_by_layer[layer_idx]
            ]

        for cached_acts_by_layer in cached_acts_biased_cot_by_layer:
            cached_acts = cached_acts_by_layer[layer_idx]  # Shape is [seq len, d_model]

            if isinstance(loc_pos, list):
                # the loc pos depends on which item we are processing
                loc_pos_for_acts = loc_pos[
                    item_counter
                ]  # This will be None if this item does not have the corresponding loc
                item_counter += 1
            else:
                loc_pos_for_acts = loc_pos

            if loc_pos_for_acts is not None:
                y.append(data["biased_cot_label"])
                if isinstance(loc_pos_for_acts, tuple):
                    X.append(
                        np.array(
                            cached_acts[loc_pos_for_acts[0] : loc_pos_for_acts[1]]
                            .mean(dim=0)  # Average activations for this span of locs
                            .float()
                            .numpy()
                        )
                    )
                else:
                    X.append(np.array(cached_acts[loc_pos_for_acts].float().numpy()))

    X = np.array(X)
    y = np.array(y)

    return X, y
