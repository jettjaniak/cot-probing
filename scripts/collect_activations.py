#!/usr/bin/env python3
import argparse
import json
import logging
import os
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from cot_probing import DATA_DIR
from cot_probing.activations import clean_run_with_cache


def parse_args():
    parser = argparse.ArgumentParser(description="Collect activations from a model")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the dataset of labeled questions",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=str,
        default=None,
        help="List of comma separated layers to cache activations for. Defaults to all layers.",
    )
    parser.add_argument(
        "-e",
        "--collect-embeddings",
        action="store_true",
        help="Collect also embeddings",
    )
    parser.add_argument(
        "-b",
        "--biased-cots-collection-mode",
        type=str,
        choices=["none", "one", "all"],
        default="one",
        help="Mode for collecting biased COTs. If one or all is selected, we filter first the biased COTs by the biased_cot_label.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def load_model_and_tokenizer(
    model_size: str,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    assert model_size in ["8B", "70B"]
    model_id = f"hugging-quants/Meta-Llama-3.1-{model_size}-BNB-NF4-BF16"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    return model, tokenizer


def build_fsp_cache(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    fsp: str,
):
    fsp_input_ids = tokenizer(fsp, return_tensors="pt").to("cuda")
    prompt_cache = StaticCache(
        config=model.config,
        max_batch_size=1,
        max_cache_len=fsp_input_ids.shape[1],
        device="cuda",
        dtype=torch.bfloat16,
    )

    with torch.no_grad():
        fsp_cache = model(**fsp_input_ids, past_key_values=prompt_cache).past_key_values

    return fsp_cache


def collect_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    dataset: Dict,
    locs_to_cache: Dict,
    layers_to_cache: List[int],
    biased_cots_collection_mode: Literal["none", "one", "all"],
    collect_embeddings: bool,
):
    activations_by_layer_by_locs = {
        loc_type: [[] for _ in range(layers_to_cache)]
        for loc_type in locs_to_cache.keys()
    }

    biased_no_fsp = dataset["biased_no_fsp"] + "\n\n"
    biased_yes_fsp = dataset["biased_yes_fsp"] + "\n\n"

    # Pre-cache FSP activations
    biased_no_fsp_cache = build_fsp_cache(model, tokenizer, biased_no_fsp)
    biased_yes_fsp_cache = build_fsp_cache(model, tokenizer, biased_yes_fsp)

    for q_data in tqdm.tqdm(dataset["qs"]):
        question_to_answer = q_data["question_to_answer"]
        expected_answer = q_data["expected_answer"]
        biased_cots = q_data["biased_cots"]
        biased_cot_label = q_data["biased_cot_label"]

        # Filter biased COTs based on the biased cot label
        biased_cots = [cot for cot in biased_cots if cot["answer"] != "other"]
        if biased_cot_label == "faithful":
            biased_cots = [
                cot for cot in biased_cots if cot["biased_cot_label"] == expected_answer
            ]
        elif biased_cot_label == "unfaithful":
            biased_cots = [
                cot for cot in biased_cots if cot["biased_cot_label"] != expected_answer
            ]

        assert (
            len(biased_cots) > 0
        ), f"No biased COTs found that match the biased CoT label {biased_cot_label}"

        # Choose the biased FSP and the answer token based on the expected answer
        if expected_answer == "yes":
            biased_fsp_cache = biased_yes_fsp_cache
            answer_tok = tokenizer.encode(" Yes", add_special_tokens=False)
        else:
            biased_fsp_cache = biased_no_fsp_cache
            answer_tok = tokenizer.encode(" No", add_special_tokens=False)

        # Build the prompt
        question_toks = tokenizer.encode(question_to_answer)

        if biased_cots_collection_mode == "one":
            # Pick one random biased COT
            random_biased_cot = random.choice(biased_cots)
            random_biased_cot.tolist()
            input_ids_to_cache = [
                question_toks + random_biased_cot.tolist() + answer_tok
            ]
        elif biased_cots_collection_mode == "all":
            # Collect activations for all biased COTs
            input_ids_to_cache = [
                question_toks + cot.tolist() + answer_tok for cot in biased_cots
            ]
        else:
            # Don't collect activations for biased COTs
            input_ids_to_cache = [question_toks]

        for input_ids in input_ids_to_cache:
            # Figure out where the last question starts
            if "last_question_tokens" in locs_to_cache:
                last_question_token_position = [
                    pos for pos, t in enumerate(input_ids) if t == question_token
                ][-1]
                locs_to_cache["last_question_tokens"] = (
                    last_question_token_position,
                    None,
                )

            resid_acts_by_layer_by_locs = clean_run_with_cache(
                model, input_ids, locs_to_cache, collect_embeddings=collect_embeddings
            )

            for loc_type in locs_to_cache.keys():
                for layer_idx in range(layers_to_cache):
                    activations_by_layer_by_locs[loc_type][layer_idx].append(
                        resid_acts_by_layer_by_locs[loc_type][layer_idx]
                    )

    return activations_by_layer_by_locs


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    input_file_path = args.file
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"File not found at {input_file_path}")

    if not input_file_path.startswith("labeled_qs_"):
        raise ValueError(
            f"Input file must start with 'labeled_qs_', got {input_file_path}"
        )

    with open(input_file_path, "r") as f:
        labeled_questions_dataset = json.load(f)

    model_size = input_file_path.split("_")[2]
    model, tokenizer = load_model_and_tokenizer(model_size)

    if args.layers:
        layers_to_cache = args.layers.split(",")
    else:
        layers_to_cache = list(range(model.config.num_hidden_layers))

    collect_embeddings = args.collect_embeddings

    locs_to_cache = {
        "last_question_tokens": (None, None),
        # "first_cot_dash": (-1, None),  # last token before CoT
        # "last_new_line": (-2, -1),  # newline before first dash in CoT
        # "step_by_step_colon": (-3, -2),  # colon before last new line.
    }

    acts = collect_activations(
        model=model,
        tokenizer=tokenizer,
        dataset=labeled_questions_dataset,
        locs_to_cache=locs_to_cache,
        layers_to_cache=layers_to_cache,
        biased_cots_collection_mode=args.biased_cots_collection_mode,
        collect_embeddings=collect_embeddings,
    )

    output_file_name = input_file_path.replace("labeled_qs_", "acts_").replace(
        ".json", ".pkl"
    )
    output_file_path = DATA_DIR / output_file_name

    with open(output_file_path, "w") as f:
        pickle.dump(acts, f)


if __name__ == "__main__":
    main(parse_args())
