#!/usr/bin/env python3
import argparse
import copy
import logging
import os
import pickle
import random
from pathlib import Path

from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing import DATA_DIR
from cot_probing.activations import (
    build_fsp_toks_cache,
    collect_resid_acts_no_pastkv,
    collect_resid_acts_with_pastkv,
)
from cot_probing.cot_evaluation import LabeledCoTs
from cot_probing.generation import BiasedCotGeneration, UnbiasedCotGeneration
from cot_probing.qs_generation import Question
from cot_probing.typing import *
from cot_probing.utils import load_any_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Collect activations from a model")
    parser.add_argument("-d", "--dataset-id", type=str, default="strategyqa")
    parser.add_argument(
        "-m",
        "--model-id",
        type=str,
        default="hugging-quants/Meta-Llama-3.1-8B-BNB-NF4-BF16",
    )
    parser.add_argument(
        "-l",
        "--layers",
        type=str,
        default=None,
        help="List of comma separated layers to cache activations for. Defaults to all layers.",
    )
    parser.add_argument(
        "-b",
        "--biased-cots-collection-mode",
        type=str,
        choices=["none", "one", "all"],
        default="all",
        help="Mode for collecting biased COTs. If one or all is selected, we filter first the biased COTs by the biased_cot_label.",
    )
    parser.add_argument(
        "-c",
        "--context",
        type=str,
        choices=["biased-fsp", "unbiased-fsp", "no-fsp"],
        default="biased-fsp",
        help="FSP context for the activations to collect.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save every N questions",
        default=50,
    )
    return parser.parse_args()


def get_last_q_toks_to_cache(
    tokenizer: PreTrainedTokenizerBase,
    question: str,
    bia_cots: list[list[int]],
    biased_cots_collection_mode: Literal["none", "one", "all"],
    fsp_context: Literal["biased-fsp", "unbiased-fsp", "no-fsp"],
):
    """May or may not include the CoT tokens"""
    question_toks = tokenizer.encode(
        question, add_special_tokens=fsp_context == "no-fsp"
    )

    if biased_cots_collection_mode == "none":
        # Don't collect activations for biased COTs
        return [question_toks]

    for cot in bia_cots:
        cot_str = tokenizer.decode(cot)
        assert "Question: " not in cot_str, f"Found 'Question: ' in {cot_str}"

    biased_cot_indexes_to_cache = []
    if biased_cots_collection_mode == "one":
        # Pick one random biased COT
        biased_cot_indexes_to_cache = [random.randint(0, len(bia_cots) - 1)]
    elif biased_cots_collection_mode == "all":
        # Collect activations for all biased COTs
        biased_cot_indexes_to_cache = list(range(len(bia_cots)))

    input_ids_to_cache = [
        question_toks + bia_cots[cot_idx] for cot_idx in biased_cot_indexes_to_cache
    ]

    return input_ids_to_cache


def collect_activations_for_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    q: Question,
    q_id: str,
    layers: list[int],
    bia_cots: list[list[int]],
    unbiased_fsp_cache: tuple,
    biased_no_fsp_cache: tuple,
    biased_yes_fsp_cache: tuple,
    biased_cots_collection_mode: Literal["none", "one", "all"],
    fsp_context: Literal["biased-fsp", "unbiased-fsp", "no-fsp"],
):
    question = q.with_step_by_step_suffix()
    assert question.endswith("by step:\n-"), question

    expected_answer = q.expected_answer

    last_q_toks_to_cache = get_last_q_toks_to_cache(
        tokenizer=tokenizer,
        question=question,
        bia_cots=bia_cots,
        biased_cots_collection_mode=biased_cots_collection_mode,
        fsp_context=fsp_context,
    )

    if fsp_context == "biased-fsp":
        # Choose the biased FSP based on the expected answer
        if expected_answer == "yes":
            fsp_cache = biased_no_fsp_cache
        else:
            fsp_cache = biased_yes_fsp_cache
    elif fsp_context == "unbiased-fsp":
        fsp_cache = unbiased_fsp_cache
    elif fsp_context == "no-fsp":
        fsp_cache = None
    else:
        raise ValueError(f"Invalid FSP context: {fsp_context}")

    resid_acts_by_layer_by_cot = []
    for last_q_toks in last_q_toks_to_cache:
        if fsp_cache is None:
            resid_acts_by_layer = collect_resid_acts_no_pastkv(
                model=model,
                all_input_ids=last_q_toks,
                layers=layers,
            )
        else:
            resid_acts_by_layer = collect_resid_acts_with_pastkv(
                model=model,
                last_q_toks=last_q_toks,
                layers=layers,
                past_key_values=fsp_cache,
            )
        resid_acts_by_layer_by_cot.append(resid_acts_by_layer)
    if biased_cots_collection_mode == "none":
        assert len(resid_acts_by_layer_by_cot) == 1
        resid_acts = resid_acts_by_layer_by_cot[0]
    else:
        resid_acts = resid_acts_by_layer_by_cot

    return {
        "question": question,
        "q_id": q_id,
        "expected_answer": expected_answer,
        "biased_cots_tokens_to_cache": last_q_toks_to_cache,
        "cached_acts": resid_acts,
    }


def get_layer_file_path(output_dir: Path, output_file_stem: str, layer: int) -> Path:
    return output_dir / f"acts_L{layer:02d}_{output_file_stem}.pkl"


def load_existing_results(
    output_dir: Path, output_file_stem: str, layers: list[int]
) -> tuple[dict, int]:
    """
    Loads existing results for all layers and returns them along with the last consistent index.
    Returns (layer_results, last_processed_index)
    """
    layer_results = {}
    last_indices = []

    for layer in layers:
        layer_file = get_layer_file_path(output_dir, output_file_stem, layer)
        print(f"Looking to load layer {layer} from {layer_file}")
        if os.path.exists(layer_file):
            try:
                with open(layer_file, "rb") as f:
                    layer_data = pickle.load(f)
                    layer_results[layer] = layer_data["qs"]
                    last_indices.append(len(layer_data["qs"]))
                    print(f"Loaded layer {layer} with {len(layer_data['qs'])} elements")
            except (EOFError, pickle.UnpicklingError):
                print(
                    f"Error loading layer {layer} from {layer_file}. Creating empty list."
                )
                layer_results[layer] = []
                last_indices.append(0)
        else:
            print(f"Layer {layer} file does not exist. Creating empty list.")
            layer_results[layer] = []
            last_indices.append(0)

    # Get minimum index across all layers
    start_idx = min(last_indices) if len(last_indices) > 0 else 0

    # Remove layer results on all layers after the start index
    for layer in layer_results:
        layer_results[layer] = layer_results[layer][:start_idx]

    assert len(layer_results) == len(layers)
    for layer in layers:
        assert (
            len(layer_results[layer]) == start_idx
        ), f"Layer {layer} has {len(layer_results[layer])} elements but should have {start_idx}"

    return layer_results, start_idx


def collect_activations(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    qs_dataset: dict[str, Question],
    unb_cots_results: UnbiasedCotGeneration,
    bia_cots_results: BiasedCotGeneration,
    bia_cots_eval_results: LabeledCoTs,
    layers: list[int],
    biased_cots_collection_mode: Literal["none", "one", "all"],
    args: argparse.Namespace,
    output_file_stem: str,
    output_dir: Path,
):
    unb_fsp_toks = unb_cots_results.unb_fsp_toks
    bia_no_fsp_toks = bia_cots_results.bia_no_fsp  # Should be list[int]
    assert isinstance(bia_no_fsp_toks, list)
    bia_yes_fsp_toks = bia_cots_results.bia_yes_fsp  # Should be list[int]
    assert isinstance(bia_yes_fsp_toks, list)

    # Pre-cache FSP activations
    unbiased_fsp_cache = build_fsp_toks_cache(model, tokenizer, unb_fsp_toks)
    biased_no_fsp_cache = build_fsp_toks_cache(model, tokenizer, bia_no_fsp_toks)
    biased_yes_fsp_cache = build_fsp_toks_cache(model, tokenizer, bia_yes_fsp_toks)

    # Load existing results and get last processed index in one pass
    layer_results, start_idx = load_existing_results(
        output_dir, output_file_stem, layers
    )

    q_ids_to_process = list(bia_cots_results.cots_by_qid.keys())

    # Create progress bar for remaining questions
    remaining_q_ids = q_ids_to_process[start_idx:]
    if start_idx > 0:
        print(f"Resuming from question {start_idx}")

    last_question_fully_processed = True

    try:
        for idx, q_id in tqdm(enumerate(remaining_q_ids), "Processing questions"):
            res = collect_activations_for_question(
                model=model,
                tokenizer=tokenizer,
                q=qs_dataset[q_id],
                q_id=q_id,
                bia_cots=bia_cots_results.cots_by_qid[q_id],
                layers=layers,
                unbiased_fsp_cache=unbiased_fsp_cache,
                biased_no_fsp_cache=biased_no_fsp_cache,
                biased_yes_fsp_cache=biased_yes_fsp_cache,
                biased_cots_collection_mode=biased_cots_collection_mode,
                fsp_context=args.context,
            )

            # Create temporary dict for new results
            new_layer_results = {}
            for layer in layers:
                layer_res = copy.deepcopy(res)
                layer_res["cached_acts"] = (
                    res["cached_acts"][layer]
                    if isinstance(res["cached_acts"], dict)
                    else [r[layer] for r in res["cached_acts"]]
                )
                new_layer_results[layer] = layer_res

            # Atomic update of all layers at once.
            # Getting interrupted before this block is not a problem: we lose one question but we save everything else.
            # Getting interrupted after (i.e., while saving the layer results) this block is not a problem: we are going to save everything again anyway.
            # Getting interrupted in the middle of this block is not a problem: we take the minimum length across layers and remove the extra element.
            last_question_fully_processed = False
            for layer in layers:
                layer_results[layer].append(new_layer_results[layer])
            last_question_fully_processed = True

            # Save if we hit the frequency or it's the last item
            if (idx + 1) % args.save_every == 0 or idx == len(remaining_q_ids) - 1:
                save_layer_results(
                    layer_results=layer_results,
                    layers=layers,
                    unbiased_fsp=unb_fsp_toks,
                    biased_no_fsp=bia_no_fsp_toks,
                    biased_yes_fsp=bia_yes_fsp_toks,
                    args=args,
                    output_file_stem=output_file_stem,
                    output_dir=output_dir,
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user. Saving current progress...")

        if not last_question_fully_processed:
            # Get minimum number of elements across all layers and remove the extra elements
            min_len = min([len(layer_results[layer]) for layer in layers])
            for layer in layers:
                layer_results[layer] = layer_results[layer][:min_len]

        # Save one final time regardless of save_frequency
        save_layer_results(
            layer_results=layer_results,
            layers=layers,
            unbiased_fsp=unb_fsp_toks,
            biased_no_fsp=bia_no_fsp_toks,
            biased_yes_fsp=bia_yes_fsp_toks,
            args=args,
            output_file_stem=output_file_stem,
            output_dir=output_dir,
        )
        raise  # Re-raise the interrupt to exit the program


def save_layer_results(
    layer_results: dict[int, dict],
    layers: list[int],
    unbiased_fsp: list[int],
    biased_no_fsp: list[int],
    biased_yes_fsp: list[int],
    args: argparse.Namespace,
    output_file_stem: str,
    output_dir: Path,
):
    """Helper function to save results for all layers"""
    for layer in layers:
        skip_args = ["verbose", "file", "layers"]
        layer_output = {
            "unbiased_fsp": unbiased_fsp,
            "biased_no_fsp": biased_no_fsp,
            "biased_yes_fsp": biased_yes_fsp,
            "qs": layer_results[layer],
            **{f"arg_{k}": v for k, v in vars(args).items() if k not in skip_args},
        }

        layer_file = get_layer_file_path(output_dir, output_file_stem, layer)
        with open(layer_file, "wb") as f:
            pickle.dump(layer_output, f)


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    model, tokenizer = load_any_model_and_tokenizer(args.model_id)

    questions_dir = DATA_DIR / "questions"
    unb_cots_dir = DATA_DIR / "unb-cots"
    bia_cots_dir = DATA_DIR / "bia-cots"
    bia_cots_eval_dir = DATA_DIR / "bia-cots-eval"
    activations_dir = Path("activations")

    model_name = args.model_id.split("/")[-1]
    output_name = f"{model_name}_{args.dataset_id}"
    output_dir = activations_dir / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(questions_dir / f"{args.dataset_id}.pkl", "rb") as f:
        questions_dataset: dict[str, Question] = pickle.load(f)

    with open(unb_cots_dir / f"{output_name}.pkl", "rb") as f:
        unb_cots_results: UnbiasedCotGeneration = pickle.load(f)

    with open(bia_cots_dir / f"{output_name}.pkl", "rb") as f:
        bia_cots_results: BiasedCotGeneration = pickle.load(f)

    with open(bia_cots_eval_dir / f"{output_name}.pkl", "rb") as f:
        bia_cots_eval_results: LabeledCoTs = pickle.load(f)

    if args.layers:
        layers = [int(l) for l in args.layers.split(",")]
    else:
        layers = list(range(model.config.num_hidden_layers + 1))

    if args.context == "no-fsp":
        output_file_stem = "no-fsp"
    elif args.context == "unbiased-fsp":
        output_file_stem = "unbiased-fsp"
    elif args.context == "biased-fsp":
        output_file_stem = "biased-fsp"

    collect_activations(
        model=model,
        tokenizer=tokenizer,
        qs_dataset=questions_dataset,
        unb_cots_results=unb_cots_results,
        bia_cots_results=bia_cots_results,
        bia_cots_eval_results=bia_cots_eval_results,
        layers=layers,
        biased_cots_collection_mode=args.biased_cots_collection_mode,
        args=args,
        output_file_stem=output_file_stem,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main(parse_args())
