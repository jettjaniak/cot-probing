#!/usr/bin/env python3
import argparse
import logging
import os
import pickle
from pathlib import Path
from typing import Callable

from tqdm import tqdm
from transformers import PreTrainedTokenizerBase

from cot_probing import DATA_DIR
from cot_probing.cot_evaluation import (
    LabeledCot,
    LabeledCoTs,
    evaluate_cots_chat,
    evaluate_cots_pretrained,
)
from cot_probing.generation import BiasedCotGeneration, UnbiasedCotGeneration
from cot_probing.qs_generation import Question
from cot_probing.utils import is_chat_model, load_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evalaute CoTs")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to the pkl of generated CoTs",
    )
    parser.add_argument(
        "-o",
        "--openai-model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model used to evaluate unbiased CoTs",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save every N questions",
        default=50,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def process_cots(
    questions_dataset: dict[str, Question],
    cots_results: UnbiasedCotGeneration | BiasedCotGeneration,
    evaluate_cots_fn: Callable[
        [Question, list[list[int]], PreTrainedTokenizerBase, str, bool],
        list[LabeledCot],
    ],
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
    dataset_id: str,
    cots_generation_file_path: Path,
    output_path: Path,
):

    results = LabeledCoTs(
        labeled_cots_by_qid={},
        model=cots_results.model,
        dataset=dataset_id,
        openai_model=args.openai_model,
        cots_generation_file_name=cots_generation_file_path.name,
        cots_generation_folder=cots_generation_file_path.parent.name,
    )
    for q_id, cots in tqdm(
        cots_results.cots_by_qid.items(), desc="Processing questions"
    ):
        q = questions_dataset[q_id]
        labeled_cots = evaluate_cots_fn(
            q,
            cots,
            tokenizer,
            args.openai_model,
            args.verbose,
        )
        results.labeled_cots_by_qid[q_id] = labeled_cots

        if len(results.labeled_cots_by_qid) % args.save_every == 0:
            with open(output_path, "wb") as f:
                pickle.dump(results, f)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    cots_path = Path(args.file)
    assert cots_path.exists()
    with open(cots_path, "rb") as f:
        cots_results: UnbiasedCotGeneration | BiasedCotGeneration = pickle.load(f)

    questions_dir = DATA_DIR / "questions"
    dataset_id = cots_path.stem.split("_")[-1]
    with open(questions_dir / f"{dataset_id}.pkl", "rb") as f:
        questions_dataset: dict[str, Question] = pickle.load(f)

    output_dir = DATA_DIR / (cots_path.parent.name + "-eval")
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir / cots_path.name

    tokenizer = load_tokenizer(cots_results.model)

    evaluate_cots_fn = (
        evaluate_cots_chat
        if is_chat_model(cots_results.model)
        else evaluate_cots_pretrained
    )

    process_cots(
        questions_dataset=questions_dataset,
        cots_results=cots_results,
        evaluate_cots_fn=evaluate_cots_fn,
        tokenizer=tokenizer,
        args=args,
        dataset_id=dataset_id,
        cots_generation_file_path=cots_path,
        output_path=output_path,
    )


if __name__ == "__main__":
    main(parse_args())
