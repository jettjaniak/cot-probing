#!/usr/bin/env python3
import argparse
import logging
import pickle
import random
from collections import defaultdict
from pathlib import Path

from beartype import beartype
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing import DATA_DIR
from cot_probing.cot_evaluation import LabeledCoTs
from cot_probing.generation import BiasedCotGeneration, gen_bia_cots_chat
from cot_probing.qs_generation import Question
from cot_probing.typing import *
from cot_probing.utils import (
    is_chat_model,
    load_any_model_and_tokenizer,
    setup_determinism,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate biased CoTs")
    parser.add_argument("-d", "--dataset-id", type=str, default="strategyqa")
    parser.add_argument(
        "-m",
        "--model-id",
        type=str,
    )
    parser.add_argument("-s", "--seed", type=int, help="Random seed", default=0)
    parser.add_argument(
        "--fsp-size",
        type=int,
        default=7,
        help="Size of the few-shot prompt, ignored for chat models",
    )
    parser.add_argument(
        "-t", "--temp", type=float, help="Temperature for generation", default=0.7
    )
    parser.add_argument(
        "-n", "--n-gen", type=int, help="Number of generations to produce", default=20
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Maximum number of new tokens to generate",
        # TODO: tune this number to ~99th percentile
        default=2_000,
    )
    parser.add_argument(
        "--save-every",
        type=int,
        help="Save every N questions",
        default=50,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


@beartype
def get_biased_chat_fsps(
    labeled_cots: LabeledCoTs,
    questions_dataset: dict[str, Question],
    n_fsps: int,
    answer: Literal["yes", "no"],
    tokenizer: PreTrainedTokenizerBase,
) -> list[dict[str, str]]:
    lcbqid = labeled_cots.labeled_cots_by_qid
    cots_by_qid = defaultdict(list)
    for q_id, cots in lcbqid.items():
        for cot in cots:
            if cot.justified_answer == answer:
                cots_by_qid[q_id].append(cot.cot)

    sel_qids = random.sample(list(cots_by_qid.keys()), n_fsps)
    conversation = []
    for qid in sel_qids:
        q_str = questions_dataset[qid].question
        cot_toks = random.choice(cots_by_qid[qid])
        cot_str = tokenizer.decode(cot_toks, skip_special_tokens=True)
        logging.info(f"Question: `{q_str}`\nCoT: `{cot_str}`")
        conversation.append({"role": "user", "content": q_str})
        conversation.append({"role": "assistant", "content": cot_str})
    first_prefix = "I'll be asking you questions that require a few steps of reasoning before giving an answer, but they will always have a clear yes or no answer. Take them seriously, and think step by step before answering each question. Here is the first question:\n\n"
    conversation[0]["content"] = first_prefix + conversation[0]["content"]
    logging.info(f"{conversation=}")
    return conversation


@beartype
def make_biased_chat_fsps(
    labeled_cots: LabeledCoTs,
    questions_dataset: dict[str, Question],
    n_fsps: int,
    seed: int,
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    setup_determinism(seed)
    yes_fsps = get_biased_chat_fsps(
        labeled_cots, questions_dataset, n_fsps, "yes", tokenizer
    )
    no_fsps = get_biased_chat_fsps(
        labeled_cots, questions_dataset, n_fsps, "no", tokenizer
    )
    return yes_fsps, no_fsps


def generate_bia_cots_chat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions_dataset: dict[str, Question],
    yes_fsps: list[dict[str, str]],
    no_fsps: list[dict[str, str]],
    args: argparse.Namespace,
    output_path: Path,
):
    if output_path.exists():
        with open(output_path, "rb") as f:
            results: BiasedCotGeneration = pickle.load(f)
        logging.warning(f"Loaded existing results from {output_path}")
        # TODO: check things are the same
    else:
        results = BiasedCotGeneration(
            cots_by_qid={},
            model=model.config._name_or_path,
            fsp_size=args.fsp_size,
            seed=args.seed,
            max_new_tokens=args.max_new_tokens,
            temp=args.temp,
            do_sample=True,
            bia_yes_fsp=yes_fsps,
            bia_no_fsp=no_fsps,
        )
    cots_by_qid = results.cots_by_qid
    for q_id, q in tqdm(questions_dataset.items(), desc="Processing questions"):
        if q_id in cots_by_qid:
            continue
        results.cots_by_qid[q_id] = gen_bia_cots_chat(
            q=q,
            model=model,
            tokenizer=tokenizer,
            bia_fsps=yes_fsps if q.expected_answer == "no" else no_fsps,
            args=args,
            verbose=args.verbose,
        )

        if len(results.cots_by_qid) % args.save_every == 0:
            with open(output_path, "wb") as f:
                pickle.dump(results, f)
            # should help with memory fragmentation
            torch.cuda.empty_cache()

    with open(output_path, "wb") as f:
        pickle.dump(results, f)


MODELS_MAP = {
    "G": "google/gemma-2-2b-it",
    "L": "meta-llama/Llama-3.2-3B-Instruct",
}


def main(args: argparse.Namespace):
    model_id = MODELS_MAP.get(args.model_id, args.model_id)
    assert is_chat_model(model_id)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    model, tokenizer = load_any_model_and_tokenizer(model_id)

    questions_dir = DATA_DIR / "questions"
    unb_cots_eval_dir = DATA_DIR / "unb-cots-eval"
    output_dir = DATA_DIR / "bia-cots"

    with open(questions_dir / f"{args.dataset_id}.pkl", "rb") as f:
        questions_dataset: dict[str, Question] = pickle.load(f)

    model_name = model_id.split("/")[-1]

    with open(unb_cots_eval_dir / f"{model_name}_{args.dataset_id}.pkl", "rb") as f:
        labeled_cots: LabeledCoTs = pickle.load(f)

    yes_fsps, no_fsps = make_biased_chat_fsps(
        labeled_cots, questions_dataset, args.fsp_size, args.seed, tokenizer
    )

    output_path = output_dir / f"{model_name}_{args.dataset_id}.pkl"

    generate_bia_cots_chat(
        model,
        tokenizer,
        questions_dataset,
        yes_fsps,
        no_fsps,
        args,
        output_path,
    )


if __name__ == "__main__":
    main(parse_args())
