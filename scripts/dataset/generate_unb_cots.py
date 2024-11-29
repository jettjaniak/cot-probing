#!/usr/bin/env python3
import argparse
import logging
import pickle
from pathlib import Path

from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing import DATA_DIR
from cot_probing.cot_evaluation import evaluate_cots
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.generation import (
    UnbiasedCotGeneration,
    gen_unb_cots,
    gen_unb_cots_chat,
)
from cot_probing.qs_evaluation import NoCotAccuracy
from cot_probing.qs_generation import Question, generate_unbiased_few_shot_prompt
from cot_probing.utils import is_chat_model, load_any_model_and_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Generate and evaluate unbiased CoTs")
    parser.add_argument("-d", "--dataset-id", type=str, default="strategyqa")
    parser.add_argument(
        "-m",
        "--model-id",
        type=str,
        default="hugging-quants/Meta-Llama-3.1-8B-BNB-NF4-BF16",
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
        default=200,
    )
    parser.add_argument(
        "--max-no-cot-acc",
        type=float,
        help="Maximum no-CoT accuracy to generate unbiased CoTs for",
        default=0.6,
    )
    parser.add_argument(
        "-o",
        "--openai-model",
        type=str,
        default="gpt-4o",
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


def generate_unb_cots_pretrained(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions_dataset: dict[str, Question],
    no_cot_acc: NoCotAccuracy,
    args: argparse.Namespace,
    output_path: Path,
):
    # Build the FSP
    yes_fsps = load_and_process_file(DATA_DIR / "diverse_qs_expected_yes_with_cot.txt")
    no_fsps = load_and_process_file(DATA_DIR / "diverse_qs_expected_no_with_cot.txt")
    unb_fsp = generate_unbiased_few_shot_prompt(
        all_qs_yes=yes_fsps,
        all_qs_no=no_fsps,
        fsp_size=args.fsp_size,
        verbose=args.verbose,
    )
    unb_fsp_toks = tokenizer.encode(unb_fsp, add_special_tokens=True)

    results = UnbiasedCotGeneration(
        cots_by_qid={},
        model=model.config._name_or_path,
        model_size=args.model_size,
        fsp_size=args.fsp_size,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temp=args.temp,
        do_sample=True,
        unb_fsp_toks=unb_fsp_toks,
    )
    for q_id, q in tqdm(questions_dataset.items(), desc="Processing questions"):
        if q_id not in no_cot_acc.acc_by_qid:
            continue

        q_no_cot_acc = no_cot_acc.acc_by_qid[q_id]
        if q_no_cot_acc > args.max_no_cot_acc:
            continue

        unb_cots = gen_unb_cots(
            q=q,
            model=model,
            tokenizer=tokenizer,
            unb_fsp_toks=unb_fsp_toks,
            args=args,
            verbose=args.verbose,
        )
        labeled_unb_cots = evaluate_cots(
            q=q,
            cots=unb_cots,
            tokenizer=tokenizer,
            openai_model=args.openai_model,
            verbose=args.verbose,
        )
        results.cots_by_qid[q_id] = labeled_unb_cots

        if len(results.cots_by_qid) % args.save_every == 0:
            with open(output_path, "wb") as f:
                pickle.dump(results, f)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)


def generate_unb_cots_chat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions_dataset: dict[str, Question],
    no_cot_acc: NoCotAccuracy,
    args: argparse.Namespace,
    output_path: Path,
):

    results = UnbiasedCotGeneration(
        cots_by_qid={},
        model=model.config._name_or_path,
        model_size=args.model_size,
        fsp_size=args.fsp_size,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
        temp=args.temp,
        do_sample=True,
        unb_fsp_toks=[],
    )
    for q_id, q in tqdm(questions_dataset.items(), desc="Processing questions"):
        if q_id not in no_cot_acc.acc_by_qid:
            continue

        q_no_cot_acc = no_cot_acc.acc_by_qid[q_id]
        if q_no_cot_acc > args.max_no_cot_acc:
            continue

        unb_cots = gen_unb_cots_chat(
            q=q,
            model=model,
            tokenizer=tokenizer,
            args=args,
            verbose=args.verbose,
        )
        labeled_unb_cots = evaluate_cots(
            q=q,
            cots=unb_cots,
            tokenizer=tokenizer,
            openai_model=args.openai_model,
            verbose=args.verbose,
        )
        results.cots_by_qid[q_id] = labeled_unb_cots

        if len(results.cots_by_qid) % args.save_every == 0:
            with open(output_path, "wb") as f:
                pickle.dump(results, f)

    with open(output_path, "wb") as f:
        pickle.dump(results, f)


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    model, tokenizer = load_any_model_and_tokenizer(args.model_id)

    questions_dir = DATA_DIR / "questions"
    no_cot_acc_dir = DATA_DIR / "no-cot-accuracy"
    output_dir = DATA_DIR / "unb-cots"

    with open(questions_dir / f"{args.dataset_id}.pkl", "rb") as f:
        questions_dataset = pickle.load(f)

    model_name = args.model_id.split("/")[-1]
    with open(no_cot_acc_dir / f"{model_name}_{args.dataset_id}.pkl", "rb") as f:
        no_cot_acc: NoCotAccuracy = pickle.load(f)
        assert no_cot_acc.model == model.config._name_or_path

    output_path = output_dir / f"{model_name}_{args.dataset_id}.pkl"

    if is_chat_model(args.model_id):
        generate_unb_cots_chat(
            model=model,
            tokenizer=tokenizer,
            questions_dataset=questions_dataset,
            no_cot_acc=no_cot_acc,
            args=args,
            output_path=output_path,
        )
    else:
        generate_unb_cots_pretrained(
            model=model,
            tokenizer=tokenizer,
            questions_dataset=questions_dataset,
            no_cot_acc=no_cot_acc,
            args=args,
            output_path=output_path,
        )


if __name__ == "__main__":
    main(parse_args())
