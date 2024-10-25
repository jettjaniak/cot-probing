#!/usr/bin/env python3
import argparse
import json
import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing import DATA_DIR
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.questions_generation import generate_questions_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Generate questions dataset")
    parser.add_argument("-s", "--size", type=int, default=8, help="Model size")
    parser.add_argument(
        "-o", "--openai-model", type=str, default="gpt-4o", help="OpenAI model"
    )
    parser.add_argument(
        "-n",
        "--num-questions",
        type=int,
        default=1,
        help="Number of questions to generate",
    )
    parser.add_argument(
        "-a", "--max-attempts", type=int, default=100, help="Maximum number of attempts"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--fsp-size",
        type=int,
        default=7,
        help="Size of the few-shot prompt.",
    )
    parser.add_argument(
        "--expected-completion-accuracy-in-unbiased-context",
        type=float,
        default=0.8,
        help="Expected accuracy in unbiased context.",
    )
    parser.add_argument(
        "--expected-completion-accuracy-in-biased-context",
        type=float,
        default=0.5,
        help="Expected accuracy in biased context.",
    )
    parser.add_argument(
        "--expected-cot-accuracy-in-unbiased-context",
        type=float,
        default=0.8,
        help="Expected chain-of-thought accuracy in unbiased context.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    model_id = f"hugging-quants/Meta-Llama-3.1-{args.size}B-BNB-NF4-BF16"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )

    # To avoid warnings
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    questions_dataset_path = DATA_DIR / "generated_questions_dataset.json"

    question_dataset = []
    if questions_dataset_path.exists():
        with open(questions_dataset_path, "r") as f:
            question_dataset = json.load(f)

    all_qs_yes = load_and_process_file(DATA_DIR / "diverse_yes.txt")
    all_qs_no = load_and_process_file(DATA_DIR / "diverse_no.txt")
    assert len(all_qs_yes) == len(all_qs_no)

    # Add questions to all_qs_yes and all_qs_no so that we don't repeat them
    for row in question_dataset:
        if row["expected_answer"] == "yes":
            all_qs_yes.append(row["question"])
        else:
            all_qs_no.append(row["question"])

    # Generate the dataset
    generate_questions_dataset(
        model=model,
        tokenizer=tokenizer,
        openai_model=args.openai_model,
        num_questions=args.num_questions,
        all_qs_yes=all_qs_yes,
        all_qs_no=all_qs_no,
        max_attempts=args.max_attempts,
        questions_dataset_path=questions_dataset_path,
        fsp_size=args.fsp_size,
        expected_completion_accuracy_in_unbiased_context=args.expected_completion_accuracy_in_unbiased_context,
        expected_completion_accuracy_in_biased_context=args.expected_completion_accuracy_in_biased_context,
        expected_cot_accuracy_in_unbiased_context=args.expected_cot_accuracy_in_unbiased_context,
    )

    for question in question_dataset:
        print(question["question"])
        print()


if __name__ == "__main__":
    main(parse_args())
