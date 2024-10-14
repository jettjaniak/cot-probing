#!/usr/bin/env python3
import argparse
import os
import pickle
from string import ascii_uppercase

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing.eval import TokenizedQuestion, get_answer_index_tokens_response
from cot_probing.task import INSTRUCTION_STR, load_task
from cot_probing.typing import *


def parse_arguments():
    # Example usage:
    # python scripts/collect_questions.py --task-name movie_recommendation --model-name google/gemma-2-2b --output-folder /workspace/cot-probing-hf --num-samples 10

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task-name", "-t", type=str, default="snarks", help="Name of the task"
    )
    parser.add_argument("--seed", "-s", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=10,
        help="Number of samples to evaluate",
    )
    parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default="google/gemma-2-2b",
        help="Model name or path",
    )
    parser.add_argument(
        "--output-folder",
        "-o",
        type=str,
        required=True,
        help="Path to the folder where the data is saved",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Check if output folder was provided
    if args.output_folder is None:
        raise ValueError(
            "Please provide an output folder with the flag --output-folder or -o"
        )

    task = load_task(args.task_name, seed=args.seed)
    num_samples = min(args.num_samples, len(task.questions))

    model = AutoModelForCausalLM.from_pretrained(args.model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Create directory for model
    model_folder_name = args.model_name.replace("/", "--")
    model_folder_path = os.path.join(args.output_folder, model_folder_name)

    # Create directory for task
    task_folder_path = os.path.join(model_folder_path, args.task_name)

    # Create directory for question collection details
    bias_type = "A"
    details_folder_name = f"bias-{bias_type}_seed-{args.seed}_total-{num_samples}"
    details_folder_path = os.path.join(task_folder_path, details_folder_name)

    # Create directory for biased context
    biased_context_folder_path = os.path.join(details_folder_path, "biased_context")
    os.makedirs(biased_context_folder_path, exist_ok=True)

    # Create directory for unbiased context
    unbiased_context_folder_path = os.path.join(details_folder_path, "unbiased_context")
    os.makedirs(unbiased_context_folder_path, exist_ok=True)

    tokenized_unbiased_fsp = tokenizer.encode(task.fsp_base)
    tokenized_biased_fsp = tokenizer.encode(task.fsp_alla)
    tokenized_instr = tokenizer.encode(
        f"\n\n{INSTRUCTION_STR}\n", add_special_tokens=False, return_tensors="pt"
    )[0]

    tokenized_unbiased_responses = []
    tokenized_biased_responses = []
    eval_questions = []
    for i in range(num_samples):
        question = task.questions[i]
        correct_idx = question.correct_idx

        # get rid of the warnings early
        model.generate(
            torch.tensor([[tokenizer.bos_token_id]]).cuda(), max_new_tokens=1
        )

        print()
        print(
            f"Evaluating question {i+1}/{num_samples}: \n`{question.question.strip()}`"
        )
        print("Choices:")
        for j, choice in enumerate(question.choices):
            print(
                f"  ({ascii_uppercase[j]}) `{choice}`{' (correct)' if j == correct_idx else ''}"
            )
        print()

        tokenized_question_with_choices = tokenizer.encode(
            question.question_with_choices, add_special_tokens=False
        )

        tokenized_unbiased_prompt = torch.cat(
            (
                tokenized_unbiased_fsp,
                tokenized_question_with_choices,
                tokenized_instr,
            ),
            dim=0,
        )

        base_idx, base_response_tokens, base_response_str = (
            get_answer_index_tokens_response(
                model, tokenizer, tokenized_unbiased_prompt, question
            )
        )
        base_correct = base_idx == correct_idx
        print(f"Unbiased response {'✅' if base_correct else '❌'}:")
        print(f"`{base_response_str}`")
        print()
        if not base_correct:
            print("Wrong answer in base context, skipping...")
            continue

        tokenized_biased_prompt = torch.cat(
            (
                tokenized_biased_fsp,
                tokenized_question_with_choices,
                tokenized_instr,
            ),
            dim=0,
        )
        alla_idx, alla_response_tokens, alla_response_str = (
            get_answer_index_tokens_response(
                model, tokenizer, tokenized_biased_prompt, question
            )
        )
        alla_correct = alla_idx == correct_idx
        print(f"Biased response {'✅' if alla_correct else '❌'}:")
        print(f"`{alla_response_str}`")
        print()

        tokenized_unbiased_responses.append(base_response_tokens)
        tokenized_biased_responses.append(alla_response_tokens)

        eval_questions.append(
            TokenizedQuestion(
                correct_answer=ascii_uppercase[correct_idx],
                tokenized_question=tokenized_question_with_choices,
            )
        )

    # Dump tokenized responses
    with open(
        os.path.join(biased_context_folder_path, "tokenized_responses.pkl"), "wb"
    ) as f:
        pickle.dump(tokenized_biased_responses, f)
    with open(
        os.path.join(unbiased_context_folder_path, "tokenized_responses.pkl"), "wb"
    ) as f:
        pickle.dump(tokenized_unbiased_responses, f)

    # Dump tokenized FSP
    with open(os.path.join(biased_context_folder_path, "tokenized_fsp.pkl"), "wb") as f:
        pickle.dump(tokenized_biased_fsp, f)
    with open(
        os.path.join(unbiased_context_folder_path, "tokenized_fsp.pkl"), "wb"
    ) as f:
        pickle.dump(tokenized_unbiased_fsp, f)

    # Dump eval questions
    with open(os.path.join(details_folder_path, "eval_questions.pkl"), "wb") as f:
        pickle.dump(eval_questions, f)


if __name__ == "__main__":
    main()
