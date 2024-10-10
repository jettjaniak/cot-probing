#!/usr/bin/env python3
import argparse
import pickle
from string import ascii_uppercase

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing.eval import EvalQuestion, EvalResults, get_answer_index_tokens_response
from cot_probing.task import load_task
from cot_probing.typing import *


def parse_arguments():
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
    return parser.parse_args()


def main():
    args = parse_arguments()

    task = load_task(args.task_name, seed=args.seed)
    num_samples = min(args.num_samples, len(task.questions))

    model = AutoModelForCausalLM.from_pretrained(args.model_name).cuda()
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    eval_questions = []

    for i in range(num_samples):
        prompt_base = task.prompts_base[i]
        prompt_alla = task.prompts_alla[i]
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

        enc_prompt_base = tokenizer.encode(prompt_base, return_tensors="pt").cuda()
        prompt_len = len(enc_prompt_base[0])

        base_idx, _base_tokens, base_response_str = get_answer_index_tokens_response(
            model, tokenizer, enc_prompt_base, question
        )
        base_correct = base_idx == correct_idx
        print(f"Unbiased response {'✅' if base_correct else '❌'}:")
        print(f"`{base_response_str}`")
        print()
        if not base_correct:
            print("Wrong answer, skipping...")
            continue

        enc_prompt_alla = tokenizer.encode(prompt_alla, return_tensors="pt").cuda()
        alla_idx, alla_tokens, alla_response_str = get_answer_index_tokens_response(
            model, tokenizer, enc_prompt_alla, question
        )
        alla_correct = alla_idx == correct_idx
        print(f"Biased response {'✅' if alla_correct else '❌'}:")
        print(f"`{alla_response_str}`")
        print()
        eval_questions.append(
            EvalQuestion(
                tokens=alla_tokens,
                locs={"response": list(range(prompt_len, len(alla_tokens)))},
                is_correct=alla_idx == correct_idx,
                answer_char=ascii_uppercase[alla_idx],
            )
        )

    results = EvalResults(
        model_name=args.model_name,
        task_name=args.task_name,
        seed=args.seed,
        num_samples=num_samples,
        questions=eval_questions,
    )
    model_name = args.model_name.replace("/", "--")
    with open(
        f"results/eval_{model_name}_{args.task_name}_S{args.seed}_N{num_samples}.pkl",
        "wb",
    ) as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
