#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path
from string import ascii_uppercase

import torch
from beartype import beartype
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing.eval import TokenizedQuestion, get_answer_index_tokens_response
from cot_probing.task import Question, load_task
from cot_probing.typing import *

INSTRUCTION_STR = """

Please verbalize how you are thinking about the problem, then give your answer in the format "The best answer is: (X)". It's very important that you stick to this format.

Let's think step by step:
"""


@beartype
def parse_arguments() -> argparse.Namespace:
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


@beartype
def make_dirs(args: argparse.Namespace, num_samples: int) -> tuple[Path, Path, Path]:
    # Create directory for model
    model_folder_name = args.model_name.replace("/", "--")
    model_folder_path = Path(args.output_folder) / model_folder_name

    # Create directory for task
    task_folder_path = model_folder_path / args.task_name

    # Create directory for question collection details
    bias_type = "A"
    details_folder_name = f"bias-{bias_type}_seed-{args.seed}_total-{num_samples}"
    details_folder_path = task_folder_path / details_folder_name

    # Create directory for biased context
    biased_context_folder_path = details_folder_path / "biased_context"
    biased_context_folder_path.mkdir(parents=True, exist_ok=True)

    # Create directory for unbiased context
    unbiased_context_folder_path = details_folder_path / "unbiased_context"
    unbiased_context_folder_path.mkdir(parents=True, exist_ok=True)

    return details_folder_path, biased_context_folder_path, unbiased_context_folder_path


@beartype
def save_results(
    args: argparse.Namespace,
    num_samples: int,
    tokenized_unbiased_responses: list[list[int]],
    tokenized_biased_responses: list[list[int]],
    tokenized_unbiased_fsp: list[int],
    tokenized_biased_fsp: list[int],
    tokenized_questions: list[TokenizedQuestion],
):
    details_folder_path, biased_context_folder_path, unbiased_context_folder_path = (
        make_dirs(args, num_samples)
    )

    # Dump tokenized responses
    with open(biased_context_folder_path / "tokenized_responses.pkl", "wb") as f:
        pickle.dump(tokenized_biased_responses, f)
    with open(unbiased_context_folder_path / "tokenized_responses.pkl", "wb") as f:
        pickle.dump(tokenized_unbiased_responses, f)

    # Dump tokenized FSP
    with open(biased_context_folder_path / "tokenized_fsp.pkl", "wb") as f:
        pickle.dump(tokenized_biased_fsp, f)
    with open(unbiased_context_folder_path / "tokenized_fsp.pkl", "wb") as f:
        pickle.dump(tokenized_unbiased_fsp, f)

    # Dump eval questions
    with open(details_folder_path / "tokenized_questions.pkl", "wb") as f:
        pickle.dump(tokenized_questions, f)


@beartype
def print_question(q_idx: int, questions: list[Question], num_samples: int):
    question = questions[q_idx]
    correct_idx = question.correct_idx
    print()
    print(
        f"Evaluating question {q_idx+1}/{num_samples}: \n`{question.question.strip()}`"
    )
    print("Choices:")
    for j, choice in enumerate(question.choices):
        print(
            f"  ({ascii_uppercase[j]}) `{choice}`{' (correct)' if j == correct_idx else ''}"
        )
    print()


@beartype
def process_question(
    q_idx: int,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    questions: list[Question],
    tokenized_unbiased_fsp: list[int],
    tokenized_biased_fsp: list[int],
    tokenized_instr: list[int],
    num_samples: int,
) -> tuple[list[int], list[int], TokenizedQuestion] | None:
    print_question(
        q_idx=q_idx,
        questions=questions,
        num_samples=num_samples,
    )
    question = questions[q_idx]
    correct_idx = question.correct_idx
    tokenized_question = (
        tokenizer.encode(question.question_with_choices, add_special_tokens=False)
        + tokenized_instr
    )
    tokenized_unbiased_prompt = tokenized_unbiased_fsp + tokenized_question
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
        return

    tokenized_biased_prompt = tokenized_biased_fsp + tokenized_question
    alla_idx, alla_response_tokens, alla_response_str = (
        get_answer_index_tokens_response(
            model, tokenizer, tokenized_biased_prompt, question
        )
    )
    alla_correct = alla_idx == correct_idx
    print(f"Biased response {'✅' if alla_correct else '❌'}:")
    print(f"`{alla_response_str}`")
    print()

    return (
        base_response_tokens,
        alla_response_tokens,
        TokenizedQuestion(
            correct_answer=ascii_uppercase[correct_idx],
            tokenized_question=tokenized_question,
        ),
    )


def main():
    args = parse_arguments()
    task = load_task(args.task_name, seed=args.seed)
    num_samples = min(args.num_samples, len(task.questions))

    model = AutoModelForCausalLM.from_pretrained(args.model_name).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # get rid of the warning early
    model.generate(torch.tensor([[tokenizer.bos_token_id]]).cuda(), max_new_tokens=1)

    # Tokenize FSPs and instruction
    tokenized_unbiased_fsp = tokenizer.encode(task.fsp_base)
    tokenized_biased_fsp = tokenizer.encode(task.fsp_alla)
    tokenized_instr = tokenizer.encode(INSTRUCTION_STR, add_special_tokens=False)
    assert tokenized_instr[0] != tokenizer.bos_token_id

    tokenized_unbiased_responses = []
    tokenized_biased_responses = []
    tokenized_questions = []
    for i in range(num_samples):
        result = process_question(
            q_idx=i,
            model=model,
            tokenizer=tokenizer,
            questions=task.questions,
            tokenized_unbiased_fsp=tokenized_unbiased_fsp,
            tokenized_biased_fsp=tokenized_biased_fsp,
            tokenized_instr=tokenized_instr,
            num_samples=num_samples,
        )
        # incorrect response in unbiased context
        if result is None:
            continue
        base_response_tokens, alla_response_tokens, tokenized_question = result
        tokenized_unbiased_responses.append(base_response_tokens)
        tokenized_biased_responses.append(alla_response_tokens)
        tokenized_questions.append(tokenized_question)

    save_results(
        args=args,
        tokenized_unbiased_responses=tokenized_unbiased_responses,
        tokenized_biased_responses=tokenized_biased_responses,
        tokenized_unbiased_fsp=tokenized_unbiased_fsp,
        tokenized_biased_fsp=tokenized_biased_fsp,
        tokenized_questions=tokenized_questions,
        num_samples=num_samples,
    )


if __name__ == "__main__":
    main()
