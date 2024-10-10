#!/usr/bin/env python3
import argparse
import pickle

from cot_probing.eval import EvalResults


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "eval_results_path",
        type=str,
        help="Path to the evaluation results pickle file",
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    eval_results_path = args.eval_results_path

    with open(eval_results_path, "rb") as f:
        eval_results: EvalResults = pickle.load(f)

    model_name = eval_results.model_name
    task_name = eval_results.task_name
    seed = eval_results.seed
    num_samples = eval_results.num_samples
    questions = eval_results.questions

    print(f"Model: {model_name}")
    print(f"Task: {task_name}")
    print(f"Seed: {seed}")
    print(f"Number of samples: {num_samples}")
    print(f"Number of filtered questions: {len(questions)}")
    num_correct = sum(q.is_correct for q in questions)
    print(f"Number correct answers: {num_correct}")
    print(f"Number incorrect answers: {len(questions) - num_correct}")
