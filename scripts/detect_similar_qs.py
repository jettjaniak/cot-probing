#!/usr/bin/env python3
import argparse
import json
import logging
import os

import tqdm
from openai import OpenAI

from cot_probing import DATA_DIR

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect similar questions in dataset of generated questions"
    )
    parser.add_argument(
        "-o", "--openai-model", type=str, default="gpt-4o", help="OpenAI model"
    )
    parser.add_argument(
        "-t", "--threshold", type=float, default=0.6, help="Threshold for similarity"
    )
    parser.add_argument("-s", "--start-index", type=int, default=0, help="Start index")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "-w",
        "--write",
        action="store_true",
        help="Remove similar questions from dataset and write to file",
    )
    return parser.parse_args()


def get_similarity_score(
    question1: str,
    question2: str,
    openai_model: str,
) -> float:
    prompt = f"""Compare the following two questions and determine if they are similar in meaning or intent. Provide only a similarity score between 0 and 1, where 0 means completely different and 1 means identical or very similar.

Question 1: {question1}
Question 2: {question2}

"""
    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant tasked with comparing the similarity of questions.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=10,
    )

    output = response.choices[0].message.content
    if ":" in output:
        str_score = output.split(":")[1].strip()
    else:
        str_score = output.strip()
    return float(str_score)


def get_keywords(question: str, openai_model: str) -> list[str]:
    prompt = f"""For the following question, provide up to three relevant keywords that capture its main concepts. 
    Respond with just the keywords in lowercase, separated by commas (no spaces after commas).
    
    Question: {question}"""

    response = client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant that extracts keywords from questions.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=50,
    )

    keywords = response.choices[0].message.content.strip().split(",")
    return keywords


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    questions_dataset_path = DATA_DIR / "generated_qs_oct28-1156.json"

    question_dataset = []
    if questions_dataset_path.exists():
        with open(questions_dataset_path, "r") as f:
            question_dataset = json.load(f)
    else:
        raise FileNotFoundError(
            f"Questions dataset file not found at {questions_dataset_path}"
        )

    for i in tqdm.tqdm(range(args.start_index, len(question_dataset))):
        ith_question = (
            question_dataset[i]["question"]
            .split("\nLet's think step by step:")[0]
            .strip()
        )

        # Get keywords only for question i
        keywords = get_keywords(ith_question, args.openai_model)

        for j in range(i + 1, len(question_dataset)):
            jth_question = (
                question_dataset[j]["question"]
                .split("\nLet's think step by step:")[0]
                .strip()
            )

            # Simple text search in jth_question
            if any(kw in jth_question.lower() for kw in keywords):
                # Only get similarity score if they share keywords
                score = get_similarity_score(
                    ith_question, jth_question, args.openai_model
                )

                if score >= args.threshold:
                    if args.verbose:
                        print(
                            f"\nThe following questions ({i} and {j}) have a similarity score of {score}:"
                        )
                        print(ith_question)
                        print(jth_question)

                    if args.write:
                        question_dataset.pop(j)
                        with open(questions_dataset_path, "w") as f:
                            json.dump(question_dataset, f)


if __name__ == "__main__":
    main(parse_args())
