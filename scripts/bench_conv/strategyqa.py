#!/usr/bin/env python3

import json
import pickle
import uuid
from pathlib import Path

import click

from cot_probing import DATA_DIR
from cot_probing.qs_generation import Question


@click.command()
@click.argument("input_json", type=click.Path(exists=True, path_type=Path))
def main(input_json: Path) -> None:
    """Convert JSON questions dataset to Question format and save as pickle.

    Args:
        input_json: Path to input JSON file
    """
    # Load JSON data
    with open(input_json) as f:
        data = json.load(f)

    # Convert to Question format
    questions: dict[str, Question] = {}

    for item in data:
        # Convert boolean answer to yes/no string
        expected_answer = "yes" if item["answer"] else "no"

        # Create extra data dict
        extra_data = {
            "facts": item["facts"],
            "original_qid": item["qid"],
        }

        # Create Question object
        question = Question(
            question=item["question"].strip(),
            expected_answer=expected_answer,  # type: ignore
            source="strategyqa",
            extra_data=extra_data,
        )

        # Use UUID for new question ID
        q_id = uuid.uuid4().hex
        questions[q_id] = question

    # Save as pickle
    with open(DATA_DIR / "strategyqa.pkl", "wb") as f:
        pickle.dump(questions, f)

    print(f"Converted {len(questions)} questions and saved")


if __name__ == "__main__":
    main()
