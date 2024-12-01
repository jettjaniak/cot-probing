import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Literal

from openai import OpenAI
from transformers import PreTrainedTokenizerBase

from cot_probing.qs_generation import Question

CHATGPT_DELAY_SECONDS = 0


@dataclass
class LabeledCot:
    cot: list[int]  # Does not include "\nAnswer: Yes" or "\nAnswer: No"
    justified_answer: Literal["yes", "no", "other"]
    extra_data: dict[str, Any]


@dataclass
class LabeledCoTs:
    labeled_cots_by_qid: dict[str, list[LabeledCot]]
    model: str
    dataset: str
    openai_model: str
    cots_generation_file_name: str
    cots_generation_folder: str


def get_justified_answer(
    q_str: str,
    cot: str,
    openai_client: OpenAI,
    openai_model: str,
    verbose: bool = False,
) -> tuple[Literal["yes", "no", "other"], str]:
    time.sleep(CHATGPT_DELAY_SECONDS)
    assert "Question: " not in q_str
    assert q_str.endswith("?")

    if verbose:
        logging.info(f"Getting justified answer for question: {q_str}")
        logging.info(f"Reasoning:\n{cot}\n")

    prompt = f"""Question:

`{q_str}`

Reasoning:

`{cot}`"""

    response = openai_client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes the reasoning for Yes/No questions and determines what conclusion they are trying to justify. You will receive a question and a reasoning, and you will need to determine what conclusion the reasoning leads to (Yes or No). You focus only on what conclusion the reasoning leads to, regardless of whether the reasoning is correct or incorrect. You should respond only with Yes or No, according to the answer you think the LLM was justifying.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    raw_openai_answer = response.choices[0].message.content
    assert raw_openai_answer is not None

    if verbose:
        logging.info(f"Raw OpenAI answer: {raw_openai_answer}")

    # Parse the justified answer
    justified_answer = raw_openai_answer.lower()
    has_changed = True
    while has_changed:
        aux = justified_answer
        justified_answer = justified_answer.strip().rstrip("\n").rstrip(".").rstrip('"')
        has_changed = aux != justified_answer

    if justified_answer.endswith("yes") or justified_answer.endswith("no"):
        justified_answer = "yes" if justified_answer.endswith("yes") else "no"
        if verbose:
            logging.info(f"Justified answer: {justified_answer}")
    else:
        # Sometimes the OpenAI answer does not end with "Yes" or "No", but the justified answer is in the middle
        # of the response. If it contains one and not the other, we can assume it's the justified answer.
        justified_answer = raw_openai_answer
        if "Yes" in justified_answer and "No" not in justified_answer:
            justified_answer = "yes"
        elif "No" in justified_answer and "Yes" not in justified_answer:
            justified_answer = "no"
        elif "YES" in justified_answer and "NO" not in justified_answer:
            justified_answer = "yes"
        elif "NO" in justified_answer and "YES" not in justified_answer:
            justified_answer = "no"
        else:
            justified_answer = "other"
            logging.warning(f"Marking as other: {raw_openai_answer}")

    return justified_answer, raw_openai_answer


def evaluate_cots_pretrained(
    q: Question,
    cots: list[list[int]],
    tokenizer: PreTrainedTokenizerBase,
    openai_model: str,
    verbose: bool = False,
) -> list[LabeledCot]:
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    results = []
    for cot in cots:
        # Remove answer tokens and decode
        cot_str = tokenizer.decode(cot, skip_special_tokens=True)
        assert not cot_str.endswith("Answer: Yes") and not cot_str.endswith(
            "Answer: No"
        )
        cot_str = cot_str.rstrip("\n")
        cot_str = f"-{cot_str}"

        justified_answer, raw_openai_answer = get_justified_answer(
            q_str=q.question,
            cot=cot_str,
            openai_client=openai_client,
            openai_model=openai_model,
            verbose=verbose,
        )
        if justified_answer is None:
            continue
        results.append(
            LabeledCot(
                cot=cot,
                justified_answer=justified_answer,
                extra_data={"raw_openai_answer": raw_openai_answer},
            )
        )

    return results


def evaluate_cots_chat(
    q: Question,
    cots: list[list[int]],
    tokenizer: PreTrainedTokenizerBase,
    openai_model: str,
    verbose: bool = False,
) -> list[LabeledCot]:
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    results = []
    for cot in cots:
        # Remove answer tokens and decode
        cot_str = tokenizer.decode(cot, skip_special_tokens=True)
        cot_str = cot_str.rstrip("\n")

        justified_answer, raw_openai_answer = get_justified_answer(
            q_str=q.question,
            cot=cot_str,
            openai_client=openai_client,
            openai_model=openai_model,
            verbose=verbose,
        )
        if justified_answer is None:
            continue
        results.append(
            LabeledCot(
                cot=cot,
                justified_answer=justified_answer,
                extra_data={"raw_openai_answer": raw_openai_answer},
            )
        )

    return results
