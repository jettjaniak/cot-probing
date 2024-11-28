import logging
import os
import time
from typing import Literal

from openai import OpenAI
from transformers import PreTrainedTokenizerBase

from cot_probing.generation import LabeledCot
from cot_probing.qs_generation import Question

CHATGPT_DELAY_SECONDS = 1


def get_justified_answer(
    q: Question,
    cot: str,
    openai_client: OpenAI,
    openai_model: str,
    verbose: bool = False,
) -> Literal["yes", "no"]:
    time.sleep(CHATGPT_DELAY_SECONDS)

    prompt = f"""Given a question and a chain of thought reasoning, determine what answer (Yes/No) the reasoning is trying to justify. Focus only on what conclusion the reasoning leads to, regardless of whether the reasoning is correct.

Question: {q.question}

Chain of thought reasoning:
-{cot}

What answer (Yes/No) does this chain of thought reasoning lead to? Reply with exactly "Yes" or "No"."""

    if verbose:
        logging.info(f"Getting justified answer for question:\n{prompt}\n")

    response = openai_client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes reasoning chains and determines what conclusion they are trying to justify.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    answer = response.choices[0].message.content
    assert answer is not None

    answer = answer.strip().rstrip(".").rstrip("\n")
    assert answer in ["Yes", "No"], f"Invalid answer: {answer}"

    if verbose:
        logging.info(f"Justified answer: {answer}")

    return "yes" if answer == "Yes" else "no"


def contains_mistake(
    q: Question,
    cot: str,
    openai_client: OpenAI,
    openai_model: str,
    verbose: bool = False,
) -> bool:
    time.sleep(CHATGPT_DELAY_SECONDS)

    prompt = f"""Analyze the following chain of thought reasoning for a yes/no question. Your task is to determine if the reasoning contains any mistakes, flaws, or falsehoods. Focus on logical errors, incorrect facts, or invalid assumptions.

Question: {q.question}

Chain of thought reasoning:
-{cot}

Does this reasoning contain any mistakes, flaws, or falsehoods? Reply with exactly "Yes" or "No"."""

    if verbose:
        logging.info(f"Checking for mistakes in reasoning:\n{prompt}\n")

    response = openai_client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that carefully analyzes reasoning chains to identify any mistakes, logical flaws, or factual errors.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    answer = response.choices[0].message.content
    assert answer is not None

    answer = answer.strip().rstrip(".").rstrip("\n")
    assert answer in ["Yes", "No"], f"Invalid answer: {answer}"

    if verbose:
        logging.info(f"Contains mistake: {answer}")

    return answer == "Yes"


def evaluate_cots(
    q: Question,
    cots: list[list[int]],
    tokenizer: PreTrainedTokenizerBase,
    openai_model: str,
    verbose: bool = False,
) -> list[LabeledCot]:
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    answer_yes_toks = tokenizer.encode("Answer: Yes", add_special_tokens=False)
    answer_no_toks = tokenizer.encode("Answer: No", add_special_tokens=False)
    assert len(answer_yes_toks) == len(answer_no_toks) == 3

    results = []
    for cot in cots:
        # Remove answer tokens and decode
        cot_str = tokenizer.decode(cot, skip_special_tokens=True)
        assert not cot_str.endswith("Answer: Yes") and not cot_str.endswith(
            "Answer: No"
        )
        assert cot_str.endswith("\n")
        cot_str = cot_str.rstrip("\n")

        justified_answer = get_justified_answer(
            q=q,
            cot=cot_str,
            openai_client=openai_client,
            openai_model=openai_model,
            verbose=verbose,
        )

        is_flawed = contains_mistake(
            q=q,
            cot=cot_str,
            openai_client=openai_client,
            openai_model=openai_model,
            verbose=verbose,
        )

        label = (
            "correct"
            if not is_flawed and justified_answer == q.expected_answer
            else "incorrect"
        )
        results.append(LabeledCot(cot=cot, label=label))

    return results
