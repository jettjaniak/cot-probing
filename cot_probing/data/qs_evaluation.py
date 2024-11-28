import math
from dataclasses import dataclass
from typing import Literal

import torch
from tqdm import tqdm
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing.activations import build_fsp_cache
from cot_probing.qs_generation import Question


@dataclass
class NoCotAccuracy:
    acc_by_qid: dict[str, float]
    model: str
    fsp_size: int
    seed: int


def get_no_cot_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_without_cot: str,
    expected_answer: Literal["yes", "no"],
    unbiased_no_cot_cache: tuple,
):
    assert question_without_cot.endswith("Answer:")
    assert "Let's think step by step:" not in question_without_cot
    assert question_without_cot.startswith("Question: ")

    yes_tok_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_tok_id = tokenizer.encode(" No", add_special_tokens=False)[0]

    prompt = tokenizer.encode(question_without_cot, add_special_tokens=False)
    logits = model(
        torch.tensor([prompt]).cuda(),
        past_key_values=unbiased_no_cot_cache,
    ).logits[0, -1]
    yes_logit = logits[yes_tok_id].item()
    no_logit = logits[no_tok_id].item()

    exp_yes = math.exp(yes_logit)
    exp_no = math.exp(no_logit)
    denom = exp_yes + exp_no
    prob_yes = exp_yes / denom
    prob_no = exp_no / denom
    if expected_answer == "yes":
        accuracy = prob_yes
    else:
        accuracy = prob_no

    return accuracy


def evaluate_no_cot_accuracy(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_dataset: dict[str, Question],
    unbiased_fsp_without_cot: str,
    fsp_size: int,
    seed: int,
) -> NoCotAccuracy:
    unbiased_no_cot_cache = build_fsp_cache(model, tokenizer, unbiased_fsp_without_cot)

    results = {}
    for q_id, q in tqdm(question_dataset.items(), desc="Evaluating no-CoT accuracy"):
        question = q.question
        expected_answer = q.expected_answer
        assert question.endswith("?")
        assert "Question: " not in question
        assert "Let's think step by step:" not in question

        no_cot_acc = get_no_cot_accuracy(
            model=model,
            tokenizer=tokenizer,
            question_without_cot=f"{q.with_question_prefix()}\nAnswer:",
            expected_answer=expected_answer,
            unbiased_no_cot_cache=unbiased_no_cot_cache,
        )

        results[q_id] = no_cot_acc

    return NoCotAccuracy(
        acc_by_qid=results,
        model=model.config._name_or_path,
        fsp_size=fsp_size,
        seed=seed,
    )
