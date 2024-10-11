from collections import Counter
from string import ascii_uppercase

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing.task import Question
from cot_probing.typing import *


def get_answer_index_tokens_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: torch.Tensor,
    question: Question,
) -> tuple[int, list[int], str]:
    prompt_len = len(input_ids[0])
    response = model.generate(
        input_ids,
        max_new_tokens=500,
        tokenizer=tokenizer,
        stop_strings=[
            f"best answer is: ({ascii_uppercase[i]}"
            for i in range(len(question.choices))
        ],
    )
    decoded_response = tokenizer.decode(response[0][prompt_len:])
    answer_char = decoded_response[-1]
    if answer_char in ascii_uppercase:
        answer_idx = ascii_uppercase.index(answer_char)
    else:
        answer_idx = -1
    return answer_idx, response[0].tolist(), decoded_response


@dataclass
class EvalQuestion:
    tokens: list[int]
    locs: dict[str, list[int]]
    is_correct: bool
    answer_char: str

    def __repr__(self):
        return f"EvalQuestion({len(self.tokens)} tokens, locs keys = {list(self.locs.keys())}, is_correct={self.is_correct}, answer_char={self.answer_char})"


@dataclass
class EvalResults:
    model_name: str
    task_name: str
    seed: int
    num_samples: int
    questions: list[EvalQuestion]

    def __repr__(self):
        return f"EvalResults(model_name={self.model_name}, task_name={self.task_name}, seed={self.seed}, num_samples={self.num_samples}, {len(self.questions)} questions)"


def get_common_tokens(
    eval_questions: list[EvalQuestion], threshold: float = 0.2
) -> list[int]:
    correct_counter = Counter()
    incorrect_counter = Counter()
    n_correct = 0
    n_incorrect = 0

    for q in eval_questions:
        locs = q.locs["response"]
        tokens = q.tokens
        response_tokens = [tokens[loc] for loc in locs]

        if q.is_correct:
            correct_counter.update(response_tokens)
            n_correct += 1
        else:
            incorrect_counter.update(response_tokens)
            n_incorrect += 1

    all_tokens = set(correct_counter.keys()) | set(incorrect_counter.keys())

    return [
        t
        for t in all_tokens
        if (correct_counter[t] > threshold * n_correct)
        and (incorrect_counter[t] > threshold * n_incorrect)
    ]


def get_correct_incorrect_idxs(
    eval_questions: list[EvalQuestion],
) -> tuple[list[int], list[int]]:
    correct_idxs = [i for i, q in enumerate(eval_questions) if q.is_correct]
    incorrect_idxs = [i for i, q in enumerate(eval_questions) if not q.is_correct]
    return correct_idxs, incorrect_idxs
