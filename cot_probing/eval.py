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
    model_output = model.generate(
        input_ids,
        max_new_tokens=500,
        tokenizer=tokenizer,
        stop_strings=[
            f"best answer is: ({ascii_uppercase[i]}"
            for i in range(len(question.choices))
        ],
    )

    #  model output includes the prompt, so we need to remove it
    response = model_output[0][prompt_len:]

    # Decode reponse and parse the model's answer
    decoded_response = tokenizer.decode(response)
    answer_char = decoded_response[-1]
    if answer_char in ascii_uppercase:
        answer_idx = ascii_uppercase.index(answer_char)
    else:
        answer_idx = -1

    return answer_idx, response[0].tolist(), decoded_response


@dataclass
class EvalQuestion:
    correct_answer: str
    question: str
    tokenized_question: list[int]

    def __repr__(self):
        return f"EvalQuestion({len(self.tokenized_question)} tokens, correct_answer={self.correct_answer}, question={self.question})"


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
