from collections import Counter
from string import ascii_uppercase

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing.task import Question
from cot_probing.typing import *


def get_answer_index_tokens_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    input_ids: list[int],
    question: Question,
) -> tuple[int, list[int], str]:
    prompt_len = len(input_ids)
    model_output = model.generate(
        torch.tensor([input_ids]).cuda(),
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

    return answer_idx, response.tolist(), decoded_response


@dataclass
class TokenizedQuestion:
    correct_answer: str
    tokenized_question: list[int]

    def __repr__(self):
        return f"TokenizedQuestion({len(self.tokenized_question)} tokens, correct_answer={self.correct_answer})"


def get_correct_incorrect_idxs(
    eval_questions: list[TokenizedQuestion],
) -> tuple[list[int], list[int]]:
    correct_idxs = [i for i, q in enumerate(eval_questions) if q.is_correct]
    incorrect_idxs = [i for i, q in enumerate(eval_questions) if not q.is_correct]
    return correct_idxs, incorrect_idxs
