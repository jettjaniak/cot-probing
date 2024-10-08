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
