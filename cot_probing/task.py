import random
import re
from string import ascii_uppercase

from cot_probing.typing import *
from cot_probing.utils import load_task_data

ANSWER_CHOICES_STR = "Answer choices:"


@dataclass(kw_only=True)
class Question:
    question: str  # Q: ...?
    choices: list[str]
    correct_idx: int
    question_with_choices: str = field(init=False)

    def __post_init__(self):
        self.question_with_choices = (
            f"{self.question}\n{ANSWER_CHOICES_STR}\n"
            + "\n".join(
                [
                    f"({ascii_uppercase[i]}) {choice}"
                    for i, choice in enumerate(self.choices)
                ]
            )
        )


@dataclass(kw_only=True)
class Task:
    name: str
    fsp_base: str
    fsp_alla: str
    questions: list[Question]


def load_task(task_name: str, seed: int) -> Task:
    random.seed(seed)
    baseline_fsp, alla_fsp, question_dicts = load_task_data(task_name)
    questions = []
    for question_dict in question_dicts:
        scores = question_dict["multiple_choice_scores"]
        assert sum(scores) == 1
        correct_idx = scores.index(1)
        input = question_dict["parsed_inputs"]
        input_lines = input.splitlines()

        answer_choices_list_idx = input_lines.index(ANSWER_CHOICES_STR)
        question_lines = input_lines[:answer_choices_list_idx]
        question = "\n".join(question_lines)
        choices_lines = input_lines[answer_choices_list_idx + 1 :]
        choices_text = []
        for line in choices_lines:
            m = re.match(r"^\((?P<choice_char>[A-Z])\) (?P<choice_text>.*)$", line)
            choice_char = m.group("choice_char")
            # choice chars should be in order A, B, C, ...
            assert choice_char == ascii_uppercase[len(choices_text)]
            choice_text = m.group("choice_text")
            choices_text.append(choice_text)

        assert len(choices_text) == len(scores)
        # correct choice should never be A
        if correct_idx == 0:
            x = random.randint(1, len(choices_text) - 1)
            choices_text[0], choices_text[x] = choices_text[x], choices_text[0]
            correct_idx = x
        questions.append(
            Question(
                question=question,
                choices=choices_text,
                correct_idx=correct_idx,
            )
        )
    random.shuffle(questions)
    return Task(
        name=task_name,
        fsp_base=baseline_fsp,
        fsp_alla=alla_fsp,
        questions=questions,
    )
