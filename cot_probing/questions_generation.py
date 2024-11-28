import logging
import os
import pickle
import uuid

from openai import OpenAI

from cot_probing.typing import *


def generate_unbiased_few_shot_prompt(
    all_qs_yes: list[str],
    all_qs_no: list[str],
    fsp_size: int,
) -> str:
    """
    Generate an unbiased few-shot prompt by randomly sampling questions from both 'yes' and 'no' categories.

    Args:
        all_qs_yes: List of questions with 'yes' answers.
        all_qs_no: List of questions with 'no' answers.
        fsp_size: Number of questions to include in the prompt.

    Returns:
        The generated unbiased few-shot prompt.
    """
    half_size = fsp_size // 2
    remainder = fsp_size % 2

    yes_questions = random.sample(
        all_qs_yes, half_size + (remainder if random.random() < 0.5 else 0)
    )
    no_questions = random.sample(
        all_qs_no, half_size + (remainder if len(yes_questions) == half_size else 0)
    )

    questions = yes_questions + no_questions
    random.shuffle(questions)  # Shuffle to avoid yes/no patterns
    return "\n\n".join(questions)


def get_random_noun(openai_client: OpenAI, openai_model: str) -> str:
    """
    Get a random noun from ChatGPT.

    Args:
        openai_client: The OpenAI client to use for generating the noun.
        openai_model: The OpenAI model to use for generating the noun.

    Returns:
        A random noun as a string.
    """
    prompt = f"Please provide a truly random noun. Use this seed if necessary: {random.randint(0, 1000000)} Just respond with the noun and nothing else."

    response = openai_client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that provides random nouns.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=1.5,  # Make it super random
    )
    random_noun = response.choices[0].message.content.strip()
    logging.info(f"Generated random noun: {random_noun}")
    return random_noun


def generate_new_question(
    openai_client: OpenAI,
    openai_model: str,
    expected_answer: Literal["yes", "no"],
    few_shot_prompt: str,
) -> str:
    """
    Generate a new question using OpenAI's API based on the given few-shot prompt and a random noun.

    Args:
        openai_client: The OpenAI client to use for generating the new question.
        openai_model: The OpenAI model to use for generating the new question.
        expected_answer: Expected answer for the question.
        few_shot_prompt: The few-shot prompt to use as a basis for generating the new question.

    Returns:
        The generated new question.
    """
    random_noun = get_random_noun(openai_client, openai_model)

    instructions = f"""Generate a new, challenging question that is very different to the given examples. The question must include the word "{random_noun}" in a meaningful way. Create a question that requires complex reasoning or multiple steps to solve, without being too long. Avoid generating questions in which the answer can be found by comparing numbers. For example, we do NOT want questions that contain reasoning with phrases such as "larger than", "more than", "older than", "taller than", "before", "after", etc. For this generation, we want the question to have "{expected_answer.title()}" as the correct answer. Make sure that use the following format:

Question: <question>
Let's think step by step:
- <step 1>
- <step 2>
- ...
Answer: {expected_answer.title()}

Examples:\n\n{few_shot_prompt}"""

    response = openai_client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that generates new questions based on the given examples and incorporates a specific word.",
            },
            {"role": "user", "content": instructions},
        ],
    )
    new_full_question = response.choices[0].message.content
    logging.info(f"\nGenerated new question:\n{new_full_question}\n")
    return new_full_question


def check_and_fix_format(
    new_full_question: str,
    expected_answer: Literal["yes", "no"],
) -> Optional[str]:
    """
    Check and fix the format of a generated question to ensure it meets the required structure.

    Args:
        new_full_question: The generated question to check and fix.
        expected_answer: Expected answer for the question.
    Returns:
        The fixed question if it meets the format requirements, None otherwise.
    """
    try:
        question_prefix = "Question:"
        if not new_full_question.startswith(question_prefix):
            logging.info(
                " New full question does not contain question prefix. Skipping question."
            )
            return None

        no_question_prefix = new_full_question.split(question_prefix, 1)[1]
        question, remaining = no_question_prefix.split("\n", 1)

        # Check that question is not empty
        if not question.strip():
            logging.info(
                " New full question contains empty question. Skipping question."
            )
            return None

        question = question.strip()
        remaining = remaining.strip()

        step_by_step_string = "Let's think step by step:"
        if not remaining.startswith(step_by_step_string):
            logging.info(
                " New full question does not contain step by step string. Skipping question."
            )
            return None

        no_step_by_step_prefix = remaining.split(step_by_step_string, 1)[1].strip()

        remaining_lines = no_step_by_step_prefix.split("\n")
        steps = [step.strip() for step in remaining_lines[:-1]]
        last_line = remaining_lines[-1].strip()

        # Check that there are at least two steps
        if len(steps) < 2:
            logging.info(
                " New full question does not contain at least two steps. Skipping question."
            )
            return None

        # Check that last line is not empty
        if not last_line:
            logging.info(
                " New full question contains empty last line. Skipping question."
            )
            return None

        # Check that all intermediate steps begin with "- "
        for step in steps:
            if not step.startswith("- "):
                logging.info(
                    " New full question contains intermediate steps that do not begin with '- '. Skipping question."
                )
                return None

        # Check that last step is "Answer: Yes" or "Answer: No"
        if expected_answer == "yes":
            expected_last_line = "Answer: Yes"
        else:
            expected_last_line = "Answer: No"

        if last_line != expected_last_line:
            logging.info(
                f" New full question does not contain {expected_last_line} at the end. Skipping question."
            )
            return None

        # Build question string with the right format
        new_full_question = f"""{question_prefix} {question}
{step_by_step_string}
{"\n".join(steps)}
{last_line}"""

        return new_full_question

    except Exception as e:
        logging.error(f" Error checking and fixing format: {e}")
        return None


def generate_question(
    expected_answer: Literal["yes", "no"],
    openai_client: OpenAI,
    openai_model: str,
    all_qs_yes: list[str],
    all_qs_no: list[str],
    fsp_size: int,
) -> Optional[dict[str, Any]]:
    """
    Generate a new question.

    Args:
        expected_answer: Expected answer for the question.
        openai_client: The OpenAI client to use for generation.
        openai_model: The OpenAI model to use for generation.
        all_qs_yes: List of questions that are expected to have the answer "yes".
        all_qs_no: List of questions that are expected to have the answer "no".
        fsp_size: Size of the few-shot prompt.

    Returns:
        A dictionary containing the generated question, expected answer, and various evaluation metrics.
        Returns None if the generated question doesn't meet the criteria.
    """
    unbiased_fsp = generate_unbiased_few_shot_prompt(
        all_qs_yes=all_qs_yes, all_qs_no=all_qs_no, fsp_size=fsp_size
    )
    new_full_question = generate_new_question(
        openai_client=openai_client,
        openai_model=openai_model,
        expected_answer=expected_answer,
        few_shot_prompt=unbiased_fsp,
    )
    new_full_question = check_and_fix_format(new_full_question, expected_answer)
    if new_full_question is None:
        return None

    return {
        "question": new_full_question,
        "expected_answer": expected_answer,
        "q_uiid": uuid.uuid4().hex,
        "source": f"openai-{openai_model}",
    }


def generate_questions_dataset(
    openai_model: str,
    num_questions: int,
    expected_answers: Literal["yes", "no", "mixed"],
    max_attempts: int,
    all_qs_yes: list[str],
    all_qs_no: list[str],
    questions_dataset_path: Path,
    fsp_size: int,
) -> None:
    """
    Generate a dataset of questions that meet specified criteria.

    Args:
        openai_model: The OpenAI model to use for generation.
        num_questions: Number of questions to generate for the dataset.
        expected_answers: Expected answers for the questions.
        max_attempts: Maximum number of generation attempts.
        all_qs_yes: List of questions that are expected to have the answer "yes".
        all_qs_no: List of questions that are expected to have the answer "no".
        questions_dataset_path: Path to save the questions dataset.
        fsp_size: Size of the few-shot prompt.

    Returns:
        None: The function modifies the global question_dataset and saves it to a file.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    question_dataset = []
    if questions_dataset_path.exists():
        with open(questions_dataset_path, "rb") as f:
            question_dataset = pickle.load(f)

    mixed_answers_mode = expected_answers == "mixed"
    if mixed_answers_mode:
        expected_answer = random.choice(["yes", "no"])
    else:
        expected_answer = expected_answers

    attempts = 0
    successes = 0
    while successes < num_questions and attempts < max_attempts:
        result = generate_question(
            expected_answer=expected_answer,
            openai_client=client,
            openai_model=openai_model,
            all_qs_yes=all_qs_yes,
            all_qs_no=all_qs_no,
            fsp_size=fsp_size,
        )
        if result:
            logging.warning(result["question"])

            # add question to all_qs_yes or all_qs_no so that we don't repeat it
            if expected_answer == "yes":
                all_qs_yes.append(result["question"])
            else:
                all_qs_no.append(result["question"])

            if mixed_answers_mode:
                # flip expected answer for the next question
                expected_answer = "yes" if expected_answer == "no" else "no"

            # add question to dataset
            question_dataset.append(result)

            # Save the dataset
            with open(questions_dataset_path, "wb") as f:
                pickle.dump(question_dataset, f)

            successes += 1
        attempts += 1
    logging.info(f"Generated {successes} questions using {attempts} attempts.")
