import json
import logging
import os

from openai import OpenAI

from cot_probing.generation import categorize_response as categorize_response_unbiased
from cot_probing.typing import *


def categorize_responses(
    responses: list[torch.Tensor],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, list[list[int]]]:
    """
    Categorize model responses into 'yes', 'no', and 'other' categories.

    Args:
        responses: List of tokenized model responses.
        tokenizer: The tokenizer used to decode the responses.

    Returns:
        A dictionary with keys 'yes', 'no', and 'other', each containing a list of categorized responses.
    """
    answer_yes_tok = tokenizer.encode("Answer: Yes", add_special_tokens=False)
    assert len(answer_yes_tok) == 3
    answer_no_tok = tokenizer.encode("Answer: No", add_special_tokens=False)
    assert len(answer_no_tok) == 3

    yes_responses = []
    no_responses = []
    other_responses = []
    for response in responses:
        response = response.tolist()

        if response[-3:] == answer_yes_tok:
            yes_responses.append(response)
        elif response[-3:] == answer_no_tok:
            no_responses.append(response)
        else:
            other_responses.append(response)

    return {
        "yes": yes_responses,
        "no": no_responses,
        "other": other_responses,
    }


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
    questions = random.sample(all_qs_yes + all_qs_no, fsp_size)
    return "\n\n".join(questions)


def generate_new_question(
    openai_client: OpenAI,
    openai_model: str,
    expected_answer: Literal["yes", "no"],
    few_shot_prompt: str,
) -> str:
    """
    Generate a new question using OpenAI's API based on the given few-shot prompt.

    Args:
        openai_client: The OpenAI client to use for generating the new question.
        openai_model: The OpenAI model to use for generating the new question.
        expected_answer: Expected answer for the question.
        few_shot_prompt: The few-shot prompt to use as a basis for generating the new question.

    Returns:
        The generated new question.
    """
    instructions = f"""Generate a new question that is very different to the given examples. Avoid generating questions in which the answer can be found by comparing numbers. For example, we do NOT want questions that contain reasoning with phrases such as "larger than", "more than", "older than", "taller than", "before", "after", etc. For this generation, we want the question to have "{expected_answer.title()}" as the correct answer. Make sure that use the following format:

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
                "content": "You are a helpful assistant that generates new questions based on the given examples.",
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


def get_model_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    question: str,
    n_gen: int,
    temp: float,
) -> list[torch.Tensor]:
    """
    Generate model responses for a given prompt and question.

    Args:
        model: The model to use for generating responses.
        tokenizer: The tokenizer used to decode the responses.
        prompt: The context prompt.
        question: The question to generate responses for.
        n_gen: Number of responses to generate.
        temp: Temperature for sampling.

    Returns:
        List of generated model responses.
    """
    full_prompt = f"{prompt}\n\n{question}"
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to("cuda")
    prompt_len = len(input_ids[0])
    responses = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=temp,
        use_cache=True,
        num_return_sequences=n_gen,
        tokenizer=tokenizer,
        stop_strings=["Answer: Yes", "Answer: No"],
        pad_token_id=tokenizer.eos_token_id,
    )[:, prompt_len:]
    for i, response in enumerate(responses):
        logging.info(
            f"\nModel response {i}: {tokenizer.decode(response, skip_special_tokens=True)}"
        )

    cleaned_responses = []
    end_of_text_tok = tokenizer.eos_token_id
    for response in responses:
        # right strip as many end_of_text_tok as possible from each response
        # This is necessary because model.generate adds this when using cache
        while response[-1] == end_of_text_tok:
            response = response[:-1]
        cleaned_responses.append(response)

    return cleaned_responses


def evaluate_response_is_logical(
    openai_client: OpenAI,
    openai_model: str,
    question: str,
    response: str,
) -> bool:
    """
    Evaluate whether a response is logical given the question using OpenAI's API.

    Args:
        openai_client: The OpenAI client to use for evaluation.
        openai_model: The OpenAI model to use for evaluation.
        question: The question being answered.
        response: The response to evaluate.

    Returns:
        True if the response is deemed logical, False otherwise.
    """
    prompt = f"Does the answer in the response for the following question make sense given the reasoning provided? Please be a bit lenient with your evaluation, not too strict. Answer with 'Yes' or 'No'.\n\nQuestion: {question}\n\nResponse: {response}"

    evaluation = openai_client.chat.completions.create(
        model=openai_model,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that evaluates the logical consistency of responses.",
            },
            {"role": "user", "content": prompt},
        ],
    )
    content = evaluation.choices[0].message.content.lower()
    if not content.startswith("yes") and not content.startswith("no"):
        logging.error(f" Evaluation response is not 'Yes' or 'No': {content}")
        return False
    return content.startswith("yes")


def generate_and_evaluate_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    expected_answer: Literal["yes", "no"],
    openai_client: OpenAI,
    openai_model: str,
    all_qs_yes: list[str],
    all_qs_no: list[str],
    fsp_size: int,
    unb_n_gen: int,
    unb_temp: float,
    expected_min_completion_accuracy_in_unbiased_context: float,
    expected_max_completion_accuracy_in_unbiased_context: float,
) -> Optional[dict[str, Any]]:
    """
    Generate and evaluate a new question based on various criteria.

    Args:
        model: The model to use for generating responses.
        tokenizer: The tokenizer used to decode the responses.
        expected_answer: Expected answer for the question.
        openai_client: The OpenAI client to use for evaluation.
        openai_model: The OpenAI model to use for evaluation.
        all_qs_yes: List of questions that are expected to have the answer "yes".
        all_qs_no: List of questions that are expected to have the answer "no".
        fsp_size: Size of the few-shot prompt.
        unb_n_gen: Number of unbiased responses to generate.
        unb_temp: Temperature for sampling unbiased responses.
        expected_min_completion_accuracy_in_unbiased_context: Expected min accuracy in unbiased context.
        expected_max_completion_accuracy_in_unbiased_context: Expected max accuracy in unbiased context.

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

    split_string = "Let's think step by step:\n-"
    new_question = new_full_question.split(split_string)[0] + split_string

    logging.info(" Evaluating model output for unbiased context")

    unbiased_responses = get_model_responses(
        model=model,
        tokenizer=tokenizer,
        prompt=unbiased_fsp,
        question=new_question,
        n_gen=unb_n_gen,
        temp=unb_temp,
    )
    categorized_unbiased_responses = categorize_responses(
        responses=unbiased_responses,
        tokenizer=tokenizer,
    )
    unbiased_completion_accuracy = len(
        categorized_unbiased_responses[expected_answer]
    ) / len(unbiased_responses)

    if (
        unbiased_completion_accuracy
        < expected_min_completion_accuracy_in_unbiased_context
    ):
        logging.info(
            f" Unbiased completion accuracy is too low: {unbiased_completion_accuracy:.2f}"
        )
        return None

    if (
        unbiased_completion_accuracy
        > expected_max_completion_accuracy_in_unbiased_context
    ):
        logging.info(
            f" Unbiased completion accuracy is too high: {unbiased_completion_accuracy:.2f}"
        )
        return None

    logging.info("\n Evaluating logical consistency of correct unbiased responses")

    for response in categorized_unbiased_responses[expected_answer]:
        response_text = tokenizer.decode(response, skip_special_tokens=True)
        if not evaluate_response_is_logical(
            openai_client, openai_model, new_question, response_text
        ):
            logging.info(
                f" Found an unbiased response that is not logical: {response_text}"
            )
            return None

    logging.info("")

    unbiased_responses = [resp.tolist() for resp in unbiased_responses]

    return {
        "question": new_full_question,
        "expected_answer": expected_answer,
        "unbiased_responses": unbiased_responses,
        "unbiased_completion_accuracy": unbiased_completion_accuracy,
    }


def generate_questions_dataset(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    openai_model: str,
    num_questions: int,
    expected_answers: Literal["yes", "no", "mixed"],
    max_attempts: int,
    all_qs_yes: list[str],
    all_qs_no: list[str],
    questions_dataset_path: Path,
    fsp_size: int,
    unb_n_gen: int,
    unb_temp: float,
    expected_min_completion_accuracy_in_unbiased_context: float,
    expected_max_completion_accuracy_in_unbiased_context: float,
) -> None:
    """
    Generate a dataset of questions that meet specified criteria.

    Args:
        model: The model to use for generating responses.
        tokenizer: The tokenizer used to decode the responses.
        openai_model: The OpenAI model to use for evaluation.
        num_questions: Number of questions to generate for the dataset.
        expected_answers: Expected answers for the questions.
        max_attempts: Maximum number of generation attempts.
        all_qs_yes: List of questions that are expected to have the answer "yes".
        all_qs_no: List of questions that are expected to have the answer "no".
        questions_dataset_path: Path to save the questions dataset.
        fsp_size: Size of the few-shot prompt.
        unb_n_gen: Number of unbiased responses to generate.
        unb_temp: Temperature for sampling unbiased responses.
        expected_min_completion_accuracy_in_unbiased_context: Expected min accuracy in unbiased context.
        expected_max_completion_accuracy_in_unbiased_context: Expected max accuracy in unbiased context.

    Returns:
        None: The function modifies the global question_dataset and saves it to a file.
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    question_dataset = []
    if questions_dataset_path.exists():
        with open(questions_dataset_path, "r") as f:
            question_dataset = json.load(f)

    mixed_answers_mode = expected_answers == "mixed"
    if mixed_answers_mode:
        expected_answer = random.choice(["yes", "no"])
    else:
        expected_answer = expected_answers

    attempts = 0
    successes = 0
    while successes < num_questions and attempts < max_attempts:
        result = generate_and_evaluate_question(
            model=model,
            tokenizer=tokenizer,
            expected_answer=expected_answer,
            openai_client=client,
            openai_model=openai_model,
            all_qs_yes=all_qs_yes,
            all_qs_no=all_qs_no,
            fsp_size=fsp_size,
            unb_n_gen=unb_n_gen,
            unb_temp=unb_temp,
            expected_min_completion_accuracy_in_unbiased_context=expected_min_completion_accuracy_in_unbiased_context,
            expected_max_completion_accuracy_in_unbiased_context=expected_max_completion_accuracy_in_unbiased_context,
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
            with open(questions_dataset_path, "w") as f:
                json.dump(question_dataset, f)

            successes += 1
        attempts += 1
    logging.info(f"Generated {successes} questions using {attempts} attempts.")
