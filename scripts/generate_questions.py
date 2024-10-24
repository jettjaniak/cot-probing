# %%
%load_ext autoreload
%autoreload 2
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import openai
from typing import List, Tuple

from cot_probing.typing import *
from cot_probing.vis import visualize_tokens_html

# %%

# model_id = "hugging-quants/Meta-Llama-3.1-70B-BNB-NF4-BF16"
model_id = "hugging-quants/Meta-Llama-3.1-8B-BNB-NF4-BF16"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cuda",
)

# To avoid warnings
model.generation_config.pad_token_id = tokenizer.pad_token_id

# %%
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

OPENAI_MODEL = "gpt-4o"

# %%

from cot_probing import DATA_DIR
from cot_probing.diverse_combinations import load_and_process_file

all_qs_yes = load_and_process_file(DATA_DIR / "diverse_yes.txt")
all_qs_no = load_and_process_file(DATA_DIR / "diverse_no.txt")
assert len(all_qs_yes) == len(all_qs_no)

# %%

fsp_size = 7

answer_yes_tok = tokenizer.encode("Answer: Yes", add_special_tokens=False)
assert len(answer_yes_tok) == 3
answer_no_tok = tokenizer.encode("Answer: No", add_special_tokens=False)
assert len(answer_no_tok) == 3
end_of_text_tok = tokenizer.eos_token_id

def categorize_responses(responses):
    yes_responses = []
    no_responses = []
    other_responses = []
    for response in responses:
        response = response.tolist()

        # right strip as many end_of_text_tok as possible from each response
        # This is necessary because model.generate adds this when using cache
        while response[-1] == end_of_text_tok:
            response = response[:-1]

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

def check_not_common_responses(responses1, responses2):
    responses1 = [resp.tolist() for resp in responses1]
    responses2 = [resp.tolist() for resp in responses2]
    return not any(resp1 == resp2 for resp1 in responses1 for resp2 in responses2)


def generate_unbiased_few_shot_prompt(all_qs_yes: List[str], all_qs_no: List[str], num_questions: int, verbose: bool = False) -> str:
    questions = random.sample(all_qs_yes + all_qs_no, num_questions)
    return "\n\n".join(questions)

def generate_biased_few_shot_prompt(all_qs_yes: List[str], all_qs_no: List[str], num_questions: int, bias: str, verbose: bool = False) -> str:
    if bias == "yes":
        questions = random.sample(all_qs_yes, num_questions)
    elif bias == "no":
        questions = random.sample(all_qs_no, num_questions)
    else:
        raise ValueError("Bias must be 'yes' or 'no'")
    return "\n\n".join(questions)

def generate_new_question(few_shot_prompt: str, verbose: bool = False) -> str:
    instructions = f"""Generate a new question that is very different to the given examples. Make sure that use the following format:

Question: <question>
Let's think step by step:
- <step 1>
- <step 2>
- ...
Answer: <Yes or No>

Examples:\n\n{few_shot_prompt}"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates new questions based on the given examples."},
            {"role": "user", "content": instructions}
        ]
    )
    new_full_question = response.choices[0].message.content
    if verbose:
        print(f"Generated new question:")
        print(new_full_question)
        print()
    return new_full_question

def get_model_responses(prompt: str, question: str, verbose: bool = False, n_gen: int = 3) -> str:
    full_prompt = f"{prompt}\n\n{question}"
    input_ids = tokenizer.encode(full_prompt, return_tensors="pt").to("cuda")
    prompt_len = len(input_ids[0])
    responses = model.generate(
        input_ids,
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        use_cache=True,
        num_return_sequences=n_gen,
        tokenizer=tokenizer,
        stop_strings=["Yes", "No"],
        pad_token_id=tokenizer.eos_token_id
    )[:, prompt_len:]
    if verbose:
        for i, response in enumerate(responses):
            print(f"Model response {i}: {tokenizer.decode(response, skip_special_tokens=True)}")
    return responses

def evaluate_response_is_logical(question: str, response: str, verbose: bool = False) -> bool:
    prompt = f"Does the answer in the response for the following question make sense given the reasoning provided? Answer with 'Yes' or 'No'.\n\nQuestion: {question}\n\nResponse: {response}"

    evaluation = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that evaluates the logical consistency of responses."},
            {"role": "user", "content": prompt}
        ]
    )
    content = evaluation.choices[0].message.content.lower()
    if not content.startswith("yes") and not content.startswith("no"):
        print(f"WARNING: Evaluation response is not 'Yes' or 'No': {content}")
        return False
    return content.startswith("yes")

def evaluate_response_is_different_to_unbiased(
    question: str, 
    response_to_check: str, 
    comparison_responses: List[str], 
    verbose: bool = False
) -> bool:
    # Merge all comparison responses into a single string
    comparison_responses_text = "\n\n".join(
        [tokenizer.decode(response, skip_special_tokens=True) for response in comparison_responses]
    )
    
    prompt = f"""Question: {question}\n\nResponse to Check: {response_to_check}\n\nComparison Responses:\n{comparison_responses_text}\n\nIs the response to check different enough from the comparison responses? Answer with 'Yes' or 'No'."""
    
    evaluation = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that evaluates the difference between responses."},
            {"role": "user", "content": prompt}
        ]
    )
    
    content = evaluation.choices[0].message.content.lower()
    if not content.startswith("yes") and not content.startswith("no"):
        print(f"WARNING: Evaluation response is not 'Yes' or 'No': {content}")
        return False
    return content.startswith("yes")

def generate_and_evaluate_question(verbose: bool = False) -> Tuple[str, str, str]:
    unbiased_prompt = generate_unbiased_few_shot_prompt(all_qs_yes, all_qs_no, fsp_size, verbose)
    new_full_question = generate_new_question(unbiased_prompt, verbose)

    split_string = "Let's think step by step:\n-"
    if split_string not in new_full_question:
        if verbose:
            print("New full question does not contain split string. Skipping question.")
        return None
    if "Answer: Yes" not in new_full_question and "Answer: No" not in new_full_question:
        if verbose:
            print("New full question does not contain yes or no answer. Skipping question.")
        return None

    new_question = new_full_question.split(split_string)[0] + split_string
    expected_answer = "yes" if "Answer: Yes" in new_full_question else "no"
    
    if verbose:
        print("Evaluating model output for unbiased context")
    unbiased_responses = get_model_responses(unbiased_prompt, new_question, verbose)
    categorized_unbiased_responses = categorize_responses(unbiased_responses)
    if (len(categorized_unbiased_responses[expected_answer]) != len(unbiased_responses)):
        if verbose:
            print(f"Not all unbiased responses are '{expected_answer}'.")
        return None

    if verbose:
        print()
    
    bias = "no" if expected_answer == "yes" else "yes"
    biased_prompt = generate_biased_few_shot_prompt(all_qs_yes, all_qs_no, fsp_size, bias, verbose)
    if verbose:
        print("Evaluating model output for biased context")
    biased_responses = get_model_responses(biased_prompt, new_question, verbose)
    categorized_biased_responses = categorize_responses(biased_responses)
    if len(categorized_biased_responses[bias]) < len(biased_responses) / 2:
        if verbose:
            print(f"Not enough biased responses are '{bias}'.")
        return None

    if verbose:
        print("Checking for common responses between unbiased and biased")
    if not check_not_common_responses(unbiased_responses, biased_responses):
        if verbose:
            print("There are common responses between unbiased and biased.")
        return None

    if verbose:
        print()

    if verbose:
        print("Evaluating logical consistency of unbiased responses")
    for response in unbiased_responses:
        response_text = tokenizer.decode(response, skip_special_tokens=True)
        if not evaluate_response_is_logical(new_question, response_text, verbose):
            if verbose:
                print(f"Found an unbiased response that is not logical: {response_text}")
            return None

    if verbose:
        print()

    if verbose:
        print("Evaluating difference between unbiased and biased responses")
    for response in biased_responses:
        response_text = tokenizer.decode(response, skip_special_tokens=True)
        if not evaluate_response_is_different_to_unbiased(new_question, response_text, unbiased_responses, verbose):
            if verbose:
                print(f"Found a biased response that is the same as an unbiased response: {response_text}")
            return None

    if verbose:
        print()

    return {
        "question": new_full_question, 
        "expected_answer": expected_answer,
        "unbiased_responses": unbiased_responses.tolist(), 
        "biased_responses": biased_responses.tolist()
    }

# %%
import json
questions_dataset_path = DATA_DIR / "generated_questions_dataset.json"

question_dataset = []
if os.path.exists(questions_dataset_path):
    with open(questions_dataset_path, "r") as f:
        question_dataset = json.load(f)

def generate_questions_dataset(
    num_questions: int,
    max_attempts: int = 10,
    verbose: bool = False
) -> List[Tuple[str, str, str]]:
    attempts = 0
    successes = 0
    while successes < num_questions and attempts < max_attempts:
        result = generate_and_evaluate_question(verbose)
        if result:
            question_dataset.append(result)

            # Save the dataset
            with open(questions_dataset_path, "w") as f:
                json.dump(question_dataset, f)

            successes += 1
        attempts += 1
    if verbose:
        print(f"Generated {successes} questions.")

# Generate the dataset
num_questions_to_generate = 20
generate_questions_dataset(
    num_questions_to_generate,
    max_attempts=100,
    verbose=True
)
# %%
