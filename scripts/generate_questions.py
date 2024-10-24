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

# %%
client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# %%

from cot_probing import DATA_DIR
from cot_probing.diverse_combinations import load_and_process_file

all_qs_yes = load_and_process_file(DATA_DIR / "diverse_yes.txt")
all_qs_no = load_and_process_file(DATA_DIR / "diverse_no.txt")
assert len(all_qs_yes) == len(all_qs_no)

# %%

fsp_size = 7

def generate_unbiased_few_shot_prompt(all_qs_yes: List[str], all_qs_no: List[str], num_questions: int, verbose: bool = False) -> str:
    questions = random.sample(all_qs_yes + all_qs_no, num_questions)
    if verbose:
        print(f"Generated unbiased few-shot prompt with {num_questions} questions.")
    return "\n\n".join(questions)

def generate_biased_few_shot_prompt(all_qs_yes: List[str], all_qs_no: List[str], num_questions: int, bias: str, verbose: bool = False) -> str:
    if bias == "yes":
        questions = random.sample(all_qs_yes, num_questions)
    elif bias == "no":
        questions = random.sample(all_qs_no, num_questions)
    else:
        raise ValueError("Bias must be 'yes' or 'no'")
    if verbose:
        print(f"Generated biased few-shot prompt with {num_questions} '{bias}' questions.")
    return "\n\n".join(questions)

def generate_new_question(few_shot_prompt: str, verbose: bool = False) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates new questions based on the given examples."},
            {"role": "user", "content": f"Given the following few-shot examples, generate a new yes or no question that is different from the examples:\n\n{few_shot_prompt}"}
        ]
    )
    new_full_question = response.choices[0].message.content
    if verbose:
        print(f"Generated new question:")
        print(new_full_question)
        print()
    return new_full_question

def get_model_response(prompt: str, question: str, verbose: bool = False) -> str:
    full_prompt = f"{prompt}\n\n{question}"
    if verbose:
        print(f"Model prompt:")
        print(full_prompt)

    inputs = tokenizer(full_prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if verbose:
        print(f"Model response for question '{question}': {response}")
    return response.split(question)[1].strip()

def evaluate_response(question: str, response: str, verbose: bool = False) -> bool:
    prompt = f"Question: {question}\n\nResponse: {response}\n\nDoes the answer in the response logically follow from the chain of thought provided? Answer with 'Yes' or 'No'."
    evaluation = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that evaluates the logical consistency of responses."},
            {"role": "user", "content": prompt}
        ]
    )
    content = evaluation.choices[0].message.content
    print("Evaluation content: ")
    print(content)
    result = content.lower().startswith("yes")
    if verbose:
        print(f"Evaluation result for question '{question}': {'Yes' if result else 'No'}")
    return result

def generate_and_evaluate_question(verbose: bool = False) -> Tuple[str, str, str]:
    unbiased_prompt = generate_unbiased_few_shot_prompt(all_qs_yes, all_qs_no, fsp_size, verbose)
    new_full_question = generate_new_question(unbiased_prompt, verbose)
    print("New full question:")
    print(new_full_question)

    split_string = "Let's think step by step:\n-"
    if split_string not in new_full_question:
        if verbose:
            print("New full question does not contain split string. Skipping question.")
        return None
    if "Answer: Yes" not in new_full_question and "Answer: No" not in new_full_question:
        if verbose:
            print("New full question does not contain answer. Skipping question.")
        return None

    new_question = new_full_question.split(split_string)[0] + split_string
    
    unbiased_response = get_model_response(unbiased_prompt, new_question, verbose)
    biased_prompt = generate_biased_few_shot_prompt(all_qs_yes, all_qs_no, fsp_size, random.choice(["yes", "no"]), verbose)
    biased_response = get_model_response(biased_prompt, new_question, verbose)
    
    if (evaluate_response(new_question, unbiased_response, verbose) and
        evaluate_response(new_question, biased_response, verbose) and
        unbiased_response != biased_response):
        return new_question, unbiased_response, biased_response
    else:
        return None

# %%
def generate_questions_dataset(num_questions: int, verbose: bool = False) -> List[Tuple[str, str, str]]:
    dataset = []
    pbar = tqdm.tqdm(total=num_questions)
    while len(dataset) < num_questions:
        result = generate_and_evaluate_question(verbose)
        if result:
            dataset.append(result)
            pbar.update(1)
    pbar.close()
    if verbose:
        print(f"Generated {len(dataset)} questions.")
    return dataset

# Generate the dataset
num_questions_to_generate = 1
question_dataset = generate_questions_dataset(num_questions_to_generate, verbose=True)

# Save the dataset
import json
with open("question_dataset.json", "w") as f:
    json.dump(question_dataset, f)

print(f"Generated and saved {len(question_dataset)} questions.")

# %%