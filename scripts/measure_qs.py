#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

from cot_probing import DATA_DIR


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
import json

from cot_probing import DATA_DIR

questions_dataset_path = DATA_DIR / "generated_questions_dataset.json"

# Load target questions

question_dataset = []
if os.path.exists(questions_dataset_path):
    with open(questions_dataset_path, "r") as f:
        question_dataset = json.load(f)

# shuffle for the sake of randomness
random.shuffle(question_dataset)

# Split into yes and no
yes_target_questions = [q for q in question_dataset if q["expected_answer"] == "yes"]
no_target_questions = [q for q in question_dataset if q["expected_answer"] == "no"]

# Split all yes and all no into train and test
test_pct = 0.20
test_yes_target_questions = yes_target_questions[
    : int(len(yes_target_questions) * test_pct)
]
train_yes_target_questions = yes_target_questions[
    int(len(yes_target_questions) * test_pct) :
]
test_no_target_questions = no_target_questions[
    : int(len(no_target_questions) * test_pct)
]
train_no_target_questions = no_target_questions[
    int(len(no_target_questions) * test_pct) :
]

train_target_questions = train_yes_target_questions + train_no_target_questions
test_target_questions = test_yes_target_questions + test_no_target_questions

# %%

# Load questions for FSPs

from cot_probing.diverse_combinations import load_and_process_file

fsp_yes_questions = load_and_process_file(DATA_DIR / "diverse_yes.txt")
fsp_no_questions = load_and_process_file(DATA_DIR / "diverse_no.txt")
assert len(fsp_yes_questions) == len(fsp_no_questions)


# %%
def generate_data(
    target_questions,
    fsp_yes_questions,
    fsp_no_questions,
    fsp_max_len,
):
    data = []

    for target_question_data in target_questions:
        question_to_answer = target_question_data["question"]
        expected_answer = target_question_data["expected_answer"]

        # Build the biased FSP
        if expected_answer == "yes":
            biased_fsp_questions = random.sample(fsp_no_questions, fsp_max_len)
        else:
            biased_fsp_questions = random.sample(fsp_yes_questions, fsp_max_len)

        # Build the unbiased FSP
        unbiased_fsp_yes_qs_num = int(fsp_max_len / 2)
        unbiased_fsp_no_qs_num = fsp_max_len - unbiased_fsp_yes_qs_num
        unbiased_fsp_questions = random.sample(
            fsp_yes_questions, unbiased_fsp_yes_qs_num
        ) + random.sample(fsp_no_questions, unbiased_fsp_no_qs_num)

        # Shuffle the FSPs
        random.shuffle(biased_fsp_questions)
        random.shuffle(unbiased_fsp_questions)

        # Remove the reasoning and answer in the question to answer
        split_string = "Let's think step by step:\n-"
        question_to_answer = question_to_answer.split(split_string)[0] + split_string

        # Add the prompt and label to the lists
        data.append(
            {
                "question_to_answer": question_to_answer,  # Includes "Let's think step by step:\n-" at the end, no CoT
                "expected_answer": expected_answer,
                "biased_fsp": "\n\n".join(biased_fsp_questions),
                "unbiased_fsp": "\n\n".join(unbiased_fsp_questions),
            }
        )

    return data


# %%
fsp_max_len = 14
temp = 0.7

train_data = generate_data(
    train_target_questions, fsp_yes_questions, fsp_no_questions, fsp_max_len
)
test_data = generate_data(
    test_target_questions, fsp_yes_questions, fsp_no_questions, fsp_max_len
)

print(f"Train data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")

# %% Dump to disk

import pickle


def dump_data():
    # Dump steering results to file, just in case
    if "-8B-" in model_id:
        data_path = (
            DATA_DIR
            / f"data_probing_for_unfaithful_answer_8B_temp_{temp}_fsp_size_{fsp_max_len}.pkl"
        )
    elif "-70B-" in model_id:
        data_path = (
            DATA_DIR
            / f"data_probing_for_unfaithful_answer_70B_temp_{temp}_fsp_size_{fsp_max_len}.pkl"
        )
    else:
        raise ValueError(f"Unknown model: {model_id}")

    for split in ["train", "test"]:
        split_data_path = data_path.with_name(
            f"{split}_data_probing_for_unfaithful_answer.pkl"
        )
        with open(split_data_path, "wb") as f:
            pickle.dump(globals()[f"{split}_data"], f)


dump_data()
# %%

# Generate a completion for each prompt and label them as faithful or unfaithful

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


def generate(input_ids, max_new_tokens=200, n_gen=3):
    prompt_len = len(input_ids[0])
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            use_cache=True,
            num_return_sequences=n_gen,
            tokenizer=tokenizer,
            stop_strings=["Answer: Yes", "Answer: No"],
            pad_token_id=tokenizer.eos_token_id,
        )
        responses = output[:, prompt_len:].cpu()

    cleaned_responses = []
    end_of_text_tok = tokenizer.eos_token_id
    for response in responses:
        # right strip as many end_of_text_tok as possible from each response
        # This is necessary because model.generate adds this when using cache
        while response[-1] == end_of_text_tok:
            response = response[:-1]
        cleaned_responses.append(response)

    return cleaned_responses


# %%


def produce_unbiased_cots(data, n_gen=10):
    for i in tqdm.tqdm(range(len(data))):
        question_to_answer = data[i]["question_to_answer"]
        unbiased_fsp = data[i]["unbiased_fsp"]

        prompt = unbiased_fsp + "\n\n" + question_to_answer
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        responses = generate(input_ids, n_gen=n_gen)

        # Filter out responses that don't end with "Answer: Yes" or "Answer: No"
        responses = [
            response
            for response in responses
            if response[-3:].tolist() == answer_yes_tok
            or response[-3:].tolist() == answer_no_tok
        ]
        assert (
            len(responses) > 0
        ), f"No responses ended with 'Answer: Yes' or 'Answer: No' for prompt: {prompt}"

        # Remove answers from the responses
        responses = [response[:-1] for response in responses]

        data[i]["unbiased_cots"] = responses


produce_unbiased_cots(train_data)
produce_unbiased_cots(test_data)

dump_data()

# %%

# Average number of unbiased COTs per prompt
print(f"Train: {np.mean([len(item['unbiased_cots']) for item in train_data])}")
print(f"Test: {np.mean([len(item['unbiased_cots']) for item in test_data])}")

# %%


def produce_biased_cots(data, n_gen=10):
    for i in tqdm.tqdm(range(len(data))):
        question_to_answer = data[i]["question_to_answer"]
        biased_fsp = data[i]["biased_fsp"]

        prompt = biased_fsp + "\n\n" + question_to_answer
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        responses = generate(input_ids, n_gen=n_gen)

        # Filter out responses that don't end with "Answer: Yes" or "Answer: No"
        responses = [
            response
            for response in responses
            if response[-3:].tolist() == answer_yes_tok
            or response[-3:].tolist() == answer_no_tok
        ]
        assert (
            len(responses) > 0
        ), f"No responses ended with 'Answer: Yes' or 'Answer: No' for prompt: {prompt}"

        # Remove answers from the responses
        responses = [response[:-1] for response in responses]

        data[i]["biased_cots"] = responses


produce_biased_cots(train_data)
produce_biased_cots(test_data)

dump_data()

# %%

# Average number of biased COTs per prompt
print(f"Train: {np.mean([len(item['biased_cots']) for item in train_data])}")
print(f"Test: {np.mean([len(item['biased_cots']) for item in test_data])}")

# %%

from cot_probing.generation import categorize_response as categorize_response_unbiased


def measure_unbiased_accuracy_for_unbiased_cots(data):
    for i in tqdm.tqdm(range(len(data))):
        question_to_answer = data[i]["question_to_answer"]
        expected_answer = data[i]["expected_answer"]
        unbiased_cots = data[i]["unbiased_cots"]
        unbiased_fsp = data[i]["unbiased_fsp"]

        correct_answer_count = 0
        for unbiased_cot in unbiased_cots:
            unbiased_fsp_with_question = f"{unbiased_fsp}\n\n{question_to_answer}"
            tokenized_unbiased_fsp_with_question = tokenizer.encode(
                unbiased_fsp_with_question
            )

            answer = categorize_response_unbiased(
                model=model,
                tokenizer=tokenizer,
                unbiased_context_toks=tokenized_unbiased_fsp_with_question,
                response=unbiased_cot.tolist(),
            )
            correct_answer_count += int(answer == expected_answer)

        unbiased_cot_accuracy = correct_answer_count / len(unbiased_cots)
        data[i]["unbiased_accuracy_for_unbiased_cots"] = unbiased_cot_accuracy


measure_unbiased_accuracy_for_unbiased_cots(train_data)
measure_unbiased_accuracy_for_unbiased_cots(test_data)

dump_data()

# %%

# Average unbiased accuracy for biased COTs
print(
    f"Train: {np.mean([item['unbiased_accuracy_for_unbiased_cots'] for item in train_data])}"
)
print(
    f"Test: {np.mean([item['unbiased_accuracy_for_unbiased_cots'] for item in test_data])}"
)

# %%

from cot_probing.generation import categorize_response as categorize_response_unbiased


def measure_unbiased_accuracy_for_biased_cots(data):
    for i in tqdm.tqdm(range(len(data))):
        question_to_answer = data[i]["question_to_answer"]
        expected_answer = data[i]["expected_answer"]
        biased_cots = data[i]["biased_cots"]
        unbiased_fsp = data[i]["unbiased_fsp"]

        correct_answer_count = 0
        for biased_cot in biased_cots:
            unbiased_fsp_with_question = f"{unbiased_fsp}\n\n{question_to_answer}"
            tokenized_unbiased_fsp_with_question = tokenizer.encode(
                unbiased_fsp_with_question
            )

            answer = categorize_response_unbiased(
                model=model,
                tokenizer=tokenizer,
                unbiased_context_toks=tokenized_unbiased_fsp_with_question,
                response=biased_cot.tolist(),
            )
            correct_answer_count += int(answer == expected_answer)

        biased_cot_accuracy = correct_answer_count / len(biased_cots)
        data[i]["unbiased_accuracy_for_biased_cots"] = biased_cot_accuracy


measure_unbiased_accuracy_for_biased_cots(train_data)
measure_unbiased_accuracy_for_biased_cots(test_data)

dump_data()

# %%

# Average unbiased accuracy for biased COTs
print(
    f"Train: {np.mean([item['unbiased_accuracy_for_biased_cots'] for item in train_data])}"
)
print(
    f"Test: {np.mean([item['unbiased_accuracy_for_biased_cots'] for item in test_data])}"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure accuracy of biased and unbiased CoTs"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)


if __name__ == "__main__":
    main(parse_args())
