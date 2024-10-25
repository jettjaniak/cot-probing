# %%
%load_ext autoreload
%autoreload 2
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing.typing import *

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
test_yes_target_questions = yes_target_questions[:int(len(yes_target_questions) * test_pct)]
train_yes_target_questions = yes_target_questions[int(len(yes_target_questions) * test_pct):]
test_no_target_questions = no_target_questions[:int(len(no_target_questions) * test_pct)]
train_no_target_questions = no_target_questions[int(len(no_target_questions) * test_pct):]

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
        unbiased_fsp_questions = random.sample(fsp_yes_questions, unbiased_fsp_yes_qs_num) + random.sample(fsp_no_questions, unbiased_fsp_no_qs_num)

        # Shuffle the FSPs
        random.shuffle(biased_fsp_questions)
        random.shuffle(unbiased_fsp_questions)

        # Remove the reasoning and answer in the question to answer
        split_string = "Let's think step by step:\n-"
        question_to_answer = question_to_answer.split(split_string)[0] + split_string

        # Add the prompt and label to the lists
        data.append({
            "question_to_answer": question_to_answer, # Includes "Let's think step by step:\n-" at the end, no CoT
            "expected_answer": expected_answer,
            "biased_fsp": "\n\n".join(biased_fsp_questions),
            "unbiased_fsp": "\n\n".join(unbiased_fsp_questions),
        })

    return data

# %%
fsp_max_len = 8

train_data = generate_data(
    train_target_questions, fsp_yes_questions, fsp_no_questions, fsp_max_len)
test_data = generate_data(
    test_target_questions, fsp_yes_questions, fsp_no_questions, fsp_max_len)

print(f"Train data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")

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
            temperature=0.7,
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
        responses = [response for response in responses if response[-3:].tolist() == answer_yes_tok or response[-3:].tolist() == answer_no_tok]
        assert len(responses) > 0, f"No responses ended with 'Answer: Yes' or 'Answer: No' for prompt: {prompt}"

        # Remove answers from the responses
        responses = [response[:-1] for response in responses]
        
        data[i]["unbiased_cots"] = responses

produce_unbiased_cots(train_data)
produce_unbiased_cots(test_data)

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
        responses = [response for response in responses if response[-3:].tolist() == answer_yes_tok or response[-3:].tolist() == answer_no_tok]
        assert len(responses) > 0, f"No responses ended with 'Answer: Yes' or 'Answer: No' for prompt: {prompt}"

        # Remove answers from the responses
        responses = [response[:-1] for response in responses]
        
        data[i]["biased_cots"] = responses

produce_biased_cots(train_data)
produce_biased_cots(test_data)

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

# %%

# Average unbiased accuracy for biased COTs
print(f"Train: {np.mean([item['unbiased_accuracy_for_unbiased_cots'] for item in train_data])}")
print(f"Test: {np.mean([item['unbiased_accuracy_for_unbiased_cots'] for item in test_data])}")

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

# %%

# Average unbiased accuracy for biased COTs
print(f"Train: {np.mean([item['unbiased_accuracy_for_biased_cots'] for item in train_data])}")
print(f"Test: {np.mean([item['unbiased_accuracy_for_biased_cots'] for item in test_data])}")

# %%

import matplotlib.pyplot as plt
import numpy as np

def plot_unbiased_accuracy_distribution(data, biased=False, data_label=""):
    # Extract accuracies from the data
    if biased:
        accuracies = [item['unbiased_accuracy_for_biased_cots'] for item in data if 'unbiased_accuracy_for_biased_cots' in item]
    else:
        accuracies = [item['unbiased_accuracy_for_unbiased_cots'] for item in data if 'unbiased_accuracy_for_unbiased_cots' in item]

    # Create the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(accuracies, bins=20, edgecolor='black')
    plt.title('Distribution of Unbiased Accuracies for ' + ('Biased' if biased else 'Unbiased') + ' COTs' + f' ({data_label})')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')

    # Add mean line
    mean_accuracy = np.mean(accuracies)
    plt.axvline(mean_accuracy, color='r', linestyle='dashed', linewidth=2)
    plt.text(mean_accuracy*1.02, plt.ylim()[1]*0.9, f'Mean: {mean_accuracy:.2f}', color='r')

    plt.tight_layout()
    plt.show()

    # Print some statistics
    print(f"Mean accuracy: {mean_accuracy:.2f}")
    print(f"Median accuracy: {np.median(accuracies):.2f}")
    print(f"Min accuracy: {min(accuracies):.2f}")
    print(f"Max accuracy: {max(accuracies):.2f}")

plot_unbiased_accuracy_distribution(train_data, biased=False, data_label="train")
plot_unbiased_accuracy_distribution(test_data, biased=False, data_label="test")
plot_unbiased_accuracy_distribution(train_data, biased=True, data_label="train")
plot_unbiased_accuracy_distribution(test_data, biased=True, data_label="test")

# %% Label each biased COT as faithful or unfaithful depending on the unbiased accuracy

faithful_accuracy_threshold = 0.8
unfaithful_accuracy_threshold = 0.4

def label_biased_cots(data):
    for item in data:
        biased_cots_accuracy = item["unbiased_accuracy_for_biased_cots"]
        unbiased_cots_accuracy = item["unbiased_accuracy_for_unbiased_cots"]
        if biased_cots_accuracy >= faithful_accuracy_threshold * unbiased_cots_accuracy:
            item["biased_cot_label"] = "faithful"
        elif biased_cots_accuracy <= unfaithful_accuracy_threshold * unbiased_cots_accuracy:
            item["biased_cot_label"] = "unfaithful"
        else:
            item["biased_cot_label"] = "mixed"

label_biased_cots(train_data)
label_biased_cots(test_data)

# Print number of faithful, unfaithful, and mixed COTs
print(f"Train faithful: {sum(item['biased_cot_label'] == 'faithful' for item in train_data)}")
print(f"Train unfaithful: {sum(item['biased_cot_label'] == 'unfaithful' for item in train_data)}")
print(f"Train mixed: {sum(item['biased_cot_label'] == 'mixed' for item in train_data)}")
print(f"Test faithful: {sum(item['biased_cot_label'] == 'faithful' for item in test_data)}")
print(f"Test unfaithful: {sum(item['biased_cot_label'] == 'unfaithful' for item in test_data)}")
print(f"Test mixed: {sum(item['biased_cot_label'] == 'mixed' for item in test_data)}")

# %%

# Prepare data for probing
min_num_train_data = min(sum(item['biased_cot_label'] == 'faithful' for item in train_data), sum(item['biased_cot_label'] == 'unfaithful' for item in train_data))
train_data_faithful = [item for item in train_data if item['biased_cot_label'] == 'faithful'][:min_num_train_data]
train_data_unfaithful = [item for item in train_data if item['biased_cot_label'] == 'unfaithful'][:min_num_train_data]
probe_train_data = train_data_faithful + train_data_unfaithful

min_num_test_data = min(sum(item['biased_cot_label'] == 'faithful' for item in test_data), sum(item['biased_cot_label'] == 'unfaithful' for item in test_data))
test_data_faithful = [item for item in test_data if item['biased_cot_label'] == 'faithful'][:min_num_test_data]
test_data_unfaithful = [item for item in test_data if item['biased_cot_label'] == 'unfaithful'][:min_num_test_data]
probe_test_data = test_data_faithful + test_data_unfaithful

# %%

from cot_probing.activations import clean_run_with_cache

question_token = tokenizer.encode("Question", add_special_tokens=False)[0]

def collect_activations(data, tokenizer, model, locs_to_cache, layers_to_cache, collect_embeddings=False):
    activations_by_layer_by_locs = {
        loc_type: [[] for _ in range(layers_to_cache)] for loc_type in locs_to_cache.keys()
    }

    for i in tqdm.tqdm(range(len(data))):
        question_to_answer = data[i]["question_to_answer"]
        expected_answer = data[i]["expected_answer"]
        biased_cots = data[i]["biased_cots"]
        biased_fsp = data[i]["biased_fsp"]
        biased_cot_label = data[i]["biased_cot_label"]

        # Build the prompt
        biased_fsp_with_question = tokenizer.encode(biased_fsp + "\n\n" + question_to_answer)
        random_biased_cot = random.choice(biased_cots)
        random_biased_cot.tolist()

        # The prompt is missing the actual answer, so we need to add it
        if expected_answer == "yes":
            answer_tok = tokenizer.encode(" Yes", add_special_tokens=False)
        else:
            answer_tok = tokenizer.encode(" No", add_special_tokens=False)

        input_ids = biased_fsp_with_question + random_biased_cot.tolist() + answer_tok

        if "last_question_tokens" in locs_to_cache:
            # Figure out where the last question starts
            last_question_token_position = [
                pos for pos, t in enumerate(input_ids) if t == question_token
            ][-1]
            locs_to_cache["last_question_tokens"] = (last_question_token_position, None)

        resid_acts_by_layer_by_locs = clean_run_with_cache(
            model, input_ids, locs_to_cache, collect_embeddings=collect_embeddings
        )
        for loc_type in locs_to_cache.keys():
            for layer_idx in range(layers_to_cache):
                activations_by_layer_by_locs[loc_type][layer_idx].append(
                    resid_acts_by_layer_by_locs[loc_type][layer_idx]
                )

    return activations_by_layer_by_locs

# %%

n_layers = model.config.num_hidden_layers
collect_embeddings = False

# Collect activations
locs_to_cache = {
    "last_question_tokens": (None, None),
    # "first_cot_dash": (-1, None),  # last token before CoT
    # "last_new_line": (-2, -1),  # newline before first dash in CoT
    # "step_by_step_colon": (-3, -2),  # colon before last new line.
}

train_activations_by_layer_by_locs = collect_activations(probe_train_data, tokenizer, model, locs_to_cache, n_layers)
test_activations_by_layer_by_locs = collect_activations(probe_test_data, tokenizer, model, locs_to_cache, n_layers)

# %%
from cot_probing.utils import to_str_tokens

last_part_of_last_question = """?
Let's think step by step:
-"""
last_part_of_last_question_tokens = tokenizer.encode(last_part_of_last_question, add_special_tokens=False)
str_tokens = to_str_tokens(last_part_of_last_question_tokens, tokenizer)

locs_to_probe = {}
loc = -1
for str_token in reversed(str_tokens):
    loc_key = f"loc_{loc}_{str_token}"
    locs_to_probe[loc_key] = loc
    loc -= 1

print(locs_to_probe)

# %%

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

def prepare_probe_data(activations_by_layer_by_locs, labels, locs_to_probe, n_layers):
    X = {}
    for loc_key, loc_pos in locs_to_probe.items():
        X[loc_key] = []
        for layer_activations in activations_by_layer_by_locs["last_question_tokens"]:
            # Each tensor has shape [seq len, d_model].
            numpy_arrays = [tensor[loc_pos].float().cpu().numpy() for tensor in layer_activations]
            layer_data = np.array(numpy_arrays)
            X[loc_key].append(layer_data)
    
    y = np.array(labels)
    
    return X, y

# Prepare train and test data
train_labels = [item["biased_cot_label"] for item in probe_train_data]
test_labels = [item["biased_cot_label"] for item in probe_test_data]
X_train, y_train = prepare_probe_data(train_activations_by_layer_by_locs, train_labels, locs_to_probe, n_layers)
X_test, y_test = prepare_probe_data(test_activations_by_layer_by_locs, test_labels, locs_to_probe, n_layers)

# Create a dictionary to store results
results = {
    "loc_type": [],
    "layer": [],
    "accuracy_train": [],
    "accuracy_test": [],
    "y_test": [],
    "y_pred_test": [],
    "probe": [],
}

# Train and evaluate logistic regression probes for each layer and loc_type
for loc_type in locs_to_probe.keys():
    for layer_idx in range(n_layers):
        # Get the activations for the current layer and flatten them
        X_train_layer = X_train[loc_type][layer_idx].reshape(X_train[loc_type][layer_idx].shape[0], -1)
        X_test_layer = X_test[loc_type][layer_idx].reshape(X_test[loc_type][layer_idx].shape[0], -1)

        # Train the logistic regression probe
        probe = LogisticRegression(random_state=42, max_iter=1000)
        probe.fit(X_train_layer, y_train)

        # Make predictions
        y_pred_train = probe.predict(X_train_layer)
        y_pred_test = probe.predict(X_test_layer)

        # Calculate metrics
        accuracy_train = accuracy_score(y_train, y_pred_train)
        accuracy_test = accuracy_score(y_test, y_pred_test)

        # Store results
        results["loc_type"].append(loc_type)
        results["layer"].append(layer_idx)
        results["accuracy_train"].append(accuracy_train)
        results["accuracy_test"].append(accuracy_test)
        results["y_test"].append(y_test)
        results["y_pred_test"].append(y_pred_test)
        results["probe"].append(probe)

        print(f"Location: {loc_type}, Layer {layer_idx}:")
        print(f"  Accuracy (train): {accuracy_train:.4f}")
        print(f"  Accuracy (test): {accuracy_test:.4f}")
        print()

# %%
import pandas as pd

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

for loc_type in locs_to_probe.keys():
    # Sort and print layers by lowest mse_test for this loc_type
    df_loc = df_results[df_results["loc_type"] == loc_type]
    df_loc = df_loc.sort_values(by="accuracy_test", ascending=False)
    print(f"Layers sorted by lowest accuracy_test for {loc_type}")
    with pd.option_context('display.max_rows', 2000):
        # Print the layer and accuracy_test, no index
        print(df_loc[["layer", "accuracy_test"]].to_string(index=False))

    top_5_layers_lowest_accuracy_test = df_loc["layer"].iloc[:5].tolist()
    print(f"Top 5 layers: {top_5_layers_lowest_accuracy_test}")

# %%

import seaborn as sns
import matplotlib.pyplot as plt

# Pivot the DataFrame to create a 2D matrix of accuracy_test values
pivot_df = df_results.pivot(index='layer', columns='loc_type', values='accuracy_test')

# Create the heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_df, cmap='viridis_r', annot=True, fmt='.4f', cbar_kws={'label': 'Accuracy Test'})

plt.title('Accuracy Test by Layer and Location Type (Probing for faithful/unfaithful answer)')
plt.xlabel('Location Type')
plt.ylabel('Layer')
plt.tight_layout()
plt.show()

# %%

# Plot the results for each loc_type
for loc_type in locs_to_probe.keys():
    df_loc = df_results[df_results["loc_type"] == loc_type]

    plt.figure(figsize=(12, 6))
    plt.plot(df_loc["layer"], df_loc["accuracy_train"], label="Accuracy (train)")
    plt.plot(df_loc["layer"], df_loc["accuracy_test"], label="Accuracy (test)")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(f"Logistic Probe Performance by Layer for {loc_type}")
    plt.legend()
    plt.show()

# %%

def plot_scatter(loc_type, layer):
    df_filtered = df_results[
        (df_results["loc_type"] == loc_type) & (df_results["layer"] == layer)
    ]

    if len(df_filtered) == 0:
        print(f"No data found for loc_type '{loc_type}' and layer {layer}")
        return

    y_test = df_filtered["y_test"].iloc[0]
    y_pred_test = df_filtered["y_pred_test"].iloc[0]

    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred_test, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(
        f"Scatter Plot of True vs Predicted Values\nLocation: {loc_type}, Layer: {layer}"
    )

    accuracy = df_filtered["accuracy_test"].iloc[0]

    plt.text(
        0.05,
        0.95,
        f"Accuracy: {accuracy:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()

k = 1
for loc_type in locs_to_probe.keys():
    df_loc = df_results[df_results["loc_type"] == loc_type]
    df_loc = df_loc.sort_values(by="accuracy_test", ascending=False)
    top_k_layers = df_loc["layer"].iloc[:k].tolist()
    for layer in top_k_layers:
        plot_scatter(loc_type, layer)

# %%

yes_acc = []
no_acc = []

print("\nAccuracy for each probe:")
for layer in range(n_layers):
    y_test = results["y_test"][layer]
    y_pred_test = results["y_pred_test"][layer]

    # Accuracy for "all yes" (y_test is 1)
    all_yes_mask = y_test == 1
    all_yes_pred = y_pred_test[all_yes_mask] > 0
    all_yes_accuracy = np.mean(all_yes_pred)
    yes_acc.append(all_yes_accuracy)
    
    # Accuracy for "all no" (y_test is -1)
    all_no_mask = y_test == -1
    all_no_pred = y_pred_test[all_no_mask] < 0
    all_no_accuracy = np.mean(all_no_pred)
    no_acc.append(all_no_accuracy)

    print(f"Layer {layer}:")
    print(f"  All Yes Accuracy: {all_yes_accuracy:.4f}")
    print(f"  All No Accuracy: {all_no_accuracy:.4f}")

plt.figure(figsize=(12, 6))

# Plot for Yes Accuracy
plt.plot(range(n_layers), yes_acc, label='All-Yes Accuracy', marker='o')
plt.plot(range(n_layers), no_acc, label='All-No Accuracy', marker='o')

plt.xlabel('Layer')
plt.ylabel('Accuracy')
plt.title('All-Yes and All-No Accuracy by Layer')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Add x-axis ticks for every 5th layer
plt.xticks(range(0, n_layers, 5))

plt.tight_layout()
plt.show()

# %%

def visualize_top_activating_tokens(loc_type, layer, remove_bos=True):
    df_results = pd.DataFrame(results)
    df_filtered = df_results[
        (df_results["loc_type"] == loc_type) & (df_results["layer"] == layer)
    ]

    if len(df_filtered) == 0:
        print(f"No data found for loc_type '{loc_type}' and layer {layer}")
        return

    probe = df_filtered["probe"].iloc[0]
    probe_coefficients = probe.coef_

    # Get an example of prompt
    prompt_idx = random.randint(0, len(test_prompts) - 1)
    print(f"Using prompt {prompt_idx}")
    prompt = test_prompts[prompt_idx]

    token_ids = tokenizer.encode(prompt) # list of len ~750
    seq_len = len(token_ids)

    # Get the corresponding values
    prompt_acts = collect_activations(
        [prompt], 
        tokenizer,
        model, 
        {"all": (0, None)}, 
        n_layers, collect_embeddings=collect_embeddings
    )["all"][layer][0] # Shape [seq_len, d_model]
    assert prompt_acts.shape == (seq_len, model.config.hidden_size), f"Expected shape ({seq_len}, {model.config.hidden_size}), got {prompt_acts.shape}"

    values = torch.tensor([probe_coefficients @ prompt_acts[i].float().cpu().numpy() for i in range(seq_len)])

    if remove_bos:
        values = values[1:]
        token_ids = token_ids[1:]

    vmin = values.min().item()
    vmax = values.max().item()
    values = values.tolist()

    print(f"vmin: {vmin}, vmax: {vmax}")

    # Visualize the tokens
    html_output = visualize_tokens_html(
        token_ids,
        tokenizer,
        token_values=values,
        vmin=vmin,
        vmax=vmax,
    )

    # Display the HTML output
    from IPython.display import HTML, display

    display(html_output)

k = 1
loc_probe_keys = list(locs_to_probe.keys())

# Keep only the 1,5,8 indexes
loc_probe_keys = [loc_probe_keys[1], loc_probe_keys[5], loc_probe_keys[8]]

for loc_type in loc_probe_keys:
    print(f"Location type: {loc_type}")
    df_loc = df_results[df_results["loc_type"] == loc_type]
    df_loc = df_loc.sort_values(by="accuracy_test", ascending=False)
    layers = df_loc["layer"].iloc[:k].tolist()
    layers = [10]
    for layer in layers:
        print(f"Layer {layer}")
        visualize_top_activating_tokens(loc_type, layer)

# %%

from functools import partial
def steer_generation(input_ids, loc_keys_to_steer, layers_to_steer, last_question_first_token_pos, steer_magnitude, max_new_tokens=200, n_gen=3):
    prompt_len = len(input_ids[0])
    
    def steering_hook_fn(module, input, output_tuple, layer_idx):
        output = output_tuple[0]
        if len(output_tuple) > 1:
            cache = output_tuple[1]
        else:
            cache = None

        # Gather probe directions for the loc_keys_to_steer and this layer
        probe_directions = []
        for loc_key in loc_keys_to_steer:
            # Select the correct probe for this loc_key and layer
            probe = df_results[(df_results["loc_type"] == loc_key) & (df_results["layer"] == layer_idx)]["probe"].iloc[0]
            probe_direction = torch.tensor(probe.coef_)
            probe_directions.append(probe_direction)
        
        # Take the mean of the probe directions
        mean_probe_dir = torch.stack(probe_directions).mean(dim=0).to(model.device)
        
        if output.shape[1] >= last_question_first_token_pos:
            # First pass, cache is empty
            activations = output[:, last_question_first_token_pos:, :]
            output[:, last_question_first_token_pos:, :] = activations + steer_magnitude * mean_probe_dir
        else:
            # We are processing a new token
            assert output.shape[1] == 1
            activations = output[:, 0, :]
            output[:, 0, :] = activations + steer_magnitude * mean_probe_dir

        if cache is not None:
            return (output, cache)
        else:
            return (output,)

    # Register hooks for the selected layers
    hooks = []
    if len(loc_keys_to_steer) > 0:
        for layer_idx in layers_to_steer:
            layer_steering_hook = partial(steering_hook_fn, layer_idx=layer_idx)
            hook = model.model.layers[layer_idx].register_forward_hook(layer_steering_hook)
            hooks.append(hook)

    try:
        # Generate text with steering
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                use_cache=True,
                num_return_sequences=n_gen,
                tokenizer=tokenizer,
                stop_strings=["Yes", "No"],
            )
            responses_tensor = output[:, prompt_len:]
    finally:
        # Remove the hooks
        for hook in hooks:
            hook.remove()

    return responses_tensor

# %%

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

# test_prompt_idx = random.randint(0, len(test_prompts) - 1)
results = []
for test_prompt_idx in tqdm.tqdm(range(len(test_prompts))):
    # print(f"Running steering on test prompt index: {test_prompt_idx}")
    test_prompt = test_prompts[test_prompt_idx]

    input_ids = tokenizer.encode(test_prompt, return_tensors="pt").to(model.device)
        
    # Find the position of the last "Question" token
    question_token_id = tokenizer.encode("Question", add_special_tokens=False)[0]
    last_question_first_token_pos = [i for i, t in enumerate(input_ids[0]) if t == question_token_id][-1]
    # print(f"Last question position first token pos: {last_question_first_token_pos}")

    # print("\nOriginal prompt:")
    # print(test_prompt)

    n_gen = 10

    # print("\nUnsteered generation:")
    unsteered_responses = steer_generation(
        input_ids, 
        [], 
        n_layers, 
        last_question_first_token_pos,
        0,
        n_gen=n_gen
    )
    # for response in unsteered_responses:
    #     print(response)

    loc_probe_keys = list(locs_to_probe.keys())
    loc_keys_to_steer = [
        loc_probe_keys[0],
        loc_probe_keys[1],
        loc_probe_keys[2],
        loc_probe_keys[5],
        loc_probe_keys[8]
    ]
    # print(f"Location keys to steer: {loc_keys_to_steer}")

    layers_to_steer = list(range(8, 19))
    # print(f"Layers to steer: {layers_to_steer}")

    # print("\nPositive steered generation:")
    positive_steered_responses = steer_generation(
        input_ids, 
        loc_keys_to_steer, 
        layers_to_steer, 
        last_question_first_token_pos,
        0.4,
        n_gen=n_gen
    )
    # for i, response in enumerate(positive_steered_responses):
    #     print(f"Response {i}: {response}")
    #     print()

    # print("\nNegative steered generation:")
    negative_steered_responses = steer_generation(
        input_ids, 
        loc_keys_to_steer, 
        layers_to_steer, 
        last_question_first_token_pos,
        -0.4,
        n_gen=n_gen
    )
    # for i, response in enumerate(negative_steered_responses):
    #     print(f"Response {i}: {response}")
    #     print()

    res = {
        "unb": categorize_responses(unsteered_responses),
        "pos_steer": categorize_responses(positive_steered_responses),
        "neg_steer": categorize_responses(negative_steered_responses),
    }

    # for variant in res.keys():
    #     print(f"{variant=}")
    #     for key in ["yes", "no", "other"]:
    #         print(f"- {key} {len(res[variant][key])}")

    results.append(res)
# %%

import pickle
# Dump steering results to file, just in case
if "-8B-" in model_id:
    steering_results_path = DATA_DIR / "steering_results_8B.pkl"
elif "-70B-" in model_id:
    steering_results_path = DATA_DIR / "steering_results_70B.pkl"
else:
    raise ValueError(f"Unknown model: {model_id}")

with open(steering_results_path, "wb") as f:
    pickle.dump(results, f)

# %%

# Plot steering results for 15 random questions
def calculate_yes_percentage(res):
    total = len(res['yes']) + len(res['no'])
    if total == 0:
        return 0
    return (len(res['yes']) - len(res['no'])) / total * 100

# Select 15 random questions
num_questions = 15
selected_indices = random.sample(range(len(results)), num_questions)

# Prepare data
unbiased_percentages = []
positive_steering_percentages = []
negative_steering_percentages = []

for idx in selected_indices:
    res = results[idx]
    unbiased_percentages.append(calculate_yes_percentage(res['unb']))
    positive_steering_percentages.append(calculate_yes_percentage(res['pos_steer']))
    negative_steering_percentages.append(calculate_yes_percentage(res['neg_steer']))

# Create the plot
fig, ax = plt.subplots(figsize=(15, 8))

x = np.arange(num_questions)
width = 0.25

ax.bar(x - width, unbiased_percentages, width, label='Unbiased', color='blue')
ax.bar(x, positive_steering_percentages, width, label='Positive Steering', color='green')
ax.bar(x + width, negative_steering_percentages, width, label='Negative Steering', color='red')

ax.set_ylabel('Percentage of All-Yes & All-No Answers')
ax.set_title('Percentage of All-Yes & All-No Answers: Unbiased vs Positive/Negative Steering')
ax.set_xticks(x)
ax.set_xticklabels([f'Q{i+1}' for i in range(num_questions)])
ax.legend()

plt.ylim(-100, 100)
plt.axhline(y=0, color='k', linestyle='-', linewidth=0.5)

# Add value labels on the bars
# def add_value_labels(ax, rects):
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate(f'{height:.1f}',
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3 if height >= 0 else -3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom' if height >= 0 else 'top',
#                     fontsize=8, rotation=90)

# add_value_labels(ax, ax.containers[0])
# add_value_labels(ax, ax.containers[1])
# add_value_labels(ax, ax.containers[2])

plt.tight_layout()
plt.show()

# %% Plot steering results for all questions

unbiased_percentages = []
positive_steering_percentages = []
negative_steering_percentages = []

for res in results:
    unbiased_percentages.append(calculate_yes_percentage(res['unb']))
    positive_steering_percentages.append(calculate_yes_percentage(res['pos_steer']))
    negative_steering_percentages.append(calculate_yes_percentage(res['neg_steer']))

# Create a DataFrame
df = pd.DataFrame({
    'Unbiased': unbiased_percentages,
    'Positive Steering': positive_steering_percentages,
    'Negative Steering': negative_steering_percentages
})

# Melt the DataFrame to long format
df_melted = df.melt(var_name='Condition', value_name='Percentage')

# Create the boxplot
plt.figure(figsize=(12, 8))
sns.boxplot(x='Condition', y='Percentage', data=df_melted)

plt.title('Distribution of All-Yes & All-No Answer Percentages Across All Questions')
plt.ylabel('Percentage of All-Yes & All-No Answers')
plt.ylim(-110, 110)  # Extend y-axis limits to make room for mean labels
plt.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Add mean values and lines for each condition
for i, condition in enumerate(['Unbiased', 'Positive Steering', 'Negative Steering']):
    mean_val = df_melted[df_melted['Condition'] == condition]['Percentage'].mean()
    plt.hlines(y=mean_val, xmin=i-0.4, xmax=i+0.4, color='red', linestyle='-', linewidth=2)
    plt.text(i, mean_val, f'Mean: {mean_val:.2f}', ha='center', va='bottom', 
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))

plt.tight_layout()
plt.show()

# Print summary statistics
print(df.describe())

# %%

# Plot the percentage of "other" responses

# Calculate percentage of "other" responses for each condition
def calculate_other_percentage(res):
    total = len(res['yes']) + len(res['no']) + len(res['other'])
    if total == 0:
        return 0
    return (len(res['other']) / total) * 100

unbiased_other = []
positive_steering_other = []
negative_steering_other = []

for res in results:
    unbiased_other.append(calculate_other_percentage(res['unb']))
    positive_steering_other.append(calculate_other_percentage(res['pos_steer']))
    negative_steering_other.append(calculate_other_percentage(res['neg_steer']))

# Calculate mean percentages
mean_unbiased_other = np.mean(unbiased_other)
mean_positive_steering_other = np.mean(positive_steering_other)
mean_negative_steering_other = np.mean(negative_steering_other)

# Create the bar plot
plt.figure(figsize=(10, 6))
conditions = ['Unbiased', 'Positive Steering', 'Negative Steering']
percentages = [mean_unbiased_other, mean_positive_steering_other, mean_negative_steering_other]

plt.bar(conditions, percentages, color=['blue', 'green', 'red'])
plt.title('Percentage of "Other" Responses Across Steering Conditions')
plt.ylabel('Percentage of "Other" Responses')
plt.ylim(0, 100)  # Set y-axis limit from 0 to 100%

# Add value labels on top of each bar
for i, v in enumerate(percentages):
    plt.text(i, v + 1, f'{v:.2f}%', ha='center', va='bottom')

plt.tight_layout()
plt.show()

# Print the mean percentages
print(f"Mean percentage of 'other' responses:")
print(f"Unbiased: {mean_unbiased_other:.2f}%")
print(f"Positive Steering: {mean_positive_steering_other:.2f}%")
print(f"Negative Steering: {mean_negative_steering_other:.2f}%")