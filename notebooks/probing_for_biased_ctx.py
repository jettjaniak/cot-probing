# %%
import tqdm

from cot_probing.typing import *
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%

model_id = "hugging-quants/Meta-Llama-3.1-70B-BNB-NF4-BF16"
# model_id = "hugging-quants/Meta-Llama-3.1-8B-BNB-NF4"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cuda",
)

# %%
from cot_probing import DATA_DIR
from cot_probing.diverse_combinations import load_and_process_file

all_qs_yes = load_and_process_file(DATA_DIR / "diverse_yes.txt")
all_qs_no = load_and_process_file(DATA_DIR / "diverse_no.txt")
assert len(all_qs_yes) == len(all_qs_no)

# %%

# Prepare and label fsps with varying degree of yes and no bias
data_size = 1000
prompts = []
labels = []

fsp_max_len = len(all_qs_yes) - 1  # All questions but one

for _ in range(data_size):
    # Choose how many yes and no questions to include
    num_yes = random.randint(1, fsp_max_len)
    num_no = fsp_max_len - num_yes

    # Split the questions into yes and no
    question_indexes = list(range(len(all_qs_yes)))

    # Pick a random question index
    question_to_answer_index = random.choice(question_indexes)

    # Randomly pick the question to answer from no and yes
    if random.random() < 0.5:
        question_to_answer = all_qs_yes[question_to_answer_index]
    else:
        question_to_answer = all_qs_no[question_to_answer_index]

    # Remove the reasoning and answer in the question to answer
    split_string = "Let's think step by step:\n-"
    question_to_answer = question_to_answer.split(split_string)[0] + split_string

    # Remove the selected question from the list
    question_indexes.remove(question_to_answer_index)

    # Pick the yes questions
    yes_question_indexes = random.sample(question_indexes, num_yes)

    # Use the other indexes for no questions
    no_question_indexes = [q for q in question_indexes if q not in yes_question_indexes]

    # Gather the questions into a list
    fsp_questions = [all_qs_yes[i] for i in yes_question_indexes] + [
        all_qs_no[i] for i in no_question_indexes
    ]

    # Shuffle the questions
    random.shuffle(fsp_questions)

    # Combine the questions into a few-shot prompt
    fsp = "\n\n".join(fsp_questions)

    # Add the question to answer to the end
    prompt = fsp + f"\n\n{question_to_answer}"

    # Add the prompt and label to the lists
    prompts.append(prompt)

    # The target label should be 1 if the fsp is only yes questions, -1 if the fsp is only no questions, and 0 if it's right in the middle (same number of yes and no questions)
    target_label = (num_yes - num_no) / fsp_max_len
    labels.append(target_label)

# %%

from cot_probing.activations import clean_run_with_cache

n_layers = model.config.num_hidden_layers

# Collect activations
locs_to_cache = {
    "before_cot": (-1, None),  # last token
}
activations_by_layer_by_locs = {
    loc_type: [[] for _ in range(n_layers)] for loc_type in locs_to_cache.keys()
}
for prompt in tqdm.tqdm(prompts):
    input_ids = tokenizer.encode(prompt)
    resid_acts_by_layer_by_locs = clean_run_with_cache(model, input_ids, locs_to_cache)
    for loc_type in locs_to_cache.keys():
        for layer_idx in range(n_layers):
            activations_by_layer_by_locs[loc_type][layer_idx].append(
                resid_acts_by_layer_by_locs[loc_type][layer_idx]
            )

# %%

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import torch

# Convert activations and labels to numpy arrays
X = [
    np.array([tensor.float().cpu().numpy() for tensor in layer_activations])
    for layer_activations in activations_by_layer_by_locs["before_cot"]
]
y = np.array(labels)

# Create a dictionary to store results
results = {
    "layer": [],
    "mse_train": [],
    "mse_test": [],
    "r2_train": [],
    "r2_test": [],
}

# Train and evaluate linear probes for each layer
for layer_idx in range(n_layers):
    # Get the activations for the current layer and flatten them
    X_layer = X[layer_idx].reshape(X[layer_idx].shape[0], -1)

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_layer, y, test_size=0.15, random_state=42
    )

    # Train the linear probe
    probe = LinearRegression()
    probe.fit(X_train, y_train)

    # Make predictions
    y_pred_train = probe.predict(X_train)
    y_pred_test = probe.predict(X_test)

    # Calculate metrics
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_test = mean_squared_error(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)

    # Store results
    results["layer"].append(layer_idx)
    results["mse_train"].append(mse_train)
    results["mse_test"].append(mse_test)
    results["r2_train"].append(r2_train)
    results["r2_test"].append(r2_test)

    print(f"Layer {layer_idx}:")
    print(f"  MSE (train): {mse_train:.4f}")
    print(f"  MSE (test): {mse_test:.4f}")
    print(f"  R2 (train): {r2_train:.4f}")
    print(f"  R2 (test): {r2_test:.4f}")
    print()

# %%

import pandas as pd
import matplotlib.pyplot as plt

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

# Plot the results
plt.figure(figsize=(12, 6))
plt.plot(df_results["layer"], df_results["mse_train"], label="MSE (train)")
plt.plot(df_results["layer"], df_results["mse_test"], label="MSE (test)")
plt.xlabel("Layer")
plt.ylabel("Mean Squared Error")
plt.title("Linear Probe Performance by Layer")
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(df_results["layer"], df_results["r2_train"], label="R2 (train)")
plt.plot(df_results["layer"], df_results["r2_test"], label="R2 (test)")
plt.xlabel("Layer")
plt.ylabel("R2 Score")
plt.title("Linear Probe Performance by Layer")
plt.legend()
plt.show()

# %%
