# %%
%load_ext autoreload
%autoreload 2
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing.typing import *
from cot_probing.vis import visualize_tokens_html

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
data_size = 500
prompts = []
labels = []

# fsp_max_len = len(all_qs_yes) - 1  # All questions but one
fsp_max_len = 10

# Set aside 20% of questions for test data only
num_test_only_questions = int(0.2 * len(all_qs_yes))
test_only_indices = set(random.sample(range(len(all_qs_yes)), num_test_only_questions))
train_indices = set(range(len(all_qs_yes))) - test_only_indices

for i in range(data_size):
    # Determine the type of prompt based on the desired distribution
    if i < 0.3 * data_size:
        # All yes
        num_yes = fsp_max_len
        num_no = 0
    elif i < 0.6 * data_size:
        # All no
        num_yes = 0
        num_no = fsp_max_len
    else:
        # Mixed
        num_yes = random.randint(1, fsp_max_len - 1)
        num_no = fsp_max_len - num_yes

    # Split the questions into yes and no, excluding test-only questions
    question_indexes = list(train_indices)

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
    indexes_available_for_no_questions = [
        q for q in question_indexes if q not in yes_question_indexes
    ]
    no_question_indexes = random.sample(indexes_available_for_no_questions, num_no)

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
collect_embeddings = True
question_token = tokenizer.encode("Question", add_special_tokens=False)[0]

# Collect activations
locs_to_cache = {
    "last_question_tokens": None,
    # "first_cot_dash": (-1, None),  # last token before CoT
    # "last_new_line": (-2, -1),  # newline before first dash in CoT
    # "step_by_step_colon": (-3, -2),  # colon before last new line.
}
activations_by_layer_by_locs = {
    loc_type: [[] for _ in range(n_layers)] for loc_type in locs_to_cache.keys()
}

tokenized_prompts = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in prompts]

# Add left padding to the prompts so that they are all the same length
# max_len = max([len(input_ids) for input_ids in tokenized_prompts])
# tokenized_prompts = [
#     [tokenizer.pad_token_id] * (max_len - len(input_ids)) + input_ids
#     for input_ids in tokenized_prompts
# ]

for input_ids in tqdm.tqdm(tokenized_prompts):
    # Figure out where does the last question start
    last_question_token_position = [
        pos for pos, t in enumerate(input_ids) if t == question_token
    ][-1]
    locs_to_cache["last_question_tokens"] = (last_question_token_position, None)

    resid_acts_by_layer_by_locs = clean_run_with_cache(
        model, input_ids, locs_to_cache, collect_embeddings=collect_embeddings
    )
    for loc_type in locs_to_cache.keys():
        for layer_idx in range(n_layers):
            activations_by_layer_by_locs[loc_type][layer_idx].append(
                resid_acts_by_layer_by_locs[loc_type][layer_idx]
            )

# %%

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Convert activations and labels to numpy arrays
X = {
    loc_type: [
        np.array([tensor.float().cpu().numpy() for tensor in layer_activations])
        for layer_activations in activations_by_layer_by_locs[loc_type]
    ]
    for loc_type in locs_to_cache.keys()
}
y = np.array(labels)

# Create a dictionary to store results
results = {
    "loc_type": [],
    "layer": [],
    "mse_train": [],
    "mse_test": [],
    "y_test": [],
    "y_pred_test": [],
    "probe": [],
}

# Create test data with test-only questions
test_size = int(0.15 * data_size)
test_prompts = []
test_labels = []

for _ in range(test_size):
    # Determine the type of prompt based on the desired distribution
    if random.random() < 0.3:
        # All yes
        num_yes = fsp_max_len
        num_no = 0
    elif random.random() < 0.6:
        # All no
        num_yes = 0
        num_no = fsp_max_len
    else:
        # Mixed
        num_yes = random.randint(1, fsp_max_len - 1)
        num_no = fsp_max_len - num_yes

    # Use only test-only indices for question selection
    question_indexes = list(test_only_indices)

    # Pick a random question index for the question to answer
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
    indexes_available_for_no_questions = [
        q for q in question_indexes if q not in yes_question_indexes
    ]
    no_question_indexes = random.sample(indexes_available_for_no_questions, num_no)

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
    test_prompts.append(prompt)

    # The target label should be 1 if the fsp is only yes questions, -1 if the fsp is only no questions, and 0 if it's right in the middle (same number of yes and no questions)
    target_label = (num_yes - num_no) / fsp_max_len
    test_labels.append(target_label)

# Collect activations for test data
test_activations_by_layer_by_locs = {
    loc_type: [[] for _ in range(n_layers)] for loc_type in locs_to_cache.keys()
}
for prompt in tqdm.tqdm(test_prompts):
    input_ids = tokenizer.encode(prompt)
    resid_acts_by_layer_by_locs = clean_run_with_cache(model, input_ids, locs_to_cache)
    for loc_type in locs_to_cache.keys():
        for layer_idx in range(n_layers):
            test_activations_by_layer_by_locs[loc_type][layer_idx].append(
                resid_acts_by_layer_by_locs[loc_type][layer_idx]
            )

# Convert test activations to numpy arrays
X_test = {
    loc_type: [
        np.array([tensor.float().cpu().numpy() for tensor in layer_activations])
        for layer_activations in test_activations_by_layer_by_locs[loc_type]
    ]
    for loc_type in locs_to_cache.keys()
}
y_test = np.array(test_labels)

# Train and evaluate linear probes for each layer and loc_type
for loc_type in locs_to_cache.keys():
    for layer_idx in range(n_layers):
        # Get the activations for the current layer and flatten them
        X_layer = X[loc_type][layer_idx].reshape(X[loc_type][layer_idx].shape[0], -1)
        X_test_layer = X_test[loc_type][layer_idx].reshape(
            X_test[loc_type][layer_idx].shape[0], -1
        )

        # Train the linear probe
        probe = LinearRegression()
        probe.fit(X_layer, y)

        # Make predictions
        y_pred_train = probe.predict(X_layer)
        y_pred_test = probe.predict(X_test_layer)

        # Calculate metrics
        mse_train = mean_squared_error(y, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)

        # Store results
        results["loc_type"].append(loc_type)
        results["layer"].append(layer_idx)
        results["mse_train"].append(mse_train)
        results["mse_test"].append(mse_test)
        results["y_test"].append(y_test)
        results["y_pred_test"].append(y_pred_test)
        results["probe"].append(probe)

        print(f"Location: {loc_type}, Layer {layer_idx}:")
        print(f"  MSE (train): {mse_train:.4f}")
        print(f"  MSE (test): {mse_test:.4f}")
        print()

# %%

import matplotlib.pyplot as plt
import pandas as pd

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

# Plot the results for each loc_type
for loc_type in locs_to_cache.keys():
    df_loc = df_results[df_results["loc_type"] == loc_type]

    plt.figure(figsize=(12, 6))
    plt.plot(df_loc["layer"], df_loc["mse_train"], label="MSE (train)")
    plt.plot(df_loc["layer"], df_loc["mse_test"], label="MSE (test)")
    plt.xlabel("Layer")
    plt.ylabel("Mean Squared Error")
    plt.title(f"Linear Probe Performance by Layer for {loc_type}")
    plt.legend()
    plt.show()

# %%


def plot_scatter(loc_type, layer):
    df_results = pd.DataFrame(results)
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

    mse = df_filtered["mse_test"].iloc[0]
    plt.text(
        0.05,
        0.95,
        f"MSE: {mse:.4f}",
        transform=plt.gca().transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    plt.tight_layout()
    plt.show()


# Example usage:
for loc_type in locs_to_cache.keys():
    for layer in [0, 79]:
        plot_scatter(loc_type, layer)


# %%


# def visualize_top_activating_tokens(loc_type, layer):
#     df_results = pd.DataFrame(results)
#     df_filtered = df_results[
#         (df_results["loc_type"] == loc_type) & (df_results["layer"] == layer)
#     ]

#     if len(df_filtered) == 0:
#         print(f"No data found for loc_type '{loc_type}' and layer {layer}")
#         return

#     probe = df_filtered["probe"].iloc[0]
#     coefficients = probe.coef_

#     # Get an example of prompt
#     prompt_idx = random.randint(0, len(prompts))
#     prompt = prompts[prompt_idx]
#     token_ids = tokenizer.encode(prompt)

#     # Get the corresponding values
#     prompt_acts = X[loc_type][layer][prompt_idx]

#     # Visualize the tokens
#     html_output = visualize_tokens_html(
#         token_ids.tolist(),
#         tokenizer,
#         token_values=values.tolist(),
#         vmin=values.min(),
#         vmax=values.max(),
#     )

#     # Display the HTML output
#     from IPython.display import HTML, display

#     display(HTML(html_output))

#     print(f"Top {top_k} activating tokens for {loc_type}, layer {layer}")


# # Example usage:
# for loc_type in locs_to_cache.keys():
#     for layer in [0, 79]:
#         visualize_top_activating_tokens(loc_type, layer)

# %%
