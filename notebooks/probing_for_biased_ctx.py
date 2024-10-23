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

train_all_qs_yes = load_and_process_file(DATA_DIR / "diverse_yes.txt")
train_all_qs_no = load_and_process_file(DATA_DIR / "diverse_no.txt")
assert len(train_all_qs_yes) == len(train_all_qs_no)

test_all_qs_yes = load_and_process_file(DATA_DIR / "probing_all_yes.txt")
test_all_qs_no = load_and_process_file(DATA_DIR / "probing_all_no.txt")
assert len(test_all_qs_yes) == len(test_all_qs_no)

# %%
def generate_prompts_and_labels(
    all_qs_yes,
    all_qs_no,
    num_samples,
    fsp_max_len,
):
    prompts = []
    labels = []
    question_indices = set(range(len(all_qs_yes)))

    for _ in range(num_samples):
        # Determine the type of prompt based on the desired distribution
        if random.random() < 0.3:
            num_yes, num_no = fsp_max_len, 0
        elif random.random() < 0.6:
            num_yes, num_no = 0, fsp_max_len
        else:
            num_yes = random.randint(1, fsp_max_len - 1)
            num_no = fsp_max_len - num_yes

        # Pick a random question index for the question to answer
        question_to_answer_index = random.choice(list(question_indices))

        # Randomly pick the question to answer from no and yes
        question_to_answer = random.choice([all_qs_yes, all_qs_no])[question_to_answer_index]

        # Remove the reasoning and answer in the question to answer
        split_string = "Let's think step by step:\n-"
        question_to_answer = question_to_answer.split(split_string)[0] + split_string

        # Remove the selected question from the list
        available_indices = question_indices - {question_to_answer_index}

        # Pick the yes and no questions
        yes_question_indexes = random.sample(list(available_indices), num_yes)
        no_question_indexes = random.sample(list(available_indices - set(yes_question_indexes)), num_no)

        # Gather and shuffle the questions
        fsp_questions = [all_qs_yes[i] for i in yes_question_indexes] + [all_qs_no[i] for i in no_question_indexes]
        random.shuffle(fsp_questions)

        # Combine the questions into a few-shot prompt
        prompt = "\n\n".join(fsp_questions) + f"\n\n{question_to_answer}"

        # Add the prompt and label to the lists
        prompts.append(prompt)
        labels.append((num_yes - num_no) / fsp_max_len)

    return prompts, labels

# %%

train_fsp_max_len = len(train_all_qs_yes) - 1  # All questions but one
train_size = 300

test_size = 50
test_fsp_max_len = min(train_fsp_max_len, len(test_all_qs_yes) - 1)

train_prompts, train_labels = generate_prompts_and_labels(
    train_all_qs_yes, train_all_qs_no, train_size, train_fsp_max_len)
test_prompts, test_labels = generate_prompts_and_labels(
    test_all_qs_yes, test_all_qs_no, test_size, test_fsp_max_len)

# %%

from cot_probing.activations import clean_run_with_cache

def collect_activations(prompts, tokenizer, model, locs_to_cache, layers_to_cache, collect_embeddings=False):
    activations_by_layer_by_locs = {
        loc_type: [[] for _ in range(layers_to_cache)] for loc_type in locs_to_cache.keys()
    }
    question_token = tokenizer.encode("Question", add_special_tokens=False)[0]

    tokenized_prompts = [tokenizer.encode(prompt) for prompt in prompts]

    # Add left padding to the prompts so that they are all the same length
    # max_len = max([len(input_ids) for input_ids in tokenized_prompts])
    # tokenized_prompts = [
    #     [tokenizer.pad_token_id] * (max_len - len(input_ids)) + input_ids
    #     for input_ids in tokenized_prompts
    # ]

    for input_ids in tqdm.tqdm(tokenized_prompts):
        # Figure out where the last question starts
        if "last_question_tokens" in locs_to_cache:
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
question_token = tokenizer.encode("Question", add_special_tokens=False)[0]

# Collect activations
locs_to_cache = {
    "last_question_tokens": None,
    # "first_cot_dash": (-1, None),  # last token before CoT
    # "last_new_line": (-2, -1),  # newline before first dash in CoT
    # "step_by_step_colon": (-3, -2),  # colon before last new line.
}

train_activations_by_layer_by_locs = collect_activations(train_prompts, tokenizer, model, locs_to_cache, n_layers)
test_activations_by_layer_by_locs = collect_activations(test_prompts, tokenizer, model, locs_to_cache, n_layers)

# %%

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

def prepare_probe_data(activations_by_layer_by_locs, labels, locs_to_cache, n_layers):
    X = {}
    for loc_type in locs_to_cache.keys():
        X[loc_type] = []
        for layer_activations in activations_by_layer_by_locs[loc_type]:
            # Each tensor has shape [seq len, d_model]. Using -1 retains only last token for all prompts
            numpy_arrays = [tensor[-1].float().cpu().numpy() for tensor in layer_activations]
            layer_data = np.array(numpy_arrays)
            X[loc_type].append(layer_data)
    
    y = np.array(labels)
    
    return X, y

# Prepare train and test data
X_train, y_train = prepare_probe_data(train_activations_by_layer_by_locs, train_labels, locs_to_cache, n_layers)
X_test, y_test = prepare_probe_data(test_activations_by_layer_by_locs, test_labels, locs_to_cache, n_layers)

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

# Train and evaluate linear probes for each layer and loc_type
for loc_type in locs_to_cache.keys():
    for layer_idx in range(n_layers):
        # Get the activations for the current layer and flatten them
        X_train_layer = X_train[loc_type][layer_idx].reshape(X_train[loc_type][layer_idx].shape[0], -1)
        X_test_layer = X_test[loc_type][layer_idx].reshape(
            X_test[loc_type][layer_idx].shape[0], -1
        )

        # Train the linear probe
        probe = LinearRegression()
        probe.fit(X_train_layer, y_train)

        # Make predictions
        y_pred_train = probe.predict(X_train_layer)
        y_pred_test = probe.predict(X_test_layer)

        # Calculate metrics
        mse_train = mean_squared_error(y_train, y_pred_train)
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

# Sort and print layers by lowest mse_test
df_results = pd.DataFrame(results)
df_results = df_results.sort_values(by="mse_test", ascending=True)
print("Layers sorted by lowest mse_test")
with pd.option_context('display.max_rows', 2000):
    # Print the layer and mse_test, no index
    print(df_results[["layer", "mse_test"]].to_string(index=False))

top_5_layers_lowest_mse_test = df_results["layer"].iloc[:5].tolist()
print(f"Top 5 layers: {top_5_layers_lowest_mse_test}")

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
    for layer in top_5_layers_lowest_mse_test[:3]:
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
    )["all"][layer][0] # Shape [seq_len, 8192]
    assert prompt_acts.shape == (seq_len, 8192), f"Expected shape ({seq_len}, 8192), got {prompt_acts.shape}"

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

    display(HTML(html_output))

# Example usage:
for loc_type in locs_to_cache.keys():
    for layer in top_5_layers_lowest_mse_test[:1]:
        visualize_top_activating_tokens(loc_type, layer)

# %%


