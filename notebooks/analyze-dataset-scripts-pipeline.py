# %%
%load_ext autoreload
%autoreload 2

import pickle
from collections import Counter

import numpy as np

from cot_probing import DATA_DIR
from cot_probing.qs_generation import Question
from cot_probing.plots import (
    plot_accuracy_difference_histogram,
    plot_accuracy_heatmap,
    plot_accuracy_histogram,
    plot_binary_histogram,
    plot_triple_histogram,
)
from cot_probing.utils import load_tokenizer

import random

# %%
questions_dir = DATA_DIR / "questions"
no_cot_acc_dir = DATA_DIR / "no-cot-accuracy"
unb_cots_eval_dir = DATA_DIR / "unb-cots-eval"
bia_cots_eval_dir = DATA_DIR / "bia-cots-eval"
labeled_qs_dir = DATA_DIR / "labeled-qs"

# %%

full_model_name = "meta-llama/Llama-3.1-8B"
model_name = "Llama-3.1-8B"
dataset_id = "gpt-4o-oct28-1156" # "strategyqa" or "gpt-4o-oct28-1156"
file_name = f"{model_name}_{dataset_id}.pkl"

# %%
tokenizer = load_tokenizer(full_model_name)

# %%
with open(questions_dir / f"{dataset_id}.pkl", "rb") as f:
    questions_dataset: dict[str, Question] = pickle.load(f)

# %%
with open(no_cot_acc_dir / file_name, "rb") as f:
    no_cot_acc_results = pickle.load(f)

print(len(no_cot_acc_results.acc_by_qid))

plot_accuracy_histogram(
    list(no_cot_acc_results.acc_by_qid.values()),
    f"No-CoT Accuracy\n{model_name} on {dataset_id}",
)

# %%
with open(unb_cots_eval_dir / file_name, "rb") as f:
    unb_cots_eval_results = pickle.load(f)

print(len(unb_cots_eval_results.labeled_cots_by_qid))

unb_cot_labels = [
    cot.justified_answer
    for cots in unb_cots_eval_results.labeled_cots_by_qid.values()
    for cot in cots
]

plot_binary_histogram(
    unb_cot_labels,
    f"Unbiased COT Labels\n{model_name} on {dataset_id} (Total CoTs: {len(unb_cot_labels)})",
)

unb_cots_accuracy_by_qid = {
    q_id: np.mean(
        [
            1 if cot.justified_answer == questions_dataset[q_id].expected_answer else 0
            for cot in cots
        ]
    )
    for q_id, cots in unb_cots_eval_results.labeled_cots_by_qid.items()
}

plot_accuracy_histogram(
    list(unb_cots_accuracy_by_qid.values()),
    f"Average accuracy of unbiased COTs per question\n{model_name} on {dataset_id} (total qs: {len(unb_cots_accuracy_by_qid)})",
)

plot_accuracy_difference_histogram(
    unb_cots_accuracy_by_qid,
    no_cot_acc_results.acc_by_qid,    
    f"Accuracy Difference: Unbiased - No-CoT\n{model_name} on {dataset_id}",
    labels=["No-CoT", "Unbiased"],
)

plot_accuracy_heatmap(
    unb_cots_accuracy_by_qid,
    no_cot_acc_results.acc_by_qid,
    f"Unbiased CoT vs No-CoT Accuracy\n{model_name} on {dataset_id}",
    "Unbiased CoT Accuracy",
    "No-CoT Accuracy",
)

# %%
with open(bia_cots_eval_dir / file_name, "rb") as f:
    bia_cots_eval_results = pickle.load(f)

print(len(bia_cots_eval_results.labeled_cots_by_qid))

bia_cot_labels = [
    cot.justified_answer
    for cots in bia_cots_eval_results.labeled_cots_by_qid.values()
    for cot in cots
]

plot_binary_histogram(
    bia_cot_labels,
    f"Biased COT Labels\n{model_name} on {dataset_id} (Total CoTs: {len(bia_cot_labels)})",
)

bia_cots_accuracy_by_qid = {
    q_id: np.mean(
        [
            1 if cot.justified_answer == questions_dataset[q_id].expected_answer else 0
            for cot in cots
        ]
    )
    for q_id, cots in bia_cots_eval_results.labeled_cots_by_qid.items()
}

plot_accuracy_histogram(
    list(bia_cots_accuracy_by_qid.values()),
    f"Average accuracy of biased COTs per question\n{model_name} on {dataset_id} (total qs: {len(bia_cots_accuracy_by_qid)})",
)

plot_accuracy_difference_histogram(
    bia_cots_accuracy_by_qid,
    no_cot_acc_results.acc_by_qid,
    f"Accuracy Difference: Biased - No-CoT\n{model_name} on {dataset_id}",
    labels=["Biased", "No-CoT"],
)

plot_accuracy_difference_histogram(
    bia_cots_accuracy_by_qid,
    unb_cots_accuracy_by_qid,
    f"Accuracy Difference: Biased Unbiased\n{model_name} on {dataset_id}",
    labels=["Unbiased", "Biased"],
)

# Plot heatmap comparing no-CoT accuracy vs biased CoT accuracy
plot_accuracy_heatmap(
    no_cot_acc_results.acc_by_qid,
    bia_cots_accuracy_by_qid,
    f"No-CoT Accuracy vs Biased CoT Accuracy\n{model_name} on {dataset_id}",
    "No-CoT Accuracy",
    "Biased CoT Accuracy",
)

# Plot heatmap comparing unbiased CoT accuracy vs biased CoT accuracy
plot_accuracy_heatmap(
    unb_cots_accuracy_by_qid,
    bia_cots_accuracy_by_qid,
    f"Unbiased CoT vs Biased CoT Accuracy\n{model_name} on {dataset_id}",
    "Unbiased CoT Accuracy",
    "Biased CoT Accuracy",
)

# %%

bia_cots_dir = DATA_DIR / "bia-cots"
with open(bia_cots_dir / file_name, "rb") as f:
    bia_cots_results = pickle.load(f)

# print("Biased CoT: Yes")
# for row in bia_cots_results.bia_yes_fsp:
#     print(f"[{row['role']}]: {row['content']}")

# print("Biased CoT: No")
# for row in bia_cots_results.bia_no_fsp:
#     print(f"[{row['role']}]: {row['content']}")

# %%

# Pick a random question
random_q_id = random.choice(list(bia_cots_accuracy_by_qid.keys()))
# random_q_id = next(
#     q_id
#     for q_id in questions_dataset.keys()
#     if "Canary" in questions_dataset[q_id].question and "European" in questions_dataset[q_id].question
# )

random_q_id = next(
    q_id
    for q_id in questions_dataset.keys()
    if "Is an electric eel technically an eel?" in questions_dataset[q_id].question
)

print(f"Question: {questions_dataset[random_q_id].question}")
expected_answer = questions_dataset[random_q_id].expected_answer
print(f"Expected Answer: {expected_answer}")

if expected_answer == "yes":
    print("Biased CoT: No")
    for row in bia_cots_results.bia_no_fsp:
        print(f"[{row['role']}]: {row['content']}")
else:
    print("Biased CoT: Yes")
    for row in bia_cots_results.bia_yes_fsp:
        print(f"[{row['role']}]: {row['content']}")

# Show the unbiased CoT for this question
for i, labeled_cot in enumerate(unb_cots_eval_results.labeled_cots_by_qid[random_q_id]):
    justified_answer = labeled_cot.justified_answer
    cot = labeled_cot.cot
    print(f"Unbiased CoT {i} ({justified_answer}): {tokenizer.decode(cot)}")

# Show the biased CoT for this question
for i, labeled_cot in enumerate(bia_cots_eval_results.labeled_cots_by_qid[random_q_id]):
    justified_answer = labeled_cot.justified_answer
    cot = labeled_cot.cot
    print(f"Biased CoT {i} ({justified_answer}): {tokenizer.decode(cot)}")

# %%

# %%
with open(labeled_qs_dir / file_name, "rb") as f:
    labeled_qs_results = pickle.load(f)

print(len(labeled_qs_results.label_by_qid))

plot_triple_histogram(
    list(labeled_qs_results.label_by_qid.values()),
    f"Labeled Questions\n{model_name} on {dataset_id} (Total qs: {len(labeled_qs_results.label_by_qid)})",
)

c = Counter(labeled_qs_results.label_by_qid.values())
print(c)

# %%
