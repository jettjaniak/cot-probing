# %%
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

# %%
questions_dir = DATA_DIR / "questions"
no_cot_acc_dir = DATA_DIR / "no-cot-accuracy"
unb_cots_eval_dir = DATA_DIR / "unb-cots-eval"
bia_cots_eval_dir = DATA_DIR / "bia-cots-eval"
labeled_qs_dir = DATA_DIR / "labeled-qs"

# %%

full_model_name = "google/gemma-2-2b-it"
model_name = "gemma-2-2b-it"
dataset_id = "gpt-4o-oct28-1156"
file_name = f"{model_name}_{dataset_id}.pkl"

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

unb_cots_accuracy = [
    np.mean(
        [
            1 if cot.justified_answer == questions_dataset[q_id].expected_answer else 0
            for cot in cots
        ]
    )
    for q_id, cots in unb_cots_eval_results.labeled_cots_by_qid.items()
]
plot_accuracy_histogram(
    unb_cots_accuracy,
    f"Average accuracy of unbiased COTs per question\n{model_name} on {dataset_id} (total qs: {len(unb_cots_accuracy)})",
)

plot_accuracy_difference_histogram(
    unb_cots_accuracy,
    list(no_cot_acc_results.acc_by_qid.values()),
    f"Accuracy Difference: Unbiased vs No-CoT\n{model_name} on {dataset_id}",
    labels=["Unbiased", "No-CoT"],
)

plot_accuracy_heatmap(
    unb_cots_accuracy,
    list(no_cot_acc_results.acc_by_qid.values()),
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

bia_cots_accuracy = [
    np.mean(
        [
            1 if cot.justified_answer == questions_dataset[q_id].expected_answer else 0
            for cot in cots
        ]
    )
    for q_id, cots in bia_cots_eval_results.labeled_cots_by_qid.items()
]
plot_accuracy_histogram(
    bia_cots_accuracy,
    f"Average accuracy of biased COTs per question\n{model_name} on {dataset_id} (total qs: {len(bia_cots_accuracy)})",
)

plot_accuracy_difference_histogram(
    bia_cots_accuracy,
    list(no_cot_acc_results.acc_by_qid.values()),
    f"Accuracy Difference: Biased vs No-CoT\n{model_name} on {dataset_id}",
    labels=["Biased", "No-CoT"],
)

plot_accuracy_difference_histogram(
    unb_cots_accuracy,
    bia_cots_accuracy,
    f"Accuracy Difference: Unbiased vs Biased\n{model_name} on {dataset_id}",
    labels=["Unbiased", "Biased"],
)

# Plot heatmap comparing no-CoT accuracy vs biased CoT accuracy
plot_accuracy_heatmap(
    list(no_cot_acc_results.acc_by_qid.values()),
    bia_cots_accuracy,
    f"Question Difficulty vs Biased CoT Accuracy\n{model_name} on {dataset_id}",
    "No-CoT Accuracy (Question Difficulty)",
    "Biased CoT Accuracy",
)

# Plot heatmap comparing unbiased CoT accuracy vs biased CoT accuracy
plot_accuracy_heatmap(
    unb_cots_accuracy,
    bia_cots_accuracy,
    f"Unbiased CoT vs Biased CoT Accuracy\n{model_name} on {dataset_id}",
    "Unbiased CoT Accuracy",
    "Biased CoT Accuracy",
)

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
