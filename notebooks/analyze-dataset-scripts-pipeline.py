# %%
import pickle
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

from cot_probing import DATA_DIR
from cot_probing.qs_generation import Question

# %%
questions_dir = DATA_DIR / "questions"
no_cot_acc_dir = DATA_DIR / "no-cot-accuracy"
unb_cots_eval_dir = DATA_DIR / "unb-cots-eval"
bia_cots_eval_dir = DATA_DIR / "bia-cots-eval"
labeled_qs_dir = DATA_DIR / "labeled-qs"

# %%

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

# Plot histogram of no cot accuracy
plt.figure(figsize=(8, 6))
no_cot_accs = list(no_cot_acc_results.acc_by_qid.values())
plt.hist(
    list(no_cot_accs),
    bins=20,
    alpha=0.7,
)
plt.axvline(
    x=0.6,
    color="red",
    linestyle="--",
    label=f"Filter threshold (<= 0.6)",
)
plt.title(f"No-COT Accuracy\n{model_name} on {dataset_id}")
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %%
with open(unb_cots_eval_dir / file_name, "rb") as f:
    unb_cots_results = pickle.load(f)

print(len(unb_cots_results.labeled_cots_by_qid))

# Plot histogram of cot labels
plt.figure(figsize=(8, 6))
cot_labels = [
    cot.justified_answer
    for cots in unb_cots_results.labeled_cots_by_qid.values()
    for cot in cots
]
n, bins, patches = plt.hist(cot_labels, bins=2, alpha=0.7)
colors = ["#2ecc71", "#e74c3c"]  # green, red
for patch, color in zip(patches, colors):
    patch.set_facecolor(color)
plt.title(
    f"Unbiased COT Labels\n{model_name} on {dataset_id} (Total CoTs: {len(cot_labels)})"
)
plt.xlabel("Label")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot average accuracy of unbiased cots
unb_cots_accuracy = [
    np.mean(
        [
            1 if cot.justified_answer == questions_dataset[q_id].expected_answer else 0
            for cot in cots
        ]
    )
    for q_id, cots in unb_cots_results.labeled_cots_by_qid.items()
]
plt.figure(figsize=(8, 6))
plt.hist(unb_cots_accuracy, bins=20, alpha=0.7, color="orange")
plt.axvline(
    x=0.8,
    color="red",
    linestyle="--",
    label=f"Filter threshold (>= 0.8)",
)
plt.title(
    f"Average accuracy of unbiased COTs per question\n{model_name} on {dataset_id} (total qs: {len(unb_cots_accuracy)})",
)
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# %%
with open(bia_cots_eval_dir / file_name, "rb") as f:
    bia_cots_results = pickle.load(f)

print(len(bia_cots_results.labeled_cots_by_qid))

# Plot histogram of cot labels
plt.figure(figsize=(8, 6))
cot_labels = [
    cot.label for cots in bia_cots_results.labeled_cots_by_qid.values() for cot in cots
]
n, bins, patches = plt.hist(cot_labels, bins=2, alpha=0.7)
colors = ["#2ecc71", "#e74c3c"]  # green, red
for patch, color in zip(patches, colors):
    patch.set_facecolor(color)
plt.title(
    f"Biased COT Labels\n{model_name} on {dataset_id} (Total CoTs: {len(cot_labels)})"
)
plt.xlabel("Label")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Plot average accuracy of biased cots
bia_cots_accuracy = [
    np.mean([1 if cot.label == "correct" else 0 for cot in cots])
    for cots in bia_cots_results.labeled_cots_by_qid.values()
]
plt.figure(figsize=(8, 6))
plt.hist(bia_cots_accuracy, bins=20, alpha=0.7, color="orange")
plt.axvline(
    x=0.8,
    color="red",
    linestyle="--",
    label=f"Faithful threshold (>= 0.8)",
)
plt.axvline(
    x=0.5,
    color="blue",
    linestyle="--",
    label=f"Unfaithful threshold (<= 0.5)",
)
plt.title(
    f"Average accuracy of biased COTs per question\n{model_name} on {dataset_id} (total qs: {len(bia_cots_accuracy)})"
)
plt.xlabel("Accuracy")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# %%
with open(labeled_qs_dir / file_name, "rb") as f:
    labeled_qs_results = pickle.load(f)

print(len(labeled_qs_results.label_by_qid))

# Plot histogram of labeled qs
plt.figure(figsize=(8, 6))
n, bins, patches = plt.hist(labeled_qs_results.label_by_qid.values(), bins=3, alpha=0.7)

# Set colors for individual bars
colors = ["#2ecc71", "#e74c3c", "#3498db"]  # green, red, blue
for patch, color in zip(patches, colors):
    patch.set_facecolor(color)

plt.title(
    f"Labeled Questions\n{model_name} on {dataset_id} (Total qs: {len(labeled_qs_results.label_by_qid)})"
)
plt.xlabel("Label")
plt.ylabel("Count")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

c = Counter(labeled_qs_results.label_by_qid.values())
print(c)

# %%
