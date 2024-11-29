# %%
import os
import pickle
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from openai import OpenAI
from tqdm import tqdm

from cot_probing.cot_evaluation import get_justified_answer

# %%
with open("../labeled_qs_with-unbiased-cots-oct28-1156.json", "r") as f:
    old_labels_dataset = json.load(f)

# %%
openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
openai_model = "gpt-4o"

# %%

save_frequency = 50
transplanted_answers_path = "transplanted_answers.pkl"
openai_answers_path = "openai_answers.pkl"

transplanted_answers = []
openai_answers = []
for q in tqdm(old_labels_dataset["qs"]):
    q_str = q["question"]
    q_str = q_str.split("\nLet's think step by step:")[0]
    q_str = q_str.split("Question: ")[1]

    for cot in q["biased_cots"]:
        transplanted_answers.append(cot["answer"])

        cot_str = cot["cot"]
        cot_str = cot_str.split("\nLet's think step by step:\n-")[1]
        cot_str = cot_str.split("\nAnswer:")[0]

        openai_answer = get_justified_answer(
            q_str=q_str,
            cot=cot_str,
            openai_client=openai_client,
            openai_model=openai_model,
        )
        if openai_answer is None:
            openai_answer = "unk"
        openai_answers.append(openai_answer)

    if len(transplanted_answers) % save_frequency == 0:
        with open(transplanted_answers_path, "wb") as f:
            pickle.dump(transplanted_answers, f)
        with open(openai_answers_path, "wb") as f:
            pickle.dump(openai_answers, f)

# %%

openai_answers = [a if a == "yes" or a == "no" else "other" for a in openai_answers]
# %%

# Get unique labels from both answer sets
unique_labels = sorted(list(set(transplanted_answers + openai_answers)))

# Create confusion matrix
cm = confusion_matrix(transplanted_answers, openai_answers, labels=unique_labels)

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    cm, annot=True, fmt="d", xticklabels=unique_labels, yticklabels=unique_labels
)

plt.title("Confusion Matrix: Transplanted vs OpenAI Answers")
plt.xlabel("OpenAI Answers")
plt.ylabel("Transplanted Answers")
plt.show()

# Optional: Print accuracy
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print(f"Accuracy: {accuracy:.2%}")

# %%

# Show examples of disagreement

max_to_show = 10
shown = 0
index = 0
for q in old_labels_dataset["qs"]:
    q_str = q["question"]
    q_str = q_str.split("\nLet's think step by step:")[0]
    q_str = q_str.split("Question: ")[1]

    for cot in q["biased_cots"]:
        cot_str = cot["cot"]
        # cot_str = cot_str.split("\nLet's think step by step:\n-")[1]
        cot_str = cot_str.split("\nAnswer:")[0]

        if transplanted_answers[index] != openai_answers[index]:
            # print(f"Question: {q_str}")
            print(f"{cot_str}")
            print(f"-> Expected: {q['expected_answer']}")
            print(f"-> Transplanted: {transplanted_answers[index]}")
            print(f"-> OpenAI: {openai_answers[index]}")
            print()
            shown += 1
        index += 1
        if shown >= max_to_show:
            break

    if shown >= max_to_show:
        break
