# %%
%load_ext autoreload
%autoreload 2
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import torch

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

# To avoid warnings
model.generation_config.pad_token_id = tokenizer.pad_token_id

# %%
import json
from cot_probing import DATA_DIR
questions_dataset_path = DATA_DIR / "generated_questions_dataset.json"

question_dataset = []
if os.path.exists(questions_dataset_path):
    with open(questions_dataset_path, "r") as f:
        question_dataset = json.load(f)

# %%

from cot_probing.diverse_combinations import load_and_process_file

all_qs_yes = load_and_process_file(DATA_DIR / "diverse_yes.txt")
all_qs_no = load_and_process_file(DATA_DIR / "diverse_no.txt")
assert len(all_qs_yes) == len(all_qs_no)

# Add questions to all_qs_yes and all_qs_no so that we don't repeat them
for row in question_dataset:
    if row["expected_answer"] == "yes":
        all_qs_yes.append(row["question"])
    else:
        all_qs_no.append(row["question"])

# %%
from cot_probing.questions_generation import generate_questions_dataset

# Generate the dataset
generate_questions_dataset(
    model=model,
    tokenizer=tokenizer,
    openai_model="gpt-4o",
    num_questions=1,
    all_qs_yes=all_qs_yes,
    all_qs_no=all_qs_no,
    max_attempts=100,
    questions_dataset_path=questions_dataset_path,
    verbose=True
)
# %%

for question in question_dataset:
    print(question["question"])
    print()

