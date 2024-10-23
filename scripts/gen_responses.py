#!/usr/bin/env python3
from tqdm.auto import trange
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing import DATA_DIR
from cot_probing.typing import *

# model_id = "hugging-quants/Meta-Llama-3.1-70B-BNB-NF4-BF16"
model_id = "hugging-quants/Meta-Llama-3.1-8B-BNB-NF4-BF16"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    device_map="cuda",
)

import os
import pickle

from cot_probing.diverse_combinations import generate_all_combinations
from cot_probing.generation import analyze_responses

responses_by_seed = {}

# Load pickle file if it exists
if os.path.exists(DATA_DIR / "responses_by_seed.pkl"):
    with open(DATA_DIR / "responses_by_seed.pkl", "rb") as f:
        responses_by_seed = pickle.load(f)

for seed in trange(3, desc="Seeds"):
    if seed in responses_by_seed:
        print(f"Skipping seed {seed} because it already exists")
        continue

    all_combinations = generate_all_combinations(seed=seed)
    all_responses = analyze_responses(
        model=model,
        tokenizer=tokenizer,
        all_combinations=all_combinations,
        max_new_tokens=120,
        temp=0.8,
        n_gen=10,
        seed=seed,
    )  # [{ "unb": {"yes": [], "no": [], "other": []}, "bias_no": {"yes": [], "no": [], "other": []}}]
    responses_by_seed[seed] = all_responses

    # Dump responses by seed to disk
    with open(DATA_DIR / "responses_by_seed.pkl", "wb") as f:
        pickle.dump(responses_by_seed, f)
