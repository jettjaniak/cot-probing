# %%
# Load data

# From json
eval_results_path = "../results/eval_google--gemma-2-2b_snarks_S0_N151.pkl"
activations_results_path = (
    "../results/activations_google--gemma-2-2b_snarks_S0_N151.pkl"
)


# %%

import pickle

from cot_probing.activations import (
    Activations,
    QuestionActivations,
    clean_run_with_cache_sigle_batch,
)
from cot_probing.typing import *

# %%
with open(eval_results_path, "rb") as f:
    eval_results = pickle.load(f)

model_name = eval_results.model_name
task_name = eval_results.task_name
seed = eval_results.seed
num_samples = eval_results.num_samples
questions = eval_results.questions

print(
    f"Processing {model_name} on {task_name} with seed {seed} and {num_samples} samples"
)

# %%
import os
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
# %%
import torch

torch.set_grad_enabled(False)
import tqdm

layers_to_cache = list(range(model.config.num_hidden_layers))
activations_by_question: list[QuestionActivations] = []
for question in tqdm.tqdm(questions):
    tokens = torch.tensor(question.tokens).cuda().unsqueeze(0)

    locs_to_cache = set()
    for key, locs in question.locs.items():
        locs_to_cache.update(locs)
    locs_to_cache = sorted(locs_to_cache)

    cache = clean_run_with_cache_sigle_batch(
        model, tokens, layers_to_cache, locs_to_cache
    )

    activations_by_question.append(QuestionActivations(cache.cpu(), locs_to_cache))
# %%
# Dump cache.cache_dict to disk
import pickle

activations = Activations(eval_results, activations_by_question, layers_to_cache)

with open(activations_results_path, "wb") as f:
    pickle.dump(activations, f)

# %%
