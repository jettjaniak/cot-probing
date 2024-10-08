# %%
import os
import pickle
import torch

os.environ["HF_HOME"] = "/workspace/hf_cache/"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache/"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache/"

# %%
from cot_probing.activations import Activations, QuestionActivations

activations_results_path = (
    "../results/activations_google--gemma-2-2b_snarks_S0_N151.pkl"
)

with open(activations_results_path, "rb") as f:
    activations = pickle.load(f)

activations
# %%
# Figure out how many positive and negatie examples

positive_activations = []
negative_activations = []
for question_idx, question in enumerate(activations.eval_results.questions):
    acts = activations.activations_by_question[
        question_idx
    ].activations  # Shape: [n_layers locs d_model]

    # Take mean across locs
    mean_acts = acts.mean(1)  # Shape: [n_layers d_model]

    if question.is_correct:
        positive_activations.append(mean_acts)
    else:
        negative_activations.append(mean_acts)

# Stack them up
positive_activations = torch.stack(
    positive_activations
)  # Shape: [n_positive n_layers d_model]
negative_activations = torch.stack(
    negative_activations
)  # Shape: [n_negative n_layers d_model]

# Mean across positive and negative examples
positive_activations = positive_activations.mean(0)  # Shape: [n_layers d_model]
negative_activations = negative_activations.mean(0)  # Shape: [n_layers d_model]

# %%
# Compute the difference

diff = positive_activations - negative_activations  # Shape: [n_layers d_model]
# %%
