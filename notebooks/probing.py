# %%
import os
import pickle

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
