# %%
# Load data

# From json
# JSON_PATH = "./generations_in_bias_ctx.json"
# import json
# with open(JSON_PATH, "r") as f:
#     data = json.load(f)

data = [
    {
        "locs": {"generation": [5]},
        "tokens": [12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        "correct": True,
        "correct_answer": "C",
    },
    {
        "locs": {"generation": [5]},
        "tokens": [12, 13, 14, 15, 99, 17, 18, 19, 20, 21],
        "correct": False,
        "correct_answer": "C",
    },
]
# %%
import os

os.environ["HF_HOME"] = "/workspace/hf_cache/"
os.environ["HF_DATASETS_CACHE"] = "/workspace/hf_cache/"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/hf_cache/"
os.environ["HF_TOKEN"] = "REPLACE_WITH_YOUR_HF_TOKEN"

model_name = "google/gemma-2-9b"

# from transformers import AutoModelForCausalLM, AutoTokenizer
# model = AutoModelForCausalLM.from_pretrained(model_name).cuda()

# Load model
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(model_name, device="cuda")

# %%
import torch

# Collect tokens from data

# Faithful examples
postiive_examples = [
    torch.tensor(data[i]["tokens"]) for i in range(len(data)) if data[i]["correct"]
]
positive_examples = torch.cat(postiive_examples)

# Unfaithful examples
negative_examples = [
    torch.tensor(data[i]["tokens"]) for i in range(len(data)) if not data[i]["correct"]
]
negative_examples = torch.cat(negative_examples)
# %%
from transformer_lens import ActivationCache

# Run model on all examples to get activations
tokens = torch.cat([positive_examples, negative_examples]).cuda()


def get_cache_fwd(model, tokens):
    model.reset_hooks()
    cache = {}

    def forward_cache_hook(act, hook):
        cache[hook.name] = act.detach()

    for layer in range(model.cfg.n_layers):
        model.add_hook(f"blocks.{layer}.hook_resid_post", forward_cache_hook, "fwd")

    torch.set_grad_enabled(True)
    logits = model(tokens.clone())
    torch.set_grad_enabled(False)

    model.reset_hooks()
    return (
        logits,
        ActivationCache(cache, model),
    )


logits, cache = get_cache_fwd(model, tokens)

# %%
# Dump cache.cache_dict to disk
import pickle

with open("activations_cache.pkl", "wb") as f:
    pickle.dump(cache.cache_dict, f)
