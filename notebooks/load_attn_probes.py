# %%
%load_ext autoreload
%autoreload 2

# %%
from pathlib import Path
import pandas as pd
from cot_probing.attn_probes import AttnProbeTrainer, collate_fn_out_to_model_out
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import cosine_similarity

from transformers import AutoTokenizer
LAYER = 15
torch.set_grad_enabled(False)

# %% Load dataset for layer 10 (needed for trainer initialization)
acts_path = Path(f"../activations/acts_L{LAYER:02d}_with-unbiased-cots-oct28-1156.pkl")
with open(acts_path, "rb") as f:
    raw_acts_dataset = pickle.load(f)
raw_acts_qs = raw_acts_dataset["qs"]
# %% Load all minimal probes for layer 10 with matching seeds 1-10
probes = []

for seed in range(11, 20+1):
    config_filters = {
        "args_probe_class": "minimal",
        "args_data_seed": seed,
        "args_weight_init_seed": seed,
        "layer": LAYER,
    }

    try:
        trainer, run, test_idxs = AttnProbeTrainer.from_wandb(
            raw_acts_dataset=raw_acts_dataset,
            config_filters=config_filters,
        )
        # Compute test metrics
        test_loss, test_acc = trainer.compute_test_loss_acc()
        print(f"Test loss: {test_loss:.3f}, test accuracy: {test_acc:.3f}")
        validation_loss, validation_acc = trainer.compute_validation_loss_acc()
        print(f"Validation loss: {validation_loss:.3f}, validation accuracy: {validation_acc:.3f}")
        probes.append(
            {
                "seed": seed,
                "trainer": trainer,
                "test_loss": test_loss,
                "test_accuracy": test_acc,
                "run": run,
                "test_idxs": test_idxs,
            }
        )
        print(f"Loaded probe for seed {seed}: test_acc={test_acc:.3f}")

    except ValueError as e:
        print(f"Could not load probe for seed {seed}: {e}")
# %%
# Sort probes by test loss
probes.sort(key=lambda x: x["test_loss"])

# Extract value vectors and create similarity matrix
value_vectors = torch.stack([p["trainer"].model.value_vector for p in probes])
n_probes = len(probes)
sim_matrix = torch.zeros((n_probes, n_probes))

for i in range(n_probes):
    for j in range(n_probes):
        sim_matrix[i,j] = cosine_similarity(
            value_vectors[i], 
            value_vectors[j], 
            dim=0
        )

# Create labels with seed and test accuracy
labels = [f"seed={p['seed']}\nloss={p['test_loss']:.3f}" for p in probes]

# Plot
plt.figure(figsize=(10, 8))
sns.heatmap(
    sim_matrix.numpy(),
    annot=True,
    fmt='.2f',
    cmap='RdBu',
    center=0,
    square=True,
    xticklabels=labels,
    yticklabels=labels
)
plt.title(f'Pairwise Cosine Similarity of Value Vectors (Layer {LAYER})\nSorted by Test Loss')
plt.tight_layout()
plt.show()

# Print test accuracies in order
print("\nTest accuracies (ordered by test loss):")
for p in probes:
    print(f"Seed {p['seed']}: {p['test_accuracy']:.3f}")

# %%
probe = probes[0]
test_idxs = probe["test_idxs"]
test_qs = [raw_acts_qs[i] for i in test_idxs]
trainer = probe["trainer"]
probe_model = trainer.model
collate_fn_out = list(trainer.test_loader)[0]
probe_model_out = collate_fn_out_to_model_out(probe_model, collate_fn_out)
probe_model_out.shape


# %%
resids = collate_fn_out.cot_acts.to(probe_model.device)
attn_mask = collate_fn_out.attn_mask.to(probe_model.device)
attn_probs = probe_model.attn_probs(resids, attn_mask)
attn_probs.shape
tokenizer = AutoTokenizer.from_pretrained("hugging-quants/Meta-Llama-3.1-8B-BNB-NF4-BF16")


# %%
cots_tokens = []
cots_labels = []
cots_answers = []
for test_q in test_qs:
    cots = test_q["biased_cots_tokens_to_cache"]
    for cot in cots:
        tokens = cot[:-4]
        cots_tokens.append(tokens)
        cots_labels.append(test_q["biased_cot_label"])
        cots_answers.append(test_q["expected_answer"])
# %%
cot_idx = 80
tokens = cots_tokens[cot_idx]
label = cots_labels[cot_idx]
answer = cots_answers[cot_idx]
print(tokenizer.decode(tokens))
print(f"label: {label}, answer: {answer}")
this_attn_probs = attn_probs[cot_idx, :len(tokens)]
print(this_attn_probs.sum())
from cot_probing.vis import visualize_tokens_html
display(visualize_tokens_html(tokens, tokenizer, this_attn_probs.tolist(), vmin=0.0, vmax=1.0))
probe_model_out[cot_idx]
# %%
