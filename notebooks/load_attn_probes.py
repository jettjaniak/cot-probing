# %%
%load_ext autoreload
%autoreload 2

# %%
from pathlib import Path
import pandas as pd
from cot_probing.attn_probes import ProbeTrainer, collate_fn_out_to_model_out
from cot_probing.activations import build_fsp_cache, collect_resid_acts_with_pastkv, collect_resid_acts_no_pastkv
from cot_probing.attn_probes_data_proc import CollateFnOutput
from cot_probing.utils import load_model_and_tokenizer
import pickle
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import cosine_similarity
import wandb
from cot_probing.typing import *

from transformers import AutoTokenizer, AutoModelForCausalLM
LAYERS = list(range(0,32+1))  
# LAYERS = list(range(0,18+1))
LAYER = 15
SEEDS = list(range(21, 40+1))
torch.set_grad_enabled(False)

# %%
api = wandb.Api()
entity = "cot-probing"
project = "attn-probes"
run_by_seed_by_layer = {}
for layer in LAYERS:
    run_by_seed_by_layer[layer] = {}
    for seed in SEEDS:
        config_filters = {
            "args_probe_class": "minimal", 
            "args_data_seed": seed,
            "args_weight_init_seed": seed,
            "layer": layer,
        }
        filters = []
        for k, v in config_filters.items():
            filters.append({f"config.{k}": v})
        runs = list(api.runs(f"{entity}/{project}", {"$and": filters}))
        assert len(runs) == 1, f"Expected 1 run, got {len(runs)}"
        run = runs[0]
        run_by_seed_by_layer[layer][seed] = run

# %%
def get_metric_by_layer(run_by_seed_by_layer: dict, metric: str) -> dict[int, list[float]]:
    """Extract metric values for each layer from runs."""
    layer_metrics = {}
    for layer in sorted(run_by_seed_by_layer.keys()):
        values = []
        for seed in run_by_seed_by_layer[layer]:
            run = run_by_seed_by_layer[layer][seed]
            value = run.summary.get(metric)
            if value is not None:
                values.append(value)
        layer_metrics[layer] = values
    return layer_metrics

def plot_metric_distribution(layer_metrics: dict[int, list[float]], 
                           title: str,
                           ylabel: str,
                           figsize=(9, 6)):
    """Plot distribution of metrics across layers."""
    plt.figure(figsize=figsize)

    # Get statistics for each layer
    layers = sorted(layer_metrics.keys())
    medians = [np.median(layer_metrics[l]) for l in layers]
    q1s = [np.percentile(layer_metrics[l], 25) for l in layers]
    q3s = [np.percentile(layer_metrics[l], 75) for l in layers]

    # Plot lines
    plt.plot(layers, medians, '-', color='black', linewidth=2, label='Median')
    plt.plot(layers, q1s, '-', color='gray', alpha=0.5, linewidth=1, label='Quartiles')
    plt.plot(layers, q3s, '-', color='gray', alpha=0.5, linewidth=1)

    # Customize the plot
    plt.title(title)
    plt.xlabel("layer")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(layers)
    plt.tight_layout()
    plt.show()

# Set larger font size globally
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 16, 
    'axes.titlesize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14
})

# Get metrics and create plots
layer_losses = get_metric_by_layer(run_by_seed_by_layer, "test_loss")
layer_accs = get_metric_by_layer(run_by_seed_by_layer, "test_accuracy")

plot_metric_distribution(layer_losses, 
                        "minimal attention probe, 10 seeds",
                        "test loss")
plot_metric_distribution(layer_accs,
                        "minimal attention probe, 10 seeds", 
                        "test accuracy")


# %% Load dataset for layer 10 (needed for trainer initialization)
acts_path = Path(f"../activations/acts_L{LAYER:02d}_biased-fsp-oct28-1156.pkl")
with open(acts_path, "rb") as f:
    raw_acts_dataset = pickle.load(f)
raw_acts_qs = raw_acts_dataset["qs"]
# %% Load all minimal probes for layer 10 with matching seeds 1-10
probes = []

for seed in SEEDS:
    config_filters = {
        "args_probe_class": "minimal",
        "args_data_seed": seed,
        "args_weight_init_seed": seed,
        "layer": LAYER,
    }

    try:
        trainer, run, test_idxs = ProbeTrainer.from_wandb(
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
collate_fn_out: CollateFnOutput = list(trainer.test_loader)[0]

# %%
model, tokenizer = load_model_and_tokenizer(8)

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
from cot_probing.vis import visualize_tokens_html
from cot_probing.typing import *
def visualize_cot(cot_idx: int, resids: Optional[Float[torch.Tensor, "1 seq d_model"]] = None):
    tokens = cots_tokens[cot_idx]
    label = cots_labels[cot_idx]
    answer = cots_answers[cot_idx]
    
    # Use provided resids or get from collate_fn_out
    if resids is None:
        resids = collate_fn_out.cot_acts[cot_idx:cot_idx+1, :len(tokens)].to(probe_model.device)
    
    attn_mask = torch.ones(1, len(tokens), dtype=torch.bool, device=probe_model.device)
    
    # Get attention probs and model output
    attn_probs = probe_model.attn_probs(resids, attn_mask)
    probe_out = probe_model(resids, attn_mask)
    
    this_attn_probs = attn_probs[0, :len(tokens)]
    print(f"Tokens: {tokenizer.decode(tokens)}")
    print(f"label: {label}, correct answer: {answer}")
    print(f"faithfulness: {probe_out.item():.2%}")
    return visualize_tokens_html(tokens, tokenizer, this_attn_probs.tolist(), vmin=0.0, vmax=1.0)

def visualize_cot_faithfulness(cot_idx: int, resids: Optional[Float[torch.Tensor, "batch seq d_model"]] = None):
    tokens = cots_tokens[cot_idx]
    label = cots_labels[cot_idx]
    answer = cots_answers[cot_idx]
    
    # Use provided resids or get from collate_fn_out
    if resids is None:
        resids = collate_fn_out.cot_acts[cot_idx:cot_idx+1, :len(tokens)].to(probe_model.device)
    
    # Calculate faithfulness for each prefix length
    faithfulness_scores = []
    for prefix_len in range(1, len(tokens) + 1):
        # Create input with just this prefix
        prefix_resids = resids[:, :prefix_len]
        prefix_mask = torch.ones(1, prefix_len, dtype=torch.bool, device=probe_model.device)
        # Get model output for this prefix
        faithfulness = probe_model(prefix_resids, prefix_mask)
        faithfulness_scores.append(faithfulness.item())
    
    print(f"Tokens: {tokenizer.decode(tokens)}")
    print(f"label: {label}, correct answer: {answer}")
    print(f"final faithfulness: {faithfulness_scores[-1]:.2%}")
    return visualize_tokens_html(tokens, tokenizer, faithfulness_scores, vmin=0.0, vmax=1.0, use_diverging_colors=True)

# Example usage with default behavior (using collate_fn_out)
for cot_idx in range(2, 100, 20):
    display(visualize_cot(cot_idx))
    display(visualize_cot(-cot_idx))

# %%
model = AutoModelForCausalLM.from_pretrained("hugging-quants/Meta-Llama-3.1-8B-BNB-NF4-BF16")
unbiased_fsp_str = raw_acts_dataset["unbiased_fsp"]
unbiased_fsp_cache = build_fsp_cache(model, tokenizer, unbiased_fsp_str)
unbiased_fsp_tokens = tokenizer.encode(unbiased_fsp_str)

# %%
def get_unbiased_resid_acts(tokens: list[int]):
    return collect_resid_acts_with_pastkv(
        model=model,
        last_q_toks=tokens,
        layers=[LAYER],
        past_key_values=unbiased_fsp_cache,
    )[LAYER].unsqueeze(0).cuda().float()

def get_no_ctx_resid_acts(tokens: list[int]):
    assert tokenizer.bos_token_id is not None
    return collect_resid_acts_no_pastkv(
        model=model,
        all_input_ids=[tokenizer.bos_token_id] + tokens,
        layers=[LAYER],
    )[LAYER][1:].unsqueeze(0).cuda().float()
# %%
# Example usage with custom residuals
for cot_idx in range(2, 100, 20):
    cot_idx = cot_idx
    tokens = cots_tokens[cot_idx]
    # custom_resids = get_unbiased_resid_acts(tokens)
    custom_resids = get_no_ctx_resid_acts(tokens)
    display(visualize_cot(cot_idx, custom_resids))
    display(visualize_cot_faithfulness(cot_idx, custom_resids))

# %%
