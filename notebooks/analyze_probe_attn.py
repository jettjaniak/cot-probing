# %%
# %%
%load_ext autoreload
%autoreload 2

import torch
from cot_probing.attn_probes_data_proc import CollateFnOutput
from cot_probing.utils import load_model_and_tokenizer
from cot_probing.activations import build_fsp_cache
from ipywidgets import Dropdown, interactive_output, VBox, FloatRangeSlider
from beartype import beartype

# %%
from cot_probing.utils import fetch_runs
from cot_probing.attn_probes import ProbeTrainer
import wandb
from cot_probing import DATA_DIR
import pickle


def load_median_probe_test_data(
    probe_class: str,
    layer: int,
    context: str,
    min_seed: int,
    max_seed: int,
    metric: str,
) -> tuple[ProbeTrainer, list[int], list[dict], str]:
    runs_by_seed_by_layer = fetch_runs(
        api=wandb.Api(),
        probe_class=probe_class,
        min_layer=layer,
        max_layer=layer,
        min_seed=min_seed,
        max_seed=max_seed,
        context=context,
    )
    assert len(runs_by_seed_by_layer) == 1
    runs_by_seed = runs_by_seed_by_layer[layer]
    seed_run_sorted = sorted(
        runs_by_seed.items(), key=lambda s_r: s_r[1].summary.get(metric)
    )

    _median_seed, median_run = seed_run_sorted[len(seed_run_sorted) // 2]
    # median_acc = median_run.summary.get(metric)
    raw_acts_path = (
        DATA_DIR / f"../../activations/acts_L{layer:02d}_{context}_oct28-1156.pkl"
    )
    with open(raw_acts_path, "rb") as f:
        raw_acts_dataset = pickle.load(f)
    trainer, _, test_idxs = ProbeTrainer.from_wandb(
        raw_acts_dataset=raw_acts_dataset,
        run_id=median_run.id,
    )
    unbiased_fsp_str = raw_acts_dataset["unbiased_fsp"]
    raw_acts_qs = [raw_acts_dataset["qs"][i] for i in test_idxs]
    return trainer, test_idxs, raw_acts_qs, unbiased_fsp_str

# %%
torch.set_grad_enabled(False)

# %%
layer = 14
context = "biased-fsp"
min_seed, max_seed = 1, 10
n_seeds = max_seed - min_seed + 1
probe_class = "QV"
metric = "test_accuracy"

trainer, test_idxs, raw_acts_qs, unbiased_fsp_str = load_median_probe_test_data(
    probe_class, layer, context, min_seed, max_seed, metric
)
collate_fn_out: CollateFnOutput = list(trainer.test_loader)[0]
from transformers import AutoTokenizer

model, tokenizer = load_model_and_tokenizer(8)
unbiased_fsp_cache = build_fsp_cache(model, tokenizer, unbiased_fsp_str)

# %%
q_idxs = []
q_and_cot_tokens = []
cots_labels = []
cots_answers = []
questions = []
for q_idx, test_q in enumerate(raw_acts_qs):
    cots = test_q["biased_cots_tokens_to_cache"]
    for cot in cots:
        tokens = cot[:-4]
        q_and_cot_tokens.append(tokens)
        cot_label = test_q["biased_cot_label"]
        cots_labels.append(cot_label)
        cots_answers.append(test_q["expected_answer"])
        questions.append(test_q["question"])
        q_idxs.append(q_idx)

# %%
from cot_probing.typing import *
from cot_probing.vis import visualize_tokens_html
from cot_probing.attn_probes import AbstractProbe

# Pre-compute all attention and probe outputs
attn_probs_cache = []
probe_outputs_cache = []
for cot_idx in range(len(q_and_cot_tokens)):
    tokens = q_and_cot_tokens[cot_idx]
    resids = collate_fn_out.cot_acts[cot_idx:cot_idx+1, :len(tokens)].to(trainer.model.device)
    attn_mask = torch.ones(1, len(tokens), dtype=torch.bool, device=trainer.model.device)
    
    attn_probs = trainer.model.attn_probs(resids, attn_mask)
    probe_out = trainer.model(resids, attn_mask)
    
    attn_probs_cache.append(attn_probs[0, :len(tokens)])
    probe_outputs_cache.append(probe_out.item())

def visualize_cot_attn(
    tokenizer: PreTrainedTokenizerBase,
    tokens: list[int],
    label: str,
    answer: str,
    cot_idx: int,
):
    this_attn_probs = attn_probs_cache[cot_idx]
    print(f"label: {label}, correct answer: {answer}")
    print(f"faithfulness: {probe_outputs_cache[cot_idx]:.2%}")
    return visualize_tokens_html(
        tokens, tokenizer, this_attn_probs.tolist(), vmin=0.0, vmax=1.0
    )

@beartype
def update_plot(cot_idx: int):
    tokens = q_and_cot_tokens[cot_idx]
    display(
        visualize_cot_attn(
            tokenizer=tokenizer,
            tokens=tokens,
            label=cots_labels[cot_idx],
            answer=cots_answers[cot_idx],
            cot_idx=cot_idx,
        )
    )

# %%
# Function to update q_idx options based on category
def get_cot_idx_options(label, answer, faith_range):
    min_faith, max_faith = faith_range
    return [
        (f"{cot_idx} ({probe_outputs_cache[cot_idx]:.0%})", cot_idx)
        for cot_idx, cot_label in enumerate(cots_labels)
        if (cot_label == label and 
            cots_answers[cot_idx] == answer and
            min_faith/100 <= probe_outputs_cache[cot_idx] <= max_faith/100)
    ]


def on_label_change(change):
    options = get_cot_idx_options(change.new, answer_dropdown.value, faith_slider.value)
    cot_idx_dropdown.options = options
    if options:
        cot_idx_dropdown.value = options[0][1]


def on_answer_change(change):
    options = get_cot_idx_options(label_dropdown.value, change.new, faith_slider.value)
    cot_idx_dropdown.options = options
    if options:
        cot_idx_dropdown.value = options[0][1]


def on_faith_change(change):
    options = get_cot_idx_options(label_dropdown.value, answer_dropdown.value, change.new)
    cot_idx_dropdown.options = options
    if options:
        cot_idx_dropdown.value = options[0][1]


label_dropdown = Dropdown(
    options=["faithful", "unfaithful"],
    description="Label:",
)
answer_dropdown = Dropdown(
    options=["yes", "no"],
    description="Answer:",
)
faith_slider = FloatRangeSlider(
    value=[0, 100],
    min=0,
    max=100,
    step=1,
    description='Faith %:',
    readout_format='.0f',
)
cot_idx_dropdown = Dropdown(
    options=get_cot_idx_options(label_dropdown.value, answer_dropdown.value, faith_slider.value),
    description="CoT index:",
)

label_dropdown.observe(on_label_change, names="value")
answer_dropdown.observe(on_answer_change, names="value")
faith_slider.observe(on_faith_change, names="value")

# %%
# Create interactive output
out = interactive_output(
    update_plot,
    {
        "cot_idx": cot_idx_dropdown,
    },
)

# Display widgets and output
widgets = VBox([label_dropdown, answer_dropdown, faith_slider, cot_idx_dropdown])
display(widgets)
display(out)  # Also display the output widget

# %%
