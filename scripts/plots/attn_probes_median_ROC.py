#!/usr/bin/env python3
import argparse
import pickle

import matplotlib.pyplot as plt
import torch
import wandb
from beartype import beartype
from torch.utils.data import DataLoader, Dataset

from cot_probing import DATA_DIR
from cot_probing.attn_probes import AttnProbeTrainer
from cot_probing.attn_probes_data_proc import (
    SequenceDataset,
    collate_fn,
    preprocess_data,
)
from cot_probing.typing import *
from cot_probing.utils import fetch_runs

plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)
torch.set_grad_enabled(False)


@beartype
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", "-l", type=int, default=15, help="Layer")
    parser.add_argument(
        "--seeds", "-s", type=str, default="21-40", help="Seed range (inclusive)"
    )
    parser.add_argument(
        "--probe-class",
        "-p",
        type=str,
        default="minimal",
        choices=["minimal", "medium"],
    )
    parser.add_argument(
        "--metric", "-m", type=str, default="test_accuracy", help="Metric"
    )
    return parser.parse_args()


@beartype
def load_median_probe_test_data(
    probe_class: str, layer: int, min_seed: int, max_seed: int, metric: str
):
    runs_by_seed_by_layer = fetch_runs(
        api=wandb.Api(),
        probe_class=probe_class,
        min_layer=layer,
        max_layer=layer,
        min_seed=min_seed,
        max_seed=max_seed,
    )
    assert len(runs_by_seed_by_layer) == 1
    runs_by_seed = runs_by_seed_by_layer[layer]
    seed_run_sorted = sorted(
        runs_by_seed.items(), key=lambda s_r: s_r[1].summary.get(metric)
    )

    # Print metric of all runs
    # for seed, run in seed_run_sorted:
    #     print(f"{seed}: {run.summary.get(metric)}")

    _median_seed, median_run = seed_run_sorted[len(seed_run_sorted) // 2]

    print(f"Median run: {median_run.id}")
    print(f"Median {metric}: {median_run.summary.get(metric)}")

    # median_acc = median_run.summary.get(metric)
    raw_acts_path = (
        f"../activations/acts_L{layer:02d}_with-unbiased-cots-oct28-1156.pkl"
    )
    with open(raw_acts_path, "rb") as f:
        raw_acts_dataset = pickle.load(f)
    trainer, _, test_idxs = AttnProbeTrainer.from_wandb(
        raw_acts_dataset=raw_acts_dataset,
        run_id=median_run.id,
    )
    return trainer, test_idxs, raw_acts_dataset


@beartype
def main(args: argparse.Namespace):
    layer = args.layer
    min_seed, max_seed = map(int, args.seeds.split("-"))
    n_seeds = max_seed - min_seed + 1
    probe_class = args.probe_class
    metric = args.metric

    trainer, test_idxs, raw_acts_dataset = load_median_probe_test_data(
        probe_class, layer, min_seed, max_seed, metric
    )

    cots_by_q, labels_by_q_list = preprocess_data(
        raw_acts_dataset,
        {"include_answer_toks": False, "data_device": "cuda", "d_model": 4096},
    )

    test_dataset = SequenceDataset(
        [cots_by_q[i] for i in test_idxs],
        [labels_by_q_list[i] for i in test_idxs],
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=len(test_dataset),
        collate_fn=collate_fn,
    )
    test_batch = next(iter(test_loader))
    resids = test_batch.cot_acts
    attn_mask = test_batch.attn_mask
    labels = test_batch.labels

    # Get model predictions (z scores)
    attn_probs = trainer.model.attn_probs(resids, attn_mask)
    values = trainer.model.values(resids)
    z = trainer.model.z(attn_probs, values)

    # Convert to numpy for sklearn
    z_np = z.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Calculate ROC curve
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(labels_np, z_np)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(
        fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})"
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"ROC Curve for Layer {layer} {probe_class.title()} (median) Probe (AUC = {roc_auc:.2f})"
    )
    plt.legend(loc="lower right")

    # Save the plot
    plt.savefig(f"roc_L{layer}_{probe_class}_probe.png", bbox_inches="tight", dpi=300)
    plt.close()

    print(f"ROC AUC: {roc_auc:.3f}")


if __name__ == "__main__":
    main(parse_args())
