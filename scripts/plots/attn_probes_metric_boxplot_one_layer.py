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
    parser.add_argument(
        "--layers", "-l", type=str, default="0-32", help="Layer range (inclusive)"
    )
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
def load_metrics(
    probe_class: str,
    min_layer: int,
    max_layer: int,
    min_seed: int,
    max_seed: int,
    metric: str,
):
    runs_by_seed_by_layer = fetch_runs(
        api=wandb.Api(),
        probe_class=probe_class,
        min_layer=min_layer,
        max_layer=max_layer,
        min_seed=min_seed,
        max_seed=max_seed,
    )

    # Collect metrics for all layers
    all_metrics = []
    for layer in range(min_layer, max_layer + 1):
        runs_by_seed = runs_by_seed_by_layer[layer]
        metrics = [run.summary.get(metric) for run in runs_by_seed.values()]
        all_metrics.append(metrics)
    return all_metrics


@beartype
def main(args: argparse.Namespace):
    min_layer, max_layer = map(int, args.layers.split("-"))
    min_seed, max_seed = map(int, args.seeds.split("-"))
    probe_class = args.probe_class
    metric = args.metric

    metrics = load_metrics(
        probe_class, min_layer, max_layer, min_seed, max_seed, metric
    )

    # Create the boxplot
    plt.figure(figsize=(15, 8))
    plt.boxplot(
        metrics,
        labels=[
            "Emb" if i == 0 else f"L{i-1:02d}" for i in range(min_layer, max_layer + 1)
        ],
    )

    # Add labels and title
    plt.ylabel(metric.replace("_", " ").title())
    if probe_class == "minimal":
        plt.title(f"V Probes Performance Across Layers")
    else:
        plt.title(f"QV Probes Performance Across Layers")

    # Rotate x-axis labels if there are many layers
    plt.xticks(rotation=45 if max_layer - min_layer > 10 else 0)

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(f"{probe_class}_L{min_layer}-{max_layer}_{metric}_boxplot.png")
    plt.close()


if __name__ == "__main__":
    main(parse_args())
