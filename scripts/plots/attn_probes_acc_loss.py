#!/usr/bin/env python3

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from matplotlib.ticker import FuncFormatter

from cot_probing.utils import fetch_runs


def get_metric_by_layer(
    run_by_seed_by_layer: dict, metric: str
) -> dict[int, list[float]]:
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


def plot_metric_distribution(
    layer_metrics: dict[int, list[float]],
    title: str,
    ylabel: str,
    save_path: Path,
    probe_class: str,
    figsize=(9, 6),
):
    """Plot distribution of metrics across layers and save to file."""
    plt.figure(figsize=figsize)

    # Get statistics for each layer
    layers = sorted(layer_metrics.keys())
    medians = [np.median(layer_metrics[l]) for l in layers]
    q1s = [np.percentile(layer_metrics[l], 25) for l in layers]
    q3s = [np.percentile(layer_metrics[l], 75) for l in layers]

    # Plot lines with consistent colors
    color = "blue" if probe_class == "minimal" else "red"
    plt.plot(layers, medians, "-", color=color, linewidth=2, label="median")
    plt.plot(layers, q1s, "-", color=color, alpha=0.5, linewidth=1, label="quartiles")
    plt.plot(layers, q3s, "-", color=color, alpha=0.5, linewidth=1)

    # Determine which layers to show on x-axis
    layers_to_show = set()
    # First and last layers
    layers_to_show.add(min(layers))
    layers_to_show.add(max(layers))

    # Best performing layer
    if "accuracy" in ylabel.lower():
        best_layer = layers[np.argmax(medians)]
        plt.ylim(0.5, 1.0)  # Set y-axis limits for accuracy plots
        # Format y-axis as percentages
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100*y:.0f}%"))
    else:  # loss
        best_layer = layers[np.argmin(medians)]
    layers_to_show.add(best_layer)

    # Add evenly spaced layers, avoiding adjacency
    step = 3
    for l in layers[::step]:
        if not any(abs(l - existing) <= 1 for existing in layers_to_show):
            layers_to_show.add(l)

    # Customize the plot
    display_name = "V" if probe_class == "minimal" else "QV"
    plt.title(title.replace("minimal", "V").replace("medium", "QV"))
    plt.xlabel("layer")
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set x-ticks: empty for unlabeled layers, number for labeled ones
    plt.xticks(layers, [str(l) if l in layers_to_show else "" for l in layers])
    plt.tight_layout()

    # Save plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def plot_both_distributions(
    minimal_metrics: dict[int, list[float]],
    medium_metrics: dict[int, list[float]],
    title: str,
    ylabel: str,
    save_path: Path,
    figsize=(9, 6),
):
    """Plot median distributions for both probe classes."""
    plt.figure(figsize=figsize)

    # Get statistics for each probe class
    layers = sorted(minimal_metrics.keys())
    minimal_medians = [np.median(minimal_metrics[l]) for l in layers]
    medium_medians = [np.median(medium_metrics[l]) for l in layers]

    # Plot lines with consistent colors
    plt.plot(layers, minimal_medians, "-", color="blue", linewidth=2, label="V probe")
    plt.plot(layers, medium_medians, "-", color="red", linewidth=2, label="QV probe")

    # Determine which layers to show on x-axis
    layers_to_show = set()
    # First and last layers
    layers_to_show.add(min(layers))
    layers_to_show.add(max(layers))

    # Best performing layers for both probes
    if "accuracy" in ylabel.lower():
        minimal_best = layers[np.argmax(minimal_medians)]
        medium_best = layers[np.argmax(medium_medians)]
        plt.ylim(0.5, 1.0)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100*y:.0f}%"))
    else:  # loss
        minimal_best = layers[np.argmin(minimal_medians)]
        medium_best = layers[np.argmin(medium_medians)]
    layers_to_show.update({minimal_best, medium_best})

    # Add evenly spaced layers, avoiding adjacency
    step = 3
    for l in layers[::step]:
        if not any(abs(l - existing) <= 1 for existing in layers_to_show):
            layers_to_show.add(l)

    # Customize the plot
    plt.title(title.replace("minimal", "V").replace("medium", "QV"))
    plt.xlabel("layer")
    plt.ylabel(f"median {ylabel}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Set x-ticks
    plt.xticks(layers, [str(l) if l in layers_to_show else "" for l in layers])
    plt.tight_layout()

    # Save plot
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layers", type=str, default="0-32", help="Layer range (inclusive)"
    )
    parser.add_argument(
        "--seeds", type=str, default="21-40", help="Seed range (inclusive)"
    )
    parser.add_argument(
        "--probe-class",
        type=str,
        default="minimal",
        choices=["minimal", "medium", "both"],
    )
    args = parser.parse_args()

    min_layer, max_layer = map(int, args.layers.split("-"))
    min_seed, max_seed = map(int, args.seeds.split("-"))

    # Set larger font size globally
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

    # Configure wandb
    api = wandb.Api()

    if args.probe_class == "both":
        print(f"Fetching runs for both probe classes...")
        minimal_runs = fetch_runs(
            api, "minimal", min_layer, max_layer, min_seed, max_seed
        )
        medium_runs = fetch_runs(
            api, "medium", min_layer, max_layer, min_seed, max_seed
        )

        print("\nCreating plots...")
        print("Plotting test losses...")
        minimal_losses = get_metric_by_layer(minimal_runs, "test_loss")
        medium_losses = get_metric_by_layer(medium_runs, "test_loss")
        plot_both_distributions(
            minimal_losses,
            medium_losses,
            f"attention probes V & QV, {max_seed - min_seed + 1} seeds",
            "test loss",
            Path("results/attn_probes_loss_acc") / "both_test_loss_by_layer.png",
        )

        print("Plotting test accuracies...")
        minimal_accs = get_metric_by_layer(minimal_runs, "test_accuracy")
        medium_accs = get_metric_by_layer(medium_runs, "test_accuracy")
        plot_both_distributions(
            minimal_accs,
            medium_accs,
            f"attention probes V & QV, {max_seed - min_seed + 1} seeds",
            "test accuracy",
            Path("results/attn_probes_loss_acc") / "both_test_accuracy_by_layer.png",
        )
    else:
        # Original single probe class plotting code
        print(
            f"Fetching runs for {max_layer - min_layer + 1} layers and {max_seed - min_seed + 1} seeds..."
        )
        print(f"Probe class: {args.probe_class}")

        run_by_seed_by_layer = fetch_runs(
            api,
            args.probe_class,
            min_layer,
            max_layer,
            min_seed,
            max_seed,
        )

        print("\nCreating plots...")
        print("Plotting test losses...")
        layer_losses = get_metric_by_layer(run_by_seed_by_layer, "test_loss")
        display_name = "V" if args.probe_class == "minimal" else "QV"
        plot_metric_distribution(
            layer_losses,
            f"{display_name} attention probe, {max_seed - min_seed + 1} seeds",
            "test loss",
            Path("results/attn_probes_loss_acc")
            / f"{args.probe_class}_test_loss_by_layer.png",
            args.probe_class,
        )

        print("Plotting test accuracies...")
        layer_accs = get_metric_by_layer(run_by_seed_by_layer, "test_accuracy")
        plot_metric_distribution(
            layer_accs,
            f"{display_name} attention probe, {max_seed - min_seed + 1} seeds",
            "test accuracy",
            Path("results/attn_probes_loss_acc")
            / f"{args.probe_class}_test_accuracy_by_layer.png",
            args.probe_class,
        )

    print(f"\nPlots saved to results/attn_probes_loss_acc")


if __name__ == "__main__":
    main()
