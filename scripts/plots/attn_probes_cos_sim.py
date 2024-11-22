#!/usr/bin/env python3
import argparse
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import wandb
from beartype import beartype
from torch.nn.functional import cosine_similarity
from wandb.apis.public.runs import Run

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
# torch.set_grad_enabled(False)


@beartype
def get_value_vector(run: Run) -> Float[torch.Tensor, " model"]:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "best_model.pt"
        best_model_file = run.file("best_model.pt")
        best_model_file.download(root=tmp_dir, replace=True)
        model_state_dict = torch.load(tmp_path, map_location="cpu", weights_only=True)
        return model_state_dict["value_vector"]


@beartype
def plot_sim_matrix(
    sim_matrix: torch.Tensor, n_seeds: int, probe_class: str, layer: int, context: str
):
    plt.figure(figsize=(9, 7))
    im = plt.imshow(
        sim_matrix,
        cmap="viridis",
    )
    ticks = set([0, n_seeds - 1])
    for tick in range(0, n_seeds, 3):
        if not any(abs(t - tick) <= 1 for t in ticks):
            ticks.add(tick)
    ticks = list(ticks)
    plt.xticks(ticks)
    plt.yticks(ticks)

    # Add text annotations with values
    for i in range(sim_matrix.shape[0]):
        for j in range(i + 1):
            plt.text(j, i, f"{sim_matrix[i, j]:.2f}", ha="center", va="center")

    plt.colorbar(im)
    assert probe_class in ["V", "QV"], probe_class
    plt.title(
        f"pairwise cosine similarity of value vectors;\nlayer {layer}, {probe_class} probes sorted by test accuracy ({context})"
    )
    plt.tight_layout()
    save_dir = Path("results/attn_probes_cos_sim")
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"L{layer:02d}_{probe_class}_{context}.png")
    plt.close()


@beartype
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", "-l", type=int, default=15, help="Layer")
    parser.add_argument(
        "--seeds", "-s", type=str, default="1-10", help="Seed range (inclusive)"
    )
    parser.add_argument(
        "--probe-class",
        "-p",
        type=str,
        default="V",
        choices=["V", "QV"],
    )
    parser.add_argument(
        "--metric", "-m", type=str, default="test_accuracy", help="Metric"
    )
    return parser.parse_args()


@beartype
def main(args: argparse.Namespace):
    layer = args.layer
    min_seed, max_seed = map(int, args.seeds.split("-"))
    n_seeds = max_seed - min_seed + 1
    probe_class = args.probe_class
    metric = args.metric

    contexts = ["no-fsp", "biased-fsp"]
    for context in contexts:
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

        value_vectors = torch.stack(
            [get_value_vector(run) for _seed, run in seed_run_sorted]
        )
        sim_matrix = torch.zeros((n_seeds, n_seeds))

        for i in range(n_seeds):
            for j in range(n_seeds):
                if j > i:
                    sim_matrix[i, j] = float("nan")
                else:
                    sim_matrix[i, j] = cosine_similarity(
                        value_vectors[i], value_vectors[j], dim=0
                    )

        plot_sim_matrix(sim_matrix, n_seeds, probe_class, layer, context)


if __name__ == "__main__":
    main(parse_args())
