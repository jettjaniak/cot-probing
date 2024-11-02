#!/usr/bin/env python3
import argparse
import pickle

import matplotlib.pyplot as plt
import torch
from beartype import beartype

import wandb
from cot_probing import DATA_DIR
from cot_probing.attn_probes import AttnProbeTrainer
from cot_probing.patching import PatchedLogitsProbs
from cot_probing.typing import *
from cot_probing.utils import fetch_runs

TOK_GROUPS = ["Question:", "[question]", "?\\n", "LTSBS:\\n-", "reasoning", "last 3"]

LOGIT_OR_PROB = "prob"
DIR = "bia_to_unb"
CATEGORIES_FILE = f"categories_{LOGIT_OR_PROB}_{DIR}_0.25_1.5_2.0_4.0.pkl"
SWAPS_FILE = f"swaps_with-unbiased-cots-oct28-1156.pkl"
LB_LAYERS = 1
PATCH_LAYERS_FILE = (
    f"patch_new_res_8B_LB{LB_LAYERS}__swaps_with-unbiased-cots-oct28-1156.pkl"
)
PATCH_ALL_FILE = "patch_new_res_8B_LB33__swaps_with-unbiased-cots-oct28-1156.pkl"


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
) -> tuple[AttnProbeTrainer, list[int], list[dict], str]:
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

    _median_seed, median_run = seed_run_sorted[len(seed_run_sorted) // 2]
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
    unbiased_fsp_str = raw_acts_dataset["unbiased_fsp"]
    raw_acts_qs = [raw_acts_dataset["qs"][i] for i in test_idxs]
    return trainer, test_idxs, raw_acts_qs, unbiased_fsp_str


@beartype
def plot_patching_heatmap(combined_values, title):
    v = combined_values
    plt.figure(figsize=(12, 6))
    plt.imshow(
        v,
        cmap="RdBu",
        origin="lower",
        vmin=-max(abs(np.min(v)), abs(np.max(v))),
        vmax=max(abs(np.min(v)), abs(np.max(v))),
    )
    plt.title(title)
    plt.colorbar()
    first_ytick = "all"
    # TODO: show only some
    if LB_LAYERS > 1:
        other_yticks = [
            f"{i*LB_LAYERS}-{(i+1)*LB_LAYERS}" for i in range(len(combined_values) - 1)
        ]
    else:
        other_yticks = [str(i - 1) for i in range(len(combined_values) - 1)]
        other_yticks[0] = "emb"
    plt.yticks(range(len(combined_values)), [first_ytick] + other_yticks)
    plt.xticks(range(len(TOK_GROUPS)), TOK_GROUPS, rotation=90)
    plt.ylabel("layers")
    plt.xlabel("token groups")
    plt.axhline(y=0.5, color="black", linewidth=1)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()


@beartype
def get_patch_values(
    plp_by_group_by_layers: dict[tuple[int, ...], dict[str, PatchedLogitsProbs]],
    prob_or_logit: Literal["prob", "logit"],
    direction: Literal["bia_to_unb", "unb_to_bia"],
) -> list[float] | list[list[float]]:
    attr = f"{prob_or_logit}_diff_change_{direction}"
    values = []
    for _layers, plp_by_group in plp_by_group_by_layers.items():
        values.append([getattr(plp, attr) for plp in plp_by_group.values()])
    if len(values) == 1:
        return values[0]
    return values


@beartype
def main(args: argparse.Namespace):
    layer = args.layer
    min_seed, max_seed = map(int, args.seeds.split("-"))
    n_seeds = max_seed - min_seed + 1
    probe_class = args.probe_class
    metric = args.metric

    trainer, test_idxs, raw_acts_qs, unbiased_fsp_str = load_median_probe_test_data(
        probe_class, layer, min_seed, max_seed, metric
    )

    with open(DATA_DIR / CATEGORIES_FILE, "rb") as f:
        categories = pickle.load(f)
    categories = {
        cat: [
            (q_idx, swap_idx)
            for q_idx, swap_idx in qidx_swap_idx_pairs
            if q_idx in test_idxs
        ]
        for cat, qidx_swap_idx_pairs in categories.items()
    }

    with open(DATA_DIR / SWAPS_FILE, "rb") as f:
        swaps_by_q = pickle.load(f)["qs"]
    swaps_by_q = [swaps_by_q[i] for i in test_idxs]

    with open(DATA_DIR / PATCH_ALL_FILE, "rb") as f:
        patch_all_by_q = pickle.load(f)
    patch_all_by_q = [patch_all_by_q[i] for i in test_idxs]

    with open(DATA_DIR / PATCH_LAYERS_FILE, "rb") as f:
        patch_layers_by_q = pickle.load(f)
    patch_layers_by_q = [patch_layers_by_q[i] for i in test_idxs]


if __name__ == "__main__":
    main(parse_args())
