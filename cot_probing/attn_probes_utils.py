import pickle

import wandb

from cot_probing import DATA_DIR
from cot_probing.attn_probes import AttnProbeTrainer
from cot_probing.typing import *
from cot_probing.utils import fetch_runs


def load_median_probe_test_data(
    probe_class: str,
    fsp_context: Literal["biased-fsp", "unbiased-fsp", "no-fsp"],
    layer: int,
    min_seed: int,
    max_seed: int,
    metric: str,
    verbose: bool = False,
) -> tuple[AttnProbeTrainer, dict[str, Any]]:
    """Loads the median probe run and the associated raw acts data (filtered for the test set)"""
    runs_by_seed_by_layer = fetch_runs(
        api=wandb.Api(),
        probe_class=probe_class,
        fsp_context=fsp_context,
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

    if verbose:
        # Print metric for each run
        for seed, run in seed_run_sorted:
            print(f"{seed}: {run.summary.get(metric)}")

    _median_seed, median_run = seed_run_sorted[len(seed_run_sorted) // 2]

    if verbose:
        print(f"Median run: {median_run.id}")
        print(f"Median {metric}: {median_run.summary.get(metric)}")

    raw_acts_path = (
        DATA_DIR / f"../../activations/acts_L{layer:02d}_biased-fsp_oct28-1156.pkl"
    )
    with open(raw_acts_path, "rb") as f:
        raw_acts_dataset = pickle.load(f)
    trainer, _, test_idxs = AttnProbeTrainer.from_wandb(
        raw_acts_dataset=raw_acts_dataset,
        run_id=median_run.id,
    )
    raw_acts_dataset["qs"] = [raw_acts_dataset["qs"][i] for i in test_idxs]
    return trainer, raw_acts_dataset
