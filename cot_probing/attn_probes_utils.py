import argparse
import pickle

import wandb
from wandb.sdk.wandb_run import Run as WandbSdkRun

from cot_probing import DATA_DIR
from cot_probing.attn_probes import (
    AbstractProbe,
    DataConfig,
    ProbeConfig,
    ProbeTrainer,
    TrainerConfig,
)
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
) -> tuple[ProbeTrainer, dict[str, Any]]:
    """Loads the median probe run and the associated raw acts data (filtered for the test set)"""
    runs_by_seed_by_layer = fetch_runs(
        api=wandb.Api(),
        probe_class=probe_class,
        context=fsp_context,
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
    trainer, _, test_idxs = ProbeTrainer.from_wandb(
        raw_acts_dataset=raw_acts_dataset,
        run_id=median_run.id,
    )
    raw_acts_dataset["qs"] = [raw_acts_dataset["qs"][i] for i in test_idxs]
    return trainer, raw_acts_dataset


def build_data_config(args: argparse.Namespace, cv_test_fold: int) -> DataConfig:
    acts_str, layer_str, context, dataset_id = args.file.stem.split("_")
    assert acts_str == "acts"
    layer = int(layer_str[1:])
    assert layer_str == f"L{layer:02d}"
    assert context in ["biased-fsp", "unbiased-fsp", "no-fsp"]

    return DataConfig(
        dataset_id=dataset_id,
        layer=layer,
        context=context,
        cv_seed=args.cv_seed,
        cv_n_folds=args.cv_n_folds,
        cv_test_fold=cv_test_fold,
        train_val_seed=args.train_seed,
        val_frac=args.val_frac,
        data_device=args.data_device,
        batch_size=args.batch_size,
    )


def build_trainer_config(
    args: argparse.Namespace,
    experiment_uuid: str,
    cv_test_fold: int,
) -> TrainerConfig:
    probe_config = ProbeConfig(
        d_model=args.d_model,
        weight_init_range=args.weight_init_range,
        weight_init_seed=args.train_seed,
        partial_seq=args.partial_seq,
    )

    data_config = build_data_config(args, cv_test_fold)

    return TrainerConfig(
        probe_class=args.probe_class,
        probe_config=probe_config,
        data_config=data_config,
        lr=args.lr,
        beta1=args.beta1,
        beta2=args.beta2,
        patience=args.patience,
        max_epochs=args.max_epochs,
        model_device=args.model_device,
        experiment_uuid=experiment_uuid,
    )


def train_attn_probe(
    raw_q_dicts: list[dict],
    args: argparse.Namespace,
    experiment_uuid: str,
    cv_test_fold: int,
) -> tuple[AbstractProbe, WandbSdkRun]:

    trainer_config = build_trainer_config(args, experiment_uuid, cv_test_fold)
    trainer = ProbeTrainer(
        c=trainer_config,
        raw_q_dicts=raw_q_dicts,
    )
    return trainer.train(project_name=args.wandb_project)
