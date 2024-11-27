#!/usr/bin/env python3
import argparse
import logging
import pickle
from pathlib import Path

import torch
from beartype import beartype

from cot_probing.attn_probes import (
    AttnProbeModelConfig,
    AttnProbeTrainer,
    DataConfig,
    ExperimentConfig,
)
from cot_probing.typing import *


@beartype
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train attn probes")
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        required=True,
        help="Path to the dataset of activations",
    )
    parser.add_argument(
        "--probe-class",
        type=str,
        required=True,
        choices=["tied", "untied"],
        help="Type of attention probe to use",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=4096,
        help="Model dimension (should match the model being probed)",
    )
    parser.add_argument(
        "--weight-init-range",
        type=float,
        default=0.02,
        help="Range for weight initialization",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for training",
    )
    parser.add_argument(
        "--beta1",
        type=float,
        default=0.9,
        help="Beta1 for Adam optimizer",
    )
    parser.add_argument(
        "--beta2",
        type=float,
        default=0.999,
        help="Beta2 for Adam optimizer",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size for training",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Number of epochs to wait for improvement before early stopping",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Maximum number of epochs to train for",
    )
    parser.add_argument(
        "--model-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--data-device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to load data on",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="attn-probes",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Wandb run name. If not provided, will be auto-generated.",
    )
    parser.add_argument(
        "--cv-n-folds",
        "--n-folds",
        type=int,
        default=10,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--cv-fold-nrs",
        "--fold-nrs",
        type=str,
        default="all",
        help="Comma-separated list of folds to train on, defaults to all folds.",
    )
    parser.add_argument(
        "--cv-seed",
        "--cvs",
        type=int,
        default=0,
        help="Random seed for reproducibility of cross-validation (data folds splitting).",
    )
    parser.add_argument(
        "--train-seed",
        "--ts",
        type=int,
        default=0,
        help="Random seed for reproducibility of training (data shuffling and weight initialization)",
    )
    parser.add_argument(
        "--partial-seq",
        action="store_true",
        help="Use partial sequences for training",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


@beartype
def build_experiment_config(
    args: argparse.Namespace,
    fold_nr: int,
) -> ExperimentConfig:
    probe_class_arg = args.probe_class

    probe_model_config = AttnProbeModelConfig(
        d_model=args.d_model,
        weight_init_range=args.weight_init_range,
        weight_init_seed=args.train_seed,
        partial_seq=args.partial_seq,
    )

    data_config = build_data_config(args, fold_nr)

    return ExperimentConfig(
        probe_class=probe_class_arg,
        probe_model_config=probe_model_config,
        data_config=data_config,
        lr=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        patience=args.patience,
        n_epochs=args.epochs,
        model_device=args.model_device,
    )


@beartype
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


@beartype
def train_attn_probe(
    raw_q_dicts: list[dict],
    args: argparse.Namespace,
    fold_nr: int,
):

    experiment_config = build_experiment_config(args, fold_nr)
    trainer = AttnProbeTrainer(
        c=experiment_config,
        raw_q_dicts=raw_q_dicts,
    )
    trainer.train(project_name=args.wandb_project)


@beartype
def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    logging.info("Running with arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    assert args.file.stem.startswith("acts_")
    with open(args.file, "rb") as f:
        raw_q_dicts = pickle.load(f)["qs"]

    if args.cv_fold_nrs == "all":
        fold_nrs = list(range(args.cv_n_folds))
    else:
        fold_nrs = [int(f) for f in args.cv_fold_nrs.split(",")]

    for fold_nr in fold_nrs:
        train_attn_probe(raw_q_dicts, args, fold_nr)


if __name__ == "__main__":
    main(parse_args())
