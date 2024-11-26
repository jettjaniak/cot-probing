#!/usr/bin/env python3
import argparse
import logging
import pickle
import uuid
from pathlib import Path
from typing import cast

import torch
from beartype import beartype

from cot_probing.attn_probes import (
    AttnProbeModelConfig,
    AttnProbeTrainer,
    DatasetConfig,
    ProbingConfig,
    get_probe_model_class,
)
from cot_probing.typing import *
from cot_probing.utils import setup_determinism


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
        choices=["V", "QV"],
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
        "--cross-validation-n-folds",
        "--n-folds",
        type=int,
        default=10,
        help="Number of folds for cross-validation.",
    )
    parser.add_argument(
        "--cross-validation-folds",
        "--folds",
        type=str,
        default=None,
        help="Comma-separated list of folds to train on. If not provided, all folds will be used.",
    )
    parser.add_argument(
        "--cross-validation-seed",
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
        "--include-answer-toks",
        action="store_true",
        help="Include answer tokens in the probe training",
    )
    parser.add_argument(
        "--partial-seq",
        action="store_true",
        help="Use partial sequences for training",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


@beartype
def build_probe_config(
    args: argparse.Namespace,
    layer: int,
    fold: int,
    n_folds: int,
) -> ProbingConfig:
    probe_class_arg = args.probe_class

    probe_config = AttnProbeModelConfig(
        d_model=args.d_model,
        weight_init_range=args.weight_init_range,
        weight_init_seed=args.train_seed,
        partial_seq=args.partial_seq,
    )

    return ProbingConfig(
        probe_model_class=get_probe_model_class(probe_class_arg),
        probe_model_config=probe_config,
        train_seed=args.train_seed,
        lr=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        batch_size=args.batch_size,
        patience=args.patience,
        n_epochs=args.epochs,
        fold=fold,
        n_folds=n_folds,
        model_device=args.model_device,
        data_device=args.data_device,
        layer=layer,
    )


@beartype
def build_dataset_config(file: Path) -> DatasetConfig:
    parts = file.stem.split("_")
    assert parts[0] == "acts"
    layer_str = parts[1]
    assert layer_str[0] == "L"
    layer = int(layer_str[1:])

    context = parts[2]
    assert context in ["biased-fsp", "unbiased-fsp", "no-fsp"]
    context_literal: Literal["biased-fsp", "unbiased-fsp", "no-fsp"] = cast(
        Literal["biased-fsp", "unbiased-fsp", "no-fsp"], context
    )

    dataset_id = parts[3]

    return DatasetConfig(
        id=dataset_id,
        context=context_literal,
        layer=layer,
    )


@beartype
def train_attn_probes(
    raw_acts_dataset: dict,
    dataset_config: DatasetConfig,
    layer: int,
    args: argparse.Namespace,
    wandb_run_name: str,
    wandb_project: str,
) -> list[dict]:
    """Train attention probe on the dataset"""
    # Set up cross-validation across folds
    setup_determinism(args.cross_validation_seed)

    # Shuffle data before splitting into folds
    raw_acts_dataset["qs"] = np.random.shuffle(raw_acts_dataset["qs"])

    fold_results = []
    n_folds = args.cross_validation_n_folds
    folds = args.cross_validation_folds
    if folds is None:
        folds = list(range(n_folds))
    else:
        folds = [int(f) for f in folds.split(",")]

    for fold in folds:
        # Create probe configuration
        probe_config = build_probe_config(args, layer, fold, n_folds)

        fold_wandb_run_name = f"{wandb_run_name}_fold{fold}"

        # Initialize trainer
        trainer = AttnProbeTrainer(
            c=probe_config,
            dataset_config=dataset_config,
            raw_acts_dataset=raw_acts_dataset,
            data_loading_kwargs={
                **vars(args),
                "fold": fold,
                "n_folds": n_folds,
            },
        )

        # Train probe
        model = trainer.train(
            run_name=fold_wandb_run_name,
            project_name=wandb_project,
            args=args,
        )
        fold_results.append(
            {
                "model": model,
                "config": probe_config,
            }
        )

    return fold_results


@beartype
def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    logging.info("Running with arguments:")
    for arg, value in vars(args).items():
        logging.info(f"  {arg}: {value}")

    assert args.file.stem.startswith("acts_")
    layer_str = args.file.stem.split("_")[1]
    layer_int = int(layer_str[1:])
    with open(args.file, "rb") as f:
        acts_dataset = pickle.load(f)

    dataset_config = build_dataset_config(args.file)

    # Create default wandb run name if none provided
    wandb_run_name = args.wandb_run_name
    if wandb_run_name is None:
        with_answer_str = "WITH_ANSWER_" if args.include_answer_toks else ""
        wandb_run_name = f"{with_answer_str}{layer_str}_{args.probe_class}_ts{args.train_seed}_cvs{args.cross_validation_seed}_{dataset_config.context}_{dataset_config.id}_{uuid.uuid4().hex[:8]}"

    training_results = train_attn_probes(
        raw_acts_dataset=acts_dataset,
        dataset_config=dataset_config,
        layer=layer_int,
        args=args,
        wandb_project=args.wandb_project,
        wandb_run_name=wandb_run_name,
    )

    # Save results
    output_path = Path("results") / f"attn_probe_{args.file.stem}.pt"
    output_path.parent.mkdir(exist_ok=True)
    torch.save(training_results, output_path)


if __name__ == "__main__":
    main(parse_args())
