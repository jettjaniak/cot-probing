#!/usr/bin/env python3
import argparse
import logging
import pickle
import uuid
from pathlib import Path

import torch
from beartype import beartype

from cot_probing.attn_probes import (
    AbstractAttnProbeModel,
    AttnProbeModelConfig,
    AttnProbeTrainer,
    FullAttnProbeModel,
    MediumAttnProbeModel,
    MinimalAttnProbeModel,
    ProbingConfig,
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
        "--test-split",
        type=float,
        default=0.1,
        help="Size of the test set split.",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Size of the validation set split.",
    )
    parser.add_argument(
        "--probe-class",
        type=str,
        required=True,
        choices=["minimal", "medium", "full"],
        help="Type of attention probe to use",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=4096,
        help="Model dimension (should match the model being probed)",
    )
    parser.add_argument(
        "--d-head",
        type=int,
        default=None,
        help="Head dimension",
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
        "--batch-size",
        type=int,
        default=4,
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
        "--data-seed",
        "--ds",
        type=int,
        default=0,
        help="Random seed for reproducibility of dataset splitting",
    )
    parser.add_argument(
        "--weight-init-seed",
        "--ws",
        type=int,
        default=0,
        help="Random seed for weight initialization",
    )
    parser.add_argument(
        "--include-answer-toks",
        action="store_true",
        help="Include answer tokens in the probe training",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


@beartype
def get_probe_model_class(probe_class_arg: str) -> type[AbstractAttnProbeModel]:
    return {
        "minimal": MinimalAttnProbeModel,
        "medium": MediumAttnProbeModel,
        "full": FullAttnProbeModel,
    }[probe_class_arg]


@beartype
def build_probe_config(
    args: argparse.Namespace,
    layer: int,
) -> ProbingConfig:
    probe_class_arg = args.probe_class
    d_head = args.d_head
    if d_head is None:
        assert args.probe_class != "full"
        d_head = args.d_model

    probe_config = AttnProbeModelConfig(
        d_model=args.d_model,
        d_head=d_head,
        weight_init_range=args.weight_init_range,
        weight_init_seed=args.weight_init_seed,
    )

    return ProbingConfig(
        probe_model_class=get_probe_model_class(probe_class_arg),
        probe_model_config=probe_config,
        data_seed=args.data_seed,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        patience=args.patience,
        n_epochs=args.epochs,
        validation_split=args.validation_split,
        test_split=args.test_split,
        model_device=args.model_device,
        data_device=args.data_device,
        layer=layer,
    )


@beartype
def train_attn_probe(
    raw_acts_dataset: dict,
    layer: int,
    args: argparse.Namespace,
    wandb_run_name: str,
    wandb_project: str,
) -> dict:
    """Train attention probe on the dataset"""
    # Create probe configuration
    probe_config = build_probe_config(args, layer)

    # Initialize trainer
    trainer = AttnProbeTrainer(
        c=probe_config,
        raw_acts_dataset=raw_acts_dataset,
        data_loading_kwargs=vars(args),
    )

    # Train probe
    model = trainer.train(
        run_name=wandb_run_name,
        project_name=wandb_project,
        args=args,
    )

    return {
        "model": model,
        "config": probe_config,
    }


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

    # Create default wandb run name if none provided
    if args.wandb_run_name is None:
        with_answer_str = "WITH_ANSWER_" if args.include_answer_toks else ""
        wandb_run_name = f"{with_answer_str}{layer_str}_{args.probe_class}_ds{args.data_seed}_ws{args.weight_init_seed}_{uuid.uuid4().hex[:8]}"
    probing_results = train_attn_probe(
        raw_acts_dataset=acts_dataset,
        layer=layer_int,
        args=args,
        wandb_project=args.wandb_project,
        wandb_run_name=wandb_run_name,
    )

    # Save results
    output_path = Path("results") / f"attn_probe_{args.file.stem}.pt"
    output_path.parent.mkdir(exist_ok=True)
    torch.save(probing_results, output_path)


if __name__ == "__main__":
    main(parse_args())
