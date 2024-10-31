#!/usr/bin/env python3
import argparse
import logging
import pickle
import random
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from cot_probing.attn_probes import (  # FullAttnProbeConfig,
    MediumAttnProbeConfig,
    MinimalAttnProbeConfig,
    ProbeTrainer,
    ProbingConfig,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train attn probes")
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        required=True,
        help="Path to the dataset of activations",
    )
    parser.add_argument(
        "--test-split-size",
        type=float,
        default=0.1,
        help="Size of the test set split.",
    )
    parser.add_argument(
        "--validation-split-size",
        type=float,
        default=0.1,
        help="Size of the validation set split.",
    )
    parser.add_argument(
        "--probe-class",
        type=str,
        default="minimal",
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
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--use-temperature",
        action="store_true",
        help="Use temperature for attention queries",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Number of epochs to wait for improvement before early stopping",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum number of epochs to train for",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device to train on",
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
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


def get_probe_class_name(probe_class_arg: str) -> str:
    return {
        "minimal": "MinimalAttnProbeModel",
        "medium": "MediumAttnProbeModel",
        "full": "FullAttnProbeModel",
    }[probe_class_arg]


def build_probe_config(
    args: argparse.Namespace,
) -> ProbingConfig:
    probe_class_arg = args.probe_class

    if probe_class_arg == "minimal":
        probe_config = MinimalAttnProbeConfig(
            d_model=args.d_model,
            d_head=args.d_model,  # For MinimalAttnProbe, d_head must equal d_model
            weight_init_range=args.weight_init_range,
            weight_init_seed=args.seed,
            use_temperature=args.use_temperature,
        )
    elif probe_class_arg == "medium":
        probe_config = MediumAttnProbeConfig(
            d_model=args.d_model,
            d_head=args.d_model,
            weight_init_range=args.weight_init_range,
            weight_init_seed=args.seed,
        )
    elif probe_class_arg == "full":
        # probe_config = FullAttnProbeConfig(
        #     d_model=args.d_model,
        #     d_head=args.d_model,
        #     weight_init_range=args.weight_init_range,
        #     weight_init_seed=args.seed,
        # )
        raise NotImplementedError("FullAttnProbeConfig is not implemented")
    else:
        raise ValueError(f"Invalid probe class: {probe_class_arg}")

    return ProbingConfig(
        probe_class_name=get_probe_class_name(probe_class_arg),
        probe_config=probe_config,
        data_seed=args.seed,
        lr=args.learning_rate,
        batch_size=args.batch_size,
        patience=args.patience,
        n_epochs=args.epochs,
        validation_split=args.validation_split_size,
        test_split=args.test_split_size,
        device=args.device,
    )


def train_attn_probe(
    acts_dataset: Dict,
    args: argparse.Namespace,
    seed: int = 42,
    wandb_project: str = "attn-probes",
    wandb_run_name: str | None = None,
    verbose: bool = False,
) -> Dict:
    """Train attention probe on the dataset"""
    # Extract sequences and labels from the dataset
    sequences = []
    labels = []

    for q_data in acts_dataset["qs"]:
        # Convert cached activations to torch tensors
        if isinstance(q_data["cached_acts"], list):
            # Multiple sequences per question (biased CoTs mode)
            for acts in q_data["cached_acts"]:
                sequences.append(torch.tensor(acts))
                # Convert yes/no to 1/0
                labels.append(1 if q_data["biased_cot_label"] == "faithful" else 0)
        else:
            # Single sequence per question
            sequences.append(torch.tensor(q_data["cached_acts"]))
            labels.append(1 if q_data["biased_cot_label"] == "faithful" else 0)

    # Create probe configuration
    probe_config = build_probe_config(args)

    # Initialize trainer
    trainer = ProbeTrainer(probe_config)

    # Train probe
    run_name = wandb_run_name or f"attn_probe_seed_{seed}"
    model = trainer.train(
        sequences=sequences,
        labels_list=labels,
        run_name=run_name,
        project_name=wandb_project,
    )

    return {
        "model": model,
        "config": probe_config,
    }


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    # Set seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    acts_file_path = Path(args.file)
    if not acts_file_path.exists():
        raise FileNotFoundError(f"File not found at {acts_file_path}")

    with open(acts_file_path, "rb") as f:
        acts_dataset = pickle.load(f)

    probing_results = train_attn_probe(
        acts_dataset=acts_dataset,
        args=args,
        seed=args.seed,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        verbose=args.verbose,
    )

    # Save results
    output_path = Path("results") / f"attn_probe_{args.file.stem}.pt"
    output_path.parent.mkdir(exist_ok=True)
    torch.save(probing_results, output_path)


if __name__ == "__main__":
    main(parse_args())
