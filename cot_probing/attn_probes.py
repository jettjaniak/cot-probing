#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
import wandb
from fancy_einsum import einsum
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from cot_probing.typing import *
from cot_probing.utils import setup_determinism


@dataclass(kw_only=True)
class AbstractAttnProbeConfig:
    d_model: int
    d_head: int
    weight_init_range: float = 0.02
    weight_init_seed: int = 0


@dataclass
class ProbingConfig:
    probe_class_name: str
    probe_config: AbstractAttnProbeConfig
    data_seed: int = 0
    lr: float = 1e-3
    batch_size: int = 32
    patience: int = 5
    n_epochs: int = 100
    validation_split: float = 0.1
    test_split: float = 0.1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class SequenceDataset(Dataset):
    def __init__(
        self, sequences: list[Float[torch.Tensor, "seq d_model"]], labels: list[int]
    ):
        self.sequences = sequences
        # Convert labels to same dtype and device as first sequence
        self.labels = torch.tensor(
            labels, dtype=self.sequences[0].dtype, device=self.sequences[0].device
        )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(
        self, idx: int
    ) -> tuple[Float[torch.Tensor, "seq d_model"], Float[torch.Tensor, ""]]:
        return self.sequences[idx], self.labels[idx]


def collate_fn(
    batch: Sequence[tuple[Float[torch.Tensor, "seq d_model"], Float[torch.Tensor, ""]]]
) -> tuple[Float[torch.Tensor, " batch seq d_model"], Float[torch.Tensor, " batch"]]:
    """Handle variable sequence lengths with padding"""
    sequences, labels = zip(*batch)
    max_len = max(seq.shape[-2] for seq in sequences)
    d_model = sequences[0].shape[-1]

    # Create padded tensor
    padded = torch.zeros(len(sequences), max_len, d_model)
    for i, seq in enumerate(sequences):
        padded[i, : seq.shape[0]] = seq

    # labels is iterable of 0-dim tensors, just stack them?
    return padded, torch.stack(list(labels))


class AbstractAttnProbeModel(nn.Module, ABC):
    def __init__(self, c: AbstractAttnProbeConfig):
        super().__init__()
        self.c = c
        self.z_bias = nn.Parameter(torch.zeros(1))

    @abstractmethod
    def _query(
        self,
    ) -> Float[torch.Tensor, " head"]:
        pass

    def query(
        self,
    ) -> Float[torch.Tensor, " head"]:
        ret = self._query()
        assert ret.shape[-1] == self.c.d_head
        return ret

    @abstractmethod
    def _keys(
        self, resids: Float[torch.Tensor, " batch seq d_model"]
    ) -> Float[torch.Tensor, " batch seq head"]:
        pass

    def keys(
        self, resids: Float[torch.Tensor, " batch seq d_model"]
    ) -> Float[torch.Tensor, " batch seq head"]:
        ret = self._keys(resids)
        assert ret.shape[-1] == self.c.d_head
        return ret

    @abstractmethod
    def values(
        self, resids: Float[torch.Tensor, " batch seq d_model"]
    ) -> Float[torch.Tensor, " batch seq"]:
        pass

    def forward(
        self,
        resids: Float[torch.Tensor, "batch seq model"],
        attn_mask: Bool[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch"]:
        # sometimes d_head will be equal to d_model (i.e. in simple probes)

        # Compute attention scores (before softmax)
        # shape (batch, seq, d_head)
        keys = self.keys(resids)
        # shape (d_head,)
        query_not_expanded = self.query()
        # shape (batch, d_head)
        query = query_not_expanded.expand(resids.shape[0], -1)
        # shape (batch, seq)
        attn_scores = einsum("batch seq head, batch head -> batch seq", keys, query)
        attn_scores = attn_scores.masked_fill(
            ~attn_mask, torch.finfo(attn_scores.dtype).min
        )
        # shape (batch, seq)
        # after softmax
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # shape (batch, seq)
        # unlike normal attention, these are 1-dimensional
        values = self.values(resids)
        z = einsum("batch seq, batch seq -> batch", attn_probs, values)
        return torch.sigmoid(z + self.z_bias)


@dataclass(kw_only=True)
class MinimalAttnProbeConfig(AbstractAttnProbeConfig):
    use_temperature: bool

    def __post_init__(self):
        assert self.d_head == self.d_model


class MinimalAttnProbeModel(AbstractAttnProbeModel):
    def __init__(self, c: MinimalAttnProbeConfig):
        super().__init__(c)
        setup_determinism(c.weight_init_seed)
        # Initialize a single vector for both query and value
        self.probe_vector = nn.Parameter(torch.randn(c.d_model) * c.weight_init_range)
        # Optional temperature parameter for scaling attention
        self.temperature = (
            nn.Parameter(torch.ones(1)) if c.use_temperature else torch.ones(1)
        )

    def _query(self) -> Float[torch.Tensor, " head"]:
        # Scale the probe vector by temperature for query
        return self.probe_vector * self.temperature

    def _keys(
        self, resids: Float[torch.Tensor, " batch seq model"]
    ) -> Float[torch.Tensor, " batch seq head"]:
        # Use residuals directly as keys since d_model == d_head
        return resids

    def values(
        self, resids: Float[torch.Tensor, " batch seq model"]
    ) -> Float[torch.Tensor, " batch seq"]:
        # Project residuals using the same probe vector
        return einsum("batch seq model, model -> batch seq", resids, self.probe_vector)


class ProbeTrainer:
    def __init__(self, c: ProbingConfig):
        self.c = c
        self.device = torch.device(c.device)

    def prepare_data(
        self, sequences: list[Float[torch.Tensor, " seq d_model"]], labels: list[int]
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        setup_determinism(self.c.data_seed)
        # Create train/val/test splits
        indices = np.random.permutation(len(sequences))
        test_size = int(len(sequences) * self.c.test_split)
        val_size = int(len(sequences) * self.c.validation_split)

        test_idx = indices[:test_size]
        val_idx = indices[test_size : test_size + val_size]
        train_idx = indices[test_size + val_size :]

        # Create datasets
        train_dataset = SequenceDataset(
            [sequences[i] for i in train_idx], [labels[i] for i in train_idx]
        )
        val_dataset = SequenceDataset(
            [sequences[i] for i in val_idx], [labels[i] for i in val_idx]
        )
        test_dataset = SequenceDataset(
            [sequences[i] for i in test_idx], [labels[i] for i in test_idx]
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.c.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.c.batch_size,
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.c.batch_size,
            collate_fn=collate_fn,
        )

        return train_loader, val_loader, test_loader

    def train(
        self,
        sequences: list[Float[torch.Tensor, "seq d_model"]],
        labels_list: list[int],
        run_name: str,
        project_name: str = "attn-probes",
    ) -> AbstractAttnProbeModel:
        # Initialize W&B
        wandb.init(project=project_name, name=run_name)
        wandb.config.update(asdict(self.c))

        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(
            sequences, labels_list
        )

        # Initialize model and optimizer
        model = globals()[self.c.probe_class_name](self.c.probe_config)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.c.lr)
        criterion = nn.BCELoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(self.c.n_epochs):
            # Training
            model.train()
            train_loss = 0
            train_acc = 0
            for batch, labels_t in tqdm(train_loader, desc=f"Epoch {epoch}"):
                batch = batch.to(self.device)
                labels_t = labels_t.to(self.device)

                optimizer.zero_grad()
                outputs = model(batch)
                loss = criterion(outputs, labels_t)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_acc += ((outputs > 0.5) == labels_t).float().mean().item()

            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            # Validation
            model.eval()
            val_loss = 0
            val_acc = 0
            with torch.no_grad():
                for batch, labels_t in val_loader:
                    batch = batch.to(self.device)
                    labels_t = labels_t.to(self.device)
                    outputs = model(batch)
                    val_loss += criterion(outputs, labels_t).item()
                    val_acc += ((outputs > 0.5) == labels_t).float().mean().item()

            val_loss /= len(val_loader)
            val_acc /= len(val_loader)

            # Logging
            wandb.log(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "epoch": epoch,
                }
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= self.c.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Save best model state to wandb
        torch.save(best_model_state, "results/best_model.pt")
        wandb.save("results/best_model.pt")

        # Load best model
        model.load_state_dict(best_model_state)

        # Final test set evaluation
        model.eval()
        test_loss = 0
        test_acc = 0
        with torch.no_grad():
            for batch, labels_t in test_loader:
                batch = batch.to(self.device)
                labels_t = labels_t.to(self.device)
                outputs = model(batch)
                test_loss += criterion(outputs, labels_t).item()
                test_acc += ((outputs > 0.5) == labels_t).float().mean().item()

        test_loss /= len(test_loader)
        test_acc /= len(test_loader)

        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})

        wandb.finish()
        return model
