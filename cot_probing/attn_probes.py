#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from fancy_einsum import einsum
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

import wandb
from cot_probing.typing import *
from cot_probing.utils import setup_determinism

torch.set_grad_enabled(True)


@dataclass(kw_only=True)
class AttnProbeModelConfig:
    d_model: int
    d_head: int
    weight_init_range: float
    weight_init_seed: int


@dataclass
class ProbingConfig:
    probe_model_class: type["AbstractAttnProbeModel"]
    probe_model_config: AttnProbeModelConfig
    data_seed: int
    lr: float
    batch_size: int
    patience: int
    n_epochs: int
    validation_split: float
    test_split: float
    device: str


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
) -> tuple[
    Float[torch.Tensor, " batch seq d_model"],
    Float[torch.Tensor, " batch"],
    Bool[torch.Tensor, " batch seq"],
]:
    """Handle variable sequence lengths with padding"""
    sequences, labels = zip(*batch)
    max_len = max(seq.shape[-2] for seq in sequences)
    d_model = sequences[0].shape[-1]

    # Create padded tensor
    padded = torch.zeros(len(sequences), max_len, d_model)
    # Create attention mask
    attn_mask = torch.zeros(len(sequences), max_len, dtype=torch.bool)
    for i, seq in enumerate(sequences):
        seq_len = seq.shape[0]
        padded[i, :seq_len] = seq
        attn_mask[i, :seq_len] = True

    # labels is iterable of 0-dim tensors, just stack them?
    return padded, torch.stack(list(labels)), attn_mask


class AbstractAttnProbeModel(nn.Module, ABC):
    def __init__(self, c: AttnProbeModelConfig):
        super().__init__()
        self.c = c
        setup_determinism(c.weight_init_seed)
        self.z_bias = nn.Parameter(torch.zeros(1))
        self.value_vector = nn.Parameter(torch.randn(c.d_model) * c.weight_init_range)

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

    def values(
        self, resids: Float[torch.Tensor, " batch seq model"]
    ) -> Float[torch.Tensor, " batch seq"]:
        # Project residuals using the value vector
        return einsum("batch seq model, model -> batch seq", resids, self.value_vector)

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


class MinimalAttnProbeModel(AbstractAttnProbeModel):
    def __init__(self, c: AttnProbeModelConfig):
        assert c.d_head == c.d_model
        super().__init__(c)
        # Use value_vector as probe_vector
        self.temperature = nn.Parameter(torch.ones(1))

    def _query(self) -> Float[torch.Tensor, " head"]:
        # Scale the value vector by temperature for query
        return self.value_vector * self.temperature

    def _keys(
        self, resids: Float[torch.Tensor, " batch seq model"]
    ) -> Float[torch.Tensor, " batch seq head"]:
        # Use residuals directly as keys since d_model == d_head
        return resids


class MediumAttnProbeModel(AbstractAttnProbeModel):
    def __init__(self, c: AttnProbeModelConfig):
        assert c.d_head == c.d_model
        super().__init__(c)
        # Only need query vector since value vector is in parent
        self.query_vector = nn.Parameter(torch.randn(c.d_model) * c.weight_init_range)

    def _query(self) -> Float[torch.Tensor, " head"]:
        return self.query_vector

    def _keys(
        self, resids: Float[torch.Tensor, " batch seq model"]
    ) -> Float[torch.Tensor, " batch seq head"]:
        # Use residuals directly as keys since d_model == d_head
        return resids


class FullAttnProbeModel(MediumAttnProbeModel):
    def __init__(self, c: AttnProbeModelConfig):
        super().__init__(c)
        # Key projection matrix from model space to attention head space
        self.W_K = nn.Parameter(torch.randn(c.d_model, c.d_head) * c.weight_init_range)

    def _keys(
        self, resids: Float[torch.Tensor, " batch seq model"]
    ) -> Float[torch.Tensor, " batch seq head"]:
        # Project residuals to key space using W_K
        return einsum("batch seq model, model head -> batch seq head", resids, self.W_K)


class ProbeTrainer:
    def __init__(self, c: ProbingConfig):
        self.c = c
        self.device = torch.device(c.device)

    def prepare_data(
        self, sequences: list[Float[torch.Tensor, " seq d_model"]], labels: list[int]
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        setup_determinism(self.c.data_seed)

        # Balance the dataset
        label_0_indices = [i for i, label in enumerate(labels) if label == 0]
        label_1_indices = [i for i, label in enumerate(labels) if label == 1]
        min_label_count = min(len(label_0_indices), len(label_1_indices))

        balanced_indices = (
            np.random.choice(label_0_indices, min_label_count, replace=False).tolist()
            + np.random.choice(label_1_indices, min_label_count, replace=False).tolist()
        )
        np.random.shuffle(balanced_indices)

        balanced_sequences = [sequences[i] for i in balanced_indices]
        balanced_labels = [labels[i] for i in balanced_indices]

        # Create train/val/test splits
        total_samples = len(balanced_sequences)
        test_size = int(total_samples * self.c.test_split)
        val_size = int(total_samples * self.c.validation_split)
        train_size = total_samples - test_size - val_size

        # Create datasets
        train_dataset = SequenceDataset(
            [balanced_sequences[i].to(self.device).float() for i in range(train_size)],
            [balanced_labels[i] for i in range(train_size)],
        )
        val_dataset = SequenceDataset(
            [
                balanced_sequences[i + train_size].to(self.device).float()
                for i in range(val_size)
            ],
            [balanced_labels[i + train_size] for i in range(val_size)],
        )
        test_dataset = SequenceDataset(
            [
                balanced_sequences[i + train_size + val_size].to(self.device).float()
                for i in range(test_size)
            ],
            [balanced_labels[i + train_size + val_size] for i in range(test_size)],
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
        project_name: str,
    ) -> AbstractAttnProbeModel:
        # Initialize W&B
        wandb.init(entity="cot-probing", project=project_name, name=run_name)
        wandb.config.update(asdict(self.c))

        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(
            sequences, labels_list
        )

        # Initialize model and optimizer
        model = self.c.probe_model_class(self.c.probe_model_config).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.c.lr)
        criterion = nn.BCELoss()

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = model.state_dict().copy()

        for epoch in range(self.c.n_epochs):
            try:
                train_loss, train_acc, val_loss, val_acc = self._train_epoch(
                    model=model,
                    optimizer=optimizer,
                    criterion=criterion,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    epoch=epoch,
                )
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
                    # Save best model state to wandb
                    torch.save(best_model_state, "results/best_model.pt")
                    wandb.save("results/best_model.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= self.c.patience:
                        print(f"Early stopping at epoch {epoch}")
                        break
            except KeyboardInterrupt:
                break

        # Load best model
        model.load_state_dict(best_model_state)
        # Final test set evaluation
        model.eval()
        test_loss, test_acc = self._compute_loss_and_acc(model, criterion, test_loader)

        wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
        wandb.finish()

        return model

    def _compute_loss_and_acc_single_batch(
        self,
        model: AbstractAttnProbeModel,
        criterion: nn.BCELoss,
        batch: torch.Tensor,
        labels_t: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> tuple[Float[torch.Tensor, ""], float]:
        batch = batch.to(self.device)
        labels_t = labels_t.to(self.device)
        attn_mask = attn_mask.to(self.device)
        outputs = model(batch, attn_mask)
        loss = criterion(outputs, labels_t)
        acc = ((outputs > 0.5) == labels_t).float().mean().item()
        return loss, acc

    def _compute_loss_and_acc(
        self,
        model: AbstractAttnProbeModel,
        criterion: nn.BCELoss,
        data_loader: DataLoader,
    ) -> tuple[float, float]:
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for batch, labels_t, attn_mask in data_loader:
                loss, acc = self._compute_loss_and_acc_single_batch(
                    model, criterion, batch, labels_t, attn_mask
                )
                total_loss += loss.item()
                total_acc += acc

        avg_loss = total_loss / len(data_loader)
        avg_acc = total_acc / len(data_loader)
        return avg_loss, avg_acc

    def _train_epoch(
        self,
        model: AbstractAttnProbeModel,
        optimizer: torch.optim.Adam,
        criterion: nn.BCELoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epoch: int,
    ) -> tuple[float, float, float, float]:
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        for batch, labels_t, attn_mask in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            loss, acc = self._compute_loss_and_acc_single_batch(
                model, criterion, batch, labels_t, attn_mask
            )
            loss.backward()
            optimizer.step()
            train_loss += loss
            train_acc += acc

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_acc = self._compute_loss_and_acc(model, criterion, val_loader)
        assert isinstance(train_loss, torch.Tensor)
        return train_loss.item(), train_acc, val_loss, val_acc
