#!/usr/bin/env python3

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import torch
from fancy_einsum import einsum
from torch import nn
from torch.optim.adam import Adam
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
    layer: int


class SequenceDataset(Dataset):
    def __init__(
        self,
        cots_by_q: list[list[Float[torch.Tensor, "seq d_model"]]],
        labels_by_q: list[int],
    ):
        self.cots_by_q = cots_by_q
        # Convert labels to same dtype and device as first sequence
        self.labels_by_q = labels_by_q

    def __len__(self) -> int:
        return len(self.labels_by_q)

    def __getitem__(
        self, idx: int
    ) -> tuple[list[Float[torch.Tensor, "seq d_model"]], int]:
        return self.cots_by_q[idx], self.labels_by_q[idx]


@dataclass
class CollateFnOutput:
    cot_acts: Float[torch.Tensor, " batch seq model"]
    attn_mask: Bool[torch.Tensor, " batch seq"]
    labels: Float[torch.Tensor, " batch"]
    q_idxs: list[int]


def collate_fn(
    batch: Sequence[tuple[list[Float[torch.Tensor, "seq d_model"]], int]]
) -> CollateFnOutput:
    flat_cot_acts = []
    flat_labels = []
    # these are indices into the batch, not the original question indices
    flat_q_idxs = []
    for i, (cots, label) in enumerate(batch):
        for cot in cots:
            flat_cot_acts.append(cot)
            flat_labels.append(label)
            flat_q_idxs.append(i)
    n_cots = len(flat_cot_acts)

    # flat_cot is shape (seq, d_model)
    max_seq_len = max(flat_cot_act.shape[-2] for flat_cot_act in flat_cot_acts)

    # Create padded tensor
    d_model = flat_cot_acts[0].shape[-1]
    padded_cot_acts = torch.zeros(n_cots, max_seq_len, d_model)
    # Create attention mask
    attn_mask = torch.zeros(n_cots, max_seq_len, dtype=torch.bool)
    for i, flat_cot in enumerate(flat_cot_acts):
        seq_len = flat_cot.shape[-2]
        padded_cot_acts[i, :seq_len] = flat_cot
        attn_mask[i, :seq_len] = True

    # labels is iterable of 0-dim tensors, just stack them?
    return CollateFnOutput(
        # upcast from BFloat16
        cot_acts=padded_cot_acts.float(),
        attn_mask=attn_mask,
        labels=torch.stack(flat_labels),
        q_idxs=flat_q_idxs,
    )


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
        self,
        cots_by_q: list[list[Float[torch.Tensor, " seq model"]]],
        labels_by_q_list: list[int],
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        setup_determinism(self.c.data_seed)

        # Separate indices by label
        pos_idxs = [i for i, label in enumerate(labels_by_q_list) if label == 1]
        neg_idxs = [i for i, label in enumerate(labels_by_q_list) if label == 0]

        # Shuffle indices
        np.random.shuffle(pos_idxs)
        np.random.shuffle(neg_idxs)

        min_size = min(len(pos_idxs), len(neg_idxs))
        # Calculate sizes for each split
        test_size_per_class = int(min_size * self.c.test_split)
        val_size_per_class = int(min_size * self.c.validation_split)

        # Split indices for each class
        pos_test = pos_idxs[:test_size_per_class]
        pos_val = pos_idxs[
            test_size_per_class : test_size_per_class + val_size_per_class
        ]
        pos_train = pos_idxs[test_size_per_class + val_size_per_class :]

        neg_test = neg_idxs[:test_size_per_class]
        neg_val = neg_idxs[
            test_size_per_class : test_size_per_class + val_size_per_class
        ]
        neg_train = neg_idxs[test_size_per_class + val_size_per_class :]

        train_idxs = pos_train + neg_train
        val_idxs = pos_val + neg_val
        test_idxs = pos_test + neg_test

        def make_dataset(idxs: list[int]) -> SequenceDataset:
            return SequenceDataset(
                [cots_by_q[i] for i in idxs],
                [labels_by_q_list[i] for i in idxs],
            )

        # Create datasets
        train_dataset = make_dataset(train_idxs)
        val_dataset = make_dataset(val_idxs)
        test_dataset = make_dataset(test_idxs)

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.c.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )
        # no batching for validation and test sets
        val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            collate_fn=collate_fn,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=len(test_dataset),
            collate_fn=collate_fn,
        )

        return train_loader, val_loader, test_loader

    def train(
        self,
        cots_by_q: list[list[Float[torch.Tensor, " seq model"]]],
        labels_by_q_list: list[int],
        run_name: str,
        project_name: str,
        args_dict: dict,
    ) -> AbstractAttnProbeModel:
        # Initialize W&B
        wandb.init(entity="cot-probing", project=project_name, name=run_name)
        wandb.config.update(asdict(self.c))
        wandb.config.update({f"args_{k}": v for k, v in args_dict.items()})
        # Prepare data
        train_loader, val_loader, test_loader = self.prepare_data(
            cots_by_q, labels_by_q_list
        )

        # Initialize model and optimizer
        model = self.c.probe_model_class(self.c.probe_model_config).to(self.device)
        optimizer = Adam(model.parameters(), lr=self.c.lr)
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
        collate_fn_output: CollateFnOutput,
    ) -> tuple[Float[torch.Tensor, ""], float]:
        cot_acts = collate_fn_output.cot_acts.to(self.device)
        attn_mask = collate_fn_output.attn_mask.to(self.device)
        labels = collate_fn_output.labels.to(self.device)
        q_idxs = collate_fn_output.q_idxs
        outputs = model(cot_acts, attn_mask)

        def compute_class_metrics(
            target_label: int,
        ) -> tuple[Float[torch.Tensor, ""], float]:
            # Get unique question indices for target label
            target_q_idxs = list(
                {q_idx for i, q_idx in enumerate(q_idxs) if labels[i] == target_label}
            )

            # Compute loss and accuracy for each question with target label
            q_losses = []
            q_accs = []
            for q_idx in target_q_idxs:
                # in each iteration we process all CoTs
                # that were present in a batch for one question
                q_mask = [i == q_idx for i in q_idxs]
                q_outputs = outputs[q_mask]
                q_labels = labels[q_mask]
                assert torch.all(q_labels == target_label)
                # Compute loss
                q_loss = criterion(q_outputs, q_labels)
                q_losses.append(q_loss)
                # Compute accuracy
                q_acc = ((q_outputs > 0.5) == q_labels.bool()).float().mean().item()
                q_accs.append(q_acc)

            # Average metrics
            if not q_losses:
                return torch.tensor(0.0).to(self.device), 0.0
            return torch.stack(q_losses).mean(), sum(q_accs) / len(q_accs)

        # Compute metrics for positive and negative classes
        pos_loss, pos_acc = compute_class_metrics(target_label=1)
        neg_loss, neg_acc = compute_class_metrics(target_label=0)

        # Take average of positive and negative metrics
        balanced_loss = (pos_loss + neg_loss) / 2
        balanced_acc = (pos_acc + neg_acc) / 2

        return balanced_loss, balanced_acc

    def _compute_loss_and_acc(
        self,
        model: AbstractAttnProbeModel,
        criterion: nn.BCELoss,
        data_loader: DataLoader,
    ) -> tuple[float, float]:
        total_loss = 0
        total_acc = 0
        with torch.no_grad():
            for collate_fn_output in data_loader:
                loss, acc = self._compute_loss_and_acc_single_batch(
                    model, criterion, collate_fn_output
                )
                total_loss += loss.item()
                total_acc += acc

        avg_loss = total_loss / len(data_loader)
        avg_acc = total_acc / len(data_loader)
        return avg_loss, avg_acc

    def _train_epoch(
        self,
        model: AbstractAttnProbeModel,
        optimizer: Adam,
        criterion: nn.BCELoss,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epoch: int,
    ) -> tuple[float, float, float, float]:
        # Training
        model.train()
        train_loss = 0
        train_acc = 0
        for collate_fn_output in tqdm(train_loader, desc=f"Epoch {epoch}"):
            optimizer.zero_grad()
            loss, acc = self._compute_loss_and_acc_single_batch(
                model, criterion, collate_fn_output
            )
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += acc

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        # Validation
        model.eval()
        val_loss, val_acc = self._compute_loss_and_acc(model, criterion, val_loader)
        return train_loss, train_acc, val_loss, val_acc
