#!/usr/bin/env python3

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cot_probing.typing import *
from cot_probing.utils import setup_determinism


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
    batch: Sequence[tuple[list[Float[torch.Tensor, "seq d_model"]], int]],
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
    device = flat_cot_acts[0].device
    padded_cot_acts = torch.zeros(n_cots, max_seq_len, d_model, device=device)
    # Create attention mask
    attn_mask = torch.zeros(n_cots, max_seq_len, dtype=torch.bool, device=device)
    for i, flat_cot in enumerate(flat_cot_acts):
        seq_len = flat_cot.shape[-2]
        padded_cot_acts[i, :seq_len] = flat_cot
        attn_mask[i, :seq_len] = True

    # labels is iterable of 0-dim tensors, just stack them?
    return CollateFnOutput(
        # upcast from BFloat16
        cot_acts=padded_cot_acts.float(),
        attn_mask=attn_mask,
        labels=torch.tensor(flat_labels, dtype=torch.float, device=device),
        q_idxs=flat_q_idxs,
    )


def preprocess_data(
    raw_acts_dataset: dict, data_loading_kwargs: dict[str, Any]
) -> tuple[list[list[Float[torch.Tensor, "seq d_model"]]], list[int]]:
    """Extract sequences and labels from the dataset for probe training"""
    include_answer_toks = data_loading_kwargs["include_answer_toks"]
    assert isinstance(include_answer_toks, bool)
    d_model = data_loading_kwargs["d_model"]
    assert isinstance(d_model, int)
    data_device = data_loading_kwargs["data_device"]
    assert isinstance(data_device, str)
    cots_by_q = []
    labels_by_q = []
    for q_data in raw_acts_dataset["qs"]:
        # labels
        biased_cot_label = q_data["biased_cot_label"]
        if biased_cot_label == "faithful":
            label = 1
        elif biased_cot_label == "unfaithful":
            label = 0
        else:
            raise ValueError(f"{biased_cot_label=}")
        labels_by_q.append(label)
        # activations
        # we have multiple CoTs per question
        acts_by_cot = [acts.to(data_device) for acts in q_data["cached_acts"]]
        assert isinstance(acts_by_cot, list)
        assert acts_by_cot[0].shape[-1] == d_model
        if not include_answer_toks:
            acts_by_cot = [acts[:-4] for acts in acts_by_cot]
        cots_by_q.append(acts_by_cot)
    return cots_by_q, labels_by_q


def split_data(
    cots_by_q: list[list[Float[torch.Tensor, " seq model"]]],
    labels_by_q_list: list[int],
    data_loading_kwargs: dict[str, Any],
) -> tuple[
    tuple[DataLoader, DataLoader, DataLoader], tuple[list[int], list[int], list[int]]
]:
    """Split data into train/val/test sets using cross-validation folds"""
    train_seed = data_loading_kwargs["train_seed"]
    assert isinstance(train_seed, int)
    batch_size = data_loading_kwargs["batch_size"]
    assert isinstance(batch_size, int)

    # Get fold information
    fold = data_loading_kwargs.get("fold", 0)
    n_folds = data_loading_kwargs.get("n_folds", 1)
    assert isinstance(fold, int) and isinstance(n_folds, int)
    assert 0 <= fold < n_folds

    setup_determinism(train_seed)

    # Separate indices by label
    pos_idxs = [i for i, label in enumerate(labels_by_q_list) if label == 1]
    neg_idxs = [i for i, label in enumerate(labels_by_q_list) if label == 0]

    # Shuffle indices
    np.random.shuffle(pos_idxs)
    np.random.shuffle(neg_idxs)

    # Calculate fold sizes
    pos_fold_size = len(pos_idxs) // n_folds
    neg_fold_size = len(neg_idxs) // n_folds

    # Split into folds
    pos_folds = [
        pos_idxs[i : i + pos_fold_size] for i in range(0, len(pos_idxs), pos_fold_size)
    ]
    neg_folds = [
        neg_idxs[i : i + neg_fold_size] for i in range(0, len(neg_idxs), neg_fold_size)
    ]

    # Append remaining indices to last fold if necessary
    if len(pos_folds) > n_folds:
        assert len(pos_folds) == n_folds + 1
        pos_folds[n_folds - 1].extend(pos_folds[n_folds])
        pos_folds = pos_folds[:n_folds]
    if len(neg_folds) > n_folds:
        assert len(neg_folds) == n_folds + 1
        neg_folds[n_folds - 1].extend(neg_folds[n_folds])
        neg_folds = neg_folds[:n_folds]

    assert len(pos_folds) == n_folds
    assert len(neg_folds) == n_folds

    # Use current fold as test set
    test_idxs = pos_folds[fold] + neg_folds[fold]

    # Use next fold (circularly) as validation set
    val_fold = (fold + 1) % n_folds
    val_idxs = pos_folds[val_fold] + neg_folds[val_fold]

    # Use remaining folds as training set
    train_folds = list(range(n_folds))
    train_folds.remove(fold)
    train_folds.remove(val_fold)

    train_idxs = []
    for train_fold in train_folds:
        train_idxs.extend(pos_folds[train_fold])
        train_idxs.extend(neg_folds[train_fold])

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
        batch_size=batch_size,
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

    return (train_loader, val_loader, test_loader), (train_idxs, val_idxs, test_idxs)


def preprocess_and_split_data(
    raw_acts_dataset: dict,
    data_loading_kwargs: dict[str, Any],
) -> tuple[
    tuple[DataLoader, DataLoader, DataLoader],
    tuple[list[int], list[int], list[int]],
]:
    cots_by_q, labels_by_q_list = preprocess_data(raw_acts_dataset, data_loading_kwargs)
    return split_data(cots_by_q, labels_by_q_list, data_loading_kwargs)
