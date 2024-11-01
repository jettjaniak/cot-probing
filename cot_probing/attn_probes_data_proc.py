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
            acts_by_cot = [acts[:-3] for acts in acts_by_cot]
        cots_by_q.append(acts_by_cot)
    return cots_by_q, labels_by_q


def split_data(
    cots_by_q: list[list[Float[torch.Tensor, " seq model"]]],
    labels_by_q_list: list[int],
    data_loading_kwargs: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    data_seed = data_loading_kwargs["data_seed"]
    assert isinstance(data_seed, int)
    test_split = data_loading_kwargs["test_split"]
    assert isinstance(test_split, float)
    validation_split = data_loading_kwargs["validation_split"]
    assert isinstance(validation_split, float)
    batch_size = data_loading_kwargs["batch_size"]
    assert isinstance(batch_size, int)
    setup_determinism(data_seed)

    # Separate indices by label
    pos_idxs = [i for i, label in enumerate(labels_by_q_list) if label == 1]
    neg_idxs = [i for i, label in enumerate(labels_by_q_list) if label == 0]

    # Shuffle indices
    np.random.shuffle(pos_idxs)
    np.random.shuffle(neg_idxs)

    min_size = min(len(pos_idxs), len(neg_idxs))
    # Calculate sizes for each split
    test_size_per_class = int(min_size * test_split)
    val_size_per_class = int(min_size * validation_split)

    # Split indices for each class
    pos_test = pos_idxs[:test_size_per_class]
    pos_val = pos_idxs[test_size_per_class : test_size_per_class + val_size_per_class]
    pos_train = pos_idxs[test_size_per_class + val_size_per_class :]

    neg_test = neg_idxs[:test_size_per_class]
    neg_val = neg_idxs[test_size_per_class : test_size_per_class + val_size_per_class]
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

    return train_loader, val_loader, test_loader


def preprocess_and_split_data(
    raw_acts_dataset: dict,
    data_loading_kwargs: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    cots_by_q, labels_by_q_list = preprocess_data(raw_acts_dataset, data_loading_kwargs)
    return split_data(cots_by_q, labels_by_q_list, data_loading_kwargs)
