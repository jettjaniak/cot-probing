#!/usr/bin/env python3

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cot_probing.typing import *
from cot_probing.utils import setup_determinism


@dataclass
class DataConfig:
    model_name: str
    dataset_id: str
    layer: int
    context: Literal["biased-fsp", "unbiased-fsp", "no-fsp"]
    cv_seed: int
    cv_n_folds: int
    cv_test_fold: int
    train_val_seed: int
    val_frac: float
    data_device: str
    batch_size: int

    def get_acts_filename(self) -> str:
        return f"{self.model_name}_{self.dataset_id}/acts_L{self.layer:02d}_{self.context}.pkl"


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


def load_data(
    raw_q_dicts: list[dict],
    data_device: str,
) -> tuple[list[list[Float[torch.Tensor, "seq d_model"]]], list[int]]:
    """Extract sequences and labels from the dataset for probe training"""
    cots_by_q = []
    labels_by_q = []
    for q_dict in raw_q_dicts:
        # labels
        biased_cot_label = q_dict["biased_cot_label"]
        if biased_cot_label == "faithful":
            label = 1
        elif biased_cot_label == "unfaithful":
            label = 0
        else:
            raise ValueError(f"{biased_cot_label=}")
        labels_by_q.append(label)
        # activations
        # we have multiple CoTs per question
        # TODO: just don't collect acts for answer tokens
        acts_by_cot = [acts.to(data_device)[:-4] for acts in q_dict["cached_acts"]]
        cots_by_q.append(acts_by_cot)
    return cots_by_q, labels_by_q


def split_data(
    cots_by_q: list[list[Float[torch.Tensor, " seq model"]]],
    labels_by_q_list: list[int],
    data_config: DataConfig,
) -> tuple[
    tuple[DataLoader, DataLoader, DataLoader], tuple[list[int], list[int], list[int]]
]:
    """Split data into train/val/test sets using cross-validation folds"""

    # Separate indices by label
    pos_idxs = [i for i, label in enumerate(labels_by_q_list) if label == 1]
    neg_idxs = [i for i, label in enumerate(labels_by_q_list) if label == 0]

    setup_determinism(data_config.cv_seed)
    random.shuffle(pos_idxs)
    random.shuffle(neg_idxs)

    # Calculate fold sizes
    n_folds = data_config.cv_n_folds
    pos_test_size = len(pos_idxs) // n_folds
    neg_test_size = len(neg_idxs) // n_folds

    test_fold_nr = data_config.cv_test_fold
    pos_test_idxs = pos_idxs[
        test_fold_nr * pos_test_size : (test_fold_nr + 1) * pos_test_size
    ]
    neg_test_idxs = neg_idxs[
        test_fold_nr * neg_test_size : (test_fold_nr + 1) * neg_test_size
    ]
    test_idxs = pos_test_idxs + neg_test_idxs
    pos_train_val_idxs = [i for i in pos_idxs if i not in pos_test_idxs]
    neg_train_val_idxs = [i for i in neg_idxs if i not in neg_test_idxs]

    pos_val_size = int(len(pos_train_val_idxs) * data_config.val_frac)
    neg_val_size = int(len(neg_train_val_idxs) * data_config.val_frac)

    setup_determinism(data_config.train_val_seed)
    random.shuffle(pos_train_val_idxs)
    random.shuffle(neg_train_val_idxs)
    pos_val_idxs = pos_train_val_idxs[:pos_val_size]
    neg_val_idxs = neg_train_val_idxs[:neg_val_size]
    val_idxs = pos_val_idxs + neg_val_idxs
    pos_train_idxs = pos_train_val_idxs[pos_val_size:]
    neg_train_idxs = neg_train_val_idxs[neg_val_size:]
    train_idxs = pos_train_idxs + neg_train_idxs

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
        batch_size=data_config.batch_size,
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


def load_and_split_data(
    raw_q_dicts: list[dict],
    data_config: DataConfig,
) -> tuple[
    tuple[DataLoader, DataLoader, DataLoader],
    tuple[list[int], list[int], list[int]],
]:
    cots_by_q, labels_by_q_list = load_data(raw_q_dicts, data_config.data_device)
    return split_data(cots_by_q, labels_by_q_list, data_config)
