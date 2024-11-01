#!/usr/bin/env python3

import argparse
import tempfile
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from fancy_einsum import einsum
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from wandb.apis.public.runs import Run

import wandb
from cot_probing.attn_probes_data_proc import CollateFnOutput, preprocess_and_split_data
from cot_probing.typing import *
from cot_probing.utils import get_git_commit_hash, safe_torch_save, setup_determinism

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
    model_device: str
    data_device: str
    layer: int


class AbstractAttnProbeModel(nn.Module, ABC):
    def __init__(self, c: AttnProbeModelConfig):
        super().__init__()
        self.c = c
        setup_determinism(c.weight_init_seed)
        self.z_bias = nn.Parameter(torch.zeros(1))
        self.value_vector = nn.Parameter(torch.randn(c.d_model) * c.weight_init_range)

    @property
    def device(self) -> torch.device:
        return self.value_vector.device

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

    def attn_probs(
        self,
        resids: Float[torch.Tensor, "batch seq model"],
        attn_mask: Bool[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch seq"]:
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
        return torch.softmax(attn_scores, dim=-1)

    def forward(
        self,
        resids: Float[torch.Tensor, "batch seq model"],
        attn_mask: Bool[torch.Tensor, "batch seq"],
    ) -> Float[torch.Tensor, "batch"]:
        # sometimes d_head will be equal to d_model (i.e. in simple probes)
        attn_probs = self.attn_probs(resids, attn_mask)
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
        super().__init__(c)
        # Only need query vector since value vector is in parent
        self.query_vector = nn.Parameter(torch.randn(c.d_head) * c.weight_init_range)

    def _query(self) -> Float[torch.Tensor, " head"]:
        return self.query_vector

    def _keys(
        self, resids: Float[torch.Tensor, " batch seq model"]
    ) -> Float[torch.Tensor, " batch seq head"]:
        # Use residuals directly as keys since d_model == d_head
        assert self.c.d_head == self.c.d_model
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


def collate_fn_out_to_model_out(
    model: AbstractAttnProbeModel,
    collate_fn_output: CollateFnOutput,
) -> Float[torch.Tensor, "batch"]:
    cot_acts = collate_fn_output.cot_acts.to(model.device)
    attn_mask = collate_fn_output.attn_mask.to(model.device)
    return model(cot_acts, attn_mask)


def compute_loss_and_acc_single_batch(
    model: AbstractAttnProbeModel,
    criterion: nn.BCELoss,
    collate_fn_output: CollateFnOutput,
) -> tuple[Float[torch.Tensor, ""], float]:
    outputs = collate_fn_out_to_model_out(model, collate_fn_output)
    labels = collate_fn_output.labels.to(model.device)
    q_idxs = collate_fn_output.q_idxs

    def compute_class_metrics(
        target_label: int,
    ) -> tuple[Float[torch.Tensor, ""], float]:
        target_q_idxs = list(
            {q_idx for i, q_idx in enumerate(q_idxs) if labels[i] == target_label}
        )

        q_losses = []
        q_accs = []
        for q_idx in target_q_idxs:
            q_mask = [i == q_idx for i in q_idxs]
            q_outputs = outputs[q_mask]
            q_labels = labels[q_mask]
            assert torch.all(q_labels == target_label)
            q_loss = criterion(q_outputs, q_labels)
            q_losses.append(q_loss)
            q_acc = ((q_outputs > 0.5) == q_labels.bool()).float().mean().item()
            q_accs.append(q_acc)

        if not q_losses:
            return torch.tensor(0.0).to(model.device), 0.0
        return torch.stack(q_losses).mean(), sum(q_accs) / len(q_accs)

    pos_loss, pos_acc = compute_class_metrics(target_label=1)
    neg_loss, neg_acc = compute_class_metrics(target_label=0)

    balanced_loss = (pos_loss + neg_loss) / 2
    balanced_acc = (pos_acc + neg_acc) / 2

    return balanced_loss, balanced_acc


def compute_loss_and_acc(
    model: AbstractAttnProbeModel,
    criterion: nn.BCELoss,
    data_loader: DataLoader,
) -> tuple[float, float]:
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for collate_fn_output in data_loader:
            loss, acc = compute_loss_and_acc_single_batch(
                model, criterion, collate_fn_output
            )
            total_loss += loss.item()
            total_acc += acc

    avg_loss = total_loss / len(data_loader)
    avg_acc = total_acc / len(data_loader)
    return avg_loss, avg_acc


class AttnProbeTrainer:
    def __init__(
        self,
        *,
        c: ProbingConfig,
        raw_acts_dataset: dict,
        data_loading_kwargs: dict[str, Any],
        model_state_dict: dict[str, Any] | None = None,
    ):
        self.c = c
        self.model_device = torch.device(c.model_device)
        self.criterion = nn.BCELoss()

        # Initialize model
        self.model = self.c.probe_model_class(self.c.probe_model_config).to(
            self.model_device
        )
        if model_state_dict is not None:
            self.model.load_state_dict(model_state_dict)

        # Create data loaders
        (
            self.train_loader,
            self.val_loader,
            self.test_loader,
        ), (
            self.train_idxs,
            self.val_idxs,
            self.test_idxs,
        ) = preprocess_and_split_data(
            raw_acts_dataset,
            data_loading_kwargs,
        )

    @classmethod
    def from_wandb(
        cls,
        raw_acts_dataset: dict,
        entity: str = "cot-probing",
        project: str = "attn-probes",
        run_id: str | None = None,
        config_filters: dict[str, Any] | None = None,
    ) -> tuple["AttnProbeTrainer", Run, list[int]]:
        """Load a model from W&B.

        Args:
            entity: W&B entity name
            project: W&B project name
            run_id: Optional W&B run ID. Must specify either run_id or config_filters
            config_filters: Optional dict of config values to filter runs by. Must specify either run_id or config_filters
            raw_acts_dataset: Dataset to use
            data_loading_kwargs: Arguments for data loading

        Raises:
            ValueError: If neither run_id nor config_filters is specified, or if both are specified
            ValueError: If config_filters matches zero or multiple runs

        Returns:
            Trainer, run, and test indices
        """
        api = wandb.Api()

        if run_id is not None:
            assert (
                config_filters is None
            ), "Must specify exactly one of run_id or config_filters, specified both"
            run = api.run(f"{entity}/{project}/{run_id}")
        else:
            assert (
                config_filters is not None
            ), "Must specify exactly one of run_id or config_filters, specified none"
            filters = []
            for k, v in config_filters.items():
                filters.append({f"config.{k}": v})
            runs = list(api.runs(f"{entity}/{project}", {"$and": filters}))

            if len(runs) == 0:
                raise ValueError("No runs found matching config filters")
            if len(runs) > 1:
                raise ValueError(
                    f"Multiple runs ({len(runs)}) found matching config filters"
                )

            run = runs[0]
        assert run.state == "finished"

        # Get config from W&B
        w_config = run.config

        # Create probe model config
        probe_model_config = AttnProbeModelConfig(**w_config["probe_model_config"])

        # Map model class name to actual class
        model_class_map = {
            "minimal": MinimalAttnProbeModel,
            "medium": MediumAttnProbeModel,
            "full": FullAttnProbeModel,
        }
        probe_model_class = model_class_map[w_config["args_probe_class"]]

        # Create probing config
        config = ProbingConfig(
            probe_model_class=probe_model_class,
            probe_model_config=probe_model_config,
            data_device="cpu",
            **{
                k: w_config[k]
                for k in [
                    "data_seed",
                    "lr",
                    "batch_size",
                    "patience",
                    "n_epochs",
                    "validation_split",
                    "test_split",
                    "model_device",
                    "layer",
                ]
            },
        )

        # Construct data loading kwargs from config
        data_loading_kwargs = {
            "batch_size": config.batch_size,
            "validation_split": config.validation_split,
            "test_split": config.test_split,
            "data_seed": config.data_seed,
            "data_device": config.data_device,
            "layer": config.layer,
            "include_answer_toks": w_config["args_include_answer_toks"],
            "d_model": probe_model_config.d_model,
        }

        # Download and load model weights
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "best_model.pt"
            best_model_file = run.file("best_model.pt")
            best_model_file.download(root=tmp_dir, replace=True)
            model_state_dict = torch.load(
                tmp_path, map_location="cpu", weights_only=True
            )
        instance = cls(
            c=config,
            raw_acts_dataset=raw_acts_dataset,
            data_loading_kwargs=data_loading_kwargs,
            model_state_dict=model_state_dict,
        )
        return instance, run, instance.test_idxs

    def train(
        self,
        run_name: str,
        project_name: str,
        args: argparse.Namespace,
    ) -> AbstractAttnProbeModel:
        # Initialize W&B
        wandb.init(entity="cot-probing", project=project_name, name=run_name)
        wandb.config.update(asdict(self.c))
        wandb.config.update({f"args_{k}": v for k, v in vars(args).items()})
        wandb.config.update({"git_commit": get_git_commit_hash()})

        optimizer = Adam(self.model.parameters(), lr=self.c.lr)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = deepcopy(self.model.state_dict())

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            for epoch in range(self.c.n_epochs):
                try:
                    train_loss, train_acc, val_loss, val_acc = train_epoch(
                        model=self.model,
                        optimizer=optimizer,
                        criterion=self.criterion,
                        train_loader=self.train_loader,
                        val_loader=self.val_loader,
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
                        best_model_state = deepcopy(self.model.state_dict())
                        # Create subdirectory for this save
                        save_subdir = tmp_dir_path / f"save_{epoch}"
                        save_subdir.mkdir(exist_ok=True)
                        tmp_path = save_subdir / "best_model.pt"
                        # Save model state to the new subdirectory
                        safe_torch_save(best_model_state, tmp_path)
                        # Always upload to the same path in wandb
                        wandb.save(str(tmp_path), base_path=str(save_subdir))

                    else:
                        patience_counter += 1
                        if patience_counter >= self.c.patience:
                            print(f"Early stopping at epoch {epoch}")
                            break
                except KeyboardInterrupt:
                    break

            self.model.load_state_dict(best_model_state)
            test_loss, test_acc = self.compute_test_loss_acc()
            wandb.log({"test_loss": test_loss, "test_accuracy": test_acc})
            wandb.finish()

        return self.model

    def compute_validation_loss_acc(self) -> tuple[float, float]:
        """Compute loss and accuracy on the validation set.

        Returns:
            tuple[float, float]: Validation loss and accuracy
        """
        self.model.eval()
        return compute_loss_and_acc(self.model, self.criterion, self.val_loader)

    def compute_test_loss_acc(self) -> tuple[float, float]:
        """Compute loss and accuracy on the test set.

        Returns:
            tuple[float, float]: Test loss and accuracy
        """
        self.model.eval()
        return compute_loss_and_acc(self.model, self.criterion, self.test_loader)


def train_epoch(
    model: AbstractAttnProbeModel,
    optimizer: Adam,
    criterion: nn.BCELoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epoch: int,
) -> tuple[float, float, float, float]:
    model.train()
    train_loss = 0
    train_acc = 0
    for collate_fn_output in tqdm(train_loader, desc=f"Epoch {epoch}"):
        optimizer.zero_grad()
        loss, acc = compute_loss_and_acc_single_batch(
            model, criterion, collate_fn_output
        )
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += acc
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    model.eval()
    val_loss, val_acc = compute_loss_and_acc(model, criterion, val_loader)
    return train_loss, train_acc, val_loss, val_acc
