#!/usr/bin/env python3
import pickle
import tempfile
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import sklearn.metrics
import torch
import wandb
from dacite import from_dict
from fancy_einsum import einsum
from torch import nn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from wandb.apis.public.runs import Run
from wandb.sdk.wandb_run import Run as WandbSdkRun

from cot_probing.attn_probes_data_proc import (
    CollateFnOutput,
    DataConfig,
    load_and_split_data,
)
from cot_probing.typing import *
from cot_probing.utils import get_git_commit_hash, safe_torch_save, setup_determinism

torch.set_grad_enabled(True)


@dataclass(kw_only=True)
class ProbeConfig:
    d_model: int
    weight_init_range: float
    weight_init_seed: int
    partial_seq: bool


@dataclass
class TrainerConfig:
    probe_class: Literal["tied", "untied"]
    probe_config: ProbeConfig
    data_config: DataConfig
    lr: float
    beta1: float
    beta2: float
    patience: int
    max_epochs: int
    model_device: str
    experiment_uuid: str

    def get_run_name(self) -> str:
        dc = self.data_config
        assert dc.train_val_seed == self.probe_config.weight_init_seed
        train_seed = dc.train_val_seed
        args = [
            f"L{dc.layer:02d}",
            self.probe_class,
            dc.context,
            f"cvs{dc.cv_seed}",
            f"ts{train_seed}",
            self.experiment_uuid,
            f"f{dc.cv_test_fold}",
        ]
        return "_".join(args)


@dataclass
class EvalMetrics:
    loss: float
    acc: float
    auc: float


class AbstractProbe(nn.Module, ABC):
    def __init__(self, c: ProbeConfig):
        super().__init__()
        self.c = c
        setup_determinism(c.weight_init_seed)
        self.z_bias = nn.Parameter(torch.zeros(1))
        self.value_vector = nn.Parameter(torch.randn(c.d_model) * c.weight_init_range)

    @property
    def device(self) -> torch.device:
        return self.value_vector.device

    @abstractmethod
    def query(self) -> Float[torch.Tensor, " model"]:
        pass

    def values(
        self, resids: Float[torch.Tensor, " batch seq model"]
    ) -> Float[torch.Tensor, " batch seq"]:
        # Project residuals using the value vector
        return einsum("batch seq model, model -> batch seq", resids, self.value_vector)

    def attn_probs(
        self,
        resids: Float[torch.Tensor, " batch seq model"],
        attn_mask: Bool[torch.Tensor, " batch seq"],
    ) -> Float[torch.Tensor, " batch seq"]:
        # Compute attention scores (before softmax)
        # shape (model,)
        query_not_expanded = self.query()
        # shape (batch, model)
        query = query_not_expanded.expand(resids.shape[0], -1)
        # shape (batch, seq)
        attn_scores = einsum("batch seq model, batch model -> batch seq", resids, query)
        attn_scores = attn_scores.masked_fill(
            ~attn_mask, torch.finfo(attn_scores.dtype).min
        )
        # shape (batch, seq)
        return torch.softmax(attn_scores, dim=-1)

    def z(
        self,
        attn_probs: Float[torch.Tensor, " batch seq"],
        values: Float[torch.Tensor, " batch seq"],
    ) -> Float[torch.Tensor, " batch"]:
        return einsum("batch seq, batch seq -> batch", attn_probs, values)

    def get_pred_scores(
        self,
        resids: Float[torch.Tensor, " batch seq model"],
        attn_mask: Bool[torch.Tensor, " batch seq"],
    ) -> Float[torch.Tensor, " batch"]:
        attn_probs = self.attn_probs(resids, attn_mask)
        # shape (batch, seq)
        # unlike normal attention, these are 1-dimensional
        values = self.values(resids)
        z = self.z(attn_probs, values)
        return z

    def forward(
        self,
        resids: Float[torch.Tensor, " batch seq model"],
        attn_mask: Bool[torch.Tensor, " batch seq"],
    ) -> Float[torch.Tensor, " batch"]:
        z = self.get_pred_scores(resids, attn_mask)
        return torch.sigmoid(z + self.z_bias)


class TiedProbe(AbstractProbe):
    def __init__(self, c: ProbeConfig):
        super().__init__(c)
        # Use value_vector as probe_vector
        self.query_scale = nn.Parameter(torch.ones(1))

    def query(self) -> Float[torch.Tensor, " model"]:
        # Scale the value vector by temperature for query
        return self.value_vector * self.query_scale


class UntiedProbe(AbstractProbe):
    def __init__(self, c: ProbeConfig):
        super().__init__(c)
        # Only need query vector since value vector is in parent
        self.query_vector = nn.Parameter(torch.randn(c.d_model) * c.weight_init_range)

    def query(self) -> Float[torch.Tensor, " model"]:
        return self.query_vector


def get_probe_class(probe_class_name: str) -> type[AbstractProbe]:
    return {
        "tied": TiedProbe,
        "untied": UntiedProbe,
    }[probe_class_name]


def collate_fn_out_to_model_out(
    model: AbstractProbe,
    collate_fn_output: CollateFnOutput,
) -> Float[torch.Tensor, " batch"]:
    cot_acts = collate_fn_output.cot_acts.to(model.device)
    attn_mask = collate_fn_output.attn_mask.to(model.device)
    return model(cot_acts, attn_mask)


def compute_loss_and_acc_single_batch(
    model: AbstractProbe,
    criterion: nn.BCELoss,
    collate_fn_output: CollateFnOutput,
) -> tuple[Float[torch.Tensor, ""], float]:
    outputs = collate_fn_out_to_model_out(model, collate_fn_output)
    labels = collate_fn_output.labels.to(model.device)
    q_idxs = collate_fn_output.q_idxs

    def compute_class_loss_acc(
        target_label: int,
    ) -> tuple[Float[torch.Tensor, ""], float] | None:
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
            return None
        return torch.stack(q_losses).mean(), sum(q_accs) / len(q_accs)

    pos_loss_acc = compute_class_loss_acc(target_label=1)
    neg_loss_acc = compute_class_loss_acc(target_label=0)

    if pos_loss_acc is None:
        assert neg_loss_acc is not None
        balanced_loss, balanced_acc = neg_loss_acc
    elif neg_loss_acc is None:
        assert pos_loss_acc is not None
        balanced_loss, balanced_acc = pos_loss_acc
    else:
        pos_loss, pos_acc = pos_loss_acc
        neg_loss, neg_acc = neg_loss_acc
        balanced_loss = (pos_loss + neg_loss) / 2
        balanced_acc = (pos_acc + neg_acc) / 2

    return balanced_loss, balanced_acc


def compute_eval_metrics(
    model: AbstractProbe,
    criterion: nn.BCELoss,
    data_loader: DataLoader,
) -> EvalMetrics:
    assert (
        len(data_loader) == 1
    ), "This function should only run on validation or test set, with a single batch"
    collate_fn_output = next(iter(data_loader))
    with torch.no_grad():
        loss, acc = compute_loss_and_acc_single_batch(
            model, criterion, collate_fn_output
        )
        outputs = collate_fn_out_to_model_out(model, collate_fn_output).cpu()
        labels = collate_fn_output.labels.cpu()
        roc_auc = float(sklearn.metrics.roc_auc_score(labels, outputs))

    return EvalMetrics(loss=loss.item(), acc=acc, auc=roc_auc)


class ProbeTrainer:
    def __init__(
        self,
        *,
        c: TrainerConfig,
        raw_q_dicts: list[dict],
        model_state_dict: dict[str, Any] | None = None,
    ):
        self.c = c
        self.model_device = torch.device(c.model_device)
        self.criterion = nn.BCELoss()

        # Initialize model
        probe_model_class = get_probe_class(self.c.probe_class)
        self.model = probe_model_class(self.c.probe_config).to(self.model_device)
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
        ) = load_and_split_data(
            raw_q_dicts,
            self.c.data_config,
        )

    @classmethod
    def from_wandb(
        cls,
        activations_dir: Path,
        entity: str = "cot-probing",
        project: str = "attn-probes",
        run_id: str | None = None,
        config_filters: dict[str, Any] | None = None,
    ) -> tuple["ProbeTrainer", Run]:
        """Load a model from W&B.

        Args:
            activations_dir: Directory containing activations
            entity: W&B entity name
            project: W&B project name
            run_id: Optional W&B run ID. Must specify either run_id or config_filters
            config_filters: Optional dict of config values to filter runs by. Must specify either run_id or config_filters

        Raises:
            ValueError: If neither run_id nor config_filters is specified, or if both are specified
            ValueError: If config_filters matches zero or multiple runs

        Returns:
            ProbeTrainer and W&B run
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

        del run.config["git_commit"]
        trainer_config = from_dict(TrainerConfig, run.config)

        # Download and load model weights
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir) / "best_model.pt"
            best_model_file = run.file("best_model.pt")
            best_model_file.download(root=tmp_dir, replace=True)
            model_state_dict = torch.load(
                tmp_path, map_location="cpu", weights_only=True
            )

        acts_filename = trainer_config.data_config.get_acts_filename()
        with open(activations_dir / acts_filename, "rb") as f:
            raw_q_dicts = pickle.load(f)["qs"]
        trainer = cls(
            c=trainer_config,
            raw_q_dicts=raw_q_dicts,
            model_state_dict=model_state_dict,
        )
        return trainer, run

    def train(
        self,
        project_name: str,
    ) -> tuple[AbstractProbe, WandbSdkRun]:
        # Initialize W&B
        run = wandb.init(
            entity="cot-probing",
            project=project_name,
            name=self.c.get_run_name(),
        )
        wandb.config.update(asdict(self.c))
        wandb.config.update({"git_commit": get_git_commit_hash()})

        optimizer = Adam(
            self.model.parameters(), lr=self.c.lr, betas=(self.c.beta1, self.c.beta2)
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = deepcopy(self.model.state_dict())

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            for epoch in range(self.c.max_epochs):
                try:
                    train_loss, train_acc, val_metrics = train_epoch(
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
                            "val_loss": val_metrics.loss,
                            "val_accuracy": val_metrics.acc,
                            "val_auc": val_metrics.auc,
                            "epoch": epoch,
                        }
                    )

                    # Early stopping
                    if val_metrics.loss < best_val_loss:
                        best_val_loss = val_metrics.loss
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
            test_metrics = self.compute_test_metrics()
            wandb.log(
                {
                    "test_loss": test_metrics.loss,
                    "test_accuracy": test_metrics.acc,
                    "test_auc": test_metrics.auc,
                }
            )
            wandb.finish()

        return self.model, run

    def compute_test_metrics(self) -> EvalMetrics:
        """Compute loss, accuracy, and AUC on the test set.

        Returns:
            EvalMetrics: Test loss, accuracy, and AUC
        """
        self.model.eval()
        return compute_eval_metrics(self.model, self.criterion, self.test_loader)


def train_epoch(
    model: AbstractProbe,
    optimizer: Adam,
    criterion: nn.BCELoss,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epoch: int,
) -> tuple[float, float, EvalMetrics]:
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
    val_metrics = compute_eval_metrics(model, criterion, val_loader)
    return train_loss, train_acc, val_metrics
