#!/usr/bin/env python3
import pickle
import tempfile
from pathlib import Path

import pytest
import torch

from cot_probing.attn_probes import DataConfig, ProbeConfig, ProbeTrainer, TrainerConfig
from cot_probing.typing import *


@pytest.fixture
def mock_data():
    # Create mock activation data
    n_batches = 5
    seq_len = 5
    d_model = 8
    n_cots = 2

    mock_acts = []
    for _ in range(n_batches):
        # Create mock activations for 2 questions, each with 2 CoTs
        for label in ["faithful", "unfaithful"]:
            cots = []
            for _ in range(n_cots):
                # Create random activations tensor
                acts = torch.randn(seq_len, d_model)
                cots.append(acts)
            mock_acts.append({"cached_acts": cots, "biased_cot_label": label})

    return mock_acts


@pytest.fixture
def mock_configs():
    probe_config = ProbeConfig(
        d_model=8,
        weight_init_range=0.02,
        weight_init_seed=0,
        partial_seq=False,
    )

    data_config = DataConfig(
        dataset_id="test",
        layer=0,
        context="biased-fsp",
        cv_seed=0,
        cv_n_folds=2,
        cv_test_fold=0,
        train_val_seed=0,
        val_frac=0.5,
        data_device="cpu",
        batch_size=2,
    )

    trainer_config = TrainerConfig(
        probe_class="tied",
        probe_config=probe_config,
        data_config=data_config,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        patience=2,
        max_epochs=2,
        model_device="cpu",
        experiment_uuid="test",
    )

    return trainer_config


def test_train_and_load_probe(mock_data, mock_configs):
    # Create temporary directory for activations
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        acts_path = tmp_path / mock_configs.data_config.get_acts_filename()

        # Save mock activations
        with open(acts_path, "wb") as f:
            pickle.dump({"qs": mock_data}, f)

        # Train probe
        trainer = ProbeTrainer(
            c=mock_configs,
            raw_q_dicts=mock_data,
        )

        # Train probe and get wandb run
        model, run = trainer.train(project_name="tests")
        run_id = run.id

        # Test that model parameters exist and have correct shapes
        assert hasattr(model, "value_vector")
        assert model.value_vector.shape == (mock_configs.probe_config.d_model,)

        if mock_configs.probe_class == "untied":
            assert hasattr(model, "query_vector")
            assert model.query_vector.shape == (mock_configs.probe_config.d_model,)
        else:
            assert hasattr(model, "query_scale")
            assert model.query_scale.shape == (1,)

        # Test forward pass
        batch_size = 2
        seq_len = 5
        test_input = torch.randn(batch_size, seq_len, mock_configs.probe_config.d_model)
        test_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

        output = model(test_input, test_mask)
        assert output.shape == (batch_size,)
        assert torch.all((output >= 0) & (output <= 1))

        # Load model from wandb and verify it works the same
        loaded_trainer, _ = ProbeTrainer.from_wandb(
            activations_dir=tmp_path,
            project="tests",
            run_id=run_id,
        )
        loaded_model = loaded_trainer.model

        # Test that loaded model produces same outputs
        loaded_output = loaded_model(test_input, test_mask)
        assert torch.allclose(output, loaded_output)

        # Clean up wandb run
        run.finish()
