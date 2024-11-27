# %%
import pickle
from typing import Literal

import torch

from cot_probing import DATA_DIR
from cot_probing.attn_probes import AbstractProbe, ProbeTrainer
from cot_probing.attn_probes_utils import fetch_runs

import wandb

# %%

LAYER = 15
PROBE_CLASS = "V"
MIN_SEED = 1
MAX_SEED = 10

raw_acts_path = (
    DATA_DIR / f"../../activations/acts_L{LAYER:02d}_biased-fsp_oct28-1156.pkl"
)
with open(raw_acts_path, "rb") as f:
    raw_acts_dataset = pickle.load(f)


def fetch_attn_probe_models(
    fsp_context: Literal["biased-fsp", "unbiased-fsp", "no-fsp"]
) -> list[AbstractProbe]:
    runs_by_seed_by_layer = fetch_runs(
        api=wandb.Api(),
        probe_class=PROBE_CLASS,
        context=fsp_context,
        min_layer=LAYER,
        max_layer=LAYER,
        min_seed=MIN_SEED,
        max_seed=MAX_SEED,
    )
    runs_by_seed = runs_by_seed_by_layer[LAYER]

    attn_probes = []
    for run in runs_by_seed.values():
        trainer, _, test_idxs = ProbeTrainer.from_wandb(
            raw_acts_dataset=raw_acts_dataset,
            run_id=run.id,
        )
        attn_probes.append(trainer.model)

    return attn_probes


# %%

biased_fsp_attn_probes = fetch_attn_probe_models("biased-fsp")
no_fsp_attn_probes = fetch_attn_probe_models("no-fsp")


# %%

averaged_biased_fsp_probe_direction = torch.stack(
    [p.value_vector.cpu() for p in biased_fsp_attn_probes]
).mean(dim=0)

averaged_no_fsp_probe_direction = torch.stack(
    [p.value_vector.cpu() for p in no_fsp_attn_probes]
).mean(dim=0)

# %%

# Calculate cosine similarity between the averaged directions
cos_similarity = torch.nn.functional.cosine_similarity(
    averaged_biased_fsp_probe_direction.unsqueeze(0),
    averaged_no_fsp_probe_direction.unsqueeze(0),
)[0]

print(f"Cosine similarity between averaged probe directions: {cos_similarity:.4f}")

# Optionally, also calculate the angle in degrees
angle_rad = torch.acos(cos_similarity)
angle_deg = torch.rad2deg(angle_rad)
print(f"Angle between directions: {angle_deg:.2f}°")
