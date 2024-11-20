# %%

import pickle

with open("../activations/acts_L15_biased-fsp-oct28-1156-A.pkl", "rb") as f:
    acts_A = pickle.load(f)

with open("../activations/acts_L15_with-unbiased-cots-oct28-1156.pkl", "rb") as f:
    acts_B = pickle.load(f)

# %%

# Check they have same keys
print(set(acts_A.keys()) == set(acts_B.keys()))

# Show difference in keys
print(set(acts_A.keys()) - set(acts_B.keys()))
print(set(acts_B.keys()) - set(acts_A.keys()))

# check that all keys have same values
for k in acts_A.keys():
    if k == "qs" or k == "arg_context" or "arg_layers":
        continue
    assert acts_A[k] == acts_B[k]
# %%

# Check same number of questions
print(len(acts_A["qs"]) == len(acts_B["qs"]))
# %%

qs_keys = set(acts_A["qs"][0].keys())

for i in range(len(acts_A["qs"])):
    assert set(acts_A["qs"][i].keys()) == qs_keys
    assert set(acts_B["qs"][i].keys()) == qs_keys
# %%

# Check values are the same
for i in range(len(acts_A["qs"])):
    for k in qs_keys:
        if k == "cached_acts":
            continue
        assert (
            acts_A["qs"][i][k] == acts_B["qs"][i][k]
        ), f"Mismatch at {i} for key {k}. {acts_A['qs'][i][k]} vs {acts_B['qs'][i][k]}"
# %%

import torch

# Check that cached_acts are the same using torch.allclose
for i in range(len(acts_A["qs"])):
    # Check same number of COTs
    assert len(acts_A["qs"][i]["cached_acts"]) == len(acts_B["qs"][i]["cached_acts"])

    for cot in range(len(acts_A["qs"][i]["cached_acts"])):
        # Assert same shape
        assert (
            acts_A["qs"][i]["cached_acts"][cot].shape
            == acts_B["qs"][i]["cached_acts"][cot].shape
        ), f"Mismatch shape at {i} for cot {cot}"

        assert torch.allclose(
            acts_A["qs"][i]["cached_acts"][cot],
            acts_B["qs"][i]["cached_acts"][cot],
            atol=1e-8,
        ), f"Mismatch content at {i} for cot {cot}"
# %%
