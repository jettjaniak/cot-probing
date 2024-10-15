# %%
import pickle
from pathlib import Path
import torch

from transformers import AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# %%

# Load activations
base_path = Path("/workspace/cot-probing-hf/google--gemma-2-2b")
task = "movie_recommendation/bias-A_seed-0_total-300"

biased_path = base_path / task / "biased_context"
unbiased_path = base_path / task / "unbiased_context"

biased_resp_acts_path = biased_path / "acts_resp_no-fsp"
unbiased_resp_acts_path = unbiased_path / "acts_resp_no-fsp"

layer = 0
biased_acts_layer_path = biased_resp_acts_path / f"L{layer:02}.pkl"
with open(biased_acts_layer_path, "rb") as f:
    biased_acts = pickle.load(f)

unbiased_acts_layer_path = unbiased_resp_acts_path / f"L{layer:02}.pkl"
with open(unbiased_acts_layer_path, "rb") as f:
    unbiased_acts = pickle.load(f)

biased_tokenized_resps_path = biased_path / "tokenized_responses.pkl"
with open(biased_tokenized_resps_path, "rb") as f:
    biased_tokenized_resps = pickle.load(f)

unbiased_tokenized_resps_path = unbiased_path / "tokenized_responses.pkl"
with open(unbiased_tokenized_resps_path, "rb") as f:
    unbiased_tokenized_resps = pickle.load(f)

# %%

to_these_ones = tokenizer.encode(" to these ones seems to be", add_special_tokens=False)

question_idx_with_different_movie = []
question_idx_with_same_movie = []

movie_token_idx = []

for q_idx, (biased_resp, unbiased_resp) in enumerate(
    zip(biased_tokenized_resps, unbiased_tokenized_resps)
):
    biased_resp_str = tokenizer.decode(biased_resp)
    unbiased_resp_str = tokenizer.decode(unbiased_resp)

    # Check that tokens in to_these_ones are in both biased and unbiased responses

    biased_to_these_ones_i = None
    for i in range(len(biased_resp) - len(to_these_ones) + 1):
        if biased_resp[i : i + len(to_these_ones)] == to_these_ones:
            biased_to_these_ones_i = i

    unbiased_to_these_ones_i = None
    for i in range(len(unbiased_resp) - len(to_these_ones) + 1):
        if unbiased_resp[i : i + len(to_these_ones)] == to_these_ones:
            unbiased_to_these_ones_i = i

    biased_movie_token_idx = biased_to_these_ones_i + len(to_these_ones) + 1
    unbiased_movie_token_idx = unbiased_to_these_ones_i + len(to_these_ones) + 1
    if biased_resp[biased_movie_token_idx] != unbiased_resp[unbiased_movie_token_idx]:
        question_idx_with_different_movie.append(q_idx)
        movie_token_idx.append((unbiased_movie_token_idx, biased_movie_token_idx))
        print(
            tokenizer.decode(
                biased_resp[
                    biased_to_these_ones_i : biased_to_these_ones_i
                    + len(to_these_ones)
                    + 2
                ]
            )
        )
        print(
            tokenizer.decode(
                unbiased_resp[
                    unbiased_to_these_ones_i : unbiased_to_these_ones_i
                    + len(to_these_ones)
                    + 2
                ]
            )
        )
        print()
    else:
        question_idx_with_same_movie.append(q_idx)

# %%
unbiased_acts_movie_token = []
biased_acts_movie_token = []
for i, q_idx in enumerate(question_idx_with_different_movie):
    unbiased_movie_token_idx, biased_movie_token_idx = movie_token_idx[i]

    unbiased_acts_movie_token.append(
        unbiased_acts[q_idx][unbiased_movie_token_idx]
    )  # Shape d_model
    biased_acts_movie_token.append(
        biased_acts[q_idx][biased_movie_token_idx]
    )  # Shape d_model

unbiased_acts_movie_token = torch.stack(unbiased_acts_movie_token)
biased_acts_movie_token = torch.stack(biased_acts_movie_token)

# %%
unbiased_acts_mean = unbiased_acts_movie_token.mean(0)  # Shape d_model
biased_acts_mean = biased_acts_movie_token.mean(0)  # Shape d_model

probe = biased_acts_mean - unbiased_acts_mean  # Shape d_model

# %%
