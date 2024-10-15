# %%
import pickle
from pathlib import Path

import torch
from transformers import AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

# %%

# Load activations
base_path = Path("../hf_results/google--gemma-2-2b")
task = "movie_recommendation/bias-A_seed-0_total-300"

task_path = base_path / task

biased_path = task_path / "biased_context"
unbiased_path = task_path / "unbiased_context"
tokenized_ques_path = task_path / "tokenized_questions.pkl"
with open(tokenized_ques_path, "rb") as f:
    tokenized_ques = pickle.load(f)

biased_resp_acts_path = biased_path / "acts_resp_no-fsp"
unbiased_resp_acts_path = unbiased_path / "acts_resp_no-fsp"

layer = 23
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
open_paren = tokenizer.encode(" (", add_special_tokens=False)[0]

q_idx_diff_movie = []
# q_idx -> (unbiased, biased)
last_movie_token_idx = {}


def get_movie_tokens_range(resp):
    # find " to these ones seems to be"
    to_these_ones_i = None
    for i in range(len(resp) - len(to_these_ones) + 1):
        if resp[i : i + len(to_these_ones)] == to_these_ones:
            to_these_ones_i = i
            break
    assert to_these_ones_i is not None
    first_movie_i = to_these_ones_i + len(to_these_ones)
    # find " ("
    open_paren_i = None
    for i in range(first_movie_i, len(resp)):
        if resp[i] == open_paren:
            open_paren_i = i
            break
    assert open_paren_i is not None
    return first_movie_i, open_paren_i


for q_idx, (biased_resp, unbiased_resp) in enumerate(
    zip(biased_tokenized_resps, unbiased_tokenized_resps)
):
    b_first_movie_i, b_open_paren_i = get_movie_tokens_range(biased_resp)
    u_first_movie_i, u_open_paren_i = get_movie_tokens_range(unbiased_resp)
    b_movie_tokens = biased_resp[b_first_movie_i:b_open_paren_i]
    u_movie_tokens = unbiased_resp[u_first_movie_i:u_open_paren_i]
    # print("biased movie tokens: ", tokenizer.decode(b_movie_tokens))
    # print("unbiased movie tokens: ", tokenizer.decode(u_movie_tokens))
    if b_movie_tokens == u_movie_tokens:
        continue
    q_idx_diff_movie.append(q_idx)
    last_movie_token_idx[q_idx] = (u_open_paren_i - 1, b_open_paren_i - 1)
print(f"{len(q_idx_diff_movie)}")

# %%


def get_probe(q_idxs: list[int]):
    u_acts_movie_token = []
    b_acts_movie_token = []
    for q_idx in q_idxs:
        u_movie_token_idx, b_movie_token_idx = last_movie_token_idx[q_idx]

        u_acts_movie_token.append(unbiased_acts[q_idx][u_movie_token_idx])
        b_acts_movie_token.append(biased_acts[q_idx][b_movie_token_idx])

    u_mean = torch.stack(u_acts_movie_token).mean(0)
    b_mean = torch.stack(b_acts_movie_token).mean(0)
    return b_mean - u_mean


import random

train_q_idxs = random.sample(q_idx_diff_movie, 55)
test_q_idxs = [q_idx for q_idx in q_idx_diff_movie if q_idx not in train_q_idxs]
probe = get_probe(train_q_idxs)

# %%
from fancy_einsum import einsum


def get_probe_acts(
    q_idxs: list[int], probe: torch.Tensor
) -> dict[int, tuple[float, float]]:
    """returns act on (unbiased, biased)"""
    acts = {}
    for q_idx in q_idxs:
        u_tok_i, b_tok_i = last_movie_token_idx[q_idx]
        acts[q_idx] = (
            (unbiased_acts[q_idx][u_tok_i] @ probe).item(),
            (biased_acts[q_idx][b_tok_i] @ probe).item(),
        )
    return acts


train_probe_acts = get_probe_acts(train_q_idxs, probe)
test_probe_acts = get_probe_acts(test_q_idxs, probe)

# %%
for q_idx, (u_act, b_act) in test_probe_acts.items():
    print(f"{q_idx}: {u_act:.3f} {b_act:.3f}")

# %%
biased_probe_acts = {}
unbiased_probe_acts = {}
for q_idx in q_idx_diff_movie:
    biased_probe_acts[q_idx] = einsum(
        "pos model, model -> pos", biased_acts[q_idx], probe
    )
    unbiased_probe_acts[q_idx] = einsum(
        "pos model, model -> pos", unbiased_acts[q_idx], probe
    )

# %%
this_i = 4
this_q_idx = test_q_idxs[this_i]

this_unbiased_resp = unbiased_tokenized_resps[this_q_idx]
this_biased_resp = biased_tokenized_resps[this_q_idx]

this_unbiased_probe_acts = unbiased_probe_acts[this_q_idx]
this_biased_probe_acts = biased_probe_acts[this_q_idx]

vmin = min(this_unbiased_probe_acts.min(), this_biased_probe_acts.min()).item()
vmax = max(this_unbiased_probe_acts.max(), this_biased_probe_acts.max()).item()

from IPython.display import HTML

from cot_probing.vis import visualize_tokens_html

display(
    HTML(
        visualize_tokens_html(tokenized_ques[this_q_idx].tokenized_question, tokenizer)
    )
)

display(
    HTML(
        visualize_tokens_html(
            this_unbiased_resp, tokenizer, this_unbiased_probe_acts.tolist(), vmin, vmax
        )
    )
)

display(
    HTML(
        visualize_tokens_html(
            this_biased_resp, tokenizer, this_biased_probe_acts.tolist(), vmin, vmax
        )
    )
)
# %%
