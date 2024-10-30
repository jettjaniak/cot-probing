# %%
%load_ext autoreload
%autoreload 2

# %%
from cot_probing.swapping import (
    SuccessfulSwap,
    PatchedLogitsProbs,
)
from cot_probing import DATA_DIR
from cot_probing.typing import *
from transformers import AutoTokenizer
import pickle
from tqdm.auto import tqdm, trange

swaps_path = DATA_DIR / f"swaps_with-unbiased-cots-oct28-1156.pkl"
with open(swaps_path, "rb") as f:
    swaps_dict = pickle.load(f)
swaps_dicts_list = swaps_dict["qs"]
swaps_by_q = [swap_dict["swaps"] for swap_dict in swaps_dicts_list]

patch_res_path = (
    DATA_DIR / f"patch_new_res_8B_LB33__swaps_with-unbiased-cots-oct28-1156.pkl"
)
with open(patch_res_path, "rb") as f:
    patch_results_by_swap_by_q = pickle.load(f)

model_id = "hugging-quants/Meta-Llama-3.1-8B-BNB-NF4-BF16"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# %%
for i, (swaps, patch_results_by_swap) in enumerate(
    zip(swaps_by_q, patch_results_by_swap_by_q)
):
    print(f"q_idx: {i}: {len(swaps)} swaps")

# %%
import matplotlib.pyplot as plt


def plot_heatmap(values, title, labels, fai_tok_str, unfai_tok_str):
    plt.imshow(
        values,
        cmap="RdBu",
        origin="lower",
        vmin=-max(abs(np.min(values)), abs(np.max(values))),
        vmax=max(abs(np.min(values)), abs(np.max(values))),
    )
    plt.title(f"{title} for `{fai_tok_str}` -> `{unfai_tok_str}`")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.show()

# %%
def get_patch_values(
    plp_by_group_by_layers: dict[tuple[int, ...], dict[str, PatchedLogitsProbs]],
    prob_or_logit: Literal["prob", "logit"],
    direction: Literal["bia_to_unb", "unb_to_bia"],
) -> np.ndarray:
    attr = f"{prob_or_logit}_diff_change_{direction}"
    values = []
    for layers, plp_by_group in plp_by_group_by_layers.items():
        values.append([getattr(plp, attr) for plp in plp_by_group.values()])
    return np.array(values)

# %%
from collections import Counter

LOGIT_OR_PROB = "prob"
top_pos_cnt = Counter()

values_bia_to_unb_by_q_by_swap = {}
values_unb_to_bia_by_q_by_swap = {}

for q_idx, (swaps, patch_results_by_swap) in enumerate(
    zip(swaps_by_q, patch_results_by_swap_by_q)
):
    print(f"{q_idx=}")
    print()
    swaps: list[SuccessfulSwap]
    patch_results_by_swap: list[
        dict[tuple[int, ...], dict[str, PatchedLogitsProbs]] | None
    ]
    for swap_idx, (swap, fpr_by_layers) in enumerate(zip(swaps, patch_results_by_swap)):
        if fpr_by_layers is None:
            continue
        print(f"{swap_idx=}")
        print(f"{swap.prob_diff:.2%}")
        unfai_tok_str = tokenizer.decode(swap.unfai_tok).replace("\n", "\\n")
        fai_tok_str = tokenizer.decode(swap.fai_tok).replace("\n", "\\n")
        print(f"`{fai_tok_str}` -> `{unfai_tok_str}`")
        print()

        values_bia_to_unb = get_patch_values(fpr_by_layers, LOGIT_OR_PROB, "bia_to_unb")
        values_unb_to_bia = get_patch_values(fpr_by_layers, LOGIT_OR_PROB, "unb_to_bia")
        values_bia_to_unb = values_bia_to_unb / np.abs(values_bia_to_unb[0]).max()
        values_unb_to_bia = values_unb_to_bia / np.abs(values_unb_to_bia[0]).max()

        values_bia_to_unb_by_q_by_swap[(q_idx, swap_idx)] = values_bia_to_unb
        values_unb_to_bia_by_q_by_swap[(q_idx, swap_idx)] = values_unb_to_bia

        # mean_abs_patch_values_per_tok = 0.5 * (
        #     np.abs(values_bia_to_unb[:, 1:]).mean(0)
        #     + np.abs(values_unb_to_bia[:, 1:]).mean(0)
        # )
        # top_seq_pos = mean_abs_patch_values_per_tok.argsort()[-3:]
        # print(f"{top_seq_pos=}")
        q_tok = tokenizer.encode("Question", add_special_tokens=False)[0]
        last_q_idx = len(swap.unb_prompt) - 1 - swap.unb_prompt[::-1].index(q_tok)
        last_q_str = tokenizer.decode(swap.unb_prompt[last_q_idx + 2 :])
        trunc_cot_str = tokenizer.decode(swap.trunc_cot)
        print(f"`{last_q_str+trunc_cot_str}`")
        # for i, toks in enumerate(toks_in_unb_prompt):
        #     print(f"{i}:\n`{tokenizer.decode(toks)}`")
        groups = list(next(iter(fpr_by_layers.values())).keys())
        # plot_heatmap(
        #     values_bia_to_unb,
        #     "bia_to_unb",
        #     groups,
        #     fai_tok_str,
        #     unfai_tok_str,
        # )
        # plot_heatmap(
        #     values_unb_to_bia,
        #     "unb_to_bia",
        #     groups,
        #     fai_tok_str,
        #     unfai_tok_str,
        # )

# %%
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Prepare data for clustering
X = []  # Will hold flattened tensors
indices = []  # Will hold (q_idx, swap_idx) pairs

for (q_idx, swap_idx), tensor in values_bia_to_unb_by_q_by_swap.items():
    # Flatten the 2D tensor into a 1D array
    flattened = tensor.flatten()
    X.append(flattened)
    indices.append((q_idx, swap_idx))

X = np.array(X)
indices = np.array(indices)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform k-means clustering
n_clusters = 10  # You can adjust this number
kmeans = KMeans(n_clusters=n_clusters, random_state=45)
cluster_labels = kmeans.fit_predict(X_scaled)

# Create a dictionary mapping cluster labels to their corresponding indices
clusters = {i: [] for i in range(n_clusters)}
for idx, label in enumerate(cluster_labels):
    q_idx, swap_idx = indices[idx]
    clusters[label].append((q_idx, swap_idx))

# Sort clusters by size
clusters = dict(sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True))

# %%

# Print cluster information
for cluster_idx, members in clusters.items():
    print(f"\nCluster {cluster_idx}: {len(members)} members")
    # for q_idx, swap_idx in members[:5]:  # Show first 5 examples from each cluster
    #     print(f"  Q{q_idx} Swap{swap_idx}")
    # if len(members) > 5:
    #     print("  ...")

# %% Compute means of each cluster

centers = []
for cluster_idx, members in clusters.items():
    center = np.mean(
        [values_bia_to_unb_by_q_by_swap[(q_idx, swap_idx)] for q_idx, swap_idx in members],
        axis=0,
    )
    centers.append(center)

# %%

groups = ['question_colon', 'toks_of_question', 'qmark_newline', 'ltsbs_newline_dash', 'reasoning', 'last_three']

def get_group_idx(group_name: str) -> int:
    return groups.index(group_name)

# Assign names to each cluster
both_important_idxs = []
ltsbs_important_idxs = []
reasoning_important_idxs = []
only_last_three_important_idxs = []
other_idxs = []

almost_zero_threshold = 0.25

for (q_idx, swap_idx), values_bia_to_unb in values_bia_to_unb_by_q_by_swap.items():
    assert values_bia_to_unb.shape == (1, len(groups))
    v = np.abs(values_bia_to_unb[0])

    max_first_three_groups = max(v[:3])
    if max_first_three_groups >= almost_zero_threshold:
        other_idxs.append((q_idx, swap_idx))
        continue

    ltsbs_score = v[3]
    reasoning_score = v[4]
    last_three_score = v[5]
    if max(ltsbs_score, reasoning_score) < last_three_score/4:
        only_last_three_important_idxs.append((q_idx, swap_idx))
        continue

    if ltsbs_score > reasoning_score * 2:
        ltsbs_important_idxs.append((q_idx, swap_idx))
        continue

    if reasoning_score > ltsbs_score * 2:
        reasoning_important_idxs.append((q_idx, swap_idx))
        continue

    bigger = max(ltsbs_score, reasoning_score)
    smaller = min(ltsbs_score, reasoning_score)

    if bigger / smaller < 1.5:
        both_important_idxs.append((q_idx, swap_idx))
        continue

    other_idxs.append((q_idx, swap_idx))

# print size of each type
print(f"both_important_idxs: {len(both_important_idxs)}")
print(f"ltsbs_important_idxs: {len(ltsbs_important_idxs)}")
print(f"reasoning_important_idxs: {len(reasoning_important_idxs)}")
print(f"only_last_three_important_idxs: {len(only_last_three_important_idxs)}")
print(f"other_idxs: {len(other_idxs)}")

# %% 

for type in [both_important_idxs, ltsbs_important_idxs, reasoning_important_idxs, only_last_three_important_idxs, other_idxs]:
    item_idx = random.randint(0, len(type) - 1)
    q_idx, swap_idx = type[item_idx]
    values = values_bia_to_unb_by_q_by_swap[(q_idx, swap_idx)]
    type_str = "both_important" if type == both_important_idxs else "ltsbs_important" if type == ltsbs_important_idxs else "reasoning_important" if type == reasoning_important_idxs else "only_last_three_important" if type == only_last_three_important_idxs else "other"
    plot_heatmap(
        values,
        f"{type_str}",
        groups,
        "",
        "",
)


# %%



# Visualize cluster centers with group names
for cluster_idx, members in clusters.items():
    plot_heatmap(
        centers[cluster_idx],
        f'Cluster {cluster_idx} Center',
        groups,
        "",
        "",
    )

# %%

# Take mean of examples in each cluster
# for cluster_idx, members in clusters.items():
#     mean_values = np.mean(
#         [values_bia_to_unb_by_q_by_swap[(q_idx, swap_idx)] for q_idx, swap_idx in members],
#         axis=0,
#     )
#     print(f"Cluster {cluster_idx} mean values: {mean_values}")

#     center = kmeans.cluster_centers_[cluster_idx]
#     print(f"Cluster {cluster_idx} center: {center}")

# %%

def show_cluster_examples(cluster_idx: int, n_examples: int = 1):
    members = clusters[cluster_idx]

    random.shuffle(members)
    
    # Take up to n_examples from this cluster
    for q_idx, swap_idx in members[:n_examples]:
        # Get the original swap and its values
        swap = swaps_by_q[q_idx][swap_idx]
        values = values_bia_to_unb_by_q_by_swap[(q_idx, swap_idx)]
        
        # Get token strings for the title
        unfai_tok_str = tokenizer.decode(swap.unfai_tok).replace("\n", "\\n")
        fai_tok_str = tokenizer.decode(swap.fai_tok).replace("\n", "\\n")
        
        # Extract the question text
        q_tok = tokenizer.encode("Question", add_special_tokens=False)[0]
        last_q_idx = len(swap.unb_prompt) - 1 - swap.unb_prompt[::-1].index(q_tok)
        question_text = tokenizer.decode(swap.unb_prompt[last_q_idx + 2:])
        trunc_cot_text = tokenizer.decode(swap.trunc_cot)
        
        # Print swap information
        print(f"\nQ{q_idx} Swap{swap_idx}")
        print(f"Q and CoT: {question_text+trunc_cot_text}")
        print(f"Prob diff: {swap.prob_diff:.2%}")
        print(f"`{fai_tok_str}` -> `{unfai_tok_str}`")
        
        # Plot the heatmap
        groups = list(next(iter(patch_results_by_swap_by_q[q_idx][swap_idx].values())).keys())
        plot_heatmap(
            values,
            f"Cluster {cluster_idx}",
            groups,
            fai_tok_str,
            unfai_tok_str,
        )

# %%
# Plot examples from each cluster
n_examples = 1  # Number of examples to show from each cluster

for cluster_idx, members in clusters.items():
    print(f"\n=== Cluster {cluster_idx} Examples ===")

    show_cluster_examples(cluster_idx, n_examples)

# %%

show_cluster_examples(3, 20)