#!/usr/bin/env python3
import pickle

import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm

from cot_probing import DATA_DIR
from cot_probing.activations import (
    collect_resid_acts_no_pastkv,
    collect_resid_acts_with_pastkv,
)
from cot_probing.attn_probes import AbstractAttnProbeModel, AttnProbeTrainer
from cot_probing.attn_probes_data_proc import CollateFnOutput
from cot_probing.patching import PatchedLogitsProbs
from cot_probing.typing import *
from cot_probing.utils import fetch_runs

TOK_GROUPS = ["Question:", "[question]", "?\\n", "LTSBS:\\n-", "reasoning", "last 3"]

LOGIT_OR_PROB = "prob"
DIR = "bia_to_unb"
CATEGORIES_FILE = f"categories_{LOGIT_OR_PROB}_{DIR}_0.25_1.5_2.0_4.0.pkl"
SWAPS_FILE = f"swaps_with-unbiased-cots-oct28-1156.pkl"
LB_LAYERS = 1
PATCH_LAYERS_FILE = (
    f"patch_new_res_8B_LB{LB_LAYERS}__swaps_with-unbiased-cots-oct28-1156.pkl"
)
PATCH_ALL_FILE = "patch_new_res_8B_LB33__swaps_with-unbiased-cots-oct28-1156.pkl"


plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
    }
)


def load_median_probe_test_data(
    probe_class: str, layer: int, min_seed: int, max_seed: int, metric: str
) -> tuple[AttnProbeTrainer, list[int], list[dict], str]:
    runs_by_seed_by_layer = fetch_runs(
        api=wandb.Api(),
        probe_class=probe_class,
        min_layer=layer,
        max_layer=layer,
        min_seed=min_seed,
        max_seed=max_seed,
    )
    assert len(runs_by_seed_by_layer) == 1
    runs_by_seed = runs_by_seed_by_layer[layer]
    seed_run_sorted = sorted(
        runs_by_seed.items(), key=lambda s_r: s_r[1].summary.get(metric)
    )

    _median_seed, median_run = seed_run_sorted[len(seed_run_sorted) // 2]
    # median_acc = median_run.summary.get(metric)
    raw_acts_path = (
        DATA_DIR / f"../../activations/acts_L{layer:02d}_biased-fsp_oct28-1156.pkl"
    )
    with open(raw_acts_path, "rb") as f:
        raw_acts_dataset = pickle.load(f)
    trainer, _, test_idxs = AttnProbeTrainer.from_wandb(
        raw_acts_dataset=raw_acts_dataset,
        run_id=median_run.id,
    )
    unbiased_fsp_str = raw_acts_dataset["unbiased_fsp"]
    raw_acts_qs = [raw_acts_dataset["qs"][i] for i in test_idxs]
    return trainer, test_idxs, raw_acts_qs, unbiased_fsp_str


def plot_patching_heatmap(combined_values, title):
    v = combined_values
    plt.figure(figsize=(12, 6))
    plt.imshow(
        v,
        cmap="RdBu",
        origin="lower",
        vmin=-max(abs(np.min(v)), abs(np.max(v))),
        vmax=max(abs(np.min(v)), abs(np.max(v))),
    )
    plt.title(title, fontsize=14)
    plt.colorbar()
    first_ytick = "all"
    # TODO: show only some
    if LB_LAYERS > 1:
        other_yticks = [
            f"{i*LB_LAYERS}-{(i+1)*LB_LAYERS}" for i in range(len(combined_values) - 1)
        ]
    else:
        other_yticks = [str(i - 1) for i in range(len(combined_values) - 1)]
        other_yticks[0] = "emb"
    plt.yticks(range(len(combined_values)), [first_ytick] + other_yticks, fontsize=10)
    plt.xticks(range(len(TOK_GROUPS)), TOK_GROUPS, rotation=90, fontsize=10)
    plt.ylabel("layers", fontsize=12)
    plt.xlabel("token groups", fontsize=12)
    plt.axhline(y=0.5, color="black", linewidth=1)
    plt.tight_layout()  # Adjust layout to prevent clipping
    plt.show()


def get_patch_values(
    plp_by_group_by_layers: dict[tuple[int, ...], dict[str, PatchedLogitsProbs]],
    prob_or_logit: Literal["prob", "logit"],
    direction: Literal["bia_to_unb", "unb_to_bia"],
) -> list[float] | list[list[float]]:
    attr = f"{prob_or_logit}_diff_change_{direction}"
    values = []
    for _layers, plp_by_group in plp_by_group_by_layers.items():
        values.append([getattr(plp, attr) for plp in plp_by_group.values()])
    if len(values) == 1:
        return values[0]
    return values


def trunc_cot_match(
    cot_tokens_list: list[list[int]], trunc_cot: list[int]
) -> int | None:
    for cot_idx, cot_tokens in enumerate(cot_tokens_list):
        for i in range(len(cot_tokens) - len(trunc_cot) + 1):
            if cot_tokens[i : i + len(trunc_cot)] == trunc_cot:
                return cot_idx
    return None


def load_filtered_data(
    raw_acts_qs: list[dict], tokenizer: PreTrainedTokenizerBase
) -> dict:
    q_and_cot_tokens = []
    cots_labels = []
    cots_answers = []
    questions = []
    for test_q in raw_acts_qs:
        cots = test_q["biased_cots_tokens_to_cache"]
        for cot in cots:
            tokens = cot[:-4]
            q_and_cot_tokens.append(tokens)
            cots_labels.append(test_q["biased_cot_label"])
            cots_answers.append(test_q["expected_answer"])
            questions.append(test_q["question"])

    with open(DATA_DIR / SWAPS_FILE, "rb") as f:
        swap_dict_by_q = pickle.load(f)["qs"]

    with open(DATA_DIR / CATEGORIES_FILE, "rb") as f:
        categories = pickle.load(f)

    categories_with_matches = {}
    for cat, qidx_swap_idx_pairs in categories.items():
        cat_pairs = []
        cat_matches = []
        for q_idx, swap_idx in tqdm(qidx_swap_idx_pairs):
            swap_dict = swap_dict_by_q[q_idx]
            assert swap_dict is not None
            swap = swap_dict["swaps"][swap_idx]

            match_idx = trunc_cot_match(q_and_cot_tokens, swap.trunc_cot)
            if match_idx is None:
                continue
            question_str = questions[match_idx]
            if question_str not in tokenizer.decode(swap.unb_prompt):
                continue
            cat_pairs.append((q_idx, swap_idx))
            cat_matches.append(match_idx)
        categories_with_matches[cat] = {"pairs": cat_pairs, "matches": cat_matches}
    categories = {cat: data["pairs"] for cat, data in categories_with_matches.items()}
    for cat, pairs_matches_dict in categories_with_matches.items():
        pairs = pairs_matches_dict["pairs"]
        matches = pairs_matches_dict["matches"]
        print(f"{cat}: {len(pairs)} {len(matches)}")

    with open(DATA_DIR / PATCH_ALL_FILE, "rb") as f:
        patch_all_by_q = pickle.load(f)

    with open(DATA_DIR / PATCH_LAYERS_FILE, "rb") as f:
        patch_layers_by_q = pickle.load(f)

    return {
        "categories_with_matches": categories_with_matches,
        "patch_all_by_q": patch_all_by_q,
        "patch_layers_by_q": patch_layers_by_q,
        "q_and_cot_tokens": q_and_cot_tokens,
        "cots_labels": cots_labels,
        "cots_answers": cots_answers,
        "swap_dict_by_q": swap_dict_by_q,
    }


from cot_probing.typing import *
from cot_probing.vis import visualize_tokens_html


def visualize_cot_attn(
    probe_model: AbstractAttnProbeModel,
    tokenizer: PreTrainedTokenizerBase,
    tokens: list[int],
    label: str,
    answer: str,
    resids: Float[torch.Tensor, "1 seq d_model"],
):
    # # Use provided resids or get from collate_fn_out
    # if resids is None:
    #     resids = collate_fn_out.cot_acts[cot_idx:cot_idx+1, :len(tokens)].to(probe_model.device)

    attn_mask = torch.ones(1, len(tokens), dtype=torch.bool, device=probe_model.device)

    # Get attention probs and model output
    attn_probs = probe_model.attn_probs(resids, attn_mask)
    probe_out = probe_model(resids, attn_mask)

    this_attn_probs = attn_probs[0, : len(tokens)]
    print(f"label: {label}, correct answer: {answer}")
    print(f"faithfulness: {probe_out.item():.2%}")
    return visualize_tokens_html(
        tokens, tokenizer, this_attn_probs.tolist(), vmin=0.0, vmax=1.0
    )


def visualize_cot_faithfulness(
    probe_model: AbstractAttnProbeModel,
    tokenizer: PreTrainedTokenizerBase,
    tokens: list[int],
    label: str,
    answer: str,
    resids: Float[torch.Tensor, "1 seq d_model"],
):
    # # Use provided resids or get from collate_fn_out
    # if resids is None:
    #     resids = collate_fn_out.cot_acts[cot_idx:cot_idx+1, :len(tokens)].to(probe_model.device)

    # Calculate faithfulness for each prefix length
    faithfulness_scores = []
    for prefix_len in range(1, len(tokens) + 1):
        # Create input with just this prefix
        prefix_resids = resids[:, :prefix_len]
        prefix_mask = torch.ones(
            1, prefix_len, dtype=torch.bool, device=probe_model.device
        )
        # Get model output for this prefix
        faithfulness = probe_model(prefix_resids, prefix_mask)
        faithfulness_scores.append(faithfulness.item())
    return visualize_tokens_html(
        tokens,
        tokenizer,
        faithfulness_scores,
        vmin=0.0,
        vmax=1.0,
        use_diverging_colors=True,
    )


# Function to update plot
def update_plot(
    category: str,
    q_idx: int,
    swap_idx: int,
    tokenizer: PreTrainedTokenizerBase,
    filtered_data: dict,
    probe_model: AbstractAttnProbeModel,
    collate_fn_out: CollateFnOutput,
    model: PreTrainedModel,
    unbiased_fsp_cache: tuple,
    layer: int,
):
    print(category, q_idx, swap_idx)
    categories_with_matches = filtered_data["categories_with_matches"]
    swap_dict_by_q = filtered_data["swap_dict_by_q"]
    patch_all_by_q = filtered_data["patch_all_by_q"]
    patch_layers_by_q = filtered_data["patch_layers_by_q"]
    assert (q_idx, swap_idx) in categories_with_matches[category]["pairs"]
    question_str = swap_dict_by_q[q_idx]["question"]
    correct_answer_str = swap_dict_by_q[q_idx]["expected_answer"]
    swap = swap_dict_by_q[q_idx]["swaps"][swap_idx]
    patch_all = patch_all_by_q[q_idx][swap_idx]
    patch_layers = patch_layers_by_q[q_idx][swap_idx]

    patch_all_values = get_patch_values(patch_all, LOGIT_OR_PROB, DIR)
    patch_layers_values = get_patch_values(patch_layers, LOGIT_OR_PROB, DIR)
    combined_values = [patch_all_values] + patch_layers_values
    trunc_cot_str = tokenizer.decode(swap.trunc_cot)
    print(question_str + trunc_cot_str)
    print()
    fai_tok_str = tokenizer.decode(swap.fai_tok).replace("\n", "\\n")
    unf_tok_str = tokenizer.decode(swap.unfai_tok).replace("\n", "\\n")
    print(f"correct answer: {correct_answer_str.upper()}")
    print(f"faithful_token:   `{fai_tok_str}`")
    print(f"unfaithful_token: `{unf_tok_str}`")
    plot_patching_heatmap(combined_values, f"change in {LOGIT_OR_PROB} diff")

    q_and_cot_tokens = filtered_data["q_and_cot_tokens"]
    cots_labels = filtered_data["cots_labels"]
    cots_answers = filtered_data["cots_answers"]
    some_idx = categories_with_matches[category]["pairs"].index((q_idx, swap_idx))
    cot_idx = categories_with_matches[category]["matches"][some_idx]
    tokens = q_and_cot_tokens[cot_idx]
    biased_resids = collate_fn_out.cot_acts[cot_idx : cot_idx + 1, : len(tokens)].to(
        probe_model.device
    )
    print("ATTN BIASED")
    display(
        visualize_cot_attn(
            probe_model=probe_model,
            tokenizer=tokenizer,
            tokens=tokens,
            label=cots_labels[cot_idx],
            answer=cots_answers[cot_idx],
            resids=biased_resids,
        )
    )
    print("FAITHFULNESS BIASED")
    display(
        visualize_cot_faithfulness(
            probe_model=probe_model,
            tokenizer=tokenizer,
            tokens=tokens,
            label=cots_labels[cot_idx],
            answer=cots_answers[cot_idx],
            resids=biased_resids,
        )
    )
    # different contexts

    def get_unbiased_resid_acts():
        return (
            collect_resid_acts_with_pastkv(
                model=model,
                last_q_toks=tokens,
                layers=[layer],
                past_key_values=unbiased_fsp_cache,
            )[layer]
            .unsqueeze(0)
            .cuda()
            .float()
        )

    def get_no_ctx_resid_acts():
        assert tokenizer.bos_token_id is not None
        return (
            collect_resid_acts_no_pastkv(
                model=model,
                all_input_ids=[tokenizer.bos_token_id] + tokens,
                layers=[layer],
            )[layer][1:]
            .unsqueeze(0)
            .cuda()
            .float()
        )

    unbiased_resids = get_unbiased_resid_acts()
    print("ATTN UNBIASED")
    display(
        visualize_cot_attn(
            probe_model=probe_model,
            tokenizer=tokenizer,
            tokens=tokens,
            label=cots_labels[cot_idx],
            answer=cots_answers[cot_idx],
            resids=unbiased_resids,
        )
    )
    print("FAITHFULNESS UNBIASED")
    display(
        visualize_cot_faithfulness(
            probe_model=probe_model,
            tokenizer=tokenizer,
            tokens=tokens,
            label=cots_labels[cot_idx],
            answer=cots_answers[cot_idx],
            resids=unbiased_resids,
        )
    )
    no_ctx_resids = get_no_ctx_resid_acts()
    print("ATTN NO CONTEXT")
    display(
        visualize_cot_attn(
            probe_model=probe_model,
            tokenizer=tokenizer,
            tokens=tokens,
            label=cots_labels[cot_idx],
            answer=cots_answers[cot_idx],
            resids=no_ctx_resids,
        )
    )
    print("FAITHFULNESS NO CONTEXT")
    display(
        visualize_cot_faithfulness(
            probe_model=probe_model,
            tokenizer=tokenizer,
            tokens=tokens,
            label=cots_labels[cot_idx],
            answer=cots_answers[cot_idx],
            resids=no_ctx_resids,
        )
    )
