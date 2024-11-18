# %%
from typing import List

import matplotlib.pyplot as plt
import wandb
from tqdm.auto import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import seaborn as sns

from cot_probing import DATA_DIR
from cot_probing.activations import (
    build_fsp_cache,
    collect_resid_acts_no_pastkv,
    collect_resid_acts_with_pastkv,
)
from cot_probing.attn_probes import AbstractAttnProbeModel, AttnProbeTrainer
from cot_probing.attn_probes_case_studies import (
    load_filtered_data,
    load_median_probe_test_data,
)
from cot_probing.attn_probes_data_proc import CollateFnOutput
from cot_probing.typing import *
from cot_probing.utils import load_model_and_tokenizer

from transformers import AutoModelForCausalLM, AutoTokenizer

LOGIT_OR_PROB = "prob"
DIR = "bia_to_unb"
CATEGORIES_FILE = f"categories_{LOGIT_OR_PROB}_{DIR}_0.25_1.5_2.0_4.0.pkl"


# %%
model, tokenizer = load_model_and_tokenizer(8)
model.eval()
model.requires_grad_(False)

# %%
layer = 15
min_seed, max_seed = 21, 40
n_seeds = max_seed - min_seed + 1
probe_class = "minimal"
metric = "test_accuracy"

trainer, test_idxs, raw_acts_qs, unbiased_fsp_str = load_median_probe_test_data(
    probe_class, layer, min_seed, max_seed, metric
)
collate_fn_out: CollateFnOutput = list(trainer.test_loader)[0]
unbiased_fsp_cache = build_fsp_cache(model, tokenizer, unbiased_fsp_str)

trainer.model.eval()
trainer.model.requires_grad_(False)

# %%
q_and_cot_tokens = []
cots_labels = []
cots_answers = []
questions = []
biased_resid_acts = []
for test_q in raw_acts_qs:
    for cot, cached_act in zip(
        test_q["biased_cots_tokens_to_cache"], test_q["cached_acts"]
    ):
        tokens = cot[:-4]
        q_and_cot_tokens.append(tokens)
        biased_resid_acts.append(cached_act)
        cots_labels.append(test_q["biased_cot_label"])
        cots_answers.append(test_q["expected_answer"])
        questions.append(test_q["question"])


# %%
# Methods for producing resid acts
def get_unbiased_resid_acts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tokens: List[int],
    layer: int,
    unbiased_fsp_cache: tuple,
):
    return (
        collect_resid_acts_with_pastkv(
            model=model,
            last_q_toks=tokens,
            layers=[layer],
            past_key_values=unbiased_fsp_cache,
        )[layer]
        .unsqueeze(0)
        .cpu()
        .float()
    )


def get_no_ctx_resid_acts(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tokens: List[int],
    layer: int,
):
    assert tokenizer.bos_token_id is not None
    return (
        collect_resid_acts_no_pastkv(
            model=model,
            all_input_ids=[tokenizer.bos_token_id] + tokens,
            layers=[layer],
        )[layer][1:]
        .unsqueeze(0)
        .cpu()
        .float()
    )


# %%
# Get predictions for different contexts
unbiased_resid_acts = []
no_ctx_resid_acts = []

for tokens in tqdm(q_and_cot_tokens):
    # Get unbiased context predictions
    unbiased_acts = get_unbiased_resid_acts(
        model=model,
        tokenizer=tokenizer,
        tokens=tokens,
        layer=layer,
        unbiased_fsp_cache=unbiased_fsp_cache,
    )
    unbiased_resid_acts.append(unbiased_acts)

    # Get no context predictions
    no_ctx_acts = get_no_ctx_resid_acts(
        model=model,
        tokenizer=tokenizer,
        tokens=tokens,
        layer=layer,
    )
    no_ctx_resid_acts.append(no_ctx_acts)


# %%
# Get probe predictions for each context
def get_probe_preds(resid_acts_list):
    pred_scores = []
    preds = []
    for resids in resid_acts_list:
        # Create attention mask for single sequence
        attn_mask = torch.ones(
            1, resids.shape[0], dtype=torch.bool, device=resids.device
        ).cuda()

        if len(resids.shape) == 2:
            resids = resids.unsqueeze(0)
        resids = resids.float().cuda()

        # Get prediction for single sequence
        pred_score = trainer.model.get_pred_scores(resids, attn_mask).cpu()[0]
        pred = trainer.model(resids, attn_mask).cpu()[0].numpy()
        pred_scores.append(pred_score.item())
        preds.append(pred > 0.5)
    return np.array(pred_scores), np.array(preds)


# Get predictions for each context type
biased_pred_scores, biased_preds = get_probe_preds(biased_resid_acts)
unbiased_pred_scores, unbiased_preds = get_probe_preds(unbiased_resid_acts)
no_ctx_pred_scores, no_ctx_preds = get_probe_preds(no_ctx_resid_acts)

# %%
# Plot histograms separately
# Biased vs Unbiased
plt.figure(figsize=(8, 5))
plt.hist(biased_pred_scores - unbiased_pred_scores, bins=50, alpha=0.7)
plt.title("Probe pred scores: Biased vs Unbiased Context")
plt.xlabel("Prediction Difference")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Biased vs No Context
plt.figure(figsize=(8, 5))
plt.hist(biased_pred_scores - no_ctx_pred_scores, bins=50, alpha=0.7)
plt.title("Probe pred scores: Biased vs No Context")
plt.xlabel("Prediction Difference")
plt.ylabel("Count")
plt.tight_layout()
plt.show()

# Print summary statistics
print("Biased vs Unbiased Context pred scores:")
print(f"Mean diff: {(biased_pred_scores - unbiased_pred_scores).mean():.3f}")
print(f"Std diff: {(biased_pred_scores - unbiased_pred_scores).std():.3f}")
print("\nBiased vs No Context pred scores:")
print(f"Mean diff: {(biased_pred_scores - no_ctx_pred_scores).mean():.3f}")
print(f"Std diff: {(biased_pred_scores - no_ctx_pred_scores).std():.3f}")

# Plot confusion matrices separately
# Biased vs Unbiased confusion matrix
plt.figure(figsize=(8, 5))
cm_biased_unbiased = confusion_matrix(biased_preds, unbiased_preds)
sns.heatmap(cm_biased_unbiased, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: Biased vs Unbiased")
plt.xlabel("Unbiased Predictions")
plt.ylabel("Biased Predictions")
plt.tight_layout()
plt.show()

# Biased vs No Context confusion matrix
plt.figure(figsize=(8, 5))
cm_biased_noctx = confusion_matrix(biased_preds, no_ctx_preds)
sns.heatmap(cm_biased_noctx, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix: Biased vs No Context")
plt.xlabel("No Context Predictions")
plt.ylabel("Biased Predictions")
plt.tight_layout()
plt.show()

# Print classification metrics
print("Biased vs Unbiased Context:")
print(f"Accuracy: {accuracy_score(biased_preds, unbiased_preds):.3f}")
print(f"F1 Score: {f1_score(biased_preds, unbiased_preds):.3f}")
print("\nBiased vs No Context:")
print(f"Accuracy: {accuracy_score(biased_preds, no_ctx_preds):.3f}")
print(f"F1 Score: {f1_score(biased_preds, no_ctx_preds):.3f}")

# %%
