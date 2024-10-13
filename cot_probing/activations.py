from functools import partial

from cot_probing.eval import EvalQuestion, EvalResults
from cot_probing.typing import *
from cot_probing.typing import Float, torch


@dataclass
class QuestionActivations:
    activations: Float[torch.Tensor, "n_layers locs d_model"]
    sorted_locs: list[int]

    def __repr__(self):
        activations_str = f"activations={list(self.activations.shape)}"
        sl_first, sl_last = self.sorted_locs[0], self.sorted_locs[-1]
        sorted_locs_str = (
            f"sorted_locs=[{sl_first}, ..., {sl_last}] ({len(self.sorted_locs)})"
        )
        return f"QuestionActivations({activations_str}, {sorted_locs_str})"


@dataclass
class Activations:
    eval_results: EvalResults
    activations_by_question: list[QuestionActivations]
    layers: list[int]


def clean_run_with_cache(
    model,
    input_ids: Int[torch.Tensor, " seq"],
    layers: list[int],
    locs_to_cache: list[int],
) -> Float[torch.Tensor, "layers locs model"]:
    resid_acts = torch.empty(
        (len(layers), len(locs_to_cache), model.config.hidden_size)
    )

    def general_hook_fn(module, input, output, layer):
        # output is a tuple, not sure what the second element is
        output = output[0]
        # 0 is for the batch dimension
        layer_idx = layers.index(layer)
        resid_acts[layer_idx] = output[0, locs_to_cache]

    hooks = []
    for name, module in model.named_modules():
        if name.startswith("model.layers."):
            layer_str = name.rsplit(".", 1)[-1]
            if not layer_str.isdigit():
                continue
            layer = int(layer_str)
            if layer not in layers:
                continue
            layer_hook_fn = partial(general_hook_fn, layer=layer)
            hooks.append(module.register_forward_hook(layer_hook_fn))
    try:
        model(
            input_ids.unsqueeze(0).to(model.device),
            output_hidden_states=False,
            output_attentions=False,
            use_cache=False,
            labels=None,
            return_dict=True,
        )
    finally:
        for hook in hooks:
            hook.remove()

    return resid_acts


# def get_mean_activations_by_common_token(
#     common_tokens: list[int], activations: Activations, q_idxs: list[int]
# ) -> dict[int, Float[torch.Tensor, "layer model"]]:
#     q_act_by_q = activations.activations_by_question
#     eval_results = activations.eval_results
#     acts_list_by_tok = {t: [] for t in common_tokens}
#     for q_idx in q_idxs:
#         q_act: QuestionActivations = q_act_by_q[q_idx]
#         eval_q: EvalQuestion = eval_results.questions[q_idx]
#         # in general, the setup was that you can have many keys in
#         # eval_q.locs, and q_act.sorted_locs would contain unique
#         # sorted locations for all the keys in eval_q.locs
#         # but for now we only have "response"
#         assert eval_q.locs["response"] == q_act.sorted_locs
#         # shape: layers, locs, model
#         acts = q_act.activations
#         # all_locs and all_toks correspond to 2nd dim of acts
#         all_locs = q_act.sorted_locs
#         all_toks = torch.tensor([eval_q.tokens[loc] for loc in all_locs])
#         for t in common_tokens:
#             this_tok_mask = all_toks == t
#             if not this_tok_mask.any():
#                 continue
#             # append activations from this question/response that
#             # correspond to this token to a list for this token
#             # acts shape: layers, locs, model
#             acts_this_tok = acts[:, this_tok_mask, :]
#             acts_list_by_tok[t].append(acts_this_tok)
#     # concatenate all the activations corresponding to each token
#     # dim 1 is the locs/tokens dimension
#     acts_by_tok = {
#         t: torch.cat(acts_list, dim=1) for t, acts_list in acts_list_by_tok.items()
#     }
#     # check if we get some acts for each token
#     assert all(acts_by_tok[t].numel() > 0 for t in common_tokens)
#     # take the mean over all questions for each token
#     return {t: acts.mean(1) for t, acts in acts_by_tok.items()}
