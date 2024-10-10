from functools import partial

from cot_probing.eval import EvalResults
from cot_probing.typing import *


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
            layer = int(name.rsplit(".", 1)[-1])
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
