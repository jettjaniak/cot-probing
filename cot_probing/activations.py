from cot_probing.typing import *


def clean_run_with_cache(
    model: PreTrainedModel,
    input_ids: list[int],
    locs_to_cache: dict[str, tuple[int, int | None]],
) -> dict[str, list[Float[torch.Tensor, " locs model"]]]:
    resid_acts_by_layer_by_locs = {loc_type: [] for loc_type in locs_to_cache.keys()}

    def hook_fn(module, input, output):
        # output is a tuple, not sure what the second element is
        output = output[0]
        for loc_type, (loc_start, loc_end) in locs_to_cache.items():
            resid_acts_by_layer = resid_acts_by_layer_by_locs[loc_type]
            # 0 is for the batch dimension
            resid_acts_by_layer.append(output[0, loc_start:loc_end].cpu())

    hooks = []
    for name, module in model.named_modules():
        if name.startswith("model.layers."):
            layer_str = name.rsplit(".", 1)[-1]
            if not layer_str.isdigit():
                continue
            hooks.append(module.register_forward_hook(hook_fn))
    try:
        model(torch.tensor([input_ids]).cuda())
    finally:
        for hook in hooks:
            hook.remove()

    return resid_acts_by_layer_by_locs
