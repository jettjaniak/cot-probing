import copy
from functools import partial

from cot_probing.typing import *


def clean_run_with_cache(
    model: PreTrainedModel,
    input_ids: list[int],
    locs_to_cache: dict[str, tuple[int | None, int | None]],
    collect_embeddings: bool = False,
) -> dict[str, list[Float[torch.Tensor, " locs model"]]]:
    resid_acts_by_layer_by_locs = {loc_type: [] for loc_type in locs_to_cache.keys()}

    def layer_hook_fn(module, input, output):
        # output is a tuple, not sure what the second element is
        output = output[0]
        for loc_type, (loc_start, loc_end) in locs_to_cache.items():
            resid_acts_by_layer = resid_acts_by_layer_by_locs[loc_type]
            # 0 is for the batch dimension
            resid_acts_by_layer.append(output[0, loc_start:loc_end].cpu())

    def embedding_hook_fn(module, input, output):
        for loc_type, (loc_start, loc_end) in locs_to_cache.items():
            resid_acts_by_layer_by_locs[loc_type].insert(
                0, output[0, loc_start:loc_end].cpu()
            )

    hooks = []
    if collect_embeddings:
        # Add hook for embeddings
        emb_module = model.get_input_embeddings()
        hooks.append(emb_module.register_forward_hook(embedding_hook_fn))

    for name, module in model.named_modules():
        if name.startswith("model.layers."):
            layer_str = name.rsplit(".", 1)[-1]
            if not layer_str.isdigit():
                continue
            hooks.append(module.register_forward_hook(layer_hook_fn))
    try:
        with torch.inference_mode():
            model(torch.tensor([input_ids]).cuda())
    finally:
        for hook in hooks:
            hook.remove()

    return resid_acts_by_layer_by_locs


def layer_to_hook_point(layer: int):
    if layer == 0:
        return "model.embed_tokens"
    return f"model.layers.{layer-1}"


def hook_point_to_layer(hook_point: str):
    if hook_point == "model.embed_tokens":
        return 0
    return int(hook_point.split(".")[-1]) + 1


def general_collect_resid_acts_hook_fn(
    module,
    input,
    output,
    layer: int,
    resid_by_layer: dict[int, Float[torch.Tensor, " seq model"]],
):
    if isinstance(output, tuple):
        assert len(output) == 2
        # second element is cache
        # this happens after decoder layers,
        # but not after embedding layer
        output = output[0]
    assert len(output.shape) == 3
    # we're running batch size 1
    output = output[0]
    # shape is (seq, model)
    resid_by_layer[layer] = output.cpu()


def _collect_resid_acts(
    model: PreTrainedModel,
    input_ids: list[int],
    layers: list[int],
    past_key_values: Optional[tuple] = None,
) -> dict[int, Float[torch.Tensor, " seq model"]]:
    resid_by_layer = {}
    hooks = []
    hook_points = set(layer_to_hook_point(i) for i in layers)
    hook_points_cnt = len(hook_points)
    for name, module in model.named_modules():
        if name in hook_points:
            hook_points_cnt -= 1
            layer = hook_point_to_layer(name)
            assert layer not in resid_by_layer
            hook_fn = partial(
                general_collect_resid_acts_hook_fn,
                layer=layer,
                resid_by_layer=resid_by_layer,
            )
            hook = module.register_forward_hook(hook_fn)
            hooks.append(hook)
    assert hook_points_cnt == 0
    try:
        model(
            torch.tensor([input_ids]).cuda(),
            past_key_values=copy.deepcopy(past_key_values),
        )
    finally:
        for hook in hooks:
            hook.remove()
    return resid_by_layer


def collect_resid_acts_no_pastkv(
    model: PreTrainedModel,
    *,
    all_input_ids: list[int],
    layers: list[int],
) -> dict[int, Float[torch.Tensor, " seq model"]]:
    return _collect_resid_acts(model, all_input_ids, layers)


def collect_resid_acts_with_pastkv(
    model: PreTrainedModel,
    *,
    last_q_toks: list[int],
    layers: list[int],
    past_key_values: tuple,
) -> dict[int, Float[torch.Tensor, " seq model"]]:
    return _collect_resid_acts(model, last_q_toks, layers, past_key_values)
