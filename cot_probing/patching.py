from functools import partial

from cot_probing.typing import *


def layer_to_hook_point(layer: int):
    if layer == 0:
        return "model.embed_tokens"
    return f"model.layers.{layer-1}"


def hook_point_to_layer(hook_point: str):
    if hook_point == "model.embed_tokens":
        return 0
    return int(hook_point.split(".")[-1]) + 1


def general_caching_hook_fn(
    module,
    input,
    output,
    pos: slice | int,
    resid_by_pos: dict[slice | int, Float[torch.Tensor, " _seq model"]],
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
    resid_by_pos[pos] = output[pos].cpu()


def general_patching_hook_fn(
    module, input, output, pos: slice | int, resid: Float[torch.Tensor, " seq model"]
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
    output[pos] = resid


def clean_run_with_cache(
    model: PreTrainedModel,
    input_ids: list[int],
    pos_by_layer: dict[int, list[slice | int]],
) -> tuple[
    Float[torch.Tensor, " vocab"],
    dict[int, dict[slice | int, Float[torch.Tensor, " _seq model"]]],
]:
    resid_by_pos_by_layer = {}
    hooks = []
    hook_points = set(layer_to_hook_point(i) for i in pos_by_layer.keys())
    hook_points_cnt = len(hook_points)
    for name, module in model.named_modules():
        if name in hook_points:
            hook_points_cnt -= 1
            layer = hook_point_to_layer(name)
            assert layer not in resid_by_pos_by_layer
            resid_by_pos = resid_by_pos_by_layer[layer] = {}
            for pos in pos_by_layer[layer]:
                hook_fn = partial(
                    general_caching_hook_fn, pos=pos, resid_by_pos=resid_by_pos
                )
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
    assert hook_points_cnt == 0
    try:
        # add and then drop batch dim
        logits = model(torch.tensor([input_ids]).cuda()).logits[0]
        # only last position and move to cpu
        logits = logits[-1].cpu()
    finally:
        for hook in hooks:
            hook.remove()
    return logits, resid_by_pos_by_layer


def patched_run(
    model: PreTrainedModel,
    input_ids: list[int],
    resid_by_pos_by_layer: dict[
        int, dict[slice | int, Float[torch.Tensor, " _seq model"]]
    ],
) -> Float[torch.Tensor, " vocab"]:
    hooks = []
    hook_points = set(layer_to_hook_point(i) for i in resid_by_pos_by_layer.keys())
    hook_points_cnt = len(hook_points)
    for name, module in model.named_modules():
        if name in hook_points:
            hook_points_cnt -= 1
            layer = hook_point_to_layer(name)
            for pos, resid in resid_by_pos_by_layer[layer].items():
                hook_fn = partial(general_patching_hook_fn, pos=pos, resid=resid)
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
    assert hook_points_cnt == 0
    try:
        # add and then drop batch dim
        logits = model(torch.tensor([input_ids]).cuda()).logits[0]
        # only last position and move to cpu
        logits = logits[-1].cpu()
    finally:
        for hook in hooks:
            hook.remove()
    return logits
