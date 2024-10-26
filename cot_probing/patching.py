from functools import partial

from cot_probing.swapping import SuccessfulSwap
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
    pos: list[int],
    layer: int,
    resid_pos_by_layer: dict[int, tuple[Float[torch.Tensor, " seq model"], list[int]]],
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
    resid_pos_by_layer[layer] = (output[pos].cpu(), pos)


def general_patching_hook_fn(
    module, input, output, pos: list[int], resid: Float[torch.Tensor, " seq model"]
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
    output[pos] = resid.cuda()


def clean_run_with_cache(
    model: PreTrainedModel,
    input_ids: list[int],
    pos_by_layer: dict[int, list[int]],
) -> tuple[
    Float[torch.Tensor, " vocab"],
    dict[int, tuple[Float[torch.Tensor, " seq model"], list[int]]],
]:
    resid_pos_by_layer = {}
    hooks = []
    hook_points = set(layer_to_hook_point(i) for i in pos_by_layer.keys())
    hook_points_cnt = len(hook_points)
    for name, module in model.named_modules():
        if name in hook_points:
            hook_points_cnt -= 1
            layer = hook_point_to_layer(name)
            assert layer not in resid_pos_by_layer
            hook_fn = partial(
                general_caching_hook_fn,
                pos=pos_by_layer[layer],
                layer=layer,
                resid_pos_by_layer=resid_pos_by_layer,
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
    return logits, resid_pos_by_layer


def patched_run(
    model: PreTrainedModel,
    input_ids: list[int],
    resid_pos_by_layer: dict[int, tuple[Float[torch.Tensor, " seq model"], list[int]]],
    pos_by_layer: dict[int, list[int]],
) -> Float[torch.Tensor, " vocab"]:
    hooks = []
    hook_points = set(layer_to_hook_point(i) for i in pos_by_layer.keys())
    hook_points_cnt = len(hook_points)
    for name, module in model.named_modules():
        if name in hook_points:
            hook_points_cnt -= 1
            layer = hook_point_to_layer(name)
            resid_acts_all, resid_pos = resid_pos_by_layer[layer]
            pos_wanted = pos_by_layer[layer]
            try:
                indices_to_patch = [resid_pos.index(pos) for pos in pos_wanted]
            except KeyError:
                print(f"resid pos {resid_pos} does not contain {pos_wanted}")
                raise
            resid_to_patch = resid_acts_all[indices_to_patch]
            hook_fn = partial(
                general_patching_hook_fn,
                pos=pos_wanted,
                resid=resid_to_patch,
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
    return logits


@dataclass
class LogitsProbs:
    logit_fai: float
    logit_unf: float
    prob_fai: float
    prob_unf: float


def get_logits_probs(
    logits: Float[torch.Tensor, "vocab"], fai_tok: int, unfai_tok: int
) -> LogitsProbs:
    probs = torch.softmax(logits, dim=-1)
    return LogitsProbs(
        logit_fai=logits[fai_tok].item(),
        logit_unf=logits[unfai_tok].item(),
        prob_fai=probs[fai_tok].item(),
        prob_unf=probs[unfai_tok].item(),
    )


@dataclass
class Cache:
    fai_tok: int
    unfai_tok: int
    input_ids_unb: list[int]
    input_ids_bia: list[int]
    logits_probs_unb: LogitsProbs
    logits_probs_bia: LogitsProbs
    # this could be a subset of all sequence positions,
    # and be different for each layer
    resid_pos_by_layer_unb: dict[
        int, tuple[Float[torch.Tensor, " seq model"], list[int]]
    ]
    resid_pos_by_layer_bia: dict[
        int, tuple[Float[torch.Tensor, " seq model"], list[int]]
    ]


def get_cache(
    model: PreTrainedModel,
    input_ids_unb: list[int],
    input_ids_bia: list[int],
    fai_tok: int,
    unfai_tok: int,
    pos_by_layer: dict[int, list[int]],
):
    logits_unbiased, resid_pos_by_layer_unbiased = clean_run_with_cache(
        model, input_ids_unb, pos_by_layer
    )
    logits_biased, resid_pos_by_layer_biased = clean_run_with_cache(
        model, input_ids_bia, pos_by_layer
    )
    logits_probs_unbiased = get_logits_probs(logits_unbiased, fai_tok, unfai_tok)
    logits_probs_biased = get_logits_probs(logits_biased, fai_tok, unfai_tok)
    if not (
        logits_probs_unbiased.prob_fai > logits_probs_biased.prob_fai
        and logits_probs_unbiased.prob_unf < logits_probs_biased.prob_unf
    ):
        return None
    return Cache(
        fai_tok=fai_tok,
        unfai_tok=unfai_tok,
        input_ids_unb=input_ids_unb,
        input_ids_bia=input_ids_bia,
        logits_probs_unb=logits_probs_unbiased,
        logits_probs_bia=logits_probs_biased,
        resid_pos_by_layer_unb=resid_pos_by_layer_unbiased,
        resid_pos_by_layer_bia=resid_pos_by_layer_biased,
    )


@dataclass
class PatchedLogitsProbs:
    bia_to_unb: LogitsProbs
    unb_to_bia: LogitsProbs

    @property
    def pd_change_unb(self):
        return self.bia_to_unb.prob_fai - self.unb_to_bia.prob_unf

    @property
    def pd_change_bia(self):
        return self.unb_to_bia.prob_fai - self.bia_to_unb.prob_unf

    @property
    def ld_change_unb(self):
        return self.bia_to_unb.logit_fai - self.unb_to_bia.logit_unf

    @property
    def ld_change_bia(self):
        return self.unb_to_bia.logit_fai - self.bia_to_unb.logit_unf


def get_patched_logits_probs(
    model: PreTrainedModel,
    cache: Cache,
    pos_by_layer: dict[int, list[int]],
):
    logits_patched_biased_to_unbiased = patched_run(
        model,
        cache.input_ids_unb,
        cache.resid_pos_by_layer_bia,
        pos_by_layer,
    )

    logits_patched_unbiased_to_biased = patched_run(
        model,
        cache.input_ids_bia,
        cache.resid_pos_by_layer_unb,
        pos_by_layer,
    )
    return PatchedLogitsProbs(
        bia_to_unb=get_logits_probs(
            logits_patched_biased_to_unbiased, cache.fai_tok, cache.unfai_tok
        ),
        unb_to_bia=get_logits_probs(
            logits_patched_unbiased_to_biased, cache.fai_tok, cache.unfai_tok
        ),
    )
