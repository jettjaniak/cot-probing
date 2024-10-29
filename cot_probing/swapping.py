import logging
import pickle

from cot_probing.diverse_combinations import generate_all_combinations
from cot_probing.generation import categorize_response
from cot_probing.patching import (
    Cache,
    PatchedLogitsProbs,
    get_cache,
    get_patched_logits_probs,
)
from cot_probing.typing import *


@dataclass
class SingleCotSwapResult:
    seq_pos: int
    swap_token: int
    prob_diff: float


@dataclass
class QuestionSwapResults:
    unfai_to_fai_swaps: list[SingleCotSwapResult | None]
    fai_to_unfai_swaps: list[SingleCotSwapResult | None]


@dataclass
class LayersFspPatchResult:
    together: PatchedLogitsProbs
    separate: list[PatchedLogitsProbs]


@dataclass
class SuccessfulSwap:
    unb_prompt: list[int]
    bia_prompt: list[int]
    trunc_cot: list[int]
    fai_tok: int
    unfai_tok: int
    swap_dir: Literal["unfai_to_fai", "fai_to_unfai"]
    prob_diff: float

    def __repr__(self) -> str:
        return f"SuccessfulSwap(fai_tok={self.fai_tok}, unfai_tok={self.unfai_tok}, swap_dir={self.swap_dir}, prob_diff={self.prob_diff})"

    def get_input_ids_unb(self) -> list[int]:
        return self.unb_prompt + self.trunc_cot

    def get_input_ids_bia(self) -> list[int]:
        return self.bia_prompt + self.trunc_cot

    def get_last_q_pos(self, tokenizer: PreTrainedTokenizerBase) -> int:
        _, q_tok = tokenizer.encode("Question")
        last_q_pos = len(self.unb_prompt) - 1 - self.unb_prompt[::-1].index(q_tok)
        assert self.unb_prompt[last_q_pos] == self.bia_prompt[last_q_pos] == q_tok
        return last_q_pos

    def get_all_q_pos(self, tokenizer: PreTrainedTokenizerBase) -> list[int]:
        _, q_tok = tokenizer.encode("Question")
        q_idxs = [i for i, tok in enumerate(self.unb_prompt) if tok == q_tok]
        for q_idx in q_idxs:
            assert self.unb_prompt[q_idx] == self.bia_prompt[q_idx] == q_tok
        return q_idxs

    def get_cache(
        self,
        model: PreTrainedModel,
        pos_by_layer: dict[int, list[int]],
    ) -> Cache | None:
        return get_cache(
            model,
            self.get_input_ids_unb(),
            self.get_input_ids_bia(),
            self.fai_tok,
            self.unfai_tok,
            pos_by_layer,
        )

    def patch_all_fsps_together(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache: Cache,
        layers: list[int],
    ) -> PatchedLogitsProbs:
        last_q_pos = self.get_last_q_pos(tokenizer)
        pos_by_layer = {l: list(range(last_q_pos)) for l in layers}
        return get_patched_logits_probs(model, cache, pos_by_layer)

    def patch_single_fsp(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache: Cache,
        layers: list[int],
        fsp_idx: int,
    ) -> PatchedLogitsProbs:
        all_q_pos = self.get_all_q_pos(tokenizer)
        begin = all_q_pos[fsp_idx]
        end = all_q_pos[fsp_idx + 1]
        unb_fsp = self.unb_prompt[begin:end]
        bia_fsp = self.bia_prompt[begin:end]
        logging.debug(
            f"fsp {fsp_idx}:\nunb:\n`{tokenizer.decode(unb_fsp)}`\nbia:\n`{tokenizer.decode(bia_fsp)}`"
        )
        pos_by_layer = {l: list(range(begin, end)) for l in layers}
        return get_patched_logits_probs(model, cache, pos_by_layer)

    def patch_every_fsp_separately(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache: Cache,
        layers: list[int],
    ) -> list[PatchedLogitsProbs]:
        all_q_pos = self.get_all_q_pos(tokenizer)
        return [
            self.patch_single_fsp(model, tokenizer, cache, layers, i)
            for i in range(len(all_q_pos) - 1)
        ]

    def patch_fsps_selected_layers(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache: Cache,
        layers: list[int],
    ) -> LayersFspPatchResult:
        together = self.patch_all_fsps_together(model, tokenizer, cache, layers)
        separate = self.patch_every_fsp_separately(model, tokenizer, cache, layers)
        return LayersFspPatchResult(together=together, separate=separate)

    def patch_fsps_all_layers_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache: Cache,
        layer_batch_size: int,
    ) -> dict[tuple[int, ...], LayersFspPatchResult]:
        n_layers = model.config.num_hidden_layers
        layers = list(range(0, n_layers + 1))
        ret = {}
        for i in range(0, n_layers + 1, layer_batch_size):
            layer_batch = layers[i : i + layer_batch_size]
            ret[tuple(layer_batch)] = self.patch_fsps_selected_layers(
                model, tokenizer, cache, layer_batch
            )
        return ret

    def patch_fsps_single_selected_layers(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache: Cache,
        layers: list[int],
    ) -> dict[tuple[int, ...], list[PatchedLogitsProbs]]:
        ret = {}
        for layer in layers:
            ret[(layer,)] = self.patch_fsps_selected_layers(
                model, tokenizer, cache, [layer]
            )
        return ret

    def patch_q_and_cot_selected_layers(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache: Cache,
        layers: list[int],
    ) -> list[PatchedLogitsProbs]:
        last_q_pos = self.get_last_q_pos(tokenizer)
        ret = []
        for seq_pos in range(last_q_pos, len(self.get_input_ids_bia())):
            pos_by_layer = {l: [seq_pos] for l in layers}
            ret.append(get_patched_logits_probs(model, cache, pos_by_layer))
        return ret

    def patch_q_and_cot_all_layers_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache: Cache,
        layer_batch_size: int,
    ) -> dict[tuple[int, ...], list[PatchedLogitsProbs]]:
        n_layers = model.config.num_hidden_layers
        layers = list(range(0, n_layers + 1))
        ret = {}
        for i in range(0, n_layers + 1, layer_batch_size):
            layer_batch = layers[i : i + layer_batch_size]
            ret[tuple(layer_batch)] = self.patch_q_and_cot_selected_layers(
                model, tokenizer, cache, layer_batch
            )
        return ret

    def patch_q_and_cot_two_slices_selected_layers(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache: Cache,
        layers: list[int],
        slice_idx: int,
    ) -> list[PatchedLogitsProbs]:
        last_q_pos = self.get_last_q_pos(tokenizer)
        ret = []
        all_pos = list(range(last_q_pos, len(self.get_input_ids_bia())))
        pos_by_layer_1st = {l: all_pos[:slice_idx] for l in layers}
        pos_by_layer_2nd = {l: all_pos[slice_idx:] for l in layers}
        ret.append(get_patched_logits_probs(model, cache, pos_by_layer_1st))
        ret.append(get_patched_logits_probs(model, cache, pos_by_layer_2nd))
        return ret

    def patch_q_and_cot_two_slices_all_layers_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        cache: Cache,
        layer_batch_size: int,
        slice_idx: int,
    ) -> dict[tuple[int, ...], list[PatchedLogitsProbs]]:
        n_layers = model.config.num_hidden_layers
        layers = list(range(0, n_layers + 1))
        ret = {}
        for i in range(0, n_layers + 1, layer_batch_size):
            layer_batch = layers[i : i + layer_batch_size]
            ret[tuple(layer_batch)] = self.patch_q_and_cot_two_slices_selected_layers(
                model, tokenizer, cache, layer_batch, slice_idx
            )
        return ret


def greedy_gen_until_answer(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    prompt_toks: list[int],
    max_new_tokens: int,
) -> list[int]:
    return model.generate(
        torch.tensor(prompt_toks).unsqueeze(0).to("cuda"),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.eos_token_id,
        stop_strings=["Answer:"],
    )[0, len(prompt_toks) :].tolist()


def get_original_swapped_contins(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    context_toks: list[int],
    trunc_cot_original: list[int],
    trunc_cot_swapped: list[int],
) -> tuple[list[int], list[int]]:
    tokens_original = context_toks + trunc_cot_original
    contin_original = greedy_gen_until_answer(
        model, tokenizer, prompt_toks=tokens_original, max_new_tokens=100
    )
    tokens_swapped = context_toks + trunc_cot_swapped
    contin_swapped = greedy_gen_until_answer(
        model, tokenizer, prompt_toks=tokens_swapped, max_new_tokens=100
    )
    return contin_original, contin_swapped


def get_resp_answer_original_swapped(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    context_toks: list[int],
    trunc_cot_toks: list[int],
    original_tok: int,
    swapped_tok: int,
    unbiased_context_toks: list[int],
) -> tuple[
    tuple[list[int], Literal["yes", "no", "other"]],
    tuple[list[int], Literal["yes", "no", "other"]],
]:
    trunc_cot_original = trunc_cot_toks + [original_tok]
    trunc_cot_swapped = trunc_cot_toks + [swapped_tok]
    contin_original, contin_swapped = get_original_swapped_contins(
        model,
        tokenizer,
        context_toks=context_toks,
        trunc_cot_original=trunc_cot_original,
        trunc_cot_swapped=trunc_cot_swapped,
    )
    # TODO: cache KV for unbiased context (and trunc cot?) to make it ~2x faster
    response_original = trunc_cot_original + contin_original
    answer_original = categorize_response(
        model,
        tokenizer,
        unbiased_context_toks=unbiased_context_toks,
        response=response_original,
    )
    response_swapped = trunc_cot_swapped + contin_swapped
    answer_swapped = categorize_response(
        model,
        tokenizer,
        unbiased_context_toks=unbiased_context_toks,
        response=response_swapped,
    )
    return (contin_original, answer_original), (contin_swapped, answer_swapped)


def try_swap_token(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    original_ctx_toks: list[int],
    unbiased_ctx_toks: list[int],
    original_cot: list[int],
    original_expected_answer: Literal["yes", "no"],
    other_tok: int,
    seq_pos: int,
) -> bool:
    original_cot_tok = original_cot[seq_pos]
    original_tok_str = tokenizer.decode([original_cot_tok])
    logging.info(f"Trying to swap original CoT token `{original_tok_str}`")
    if original_cot_tok == other_tok:
        logging.info("Original CoT token and other token are the same, skipping...")
        return False
    # if original_top_tok == other_tok:
    #     print("Original top token and other top token are the same, skipping...")
    #     return
    other_tok_str = tokenizer.decode([other_tok])
    logging.info(f"Swapping with other token `{other_tok_str}`")
    # top0 is different than what was sampled
    # truncate it and evaluate with and without swapping (in the unbiased context)
    # if we get a different answer, we've found a swap
    trunc_cot_toks = original_cot[:seq_pos]
    (resp_original, answer_original), (resp_swapped, answer_swapped) = (
        get_resp_answer_original_swapped(
            model,
            tokenizer,
            context_toks=original_ctx_toks,
            trunc_cot_toks=trunc_cot_toks,
            original_tok=original_cot_tok,
            swapped_tok=other_tok,
            unbiased_context_toks=unbiased_ctx_toks,
        )
    )
    resp_original_str = tokenizer.decode(resp_original)
    resp_swapped_str = tokenizer.decode(resp_swapped)
    if answer_original != original_expected_answer:
        logging.info("Original response didn't match expected answer, skipping...")
        logging.debug(f"original response:\n`{resp_original_str}`")
        return False
    if answer_swapped == "other":
        logging.info("Swapped response didn't result in an answer, skipping...")
        logging.debug(f"swapped response:\n`{resp_swapped_str}`")
        return False
    if answer_original == answer_swapped:
        logging.info("Swapping didn't change the answer, skipping...")
        logging.debug(f"original response:\n`{resp_original_str}`")
        logging.debug(f"swapped response:\n`{resp_swapped_str}`")
        return False
    logging.info("truncated cot:")
    logging.info(tokenizer.decode(trunc_cot_toks))
    logging.info("###")
    logging.info(f"original answer: {answer_original}")
    logging.info(f"`{resp_original_str}`")
    logging.info("###")
    logging.info(f"swapped answer: {answer_swapped}")
    logging.info(f"`{resp_swapped_str}`")
    return True


def try_swap_position(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    original_ctx_toks: list[int],
    unbiased_ctx_toks: list[int],
    original_cot: list[int],
    original_expected_answer: Literal["yes", "no"],
    probs_other_orig_diff: Float[torch.Tensor, " seq vocab"],
    seq_pos: int,
    topk_tok: int,
    prob_diff_threshold: float,
) -> tuple[int, float] | None:
    this_diff = probs_other_orig_diff[seq_pos]
    diff_abv_thresh_mask = (this_diff > prob_diff_threshold).cuda()
    if not diff_abv_thresh_mask.any():
        logging.info("All prob diffs are below threshold, skipping...")
        return None
    logging.info(
        f"Found {diff_abv_thresh_mask.sum().item()} other tokens above threshold"
    )
    vocab_size = len(this_diff)
    other_tokens = torch.arange(vocab_size).cuda()[diff_abv_thresh_mask]
    other_tokens_prob_diffs = this_diff[diff_abv_thresh_mask]
    sorted_indices = torch.argsort(other_tokens_prob_diffs, descending=True)
    sorted_other_tokens = other_tokens[sorted_indices].tolist()[:topk_tok]
    logging.info(f"Trying {len(sorted_other_tokens)} other tokens")
    for other_tok in sorted_other_tokens:
        if try_swap_token(
            model,
            tokenizer,
            original_ctx_toks=original_ctx_toks,
            unbiased_ctx_toks=unbiased_ctx_toks,
            original_cot=original_cot,
            original_expected_answer=original_expected_answer,
            other_tok=other_tok,
            seq_pos=seq_pos,
        ):
            prob_diff = this_diff[other_tok].item()
            return other_tok, prob_diff
    return None


def get_logits(
    model: PreTrainedModel, prompt_toks: list[int], q_toks: list[int]
) -> torch.Tensor:
    with torch.inference_mode():
        tok_tensor = torch.tensor(prompt_toks + q_toks).unsqueeze(0).to("cuda")
        logits = model(tok_tensor).logits
        return logits[0, len(prompt_toks) - 1 : -1]


def process_cot(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    original_cot: list[int],
    original_ctx_toks: list[int],
    original_expected_answer: str,
    other_ctx_toks: list[int],
    unbiased_ctx_toks: list[int],
    topk_pos: int,
    topk_tok: int,
    prob_diff_threshold: float,
) -> SingleCotSwapResult | None:
    best_prob_diff = 0.0
    best_swap_tok = None
    best_seq_pos = None
    original_logits = get_logits(model, original_ctx_toks, original_cot)
    other_logits = get_logits(model, other_ctx_toks, original_cot)
    original_probs = torch.softmax(original_logits, dim=-1)
    other_probs = torch.softmax(other_logits, dim=-1)
    probs_other_orig_diff = other_probs - original_probs
    max_probs_other_orig_diff = probs_other_orig_diff.max(dim=-1).values
    for seq_pos in max_probs_other_orig_diff.topk(k=topk_pos).indices.tolist():
        if max_probs_other_orig_diff[seq_pos] < prob_diff_threshold:
            break
        swap_result = try_swap_position(
            model,
            tokenizer,
            original_ctx_toks=original_ctx_toks,
            unbiased_ctx_toks=unbiased_ctx_toks,
            original_cot=original_cot,
            original_expected_answer=original_expected_answer,
            probs_other_orig_diff=probs_other_orig_diff,
            seq_pos=seq_pos,
            prob_diff_threshold=prob_diff_threshold,
            topk_tok=topk_tok,
        )
        if swap_result is None:
            continue
        swap_tok, prob_diff = swap_result
        if prob_diff > best_prob_diff:
            best_prob_diff = prob_diff
            best_swap_tok = swap_tok
            best_seq_pos = seq_pos
    if best_swap_tok is None:
        return None
    return SingleCotSwapResult(
        seq_pos=best_seq_pos, swap_token=best_swap_tok, prob_diff=best_prob_diff
    )


def process_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    unbiased_prompt_str: str,
    bias_no_prompt_str: str,
    responses_by_answer_by_ctx: dict[str, dict[str, list[list[int]]]],
    topk_pos: int,
    topk_tok: int,
    prob_diff_threshold: float,
    debug: bool,
) -> QuestionSwapResults:
    unbiased_ctx_toks = tokenizer.encode(unbiased_prompt_str)
    bias_no_ctx_toks = tokenizer.encode(bias_no_prompt_str)

    # swaying from unfaithful to faithful
    unf_resps = responses_by_answer_by_ctx["bias_no"]["no"]
    if debug:
        unf_resps = unf_resps[:2]
    unf_to_fai_swaps = []
    for unf_resp in unf_resps:
        result = process_cot(
            model,
            tokenizer,
            original_cot=unf_resp,
            original_ctx_toks=bias_no_ctx_toks,
            original_expected_answer="no",
            other_ctx_toks=unbiased_ctx_toks,
            unbiased_ctx_toks=unbiased_ctx_toks,
            topk_pos=topk_pos,
            topk_tok=topk_tok,
            prob_diff_threshold=prob_diff_threshold,
        )
        unf_to_fai_swaps.append(result)

    # swaying from faithful to unfaithful
    fai_resps = responses_by_answer_by_ctx["unb"]["yes"]
    if debug:
        fai_resps = fai_resps[:2]
    fai_to_unf_swaps = []
    for fai_resp in fai_resps:
        result = process_cot(
            model,
            tokenizer,
            original_cot=fai_resp,
            original_ctx_toks=unbiased_ctx_toks,
            original_expected_answer="yes",
            other_ctx_toks=bias_no_ctx_toks,
            unbiased_ctx_toks=unbiased_ctx_toks,
            topk_pos=topk_pos,
            topk_tok=topk_tok,
            prob_diff_threshold=prob_diff_threshold,
        )
        fai_to_unf_swaps.append(result)

    return QuestionSwapResults(
        unfai_to_fai_swaps=unf_to_fai_swaps, fai_to_unfai_swaps=fai_to_unf_swaps
    )


def extract_str_question(prompt_str: str) -> str:
    substr = "Question: "
    idx = prompt_str.rfind(substr)
    return prompt_str[idx:]


def process_successful_swaps_single_q_single_dir(
    seen_responses: set[tuple[int, ...]],
    unb_prompt: list[int],
    bias_no_prompt: list[int],
    responses: list[list[int]],
    swaps: list[SingleCotSwapResult | None],
    swap_dir: Literal["fai_to_unfai", "unfai_to_fai"],
) -> list[SuccessfulSwap]:
    ret = []
    seen_swap_toks = set()
    for resp, swap in zip(responses, swaps):
        if swap is None or tuple(resp) in seen_responses:
            continue
        swap_seq_pos = swap.seq_pos
        original_tok = resp[swap_seq_pos]
        swap_tok = swap.swap_token
        if (original_tok, swap_tok) in seen_swap_toks:
            continue
        seen_responses.add(tuple(resp))
        seen_swap_toks.add((original_tok, swap_tok))
        prob_diff = swap.prob_diff
        fai_tok = original_tok if swap_dir == "fai_to_unfai" else swap_tok
        unfai_tok = swap_tok if swap_dir == "fai_to_unfai" else original_tok
        successful_swap = SuccessfulSwap(
            unb_prompt=unb_prompt,
            bia_prompt=bias_no_prompt,
            trunc_cot=resp[:swap_seq_pos],
            fai_tok=fai_tok,
            unfai_tok=unfai_tok,
            swap_dir=swap_dir,
            prob_diff=prob_diff,
        )
        ret.append(successful_swap)
    return ret


def process_successful_swaps_single_q(
    q_idx: int,
    swap_results: QuestionSwapResults,
    all_combinations: list[dict[str, str]],
    responses_by_answer_by_ctx_by_q: list[dict[str, dict[str, list[list[int]]]]],
    tokenizer: PreTrainedTokenizerBase,
) -> list[SuccessfulSwap]:
    combined_prompts = all_combinations[q_idx]
    responses_by_answer_by_ctx = responses_by_answer_by_ctx_by_q[q_idx]
    fai_responses = responses_by_answer_by_ctx["unb"]["yes"]
    unfai_responses = responses_by_answer_by_ctx["bias_no"]["no"]

    fai_to_unfai_swaps = swap_results.fai_to_unfai_swaps
    unfai_to_fai_swaps = swap_results.unfai_to_fai_swaps

    unb_prompt_str = combined_prompts["unb_yes"]
    bias_no_prompt_str = combined_prompts["no_yes"]
    unb_prompt = tokenizer.encode(unb_prompt_str)
    bias_no_prompt = tokenizer.encode(bias_no_prompt_str)

    assert extract_str_question(unb_prompt_str) == extract_str_question(
        bias_no_prompt_str
    )

    seen_responses = set()
    successful_swaps = process_successful_swaps_single_q_single_dir(
        seen_responses=seen_responses,
        unb_prompt=unb_prompt,
        bias_no_prompt=bias_no_prompt,
        responses=fai_responses,
        swaps=fai_to_unfai_swaps,
        swap_dir="fai_to_unfai",
    ) + process_successful_swaps_single_q_single_dir(
        seen_responses=seen_responses,
        unb_prompt=unb_prompt,
        bias_no_prompt=bias_no_prompt,
        responses=unfai_responses,
        swaps=unfai_to_fai_swaps,
        swap_dir="unfai_to_fai",
    )

    return successful_swaps


def process_successful_swaps(
    responses_path: Path,
    swap_results_path: Path,
    tokenizer: PreTrainedTokenizerBase,
    seed_i: int,
) -> list[list[SuccessfulSwap]]:
    with open(responses_path, "rb") as f:
        responses_by_seed = pickle.load(f)
    seed = list(responses_by_seed.keys())[seed_i]
    responses_by_answer_by_ctx_by_q = responses_by_seed[seed]
    all_combinations = generate_all_combinations(seed=seed)

    with open(swap_results_path, "rb") as f:
        swap_results_by_q = pickle.load(f)

    successful_swaps_by_q = []

    for q_idx, swap_results in enumerate(swap_results_by_q):
        successful_swaps = process_successful_swaps_single_q(
            q_idx=q_idx,
            swap_results=swap_results,
            all_combinations=all_combinations,
            responses_by_answer_by_ctx_by_q=responses_by_answer_by_ctx_by_q,
            tokenizer=tokenizer,
        )
        successful_swaps_by_q.append(successful_swaps)

    return successful_swaps_by_q
