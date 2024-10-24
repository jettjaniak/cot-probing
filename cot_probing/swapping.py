import logging

from cot_probing.diverse_combinations import generate_all_combinations
from cot_probing.generation import categorize_response
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
    other_tokens = torch.arange(len(this_diff)).cuda()[diff_abv_thresh_mask]
    sorted_indices = torch.argsort(this_diff[diff_abv_thresh_mask], descending=True)
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
