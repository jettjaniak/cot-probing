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
    unfai_to_fai_swaps: list[SingleCotSwapResult]
    fai_to_unfai_swaps: list[SingleCotSwapResult]


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
    print(f"Trying to swap original CoT token `{original_tok_str}`")
    if original_cot_tok == other_tok:
        print("Original CoT token and other token are the same, skipping...")
        return False
    # if original_top_tok == other_tok:
    #     print("Original top token and other top token are the same, skipping...")
    #     return
    other_tok_str = tokenizer.decode([other_tok])
    print(f"Swapping with other token `{other_tok_str}`")
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
        print("Original response didn't match expected answer, skipping...")
        print(f"original response:\n`{resp_original_str}`")
        return False
    if answer_swapped == "other":
        print("Swapped response didn't result in an answer, skipping...")
        print(f"swapped response:\n`{resp_swapped_str}`")
        return False
    if answer_original == answer_swapped:
        print("Swapping didn't change the answer, skipping...")
        print(f"original response:\n`{resp_original_str}`")
        print(f"swapped response:\n`{resp_swapped_str}`")
        return False
    print("truncated cot:")
    print(tokenizer.decode(trunc_cot_toks))
    print("###")
    print(f"original answer: {answer_original}")
    print(f"`{resp_original_str}`")
    print("###")
    print(f"swapped answer: {answer_swapped}")
    print(f"`{resp_swapped_str}`")
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
    prob_diff_threshold: float,
) -> tuple[int, float] | None:
    this_diff = probs_other_orig_diff[seq_pos]
    diff_abv_thresh_mask = (this_diff > prob_diff_threshold).cuda()
    if not diff_abv_thresh_mask.any():
        print("All prob diffs are below threshold, skipping...")
        return None
    print(f"Trying {diff_abv_thresh_mask.sum().item()} other tokens")
    other_tokens = torch.arange(len(this_diff)).cuda()[diff_abv_thresh_mask]
    sorted_indices = torch.argsort(this_diff[diff_abv_thresh_mask], descending=True)
    sorted_other_tokens = other_tokens[sorted_indices].tolist()

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
    probs_other_orig_diff: Float[torch.Tensor, " seq vocab"],
    unbiased_ctx_toks: list[int],
    topk: int,
    prob_diff_threshold: float,
) -> SingleCotSwapResult | None:
    best_prob_diff = 0.0
    best_swap_tok = None
    best_seq_pos = None
    max_probs_other_orig_diff = probs_other_orig_diff.max(dim=-1).values
    for seq_pos in max_probs_other_orig_diff.topk(k=topk).indices.tolist():
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
    topk: int,
    prob_diff_threshold: float,
) -> QuestionSwapResults:
    unbiased_prompt = tokenizer.encode(unbiased_prompt_str)
    bias_no_prompt = tokenizer.encode(bias_no_prompt_str)
    unbiased_logits = get_logits(model, unbiased_prompt, fai_resp)
    bias_no_logits = get_logits(model, bias_no_prompt, fai_resp)
    unbiased_probs = torch.softmax(unbiased_logits, dim=-1)
    bias_no_probs = torch.softmax(bias_no_logits, dim=-1)

    # swaying from unfaithful to faithful
    unf_resps = responses_by_answer_by_ctx["bias_no"]["no"]
    unf_to_fai_swaps = []
    for unf_resp in unf_resps:
        result = process_cot(
            model,
            tokenizer,
            original_cot=unf_resp,
            original_ctx_toks=bias_no_prompt,
            original_expected_answer="no",
            probs_other_orig_diff=unbiased_probs - bias_no_probs,
            unbiased_ctx_toks=unbiased_prompt,
            topk=topk,
            prob_diff_threshold=prob_diff_threshold,
        )
        unf_to_fai_swaps.append(result)

    # swaying from faithful to unfaithful
    fai_resps = responses_by_answer_by_ctx["unb"]["yes"]
    fai_to_unf_swaps = []
    for fai_resp in fai_resps:
        result = process_cot(
            model,
            tokenizer,
            original_cot=fai_resp,
            original_ctx_toks=unbiased_prompt,
            original_expected_answer="yes",
            probs_other_orig_diff=bias_no_probs - unbiased_probs,
            unbiased_ctx_toks=bias_no_prompt,
            topk=topk,
            prob_diff_threshold=prob_diff_threshold,
        )
        fai_to_unf_swaps.append(result)

    return QuestionSwapResults(
        unfai_to_fai_swaps=unf_to_fai_swaps, fai_to_unfai_swaps=fai_to_unf_swaps
    )
