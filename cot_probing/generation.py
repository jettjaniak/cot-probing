import random

import numpy as np
import torch
import tqdm

from cot_probing.typing import *
from cot_probing.utils import setup_determinism


def hf_generate_many(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_toks: list[int],
    max_new_tokens: int,
    temp: float,
    n_gen: int,
    seed: int,
    do_sample: bool,
) -> list[list[int]]:
    prompt_len = len(prompt_toks)
    setup_determinism(seed)
    responses_tensor = model.generate(
        torch.tensor([prompt_toks]).cuda(),
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer,
        do_sample=do_sample,
        temperature=temp,
        num_return_sequences=n_gen,
        stop_strings=["Answer:"],
    )[:, prompt_len:]
    ret = []
    for response_toks in responses_tensor:
        response_toks = response_toks.tolist()
        if tokenizer.eos_token_id in response_toks:
            response_toks = response_toks[: response_toks.index(tokenizer.eos_token_id)]
        ret.append(response_toks)
    return ret


def categorize_response(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    unbiased_context_toks: list[int],
    response: list[int],
) -> Literal["yes", "no", "other"]:
    yes_tok_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_tok_id = tokenizer.encode(" No", add_special_tokens=False)[0]
    answer_toks = tokenizer.encode("Answer:", add_special_tokens=False)
    assert len(answer_toks) == 2

    if response[-2:] != answer_toks:
        # Last two tokens were not "Answer:"
        return "other"

    # response_str = tokenizer.decode(response)
    # print(f"Categorizing response: `{response_str}`")
    full_prompt = unbiased_context_toks + response
    logits = model(torch.tensor([full_prompt]).cuda()).logits[0, -1]
    yes_logit = logits[yes_tok_id].item()
    no_logit = logits[no_tok_id].item()
    if yes_logit >= no_logit:
        return "yes"
    else:
        return "no"


def categorize_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    unbiased_context_toks: list[int],
    responses: list[list[int]],
) -> dict[str, list[list[int]]]:
    ret = {"yes": [], "no": [], "other": []}
    for response in responses:
        category = categorize_response(
            model,
            tokenizer,
            unbiased_context_toks,
            response,
        )
        ret[category].append(response)
    return ret


def analyze_responses_single_question(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    combined_prompts: dict[str, str],
    max_new_tokens: int,
    temp: float,
    n_gen: int,
    seed: int,
    do_sample: bool,
):
    prompt_unb = combined_prompts["unb_yes"]
    prompt_no = combined_prompts["no_yes"]
    question = prompt_unb.rsplit("Question:", 1)[-1][1:]
    print("###")
    print(question)
    prompt_toks_unb = tokenizer.encode(prompt_unb)
    prompt_toks_no = tokenizer.encode(prompt_no)
    resp_unb = hf_generate_many(
        model,
        tokenizer,
        prompt_toks_unb,
        max_new_tokens=max_new_tokens,
        temp=temp,
        n_gen=n_gen,
        seed=seed,
        do_sample=do_sample,
    )
    resp_no = hf_generate_many(
        model,
        tokenizer,
        prompt_toks_no,
        max_new_tokens=max_new_tokens,
        temp=temp,
        n_gen=n_gen,
        seed=seed,
        do_sample=do_sample,
    )
    res = {
        "unb": categorize_responses(model, tokenizer, prompt_toks_unb, resp_unb),
        "bias_no": categorize_responses(model, tokenizer, prompt_toks_unb, resp_no),
    }
    for variant in ["unb", "bias_no"]:
        print(f"{variant=}")
        for key in ["yes", "no", "other"]:
            print(f"- {key} {len(res[variant][key])}")
    return res


def analyze_responses(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    all_combinations: list[dict[str, str]],
    max_new_tokens: int,
    temp: float,
    n_gen: int,
    seed: int,
    do_sample: bool,
):
    results = []
    for i, combined_prompts in tqdm.tqdm(enumerate(all_combinations), desc="Questions"):
        res = analyze_responses_single_question(
            model,
            tokenizer,
            combined_prompts,
            max_new_tokens,
            temp,
            n_gen,
            seed,
            do_sample,
        )
        results.append(res)
    return results
