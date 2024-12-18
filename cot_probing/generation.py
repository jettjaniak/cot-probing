import argparse
import logging

import torch
from beartype import beartype

from cot_probing.qs_generation import Question
from cot_probing.typing import *
from cot_probing.utils import (
    conversation_to_str_prompt,
    make_chat_prompt,
    setup_determinism,
)


@dataclass
class UnbiasedCotGeneration:
    cots_by_qid: dict[str, list[list[int]]]
    model: str
    fsp_size: int
    seed: int
    max_new_tokens: int
    temp: float
    do_sample: bool
    unb_fsp_toks: list[int]


@dataclass
class BiasedCotGeneration:
    cots_by_qid: dict[str, list[list[int]]]
    model: str
    fsp_size: int
    seed: int
    max_new_tokens: int
    temp: float
    do_sample: bool
    bia_yes_fsp: Any
    bia_no_fsp: Any


def generate_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_toks: list[int],
    max_new_tokens: int,
    temp: float,
    n_gen: int,
    seed: int,
    do_sample: bool,
    verbose: bool = False,
) -> list[list[int]]:
    prompt_len = len(prompt_toks)
    setup_determinism(seed)
    answer_yes_toks = tokenizer.encode("Answer: Yes", add_special_tokens=False)
    answer_no_toks = tokenizer.encode("Answer: No", add_special_tokens=False)
    assert len(answer_yes_toks) == len(answer_no_toks) == 3

    responses_tensor = model.generate(
        torch.tensor([prompt_toks]).cuda(),
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        tokenizer=tokenizer,
        do_sample=do_sample,
        temperature=temp,
        num_return_sequences=n_gen,
        use_cache=True,
        stop_strings=["Answer: Yes", "Answer: No"],
    )[:, prompt_len:]

    ret = []
    for response_toks in responses_tensor:
        response_toks = response_toks.tolist()
        if tokenizer.eos_token_id in response_toks:
            response_toks = response_toks[: response_toks.index(tokenizer.eos_token_id)]

        response_str = tokenizer.decode(response_toks)
        if response_toks[-3:] not in [answer_yes_toks, answer_no_toks]:
            logging.warning(
                f"Generated completion does not end in 'Answer: Yes' or 'Answer: No': `{response_str}`"
            )
            continue

        if "Question: " in response_str:
            logging.warning(
                f"Generated completion contains 'Question: ': `{response_str}`"
            )
            continue

        if verbose:
            logging.info(f"Generated completion: `{response_str}`")

        # drop "\nAnswer: Yes/No"
        ret.append(response_toks[:-4])

    return ret


def generate_completions_chat(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    prompt_toks: list[int],
    max_new_tokens: int,
    temp: float,
    n_gen: int,
    seed: int,
    do_sample: bool,
    verbose: bool = False,
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
        use_cache=True,
    )[:, prompt_len:]

    ret = []
    for response_toks in responses_tensor:
        response_toks = response_toks.tolist()
        if tokenizer.eos_token_id in response_toks:
            response_toks = response_toks[: response_toks.index(tokenizer.eos_token_id)]

        if len(response_toks) == max_new_tokens:
            logging.info(
                f"Generated completion is too long: `{tokenizer.decode(response_toks)}`"
            )
            continue

        if verbose:
            response_str = tokenizer.decode(response_toks)
            logging.info(f"Generated completion: `{response_str}`")

        ret.append(response_toks)

    return ret


def gen_unb_cots(
    q: Question,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    unb_fsp_toks: list[int],
    args: argparse.Namespace,
    verbose: bool = False,
) -> list[list[int]]:
    question_toks = tokenizer.encode(
        q.with_step_by_step_suffix(), add_special_tokens=False
    )

    prompt_toks = unb_fsp_toks + question_toks
    return generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompt_toks=prompt_toks,
        max_new_tokens=args.max_new_tokens,
        n_gen=args.n_gen,
        temp=args.temp,
        seed=args.seed,
        do_sample=True,
        verbose=verbose,
    )


def gen_unb_cots_chat(
    q: Question,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    args: argparse.Namespace,
    verbose: bool = False,
) -> list[list[int]]:

    instruction = f"""Here is a question that requires a few steps of reasoning before giving an answer, but it has a clear yes or no answer:

{q.question}

Take it seriously, and think step by step before answering it."""
    chat_prompt = make_chat_prompt(
        instruction=instruction,
        tokenizer=tokenizer,
    )
    prompt_toks = tokenizer.encode(chat_prompt, add_special_tokens=False)

    return generate_completions_chat(
        model=model,
        tokenizer=tokenizer,
        prompt_toks=prompt_toks,
        max_new_tokens=args.max_new_tokens,
        n_gen=args.n_gen,
        temp=args.temp,
        seed=args.seed,
        do_sample=True,
        verbose=verbose,
    )


def gen_bia_cots_chat(
    q: Question,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    bia_fsps: list[dict[str, str]],
    args: argparse.Namespace,
    verbose: bool = False,
) -> list[list[int]]:

    conversation = bia_fsps + [{"role": "user", "content": q.question}]
    str_prompt = conversation_to_str_prompt(conversation, tokenizer)
    prompt_toks = tokenizer.encode(str_prompt, add_special_tokens=False)

    return generate_completions_chat(
        model=model,
        tokenizer=tokenizer,
        prompt_toks=prompt_toks,
        max_new_tokens=args.max_new_tokens,
        n_gen=args.n_gen,
        temp=args.temp,
        seed=args.seed,
        do_sample=True,
        verbose=verbose,
    )


def gen_bia_cots(
    q: Question,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    bia_fsp_toks: list[int],
    args: argparse.Namespace,
    verbose: bool = False,
) -> list[list[int]]:
    question_toks = tokenizer.encode(
        q.with_step_by_step_suffix(), add_special_tokens=False
    )

    prompt_toks = bia_fsp_toks + question_toks
    if verbose:
        logging.info(
            f"Generating biased CoTs for prompt:\n{tokenizer.decode(prompt_toks)}\n\n"
        )

    return generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompt_toks=prompt_toks,
        max_new_tokens=args.max_new_tokens,
        n_gen=args.n_gen,
        temp=args.temp,
        seed=args.seed,
        do_sample=True,
        verbose=verbose,
    )


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
