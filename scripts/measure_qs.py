#!/usr/bin/env python3
import argparse
import copy
import json
import logging
import random
from pathlib import Path

import torch
from beartype import beartype
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StaticCache

from cot_probing import DATA_DIR
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.typing import *


@beartype
def generate_cots(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    question_toks: list[int],
    fsp_toks: list[int],
    fsp_cache: tuple,
    max_new_tokens: int,
    n_gen: int,
    temp: float,
    seed: int,
) -> list[list[int]]:
    # TODO: batching
    # past_key_values = copy.deepcopy(fsp_cache)
    ret = []
    with torch.inference_mode():
        # for _ in range(n_gen):
        #     response = model.generate(
        #         input_ids=torch.tensor([fsp_toks + question_toks]).to("cuda"),
        #         max_new_tokens=max_new_tokens,
        #         do_sample=True,
        #         temperature=temp,
        #         tokenizer=tokenizer,
        #         stop_strings=["Answer:"],
        #         pad_token_id=tokenizer.eos_token_id,
        #         past_key_values=past_key_values,
        #     )[0, len(fsp_toks) :].tolist()
        #     if tokenizer.eos_token_id in response:
        #         response = response[: response.index(tokenizer.eos_token_id)]
        #     ret.append(response)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        responses = model.generate(
            input_ids=torch.tensor([fsp_toks + question_toks]).to("cuda"),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            tokenizer=tokenizer,
            stop_strings=["Answer:"],
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
            num_return_sequences=n_gen,
        )[:, len(fsp_toks) :].tolist()
        for response in responses:
            if tokenizer.eos_token_id in response:
                response = response[: response.index(tokenizer.eos_token_id)]
            ret.append(response)
    return ret


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate biased and unbiased CoTs and measure their accuracy"
    )
    parser.add_argument(
        "-m", "--model-size", type=int, help="Model size in billions of parameters"
    )
    parser.add_argument(
        "-q", "--questions-path", type=Path, help="Path to the questions JSON"
    )
    parser.add_argument("-s", "--seed", type=int, help="Random seed", default=0)
    parser.add_argument(
        "-f", "--fsp-size", type=int, help="Number of FSPs to use", default=16
    )
    parser.add_argument(
        "-t", "--temp", type=float, help="Temperature for generation", default=0.7
    )
    parser.add_argument(
        "-n", "--n-gen", type=int, help="Number of generations to produce", default=10
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        help="Maximum number of new tokens to generate",
        default=200,
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser.parse_args()


@beartype
def build_fsps(args: argparse.Namespace, seed: int) -> tuple[str, str, str]:
    n = args.fsp_size
    yes_fsps = load_and_process_file(DATA_DIR / "diverse_yes.txt")
    no_fsps = load_and_process_file(DATA_DIR / "diverse_no.txt")
    random.seed(seed)
    shuffled_indices = random.sample(range(len(yes_fsps)), n)
    yes_fsps = [yes_fsps[i] for i in shuffled_indices]
    no_fsps = [no_fsps[i] for i in shuffled_indices]

    unb_yes_idxs = random.sample(range(n), n // 2)
    unb_fsps = [yes_fsps[i] if i in unb_yes_idxs else no_fsps[i] for i in range(n)]

    unb_fsps = "\n\n".join(unb_fsps) + "\n\n"
    yes_fsps = "\n\n".join(yes_fsps) + "\n\n"
    no_fsps = "\n\n".join(no_fsps) + "\n\n"

    return unb_fsps, yes_fsps, no_fsps


def categorize_response_cache(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    unbiased_cache: StaticCache,
    response: list[int],
) -> Literal["yes", "no", "other"]:
    yes_tok_id = tokenizer.encode(" Yes", add_special_tokens=False)[0]
    no_tok_id = tokenizer.encode(" No", add_special_tokens=False)[0]
    answer_toks = tokenizer.encode("Answer:", add_special_tokens=False)
    assert len(answer_toks) == 2

    if response[-2:] != answer_toks:
        # Last two tokens were not "Answer:"
        return "other"

    logits = model(
        torch.tensor([response]).cuda(), past_key_values=unbiased_cache
    ).logits[0, -1]
    yes_logit = logits[yes_tok_id].item()
    no_logit = logits[no_tok_id].item()
    if yes_logit >= no_logit:
        return "yes"
    else:
        return "no"


@beartype
def process_question(
    q: dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    unb_fsp_toks: list[int],
    yes_fsp_toks: list[int],
    no_fsp_toks: list[int],
    unb_fsp_cache: tuple,
    yes_fsp_cache: tuple,
    no_fsp_cache: tuple,
    args: argparse.Namespace,
) -> None:
    substring = "\nLet's think step by step:\n-"
    question_str = q["question"]
    idx = question_str.find(substring)
    assert idx != -1
    question_str = question_str[: idx + len(substring)]
    q["question"] = question_str
    question_toks = tokenizer.encode(question_str, add_special_tokens=False)
    unb_cots = generate_cots(
        model=model,
        tokenizer=tokenizer,
        question_toks=question_toks,
        fsp_toks=unb_fsp_toks,
        fsp_cache=unb_fsp_cache,
        max_new_tokens=args.max_new_tokens,
        n_gen=args.n_gen,
        temp=args.temp,
        seed=args.seed,
    )
    bia_fsp_cache = no_fsp_cache if q["expected_answer"] == "yes" else yes_fsp_cache
    bia_fsp_toks = no_fsp_toks if q["expected_answer"] == "yes" else yes_fsp_toks
    biased_cots = generate_cots(
        model=model,
        tokenizer=tokenizer,
        question_toks=question_toks,
        fsp_toks=bia_fsp_toks,
        fsp_cache=bia_fsp_cache,
        max_new_tokens=args.max_new_tokens,
        n_gen=args.n_gen,
        temp=args.temp,
        seed=args.seed,
    )
    q["n_gen"] = args.n_gen

    n_correct_unb = 0
    ret_unbiased_cots = []
    for unb_cot_tok in unb_cots:
        unb_cot_str = tokenizer.decode(unb_cot_tok, skip_special_tokens=True)
        answer = categorize_response_cache(model, tokenizer, unb_fsp_cache, unb_cot_tok)
        n_correct_unb += answer == q["expected_answer"]
        ret_unbiased_cots.append(dict(cot=unb_cot_str, answer=answer))
    q["n_correct_unbiased"] = n_correct_unb
    q["unbiased_cots"] = ret_unbiased_cots

    n_correct_bia = 0
    ret_biased_cots = []
    for biased_cot_tok in biased_cots:
        biased_cot_str = tokenizer.decode(biased_cot_tok, skip_special_tokens=True)
        answer = categorize_response_cache(
            model, tokenizer, unb_fsp_cache, biased_cot_tok
        )
        n_correct_bia += answer == q["expected_answer"]
        ret_biased_cots.append(dict(cot=biased_cot_str, answer=answer))
    q["n_correct_biased"] = n_correct_bia
    q["biased_cots"] = ret_biased_cots

    # from the LLM-generated question file
    if "unbiased_responses" in q:
        del q["unbiased_responses"]
    if "unbiased_completion_accuracy" in q:
        del q["unbiased_completion_accuracy"]


def get_cache(model: PreTrainedModel, fsp_toks: list[int]) -> tuple:
    with torch.inference_mode():
        return model(
            torch.tensor([fsp_toks]).to("cuda"),
        ).past_key_values


def main(args: argparse.Namespace):
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    model_id = f"hugging-quants/Meta-Llama-3.1-{args.model_size}B-BNB-NF4-BF16"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    # get rid of the warnings early
    model(torch.tensor([[tokenizer.bos_token_id]]).cuda())
    model.generate(
        torch.tensor([[tokenizer.bos_token_id]]).cuda(),
        max_new_tokens=1,
        pad_token_id=tokenizer.eos_token_id,
    )

    with open(args.questions_path, "r") as f:
        qs = json.load(f)

    unb_fsps, yes_fsps, no_fsps = build_fsps(args, args.seed)
    unb_fsp_toks = tokenizer.encode(unb_fsps)
    yes_fsp_toks = tokenizer.encode(yes_fsps)
    no_fsp_toks = tokenizer.encode(no_fsps)
    print("Getting FSP caches")
    unb_fsp_cache = get_cache(model, unb_fsp_toks)
    yes_fsp_cache = get_cache(model, yes_fsp_toks)
    no_fsp_cache = get_cache(model, no_fsp_toks)

    skip_args = ["verbose", "questions_path"]
    ret = dict(
        biased_no_fsp=no_fsps,
        biased_yes_fsp=yes_fsps,
        unbiased_fsp=unb_fsps,
        qs=qs,
        **{f"arg_{k}": v for k, v in vars(args).items() if k not in skip_args},
    )
    file_identifier = args.questions_path.stem.split("_")[-1]
    for q in tqdm(qs, desc="Processing questions"):
        process_question(
            q=q,
            model=model,
            tokenizer=tokenizer,
            unb_fsp_toks=unb_fsp_toks,
            yes_fsp_toks=yes_fsp_toks,
            no_fsp_toks=no_fsp_toks,
            unb_fsp_cache=unb_fsp_cache,
            yes_fsp_cache=yes_fsp_cache,
            no_fsp_cache=no_fsp_cache,
            args=args,
        )
        with open(DATA_DIR / f"measured_qs_{file_identifier}.json", "w") as f:
            json.dump(ret, f)


if __name__ == "__main__":
    main(parse_args())
