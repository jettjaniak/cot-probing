#!/usr/bin/env python3
import argparse
import json
import logging
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing import DATA_DIR
from cot_probing.diverse_combinations import load_and_process_file
from cot_probing.generation import categorize_response as categorize_response
from cot_probing.typing import *


def generate_cots(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    fsp_toks: list[int],
    question_toks: list[int],
    max_new_tokens: int,
    n_gen: int,
    temp: float,
) -> list[list[int]]:
    # TODO: batching, cache
    # TODO: use hf_generate_many
    # TODO: make sure we return n_gen responses
    input_ids_toks = fsp_toks + question_toks
    prompt_len = len(input_ids_toks)
    with torch.inference_mode():
        output = model.generate(
            torch.tensor([input_ids_toks]).to(model.device),
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            use_cache=True,
            num_return_sequences=n_gen,
            tokenizer=tokenizer,
            stop_strings=["Answer: Yes", "Answer: No"],
            pad_token_id=tokenizer.eos_token_id,
        )
        responses = output[:, prompt_len:].tolist()
    ret = []
    for response_toks in responses:
        if tokenizer.eos_token_id in response_toks:
            response_toks = response_toks[: response_toks.index(tokenizer.eos_token_id)]
        ret.append(response_toks)
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


def process_question(
    q: dict,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    unb_fsp_toks: list[int],
    yes_fsp_toks: list[int],
    no_fsp_toks: list[int],
    args: argparse.Namespace,
) -> None:
    question_toks = tokenizer.encode(q["question"], add_special_tokens=False)
    unb_cots = generate_cots(
        model=model,
        tokenizer=tokenizer,
        fsp_toks=unb_fsp_toks,
        question_toks=question_toks,
        max_new_tokens=args.max_new_tokens,
        n_gen=args.n_gen,
        temp=args.temp,
    )
    bia_fsp_toks = yes_fsp_toks if q["expected_answer"] == "yes" else no_fsp_toks
    biased_cots = generate_cots(
        model=model,
        tokenizer=tokenizer,
        fsp_toks=bia_fsp_toks,
        question_toks=question_toks,
        max_new_tokens=args.max_new_tokens,
        n_gen=args.n_gen,
        temp=args.temp,
    )
    q["n_gen"] = args.n_gen
    q["n_correct_unbiased"] = sum(
        categorize_response(model, tokenizer, unb_fsp_toks, cot) == q["expected_answer"]
        for cot in unb_cots
    )
    n_correct_bia = 0
    ret_biased_cots = []
    for biased_cot_tok in biased_cots:
        biased_cot_str = tokenizer.decode(biased_cot_tok, skip_special_tokens=True)
        answer = categorize_response(model, tokenizer, bia_fsp_toks, biased_cot_tok)
        n_correct_bia += answer == q["expected_answer"]
        ret_biased_cots.append(dict(cot=biased_cot_str, answer=answer))
    q["n_correct_biased"] = n_correct_bia
    q["biased_cots"] = ret_biased_cots
    # from the LLM-generated question file
    if "unbiased_responses" in q:
        del q["unbiased_responses"]
    if "unbiased_completion_accuracy" in q:
        del q["unbiased_completion_accuracy"]


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

    skip_args = ["verbose", "questions_path"]
    ret = dict(
        biased_no_fsp=no_fsps,
        biased_yes_fsp=yes_fsps,
        unbiased_fsp=unb_fsps,
        qs=qs,
        **{f"arg_{k}": v for k, v in vars(args).items() if k not in skip_args},
    )
    file_identifier = args.questions_path.stem.split("_")[-1]
    for q in tqdm(qs[:3]):  # TODO
        process_question(
            q=q,
            model=model,
            tokenizer=tokenizer,
            unb_fsp_toks=unb_fsp_toks,
            yes_fsp_toks=yes_fsp_toks,
            no_fsp_toks=no_fsp_toks,
            args=args,
        )
        with open(DATA_DIR / f"measured_qs_{file_identifier}.json", "a") as f:
            f.write(json.dumps(ret))


if __name__ == "__main__":
    main(parse_args())
