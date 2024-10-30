#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cot_probing import DATA_DIR
from cot_probing.swapping import SuccessfulSwap
from cot_probing.typing import *


def process_swaps(
    swaps_by_q: list[list[SuccessfulSwap]],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    layer_batch_size: int,
    output_path: Path,
):
    n_layers = model.config.num_hidden_layers
    q_and_cot_patch_results_by_swap_by_q = []
    for swaps in tqdm(swaps_by_q, desc="Questions"):
        q_and_cot_patch_results_by_swap_by_q.append([])
        q_and_cot_patch_results_by_swap = q_and_cot_patch_results_by_swap_by_q[-1]
        seen_toks = set()
        for swap in tqdm(swaps, desc="Swaps", leave=False):
            if (swap.fai_tok, swap.unfai_tok) in seen_toks:
                q_and_cot_patch_results_by_swap.append(None)
                continue
            seen_toks.add((swap.fai_tok, swap.unfai_tok))
            last_q_pos = swap.get_last_q_pos(tokenizer)
            len_input_ids = len(swap.get_input_ids_bia())
            assert len_input_ids == len(swap.get_input_ids_unb())
            pos_by_layer_cache = {
                layer: list(range(last_q_pos, len_input_ids))
                for layer in range(n_layers + 1)
            }
            cache = swap.get_cache(model, pos_by_layer_cache)
            q_and_cot_patch_results = (
                swap.patch_q_and_cot_groups_all_layers_batch(
                    model, tokenizer, cache, layer_batch_size
                )
                if cache is not None
                else None
            )
            q_and_cot_patch_results_by_swap.append(q_and_cot_patch_results)

            # Update the file after processing each swap
            with open(output_path, "wb") as f:
                pickle.dump(q_and_cot_patch_results_by_swap_by_q, f)


def main(args):
    model_id = f"hugging-quants/Meta-Llama-3.1-{args.model_size}B-BNB-NF4-BF16"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="cuda",
    )
    # get rid of the warning early
    model(torch.tensor([[tokenizer.bos_token_id]]).cuda())
    with open(args.swaps_path, "rb") as f:
        swaps_dict = pickle.load(f)
    swaps_dicts_list = swaps_dict["qs"]
    swaps_by_q = [swap_dict["swaps"] for swap_dict in swaps_dicts_list]

    output_path = (
        DATA_DIR
        / f"patch_new_res_{args.model_size}B_LB{args.layer_batch_size}__{args.swaps_path.stem}.pkl"
    )
    process_swaps(
        swaps_by_q,
        model,
        tokenizer,
        args.layer_batch_size,
        output_path,
    )

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process FSP patches for successful swaps"
    )
    parser.add_argument(
        "-s",
        "--swaps-path",
        type=Path,
        help="Path to the swap results pickle file",
    )
    parser.add_argument(
        "-m", "--model-size", type=int, help="Model size in billions of parameters"
    )
    parser.add_argument(
        "-l", "--layer-batch-size", type=int, help="Batch size for processing layers"
    )
    parser.add_argument("--slice-idx", type=int, help="Index of the slice to patch")
    args = parser.parse_args()

    main(args)
