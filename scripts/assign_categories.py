#!/usr/bin/env python3
import argparse
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from cot_probing import DATA_DIR
from cot_probing.typing import *


@dataclass
class GroupingThresholds:
    almost_zero: float = 0.25
    ratio_threshold: float = 1.5
    strong_ratio: float = 2.0
    last_three_ratio: float = 4.0


def get_category_name(values: list[float], thresholds: GroupingThresholds) -> str:
    assert len(values) == 6
    # Check if first three groups have significant values
    max_first_three_groups = max(values[:3])
    if max_first_three_groups >= thresholds.almost_zero:
        return "other"

    ltsbs_score = values[3]
    reasoning_score = values[4]
    last_three_score = values[5]
    ltsbs_reasoning_max = max(ltsbs_score, reasoning_score)

    # Check if only last three is important
    if ltsbs_reasoning_max < last_three_score / thresholds.last_three_ratio:
        return "only_last_three"

    # Check if ltsbs is strongly dominant
    if ltsbs_score > reasoning_score * thresholds.strong_ratio:
        return "ltsbs"

    # Check if reasoning is strongly dominant
    if reasoning_score > ltsbs_score * thresholds.strong_ratio:
        return "reasoning"

    # Check if both are similarly important
    if (
        ltsbs_reasoning_max / min(ltsbs_score, reasoning_score)
        < thresholds.ratio_threshold
    ):
        return "both"

    return "other"


def main(args):

    # Load data
    with open(args.swaps_path, "rb") as f:
        swaps_dict = pickle.load(f)
    swaps_dicts_list = swaps_dict["qs"]
    swaps_by_q = [swap_dict["swaps"] for swap_dict in swaps_dicts_list]

    with open(args.patch_results_path, "rb") as f:
        patch_results_by_swap_by_q = pickle.load(f)

    # Assign groups
    thresholds = GroupingThresholds(
        almost_zero=args.almost_zero_threshold,
        ratio_threshold=args.ratio_threshold,
        strong_ratio=args.strong_ratio_threshold,
        last_three_ratio=args.last_three_ratio_threshold,
    )
    categories = defaultdict(list)
    # Process values
    for q_idx, (swaps, patch_results_by_swap) in enumerate(
        zip(swaps_by_q, patch_results_by_swap_by_q)
    ):
        for swap_idx, plp_by_layers in enumerate(patch_results_by_swap):
            if plp_by_layers is None:
                continue
            assert len(plp_by_layers) == 1
            # Get the only layer group
            plp_by_group = next(iter(plp_by_layers.values()))
            attr = f"{args.prob_or_logit}_diff_change_{args.direction}"
            values = [abs(getattr(plp, attr)) for plp in plp_by_group.values()]
            vmax = max(values)
            values = [v / vmax for v in values]
            category_name = get_category_name(values, thresholds)
            categories[category_name].append((q_idx, swap_idx))

    # Print statistics
    print("\nCategory sizes:")
    for category_name, members in categories.items():
        print(f"{category_name}: {len(members)}")

    output_path = (
        DATA_DIR
        / f"categories_{args.prob_or_logit}_{args.direction}_{args.almost_zero_threshold}_{args.ratio_threshold}_{args.strong_ratio_threshold}_{args.last_three_ratio_threshold}.pkl"
    )
    with open(output_path, "wb") as f:
        pickle.dump(categories, f)

    print(f"\nResults saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--swaps-path", "-s", type=Path, required=True)
    parser.add_argument("--patch-results-path", "-p", type=Path, required=True)
    parser.add_argument(
        "--prob-or-logit", type=str, choices=["prob", "logit"], default="prob"
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["bia_to_unb", "unb_to_bia"],
        default="bia_to_unb",
    )
    parser.add_argument("--almost-zero-threshold", type=float, default=0.25)
    parser.add_argument("--ratio-threshold", type=float, default=1.5)
    parser.add_argument("--strong-ratio-threshold", type=float, default=2.0)
    parser.add_argument("--last-three-ratio-threshold", type=float, default=4.0)
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
