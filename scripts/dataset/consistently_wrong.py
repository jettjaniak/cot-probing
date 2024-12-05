#!/usr/bin/env python3
import argparse
import curses
import os
import pickle

from openai import OpenAI

from cot_probing import DATA_DIR
from cot_probing.cot_evaluation import (
    LabeledCoTs,
    get_honesty_questions,
    get_justified_answer,
    get_subjective_score,
)
from cot_probing.qs_evaluation import NoCotAccuracy
from cot_probing.qs_generation import Question
from cot_probing.typing import *
from cot_probing.utils import is_chat_model, load_tokenizer

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def parse_args():
    parser = argparse.ArgumentParser(description="Generate biased CoTs")
    parser.add_argument("-d", "--dataset-id", type=str, default="strategyqa")
    parser.add_argument(
        "-m",
        "--model-id",
        type=str,
    )
    parser.add_argument("--max-p-correct", "-p", type=float, default=0.1)
    parser.add_argument("--max-cot-acc", "-a", type=float, default=0.2)
    return parser.parse_args()


MODELS_MAP = {
    "G": "google/gemma-2-2b-it",
    "L": "meta-llama/Llama-3.2-3B-Instruct",
}


def display_interface(
    stdscr: curses.window,
    qs_dataset: dict[str, Question],
    no_cot_acc: NoCotAccuracy,
    labeled_cots: LabeledCoTs,
    consistently_wrong_qids: set[str],
    tokenizer,
    cot_acc_by_qid: dict[str, float],
):

    curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)

    current_qid_idx = 0
    current_cot_idx = 0
    qids = sorted(consistently_wrong_qids)
    re_evaluated_justified_answer = None
    re_evaluated_raw_openai_answer = None
    honesty_questions = None
    subjective_score = None

    while True:
        stdscr.clear()
        qid = qids[current_qid_idx]
        question = qs_dataset[qid]
        wrong_cots = [
            cot
            for cot in labeled_cots.labeled_cots_by_qid[qid]
            if cot.justified_answer != question.expected_answer
        ]

        # Display question information
        stdscr.addstr(
            0,
            0,
            f"Question {current_qid_idx + 1}/{len(qids)} (ID: {qid})",
            curses.A_BOLD,
        )
        stdscr.addstr(2, 0, f"Question: {question.question}")
        stdscr.addstr(4, 0, f"Expected answer: ", curses.A_BOLD)
        stdscr.addstr(f"{question.expected_answer}", curses.color_pair(1))
        stdscr.addstr(5, 0, f"No CoT p_correct: {no_cot_acc.acc_by_qid[qid]:.0%}")
        stdscr.addstr(6, 0, f"CoT accuracy: {cot_acc_by_qid[qid]:.0%}")

        # Display current CoT
        if wrong_cots:
            stdscr.addstr(
                8,
                0,
                f"Wrong CoT {current_cot_idx + 1}/{len(wrong_cots)}:",
                curses.A_BOLD,
            )
            cot_string = tokenizer.decode(wrong_cots[current_cot_idx].cot)

            # Split into lines and handle wrapping
            max_y, max_x = stdscr.getmaxyx()
            y_pos = 9

            # Split by newlines first
            lines = cot_string.split("\n")
            for line in lines:
                # Then wrap each line to fit screen width
                while line and y_pos < max_y - 3:  # Leave space for instructions
                    # Calculate how much of the line can fit
                    available_width = max_x - 2
                    chunk = line[:available_width]

                    # If this isn't the end of the line, try to break at a space
                    if len(line) > available_width:
                        last_space = chunk.rfind(" ")
                        if last_space > 0:  # If we found a space
                            chunk = chunk[:last_space]
                            line = line[last_space + 1 :]
                        else:
                            line = line[available_width:]
                    else:
                        line = ""

                    try:
                        stdscr.addstr(y_pos, 0, chunk)
                    except curses.error:
                        pass  # Ignore if we try to write beyond screen bounds
                    y_pos += 1

            if re_evaluated_justified_answer is None:
                y_pos += 1
                stdscr.addstr(
                    y_pos,
                    0,
                    f"Justified answer: ",
                    curses.A_BOLD,
                )
                stdscr.addstr(
                    wrong_cots[current_cot_idx].justified_answer,
                )

            else:
                assert re_evaluated_raw_openai_answer is not None
                y_pos += 1
                stdscr.addstr(
                    y_pos,
                    0,
                    f"(Re-evaluated) Justified answer: ",
                    curses.A_BOLD,
                )
                stdscr.addstr(
                    re_evaluated_justified_answer,
                )
                y_pos += 1
                stdscr.addstr(
                    y_pos,
                    0,
                    f"Raw OpenAI answer: ",
                    curses.A_BOLD,
                )
                stdscr.addstr(
                    re_evaluated_raw_openai_answer,
                )

            if honesty_questions is not None:
                y_pos += 1
                stdscr.addstr(
                    y_pos,
                    0,
                    f"Honesty questions: ",
                    curses.A_BOLD,
                )
                stdscr.addstr("\n".join(honesty_questions))

            if subjective_score is not None:
                y_pos += 1
                stdscr.addstr(
                    y_pos,
                    0,
                    f"Subjective score: ",
                    curses.A_BOLD,
                )
                stdscr.addstr(str(subjective_score))

        # Display instructions
        stdscr.addstr(max_y - 2, 0, "Controls: ", curses.A_BOLD)
        stdscr.addstr(
            "← → (navigate CoTs) | ↑ ↓ (navigate questions) | e (re-eval just. ans.) | h (eval honest.) | s (eval subjective) | q (quit)",
            curses.color_pair(2),
        )

        stdscr.refresh()

        # Handle input
        key = stdscr.getch()
        if key == ord("q"):
            break
        elif key == curses.KEY_RIGHT and wrong_cots:
            current_cot_idx = (current_cot_idx + 1) % len(wrong_cots)
            re_evaluated_justified_answer = None
            re_evaluated_raw_openai_answer = None
            honesty_questions = None
            subjective_score = None
        elif key == curses.KEY_LEFT and wrong_cots:
            current_cot_idx = (current_cot_idx - 1) % len(wrong_cots)
            re_evaluated_justified_answer = None
            re_evaluated_raw_openai_answer = None
            honesty_questions = None
            subjective_score = None
        elif key == curses.KEY_DOWN:
            current_qid_idx = (current_qid_idx + 1) % len(qids)
            current_cot_idx = 0
            re_evaluated_justified_answer = None
            re_evaluated_raw_openai_answer = None
            honesty_questions = None
            subjective_score = None
        elif key == curses.KEY_UP:
            current_qid_idx = (current_qid_idx - 1) % len(qids)
            current_cot_idx = 0
            re_evaluated_justified_answer = None
            re_evaluated_raw_openai_answer = None
            honesty_questions = None
            subjective_score = None
        elif key == ord("e") and wrong_cots:
            # Re-evaluate the justified answer
            cot = wrong_cots[current_cot_idx]
            cot_str = tokenizer.decode(cot.cot)
            question = qs_dataset[qid]
            re_evaluated_justified_answer, re_evaluated_raw_openai_answer = (
                get_justified_answer(
                    q_str=question.question,
                    cot=cot_str,
                    openai_client=openai_client,
                    openai_model=labeled_cots.openai_model,
                    verbose=True,
                )
            )
        elif key == ord("h") and wrong_cots:
            # Re-evaluate the justified answer
            cot = wrong_cots[current_cot_idx]
            cot_str = tokenizer.decode(cot.cot)
            question = qs_dataset[qid]
            honesty_questions, _ = get_honesty_questions(
                q_str=question.question,
                cot=cot_str,
                openai_client=openai_client,
                openai_model=labeled_cots.openai_model,
                verbose=True,
            )
        elif key == ord("s") and wrong_cots:
            # Re-evaluate the justified answer
            cot = wrong_cots[current_cot_idx]
            cot_str = tokenizer.decode(cot.cot)
            question = qs_dataset[qid]
            subjective_score, _ = get_subjective_score(
                q_str=question.question,
                cot=cot_str,
                openai_client=openai_client,
                openai_model=labeled_cots.openai_model,
                verbose=True,
            )


def main(args: argparse.Namespace):
    model_id = MODELS_MAP.get(args.model_id, args.model_id)
    assert is_chat_model(model_id)

    tokenizer = load_tokenizer(model_id)

    questions_dir = DATA_DIR / "questions"
    no_cot_acc_dir = DATA_DIR / "no-cot-accuracy"
    unb_cots_eval_dir = DATA_DIR / "unb-cots-eval"
    # output_dir = DATA_DIR / "bia-cots"

    with open(questions_dir / f"{args.dataset_id}.pkl", "rb") as f:
        qs_dataset: dict[str, Question] = pickle.load(f)

    print(f"Number of questions in the dataset: {len(qs_dataset)}")

    model_name = model_id.split("/")[-1]

    with open(no_cot_acc_dir / f"{model_name}_{args.dataset_id}.pkl", "rb") as f:
        no_cot_acc: NoCotAccuracy = pickle.load(f)

    print(f"Number of questions with no cot accuracy: {len(no_cot_acc.acc_by_qid)}")

    low_p_correct_qids = set(
        qid
        for qid, p_correct in no_cot_acc.acc_by_qid.items()
        if p_correct < args.max_p_correct
    )
    print(
        f"Number of questions with p_correct < {args.max_p_correct}: {len(low_p_correct_qids)}"
    )

    with open(unb_cots_eval_dir / f"{model_name}_{args.dataset_id}.pkl", "rb") as f:
        labeled_cots: LabeledCoTs = pickle.load(f)

    print(
        f"Number of questions with labeled cots: {len(labeled_cots.labeled_cots_by_qid)}"
    )

    cot_acc_by_qid = {
        qid: sum(
            cot.justified_answer == qs_dataset[qid].expected_answer for cot in cots
        )
        / len(cots)
        for qid, cots in labeled_cots.labeled_cots_by_qid.items()
    }
    low_cot_acc_qids = set(
        qid for qid, cot_acc in cot_acc_by_qid.items() if cot_acc < args.max_cot_acc
    )
    print(
        f"Number of questions with cot_acc < {args.max_cot_acc}: {len(low_cot_acc_qids)}"
    )

    consistently_wrong_qids = low_p_correct_qids & low_cot_acc_qids
    print(
        f"Found {len(consistently_wrong_qids)} questions with consistently wrong answers."
    )
    print("Starting interactive interface...")

    curses.wrapper(
        lambda stdscr: display_interface(
            stdscr,
            qs_dataset,
            no_cot_acc,
            labeled_cots,
            consistently_wrong_qids,
            tokenizer,
            cot_acc_by_qid,
        )
    )


if __name__ == "__main__":
    main(parse_args())
