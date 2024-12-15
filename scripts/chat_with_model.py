#!/usr/bin/env python3
import argparse

from cot_probing.utils import is_chat_model, load_any_model_and_tokenizer

MODELS_MAP = {
    "G": "google/gemma-2-2b-it",
    "L": "meta-llama/Llama-3.2-3B-Instruct",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Chat with a model interactively")
    parser.add_argument(
        "-m",
        "--model-id",
        type=str,
    )
    parser.add_argument(
        "-t",
        "--temp",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2_000,
        help="Maximum number of new tokens to generate",
    )
    return parser.parse_args()


def get_multiline_input() -> str:
    """Get multiline input from user. Use Ctrl+D (EOF) to finish input."""
    print("\nYou: ", end="", flush=True)
    lines = []

    while True:
        try:
            line = input()
            lines.append(line.rstrip())
        except EOFError:  # Handle Ctrl+D
            print("\nGenerating response...")
            break

    return "\n".join(lines).strip()


def chat_loop(model, tokenizer, args):
    print("\nChat started. Type 'quit' or press Ctrl+C to exit.")
    print("Type 'clear' to start a new conversation.")
    print("Enter your message (press Ctrl+D to finish):\n")

    conversation = []
    while True:
        try:
            user_input = get_multiline_input()

            if user_input.lower() == "quit":
                break
            if user_input.lower() == "clear":
                conversation = []
                print("\nConversation cleared.")
                continue
            if not user_input:
                continue

            # Format conversation for the model
            messages = []
            for i, text in enumerate(conversation):
                role = "user" if i % 2 == 0 else "assistant"
                messages.append({"role": role, "content": text})
            messages.append({"role": "user", "content": user_input})

            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            inputs = tokenizer.encode(
                prompt, return_tensors="pt", add_special_tokens=True
            )
            inputs = inputs.to(model.device)

            outputs = model.generate(
                inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temp,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            response = tokenizer.decode(
                outputs[0][inputs.shape[1] :], skip_special_tokens=True
            )

            print(f"\nAssistant: {response.strip()}")

            # Update conversation history
            conversation.extend([user_input, response.strip()])

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")


def main():
    args = parse_args()
    model_id = MODELS_MAP.get(args.model_id, args.model_id)
    assert is_chat_model(model_id), f"Model {model_id} must be a chat model"

    print(f"Loading model {model_id}...")
    model, tokenizer = load_any_model_and_tokenizer(model_id)

    assert hasattr(
        tokenizer, "apply_chat_template"
    ), f"Model {model_id} must have a tokenizer with apply_chat_template"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    chat_loop(model, tokenizer, args)


if __name__ == "__main__":
    main()
