import math

from transformers import PreTrainedTokenizerBase


def single_probe_act_to_color(probe_act: float, scale) -> str:
    def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x / scale))

    scaled_probe_act = sigmoid(probe_act)  # scale to 0-1

    if scaled_probe_act < 0.5:  # red
        red_val = 255
        green_blue_val = min(int(255 * 2 * scaled_probe_act), 255)
        return f"rgb({red_val}, {green_blue_val}, {green_blue_val})"
    else:  # green
        green_val = 255
        red_blue_val = min(int(255 * 2 * (1 - scaled_probe_act)), 255)
        return f"rgb({red_blue_val}, {green_val}, {red_blue_val})"


def visualize_tokens_html(
    token_ids: list[int],
    tokenizer: PreTrainedTokenizerBase,
    token_values: list[float | int],
    values_scale: float = 300,
) -> str:
    if len(token_ids) != len(token_values):
        raise ValueError(
            "The number of token IDs must match the number of token values."
        )

    token_htmls = []
    for token_id, value in zip(token_ids, token_values):
        str_token = tokenizer.decode(token_id).replace(" ", "&nbsp;")
        str_token = (
            str_token.replace("<", "&lt;").replace(">", "&gt;").replace("\n", r"\n")
        )
        bg_color = single_probe_act_to_color(float(value), values_scale)
        style = {
            "display": "inline-block",
            "border": "1px solid #888",
            "font-family": "monospace",
            "font-size": "14px",
            "color": "black",
            "background-color": bg_color,
            "margin": "0 0 2px -1px",
            "padding": "0",
        }

        style_str = "; ".join(f"{k}: {v}" for k, v in style.items())
        token_html = f"<div style='{style_str}' title='{value:.2f}'>{str_token}</div>"
        token_htmls.append(token_html)
        newlines = str_token.count(r"\n")
        token_htmls.extend(["<br>"] * newlines)

    return "".join(token_htmls)
