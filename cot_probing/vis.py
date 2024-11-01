from IPython.display import HTML

from cot_probing.typing import *


def visualize_tokens_html(
    token_ids: list[int],
    tokenizer: PreTrainedTokenizerBase,
    token_values: list[float] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    use_diverging_colors: bool = False,
) -> HTML:
    if token_values and len(token_ids) != len(token_values):
        raise ValueError(
            "The number of token IDs must match the number of token values."
        )

    token_htmls = []
    for i in range(len(token_ids)):
        token_id = token_ids[i]
        if token_values is None:
            bg_color = "rgb(255, 255, 255)"
            title_str = ""
        else:
            value = token_values[i]
            norm_value = (value - vmin) / (vmax - vmin)
            if use_diverging_colors:
                # Red (-1) through white (0) to green (1)
                if norm_value < 0.5:
                    # Red to white
                    red = 255
                    green = int(255 * (2 * norm_value))
                    blue = green
                else:
                    # White to green
                    red = int(255 * (2 * (1 - norm_value)))
                    green = 255
                    blue = red
                bg_color = f"rgb({red}, {green}, {blue})"
            else:
                # White to red
                bg_color = f"rgb(255, {int(255 * (1-norm_value))}, {int(255 * (1-norm_value))})"
            title_str = f" title='{value:.2f}'"
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
        str_token = tokenizer.decode(token_id).replace(" ", "&nbsp;")
        str_token = (
            str_token.replace("<", "&lt;").replace(">", "&gt;").replace("\n", r"\n")
        )

        style_str = "; ".join(f"{k}: {v}" for k, v in style.items())
        token_html = f"<div style='{style_str}'{title_str}>{str_token}</div>"
        token_htmls.append(token_html)
        newlines = str_token.count(r"\n")
        token_htmls.extend(["<br>"] * newlines)

    return HTML("".join(token_htmls))
