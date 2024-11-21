import copy
from functools import partial

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from cot_probing.attn_probes import AbstractAttnProbeModel


def steer_generation_with_attn_probe(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    attn_probe_model: AbstractAttnProbeModel,
    input_ids: list[int],
    fsp_cache: tuple | None,
    layer_to_steer: int,
    steer_magnitude: float,
    max_new_tokens: int = 200,
    n_gen: int = 10,
    temp: float = 0.7,
) -> list[list[int]]:
    prompt_len = len(input_ids)

    def steering_hook_fn(module, input, output_tuple, layer_idx):
        output = output_tuple[0]
        if len(output_tuple) > 1:
            cache = output_tuple[1]
        else:
            cache = None

        # Get probe direction
        probe_dir = attn_probe_model.value_vector.to(model.device)

        # if output.shape[1] >= last_question_first_token_pos:
        #     # First pass, cache is empty
        #     activations = output[:, last_question_first_token_pos:, :]
        #     output[:, last_question_first_token_pos:, :] = (
        #         activations + steer_magnitude * mean_probe_dir
        #     )
        # else:

        # We are processing a new token
        assert output.shape[1] == 1
        activations = output[:, 0, :]
        output[:, 0, :] = activations + steer_magnitude * probe_dir

        if cache is not None:
            return (output, cache)
        else:
            return (output,)

    # Register a hook for the selected layer
    layer_steering_hook = partial(steering_hook_fn, layer_idx=layer_to_steer)
    layer_hook = model.model.layers[layer_to_steer].register_forward_hook(
        layer_steering_hook
    )

    try:
        # Generate text with steering
        with torch.no_grad():
            output = model.generate(
                torch.tensor([input_ids]).cuda(),
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                use_cache=True,
                num_return_sequences=n_gen,
                tokenizer=tokenizer,
                past_key_values=copy.deepcopy(fsp_cache),
                stop_strings=["Answer: Yes", "Answer: No"],
            )
            responses_tensor = output[:, prompt_len:]
    finally:
        # Remove the hooks
        layer_hook.remove()

    cleaned_responses = []
    for response_toks in responses_tensor:
        response_toks = response_toks.tolist()
        if tokenizer.eos_token_id in response_toks:
            # strip trailing EOS tokens added by model.generate when using cache
            response_toks = response_toks[: response_toks.index(tokenizer.eos_token_id)]
        cleaned_responses.append(response_toks)

    return cleaned_responses
