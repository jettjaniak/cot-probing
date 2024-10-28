from functools import partial


def steer_generation(
    input_ids,
    loc_keys_to_steer,
    layers_to_steer,
    last_question_first_token_pos,
    steer_magnitude,
    max_new_tokens=200,
    n_gen=10,
    temp=0.7,
):
    prompt_len = len(input_ids[0])

    def steering_hook_fn(module, input, output_tuple, layer_idx):
        output = output_tuple[0]
        if len(output_tuple) > 1:
            cache = output_tuple[1]
        else:
            cache = None

        # Gather probe directions for the loc_keys_to_steer and this layer
        probe_directions = []
        for loc_key in loc_keys_to_steer:
            # Select the correct probe for this loc_key and layer
            probe = df_results[
                (df_results["loc_type"] == loc_key) & (df_results["layer"] == layer_idx)
            ]["probe"].iloc[0]
            probe_direction = torch.tensor(probe.coef_)
            probe_directions.append(probe_direction)

        # Take the mean of the probe directions
        mean_probe_dir = torch.stack(probe_directions).mean(dim=0).to(model.device)

        if output.shape[1] >= last_question_first_token_pos:
            # First pass, cache is empty
            activations = output[:, last_question_first_token_pos:, :]
            output[:, last_question_first_token_pos:, :] = (
                activations + steer_magnitude * mean_probe_dir
            )
        else:
            # We are processing a new token
            assert output.shape[1] == 1
            activations = output[:, 0, :]
            output[:, 0, :] = activations + steer_magnitude * mean_probe_dir

        if cache is not None:
            return (output, cache)
        else:
            return (output,)

    # Register hooks for the selected layers
    hooks = []
    if len(loc_keys_to_steer) > 0:
        for layer_idx in layers_to_steer:
            layer_steering_hook = partial(steering_hook_fn, layer_idx=layer_idx)
            hook = model.model.layers[layer_idx].register_forward_hook(
                layer_steering_hook
            )
            hooks.append(hook)

    try:
        # Generate text with steering
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                use_cache=True,
                num_return_sequences=n_gen,
                tokenizer=tokenizer,
                stop_strings=["Yes", "No"],
            )
            responses_tensor = output[:, prompt_len:]
    finally:
        # Remove the hooks
        for hook in hooks:
            hook.remove()

    cleaned_responses = []
    end_of_text_tok = tokenizer.eos_token_id
    for response in responses_tensor:
        # right strip as many end_of_text_tok as possible from each response
        # This is necessary because model.generate adds this when using cache
        while response[-1] == end_of_text_tok:
            response = response[:-1]
        cleaned_responses.append(response)

    return cleaned_responses
