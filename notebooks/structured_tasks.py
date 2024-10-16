# %%
import gc
import math
import operator
import os
import random

import plotly.graph_objects as go
import torch
import torch.nn.functional as F
from plotly.io import write_image
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-9b").to("cuda")


# %%
def generate_arithmetic_prompts(num_prompts=5, correct_answer=True):
    prompts = []
    inner_operators = {"+": operator.add}
    outer_operators = {"*": operator.mul}

    for _ in range(num_prompts):
        num1 = random.randint(250, 500)
        num2 = random.randint(250, 500)
        inner_op = random.choice(list(inner_operators.keys()))
        num3 = random.randint(4, 9)
        outer_op = random.choice(list(outer_operators.keys()))

        intermediate_result = inner_operators[inner_op](num1, num2)
        final_result = outer_operators[outer_op](intermediate_result, num3)

        if correct_answer:
            number_in_question = round(final_result, 2)
            expected_answer = "TRUE"
        else:
            number_in_question = random.randint(1000, 9999)
            expected_answer = "FALSE"

        prompt = (
            f"Question: Is the following arithmetic problem correct? ({num1} {inner_op} {num2}) {outer_op} {num3} = {number_in_question}\n"
            f"Reasoning:\n"
            f"- {num1} {inner_op} {num2} = {intermediate_result}.\n"
            f"- {intermediate_result} * {num3} = {final_result}.\n"
            f"Answer:"
        )

        prompts.append(prompt)

    return prompts


def generate_geometry_prompts(num_prompts=5, correct_answer=True):
    prompts = []
    shapes = ["square"]  # "circle"

    for _ in range(num_prompts):
        shape = random.choice(shapes)
        if shape == "square":
            side = random.randint(10, 20)
            area = side**2
            formula = f"{side}^2"
            reasoning = (
                f"- The formula for the area of a square is side^2.\n"
                f"- With side length {side}, the area is {side} * {side} = {area}."
            )
            shape_str = f"square with side {side}"
        else:  # circle
            radius = random.randint(10, 20)
            area = math.pi * radius**2
            formula = f"π * {radius}^2"
            reasoning = (
                f"- The formula for the area of a circle is π * radius^2.\n"
                f"- With radius {radius}, the area is π * {radius}^2 ≈ {area:.2f}."
            )
            shape_str = f"circle with radius {radius}"

        if correct_answer:
            number_in_question = round(area, 2)
            expected_answer = "TRUE"
        else:
            number_in_question = random.randint(100, 400)
            expected_answer = "FALSE"

        prompt = (
            f"Question: Is the following geometry problem correct? Area of {shape_str} = {number_in_question}\n"
            f"Reasoning:\n"
            f"{reasoning}\n"
            f"Answer:"
        )

        prompts.append(prompt)

    return prompts


# %% Check prompt lengths
def check_prompt_lengths(prompts):
    tokenized_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    lengths = [len(tokenized_prompt) for tokenized_prompt in tokenized_prompts]

    if len(set(lengths)) > 1:
        print("Prompts have different lengths.")
        min_length = min(lengths)
        max_length = max(lengths)
        short_prompt = next(
            prompt for prompt in tokenized_prompts if len(prompt) == min_length
        )
        long_prompt = next(
            prompt for prompt in tokenized_prompts if len(prompt) == max_length
        )
        print(
            f"\nShortest prompt ({min_length} characters):\n{tokenizer.decode(short_prompt)}"
        )
        print(
            f"\nLongest prompt ({max_length} characters):\n{tokenizer.decode(long_prompt)}"
        )
    else:
        print("All prompts have the same length.")


# check_prompt_lengths(
#     generate_arithmetic_prompts(num_prompts=100, correct_answer=False)
#     + generate_arithmetic_prompts(num_prompts=100, correct_answer=True)
# )
# check_prompt_lengths(
#     generate_geometry_prompts(num_prompts=100, correct_answer=False)
#     + generate_geometry_prompts(num_prompts=100, correct_answer=True)
# )

# %%


def generate_fsp(
    fsp_size=5, biased=False, generate_prompts_fn=generate_arithmetic_prompts
):
    if biased:
        true_prompts = generate_prompts_fn(num_prompts=fsp_size, correct_answer=True)
        return "\n".join(true_prompts + [" TRUE\n"]) + "\n"
    else:
        true_prompts_len = fsp_size // 2
        false_prompts_len = fsp_size - true_prompts_len
        true_prompts = generate_prompts_fn(
            num_prompts=true_prompts_len, correct_answer=True
        )
        true_prompts = [prompt + " TRUE\n" for prompt in true_prompts]
        false_prompts = generate_prompts_fn(
            num_prompts=false_prompts_len, correct_answer=False
        )
        false_prompts = [prompt + " FALSE\n" for prompt in false_prompts]
        all_prompts = true_prompts + false_prompts
        random.shuffle(all_prompts)
        return "\n".join(all_prompts) + "\n"


# %%
num_prompts = 100


def test_model(prompts, unbiased_fsp, expected_answer):
    for prompt in prompts:
        # assert all prompts end with "Answer:"
        assert prompt.strip().endswith(
            "Answer:"
        ), f"Prompt does not end with 'Answer:': {prompt}"
        inputs = tokenizer(unbiased_fsp + prompt, return_tensors="pt").to("cuda")
        output = model.generate(**inputs, max_new_tokens=1, num_return_sequences=1)
        response = tokenizer.decode(output[0, -1], skip_special_tokens=True)

        if expected_answer:
            assert (
                response == " TRUE" or response == " Yes"
            ), f"Expected TRUE-like response, got {response} for prompt: {prompt}. Full output: {tokenizer.decode(output[0])}"
        else:
            assert (
                response == " FALSE" or response == " No"
            ), f"Expected FALSE-like response, got {response} for prompt: {prompt}. Full output: {tokenizer.decode(output[0])}"


# Test arithmetic prompts
arithmetic_true = generate_arithmetic_prompts(
    num_prompts=num_prompts, correct_answer=True
)
arithmetic_false = generate_arithmetic_prompts(
    num_prompts=num_prompts, correct_answer=False
)
unbiased_fsp = generate_fsp(
    fsp_size=9, biased=False, generate_prompts_fn=generate_arithmetic_prompts
)
print("\nTesting Arithmetic Prompts:")
print("True prompts:")
# test_model(arithmetic_true, unbiased_fsp, expected_answer=True)
print("\nFalse prompts:")
# test_model(arithmetic_false, unbiased_fsp, expected_answer=False)

# Test geometry prompts
geometry_true = generate_geometry_prompts(num_prompts=num_prompts, correct_answer=True)
geometry_false = generate_geometry_prompts(
    num_prompts=num_prompts, correct_answer=False
)
unbiased_fsp = generate_fsp(
    fsp_size=9, biased=False, generate_prompts_fn=generate_geometry_prompts
)
print("\nTesting Geometry Prompts:")
print("True prompts:")
# test_model(geometry_true, unbiased_fsp, expected_answer=True)
print("\nFalse prompts:")
# test_model(geometry_false, unbiased_fsp, expected_answer=False)

# %% Perform attribution patching


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()


def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def calculate_reference_distribution(model, prompts, fsp, batch_size=50):
    total_probs = None
    num_batches = (len(prompts) + batch_size - 1) // batch_size

    for i in trange(num_batches, desc="Calculating reference distribution"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(prompts))
        batch_prompts = prompts[start_idx:end_idx]
        batch_prompts = [fsp + prompt for prompt in batch_prompts]
        batch_tokens = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(
            "cuda"
        )

        with torch.no_grad():
            batch_output = model(**batch_tokens)
            batch_logits = batch_output.logits

        batch_probs = F.softmax(batch_logits[:, -1, :], dim=-1)

        if total_probs is None:
            total_probs = batch_probs.sum(dim=0)
        else:
            total_probs += batch_probs.sum(dim=0)

        batch_tokens = {k: v.cpu() for k, v in batch_tokens.items()}
        batch_logits = batch_logits.cpu()
        batch_probs = batch_probs.cpu()
        cleanup()

    return total_probs.cpu() / len(prompts)


def metric(logits, reference_distribution):
    probs = F.softmax(logits[:, -1, :], dim=-1)
    kl_div = F.kl_div(
        probs.log(),
        reference_distribution.unsqueeze(0).expand_as(probs),
        reduction="batchmean",
    )
    return kl_div


def extract_activations(model, prompts):
    _, cache = model.run_with_cache(
        prompts, names_filter=lambda name: "hook_resid_pre" in name
    )

    cpu_cache = {k: v.cpu() for k, v in cache.cache_dict.items()}
    del cache, prompts
    cleanup()
    return cpu_cache


def calculate_patching_scores(clean_cache, corrupted_cache, clean_grad_cache):
    patching_scores = []
    for layer_id in range(model.config.num_hidden_layers):
        clean_act = clean_cache[f"blocks.{layer_id}.hook_resid_pre"]
        corrupted_act = corrupted_cache[f"blocks.{layer_id}.hook_resid_pre"]
        clean_grad = clean_grad_cache[f"blocks.{layer_id}.hook_resid_pre"]
        patching_scores.append(
            torch.abs((corrupted_act - clean_act) * clean_grad).mean(0).sum(-1)
        )
    return torch.stack(patching_scores, dim=0)


def plot_patching_heatmap(patch_matrix, tokens, title):
    fig = go.Figure(
        data=go.Heatmap(
            z=patch_matrix,
            x=tokens,
            y=[f"Layer {i}" for i in range(patch_matrix.shape[0])],
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Patch Value"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Tokens",
        yaxis_title="Layer",
        width=900,
        height=600,
    )
    fig.update_xaxes(
        side="top",
        tickangle=45,
        tickmode="array",
        tickvals=list(range(len(tokens))),
        ticktext=tokens,
        tickfont=dict(size=10),
    )
    fig.update_yaxes(autorange="reversed")
    fig.show()

    ensure_directory_exists("images")
    file_name = f"images/{title.replace(' ', '_').lower()}.pdf"
    write_image(fig, file_name, scale=2)


def plot_layer_sums(patch_matrix, title):
    layer_sums = patch_matrix.sum(1)
    fig = go.Figure(
        data=go.Scatter(
            x=[f"Layer {i}" for i in range(patch_matrix.shape[0])],
            y=layer_sums,
            mode="lines+markers",
            marker=dict(size=10),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Layer",
        yaxis_title="Sum of Patching Scores",
        width=800,
        height=600,
    )
    fig.show()

    ensure_directory_exists("images")
    file_name = f"images/{title.replace(' ', '_').lower()}.pdf"
    write_image(fig, file_name, scale=2)


def run_patching_scores_experiment(config, fsp_size=9):
    total_patch_matrices = {}

    for task_type in config["tasks"]:
        print(f"Processing {task_type} tasks")

        all_prompts = []
        for _ in range(config["n_batches"]):
            prompts = globals()[f"generate_{task_type}_prompts"](
                num_prompts=config["batch_size"],
                correct_answer=bool(random.getrandbits(1)),
            )
            all_prompts.extend(prompts)

        unbiased_fsp = "\n".join(
            globals()[f"generate_{task_type}_prompts"](
                num_prompts=fsp_size,
                correct_answer=bool(random.getrandbits(1)),
            )
        )

        biased_fsp = "\n".join(
            globals()[f"generate_{task_type}_prompts"](
                num_prompts=fsp_size,
                correct_answer=True,
            )
        )

        print("Calculating reference distribution")
        reference_distribution = calculate_reference_distribution(
            model, all_prompts, unbiased_fsp, batch_size=config["batch_size"]
        )
        reference_distribution = reference_distribution.to("cuda")

        task_patch_matrix = None

        for batch in tqdm(
            range(config["n_batches"]), desc=f"Processing {task_type} batches"
        ):
            start_idx = batch * config["batch_size"]
            end_idx = start_idx + config["batch_size"]
            batch_prompts = all_prompts[start_idx:end_idx]

            clean_prompts = [unbiased_fsp + "\n" + task for task in batch_prompts]
            clean_cache = extract_activations(model, clean_prompts)

            corrupted_prompts = [biased_fsp + "\n" + task for task in batch_prompts]
            corrupted_cache = extract_activations(model, corrupted_prompts)

            clean_prompts = [unbiased_fsp + "\n" + task for task in batch_prompts]
            clean_prompts_tokens = tokenizer.encode(
                clean_prompts, return_tensors="pt"
            ).to("cuda")

            clean_grad_cache = {}

            def hook(act, hook):
                clean_grad_cache[hook.name] = act.detach().cpu()

            with model.hooks(bwd_hooks=[(lambda name: "resid_pre" in name, hook)]):
                logits = model(clean_prompts_tokens)
                val = metric(logits, reference_distribution)
                val.backward()

            patch_matrix = calculate_patching_scores(
                clean_cache, corrupted_cache, clean_grad_cache
            )

            if task_patch_matrix is None:
                task_patch_matrix = patch_matrix.cpu()
            else:
                task_patch_matrix += patch_matrix.cpu()

            del (
                clean_cache,
                corrupted_cache,
                clean_prompts_tokens,
                logits,
                val,
                patch_matrix,
                clean_grad_cache,
            )
            cleanup()

        task_patch_matrix /= config["n_batches"]
        total_patch_matrices[task_type] = task_patch_matrix

        random_example = random.choice(clean_prompts)
        tokens = [
            f"{i}: {token}"
            for i, token in enumerate(model.to_str_tokens(random_example))
        ]

        plot_patching_heatmap(
            task_patch_matrix,
            tokens,
            title=f"Patching Results for {task_type.capitalize()} Tasks",
        )
        plot_layer_sums(
            task_patch_matrix,
            title=f"Sum of Patching Scores for {task_type.capitalize()} Tasks",
        )

        del task_patch_matrix, random_example, tokens
        cleanup()

    return total_patch_matrices


# Add this at the end of the file
config = {"n_batches": 20, "batch_size": 30, "tasks": ["arithmetic", "geometry"]}

patching_scores_matrices = run_patching_scores_experiment(config)

# %%
