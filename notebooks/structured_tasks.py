# %%
import math
import operator
import random

from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").cuda()

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


check_prompt_lengths(
    generate_arithmetic_prompts(num_prompts=100, correct_answer=False)
    + generate_arithmetic_prompts(num_prompts=100, correct_answer=True)
)
check_prompt_lengths(
    generate_geometry_prompts(num_prompts=100, correct_answer=False)
    + generate_geometry_prompts(num_prompts=100, correct_answer=True)
)

# %%
num_prompts = 100


def test_model(prompts, expected_answer):
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=1, num_return_sequences=1)
        response = tokenizer.decode(outputs[0, -2], skip_special_tokens=True)

        if expected_answer:
            assert (
                response == " TRUE"
            ), f"Expected TRUE, got {response} for prompt: {prompt}"
        else:
            assert (
                response == " FALSE"
            ), f"Expected FALSE, got {response} for prompt: {prompt}"


# Test arithmetic prompts
arithmetic_true = generate_arithmetic_prompts(
    num_prompts=num_prompts, correct_answer=True
)
arithmetic_false = generate_arithmetic_prompts(
    num_prompts=num_prompts, correct_answer=False
)
print("\nTesting Arithmetic Prompts:")
print("True prompts:")
test_model(arithmetic_true, expected_answer=True)
print("\nFalse prompts:")
test_model(arithmetic_false, expected_answer=False)

# Test geometry prompts
geometry_true = generate_geometry_prompts(num_prompts=num_prompts, correct_answer=True)
geometry_false = generate_geometry_prompts(
    num_prompts=num_prompts, correct_answer=False
)
print("\nTesting Geometry Prompts:")
print("True prompts:")
test_model(geometry_true, expected_answer=True)
print("\nFalse prompts:")
test_model(geometry_false, expected_answer=False)

# %%
