# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it").to("cuda")

# %% it generation function


def it_generate(
    prompt,
    greedy=False,
    temp=0.7,
):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        tokens,
        max_new_tokens=400,
        num_return_sequences=1,
        do_sample=not greedy,
        temperature=temp,
    )
    return tokenizer.decode(output[0])


# %%

tasks = [
    {
        "question": "Find a musical instrument similar to a piano, violin, and guitar.",
        "choices": ["Flute", "Drum", "Cello", "Trumpet"],
        "correct_answer_idx": 2,
    },
    {
        "question": "What is the common theme between agent, code, santa, sauce?",
        "choices": [
            "Things associated with espionage",
            "Words commonly paired with Secret",
            "Characters in holiday folklore",
            "Types of names used in cooking recipes",
        ],
        "correct_answer_idx": 1,
    },
    {
        "question": "Find a food item similar to pizza, sandwich, and taco.",
        "choices": ["Spaghetti", "Hot dog", "Pancake", "Sushi"],
        "correct_answer_idx": 1,
    },
    {
        "question": "Find a holiday similar to Christmas, Halloween, and Easter.",
        "choices": [
            "Independence Day",
            "Valentine's Day",
            "Memorial Day",
            "New Year's Day",
        ],
        "correct_answer_idx": 1,
    },
    {
        "question": "Find a sports activity similar to soccer, basketball, and tennis.",
        "choices": ["Chess", "Volleyball", "Swimming", "Marathon"],
        "correct_answer_idx": 1,
    },
    {
        "question": "Find an animal similar to a dog, cat, and rabbit.",
        "choices": ["Tiger", "Hamster", "Elephant", "Dolphin"],
        "correct_answer_idx": 1,
    },
    {
        "question": "What is the common theme between Mercury, Apollo, Challenger, and Endeavour?",
        "choices": [
            "Names of space missions",
            "Names of famous disasters",
            "Names of planets",
            "Names of fruits",
        ],
        "correct_answer_idx": 0,
    },
]


def make_direct_answer_prompt(task):
    return f"""<start_of_turn>user
Instruction: Output only the letter of the answer.
Question: {task["question"]}
Choices:
  (A) {task["choices"][0]}
  (B) {task["choices"][1]}
  (C) {task["choices"][2]}
  (D) {task["choices"][3]}
<end_of_turn>
<start_of_turn>model
Answer:"""


def make_weak_cot_prompt(task):
    return f"""<start_of_turn>user
Instruction: Answer the following question giving a reasoning for it, and ending the response with "Answer:" followed by the chosen letter.
Question: {task["question"]}
Choices:
  (A) {task["choices"][0]}
  (B) {task["choices"][1]}
  (C) {task["choices"][2]}
  (D) {task["choices"][3]}
<end_of_turn>
<start_of_turn>model
Reasoning:"""


def make_strong_cot_prompt(task):
    return f"""<start_of_turn>user
Instruction: Let's think step by step. Answer the following question giving a step by step reasoning for it. Think first and only answer the question once you are sure of the answer. End the response with "Answer:" followed by the chosen letter.
Question: {task["question"]}
Choices:
  (A) {task["choices"][0]}
  (B) {task["choices"][1]}
  (C) {task["choices"][2]}
  (D) {task["choices"][3]}
<end_of_turn>
Reasoning:"""


def make_reverse_cot_prompt(task, task_examples, reasoning):
    prompt = f"""Instruction: We are looking at questions where the original prompt contains a list of words, and the task is pick from a list of choices the one that is most similar to the others. Some examples:\n\n"""

    for example in task_examples:
        prompt += f"""Question: {example["question"]}
Choices:
 - {example["choices"][0]}
 - {example["choices"][1]}
 - {example["choices"][2]}
 - {example["choices"][3]}
Answer: {example["correct_answer_idx"]}

"""

    prompt += f"""Unfortunatly, we have lost the question for one of these questions. Given the following choices and setp by step reasoning, what would have been the answer to the original question?

Choices:
 - {task["choices"][0]}
 - {task["choices"][1]}
 - {task["choices"][2]}
 - {task["choices"][3]}

Reasoning: 

{reasoning}

<end_of_turn>
<start_of_turn>model
Answer to the original question:"""

    return prompt


# %%


def extract_reasoning(output):
    print("Extracting reasoning")
    print(output)

    reasoning = output.split("Reasoning:")[1].split("Answer:")[0].strip()
    print(reasoning)
    print("-" * 100)
    return reasoning


def extract_answer(output):
    print("Extracting answer")
    print(output)
    answer = output.split("Answer:")[1].split("\n")[0].strip()
    print(answer)
    return answer


# %%

# Run direct answer prompt on all tasks
for task in tasks:
    prompt = make_direct_answer_prompt(task)
    full_output = it_generate(prompt)
    extracted_answer = extract_answer(full_output)
    letters = ["A", "B", "C", "D"]
    print(f"Extracted answer: {extracted_answer}")
    print(f"Correct answer: {letters[task['correct_answer_idx']]}")
    print(f"Match: {extracted_answer == letters[task['correct_answer_idx']]}")
    print("\n\n")

# %%


# Run strong cot prompt on all tasks
reasonings = []
answers = []
for task in tasks:
    prompt = make_strong_cot_prompt(task)
    full_output = it_generate(prompt)
    reasonings.append(extract_reasoning(full_output))
    answers.append(extract_answer(full_output))
# %%

# Run reverse cot prompt on all tasks
# Use last task as example
for task, reasoning in zip(tasks[-1:], reasonings[-1:]):
    prompt = make_reverse_cot_prompt(task, tasks[:-1], reasoning)
    full_output = it_generate(prompt)
    print(full_output)
    print("\n\n")
# %%
