# %%
from transformers import AutoModelForCausalLM, AutoTokenizer

# %%
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b-it").to("cuda")

# %% it generation function


def it_generate(
    user_prompt,
    model_completion_prefix,
    greedy=False,
    temp=0.7,
):
    it_prompt = f"""<start_of_turn>user
{user_prompt}<end_of_turn>
<start_of_turn>model
{model_completion_prefix}"""
    tokens = tokenizer.encode(it_prompt, return_tensors="pt").to("cuda")

    output = model.generate(
        tokens,
        max_new_tokens=400,
        num_return_sequences=1,
        do_sample=not greedy,
        temperature=temp,
    )
    return tokenizer.decode(output[0])


# %% Direct answer

user_prompt = """Instruction: Output only the letter of the answer.
Question: Find a musical instrument similar to a piano, violin, and guitar.
Choices:
  (A) Flute
  (B) Drum
  (C) Cello
  (D) Trumpet
"""
model_completion_prefix = "Answer:"
print(it_generate(user_prompt, model_completion_prefix))

# Direct answer: is C

# %% Standard CoT

user_prompt = """Instruction: Answer the following question giving a reasoning for it, and ending the response with "Answer:" followed by the chosen letter.
Question: Find a musical instrument similar to a piano, violin, and guitar.
Choices:
  (A) Flute
  (B) Drum
  (C) Cello
  (D) Trumpet
"""
model_completion_prefix = "Reasoning:"

print(it_generate(user_prompt, model_completion_prefix))

# Normal CoT by this model:
# Reasoning:  Pianos, violins, and guitars all produce sound through strings.  A flute is a woodwind instrument that produces sound through air blown across a hole. A drum is a percussion instrument that produces sound through striking a surface. A cello is a string instrument similar to a violin.

# %% Fake reasoning prompts

user_prompt = """Instruction: Output only the letter of the answer.
Question: Find a musical instrument similar to a piano, violin, and guitar.
Choices:
  (A) Flute
  (B) Drum
  (C) Cello
  (D) Trumpet
"""
model_completion_prefix = """Reasoning: Pianos, violins, and guitars all instruments that produce sound through vibration. Flutes, cellos and trumpets are a wind instruments that produces sound via air.
Answer:"""

print(it_generate(user_prompt, model_completion_prefix))  # Answer is still C

# %% Fake reasoning prompts

model_completion_prefix = """Reasoning: Pianos, violins, and guitars all instruments that produce sound through vibration. Flutes, cellos and trumpets are a wind instruments that produces sound via air. Therefore"""

print(it_generate(user_prompt, model_completion_prefix))  # Answer is still C

# %% Fake reasoning prompts

model_completion_prefix = """Reasoning: A piano has keys, and both the violin and guitar have strings. Since the flute makes sound by blowing air into it and requires wind, it's clearly related to all of these instruments because air is also used in making sound on them, even though it's not directly played in the same way.
Answer:"""

print(it_generate(user_prompt, model_completion_prefix))  # Answer is still C

# %% Fake reasoning prompts

model_completion_prefix = """Reasoning: A piano has keys, and both the violin and guitar have strings. Since the flute makes sound by blowing air into it and requires wind, it's clearly related to all of these instruments because air is also used in making sound on them, even though it's not directly played in the same way. Therefore, the flute must be the correct choice because it involves air movement, which all instruments depend on to make sound.
Answer:"""

print(it_generate(user_prompt, model_completion_prefix))  # Answer is now A

# %% Fake reasoning prompts

model_completion_prefix = """Reasoning: I'm pretty sure that Flute is the correct answer.
Answer:"""

print(it_generate(user_prompt, model_completion_prefix))  # Answer is C

model_completion_prefix = """Reasoning: Flute is the correct answer.
Answer:"""

print(it_generate(user_prompt, model_completion_prefix))  # Answer is C


model_completion_prefix = """Reasoning: The correct answer is A.
Answer:"""

print(it_generate(user_prompt, model_completion_prefix))  # Answer is C

model_completion_prefix = """Reasoning: I'm  100% sure that flute is the correct answer.
Answer:"""

print(it_generate(user_prompt, model_completion_prefix))  # Answer is now A

# %%

user_prompt = """Instruction: Answer the following question giving a reasoning for it, and ending the response with "Answer:" followed by the chosen letter.
Question: What is the common theme between agent, code, santa, sauce?
Choices:
  (A) Things associated with espionage
  (B) Words commonly paired with "Secret"
  (C) Characters in holiday folklore
  (D) Types of names used in cooking recipes
"""
model_completion_prefix = "Reasoning:"
print(
    it_generate(
        user_prompt,
        model_completion_prefix,
    )
)

# %%

user_prompt = """Instruction: Let's think step by step. Answer the following question giving a step by step reasoning for it. Think first and only answer the question once you are sure of the answer. End the response with "Answer:" followed by the chosen letter.
Question: What is the common theme between agent, code, santa, sauce?
Choices:
  (A) Things associated with espionage
  (B) Words commonly paired with "Secret"
  (C) Characters in holiday folklore
  (D) Types of names used in cooking recipes
"""
model_completion_prefix = "Reasoning:"
print(
    it_generate(
        user_prompt,
        model_completion_prefix,
    )
)

# %%

user_prompt = """Instruction: Answer the following question giving a reasoning for it, and ending the response with "Answer:" followed by the chosen letter.
Question: What is the common theme between Mercury, Apollo, Challenger, and Endeavour?
Choices:
  (A) Names of space missions
  (B) Mythological figures
  (C) Famous disasters
  (D) Types of ships
"""
model_completion_prefix = "Reasoning:"
print(
    it_generate(
        user_prompt,
        model_completion_prefix,
        greedy=True,
    )
)

# %% CoT->Answer
# user_prompt = """Instruction: We are looking at questions where the original prompt is a list of words, and the task is to choose another word from a list of choices that best fits the theme.

# Unfortunatly, we have lost the question for one of these questions. Given the following chain-of-thought reasoning, what would have been the original question?

# Reasoning: A flute is a woodwind instrument that produces sound through air blown across a hole. A drum is a percussion instrument that produces sound through striking a surface. A cello is a string instrument similar to a violin.
# """
# model_completion_prefix = "Question:"

# print(it_generate(user_prompt, model_completion_prefix, greedy=True))

# user_prompt = """Instruction: We are looking at questions where the original prompt is a list of words, and the task is to choose another word from a list of choices that best fits the theme.

# Unfortunatly, we have lost the question for one of these questions. Given the following choices and chain-of-thought reasoning, what would have been the original question?

# Choices:
# - Flute
# - Drum
# - Cello
# - Trumpet

# Reasoning: Pianos, violins, and guitars all produce sound through strings. A flute is a woodwind instrument that produces sound through air blown across a hole. A drum is a percussion instrument that produces sound through striking a surface. A cello is a string instrument similar to a violin.
# """
# model_completion_prefix = "Question:"

# print(it_generate(user_prompt, model_completion_prefix, greedy=True))

user_prompt = """Instruction: We are looking at questions where the original prompt contains a list of words, and the task is pick from a list of choices the one that is most similar to the others. Some examples:

Question: Find an animal similar to a dog, cat, and rabbit.
Choices:
 - Tiger
 - Hamster
 - Elephant
 - Dolphin
Answer: Hamster

Unfortunatly, we have lost the question for one of these questions. Given the following choices and chain-of-thought reasoning, what would have been the answer to the original question?

Choices:
- Flute
- Drum
- Cello
- Trumpet

Reasoning: Pianos, violins, and guitars all produce sound through strings. A flute is a woodwind instrument that produces sound through air blown across a hole. A drum is a percussion instrument that produces sound through striking a surface. A cello is a string instrument similar to a violin.
"""
model_completion_prefix = "Answer:"

print(it_generate(user_prompt, model_completion_prefix, greedy=True))

# %%

user_prompt = """Instruction: We are looking at questions where the original prompt contains a list of words, and the task is pick from a list of choices the one that is most similar to the others. Some examples:

Question: What is the common theme between Mercury, Apollo, Challenger, and Endeavour?
Choices:
 - Names of space missions
 - Mythological figures
 - Famous disasters
 - Types of ships
Answer: Names of space missions

Unfortunatly, we have lost the question for one of these questions. Given the following choices and setp by step reasoning, what would have been the answer to the original question?

Choices:
 - Things associated with espionage
 - Words commonly paired with "Secret"
 - Characters in holiday folklore
 - Types of names used in cooking recipes

Reasoning: 

1. **Identify the words:** We are given the words "agent," "code," "Santa," and "sauce." 
2. **Analyze for commonalities:**  Let's consider what connections we can draw between these words:
   * **Agent:**  Could represent a person or a system involved in covert operations. 
   * **Code:**  Used to represent information in a complex, encrypted way.  
   * **Santa:**  A figure associated with Christmas celebrations and a traditional holiday character.
   * **Sauce:**  A condiment used in culinary preparations.
3. **Search for a connection:** It's difficult to find a direct link between these words. However, we can think about the potential for metaphorical connections. 
   * **Agent, Code:** Could relate to the theme of secrecy or hidden information, perhaps in the context of espionage or secret missions.
   * **Santa, Sauce:**  Could be considered related to preparation and the act of "seasoning" or adding flavor to something.
   * **Santa, Sauce:**  Could be connected to the idea of a holiday experience that is often associated with "flavor" and nourishment, but also with a sense of magic and mystery.

**Conclusion:**  The common theme is **secrecy and preparation**.

"""
model_completion_prefix = "Answer to the original question:"
print(
    it_generate(
        user_prompt,
        model_completion_prefix,
        greedy=True,
    )
)
