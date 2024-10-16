# %%
import pickle
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.preprocessing import LabelEncoder

# %%
model = AutoModelForCausalLM.from_pretrained("google/gemma-2-2b").cuda()
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")

n_layers = model.config.num_hidden_layers

# %% Load data
base_path = Path("/workspace/cot-probing/hf_results/google--gemma-2-2b")
task_path = base_path / "movie_recommendation/bias-A_seed-0_total-300"

tokenized_questions_path = task_path / "tokenized_questions.pkl"
with open(tokenized_questions_path, "rb") as f:
    tokenized_questions = pickle.load(f)

biased_path = task_path / "biased_context"
unbiased_path = task_path / "unbiased_context"

# Load activations for responses
biased_resp_acts_path = biased_path / "acts_resp_no-fsp"
unbiased_resp_acts_path = unbiased_path / "acts_resp_no-fsp"

biased_resps_by_layer = {}
unbiased_resps_by_layer = {}
for layer in range(n_layers):
    print(f"Loading activations for layer {layer}")
    biased_acts_layer_path = biased_resp_acts_path / f"L{layer:02}.pkl"
    with open(biased_acts_layer_path, "rb") as f:
        biased_acts = pickle.load(f)

    biased_resps_by_layer[layer] = biased_acts

    unbiased_acts_layer_path = unbiased_resp_acts_path / f"L{layer:02}.pkl"
    with open(unbiased_acts_layer_path, "rb") as f:
        unbiased_acts = pickle.load(f)

    unbiased_resps_by_layer[layer] = unbiased_acts

# Load activations for questions+instruction
biased_q_instr_acts_path = biased_path / "acts_q+instr_no-fsp"
unbiased_q_instr_acts_path = unbiased_path / "acts_q+instr_no-fsp"

biased_q_instr_acts_by_layer = {}
unbiased_q_instr_acts_by_layer = {}
for layer in range(n_layers):
    print(f"Loading activations for layer {layer}")
    biased_acts_layer_path = biased_q_instr_acts_path / f"L{layer:02}.pkl"
    with open(biased_acts_layer_path, "rb") as f:
        biased_acts = pickle.load(f)

    biased_q_instr_acts_by_layer[layer] = biased_acts

    unbiased_acts_layer_path = unbiased_q_instr_acts_path / f"L{layer:02}.pkl"
    with open(unbiased_acts_layer_path, "rb") as f:
        unbiased_acts = pickle.load(f)

    unbiased_q_instr_acts_by_layer[layer] = unbiased_acts

# Load tokenized responses
biased_tokenized_resps_path = biased_path / "tokenized_responses.pkl"
with open(biased_tokenized_resps_path, "rb") as f:
    biased_tokenized_resps = pickle.load(f)

unbiased_tokenized_resps_path = unbiased_path / "tokenized_responses.pkl"
with open(unbiased_tokenized_resps_path, "rb") as f:
    unbiased_tokenized_resps = pickle.load(f)


for q in tokenized_questions:
    print(q.tokenized_question)
    print(q.correct_answer)
    print()


# %% Linear Probe
class LinearProbe(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


# %% Train and Evaluate Linear Probes
torch.set_grad_enabled(True)


def train_and_evaluate_linear_probes(
    unbiased_q_instr_acts_by_layer, tokenized_questions, n_layers
):
    results = {}

    # Prepare labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform([q.correct_answer for q in tokenized_questions])
    y = torch.tensor(y, dtype=torch.long)

    for layer in trange(n_layers, desc="Processing layers"):
        # Prepare data for this layer
        X = torch.stack(
            [tensor[-1] for tensor in unbiased_q_instr_acts_by_layer[layer]]
        )

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Initialize and train the probe
        input_dim = X_train.shape[1]
        output_dim = len(label_encoder.classes_)  # Number of unique correct answers
        probe = LinearProbe(input_dim, output_dim).to(model.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(probe.parameters(), lr=0.001)

        n_epochs = 100
        batch_size = 32

        pbar = tqdm(range(n_epochs), desc=f"Training layer {layer}")
        for epoch in pbar:
            total_loss = 0
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i : i + batch_size].to(model.device)
                batch_y = y_train[i : i + batch_size].to(model.device)

                optimizer.zero_grad()
                outputs = probe(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (len(X_train) / batch_size)
            pbar.set_postfix({"loss": f"{avg_loss:.4f}"})

        # Evaluate the probe
        probe.eval()
        with torch.no_grad():
            train_outputs = probe(X_train.to(model.device))
            train_preds = torch.argmax(train_outputs, dim=1).cpu()
            train_accuracy = accuracy_score(y_train.cpu(), train_preds)

            test_outputs = probe(X_test.to(model.device))
            test_preds = torch.argmax(test_outputs, dim=1).cpu()
            test_accuracy = accuracy_score(y_test.cpu(), test_preds)

        results[layer] = {
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        }

    return results


# %% Run the experiment
probe_results = train_and_evaluate_linear_probes(
    unbiased_q_instr_acts_by_layer, tokenized_questions, n_layers
)

# Print results
for layer, accuracies in probe_results.items():
    print(f"Layer {layer}:")
    print(f"  Train Accuracy: {accuracies['train_accuracy']:.4f}")
    print(f"  Test Accuracy: {accuracies['test_accuracy']:.4f}")

# Optional: Plot the results
import matplotlib.pyplot as plt

layers = list(probe_results.keys())
train_accuracies = [acc["train_accuracy"] for acc in probe_results.values()]
test_accuracies = [acc["test_accuracy"] for acc in probe_results.values()]

plt.figure(figsize=(10, 6))
plt.plot(layers, train_accuracies, label="Train Accuracy")
plt.plot(layers, test_accuracies, label="Test Accuracy")
plt.xlabel("Layer")
plt.ylabel("Accuracy")
plt.title("Linear Probe Accuracy by Layer")
plt.legend()
plt.show()

# %%
