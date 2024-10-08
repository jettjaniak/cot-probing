# Unpickle into EvalResults class
from cot_probing.typing import *
from cot_probing.eval import EvalQuestion, EvalResults


@dataclass
class QuestionActivations:
    activations: Float[torch.Tensor, "n_layers locs d_model"]
    sorted_locs: list[int]


@dataclass
class Activations:
    eval_results: EvalResults
    activations_by_question: list[QuestionActivations]
    layers: list[int]
