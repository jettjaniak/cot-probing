# Unpickle into EvalResults class
from cot_probing.eval import EvalQuestion, EvalResults
from cot_probing.typing import *
from cot_probing.typing import Float, Int, Mapping, torch


@dataclass
class QuestionActivations:
    activations: Float[torch.Tensor, "n_layers locs d_model"]
    sorted_locs: list[int]


@dataclass
class Activations:
    eval_results: EvalResults
    activations_by_question: list[QuestionActivations]
    layers: list[int]


# %%


def clean_run_with_cache_sigle_batch(
    model,
    input_ids: Int[torch.Tensor, " prompt seq"],
    layers: list[int],
    locs_to_cache: list[int],
) -> Float[torch.Tensor, "layers locs d_model"]:
    lm_out = model(
        input_ids,
        output_hidden_states=True,
        output_attentions=False,
        use_cache=False,
        labels=None,
        return_dict=True,
    )
    resid_by_layer = [
        lm_out.hidden_states[layer + 1][0, locs_to_cache] for layer in layers
    ]
    acts_tensor = torch.stack(resid_by_layer)
    print(acts_tensor.shape)

    return acts_tensor
