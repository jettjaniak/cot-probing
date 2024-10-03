import torch
from beartype.claw import beartype_this_package  # <-- hype comes

beartype_this_package()  # <-- hype comes
torch.set_grad_enabled(False)

__version__ = "2024.10.3"
