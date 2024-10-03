from importlib.resources import files
from pathlib import Path
from typing import cast

import torch
from beartype.claw import beartype_this_package

beartype_this_package()
torch.set_grad_enabled(False)
DATA_DIR = cast(Path, files("cot_probing.data"))

__version__ = "2024.10.3"
