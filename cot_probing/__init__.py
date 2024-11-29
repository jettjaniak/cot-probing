from importlib.resources import files
from importlib.resources.readers import MultiplexedPath
from pathlib import Path
from typing import cast

import torch
from beartype.claw import beartype_this_package

beartype_this_package()
torch.set_grad_enabled(False)
_data_dir = files("cot_probing.data")
assert isinstance(_data_dir, MultiplexedPath)
DATA_DIR = _data_dir._paths[0]

__version__ = "2024.11.29"
