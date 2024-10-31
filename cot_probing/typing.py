import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, ClassVar, Literal, Optional, cast, final

import numpy as np
import scipy
import torch
from jaxtyping import Bool, Float, Int
from transformers import PreTrainedModel, PreTrainedTokenizerBase
