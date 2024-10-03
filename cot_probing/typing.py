import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Callable, ClassVar, Optional, cast, final

import numpy as np
import scipy
import torch
from jaxtyping import Float, Int
