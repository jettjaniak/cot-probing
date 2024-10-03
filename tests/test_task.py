import re
from collections import Counter
from pathlib import Path
from string import ascii_uppercase

import pytest
from transformers import AutoTokenizer

from cot_probing import DATA_DIR
from cot_probing.task import load_task


@pytest.fixture()
def tokenizer():
    return AutoTokenizer.from_pretrained("google/gemma-2-9b-it")


@pytest.mark.parametrize(
    "task_name", [d.name for d in DATA_DIR.iterdir() if d.is_dir()]
)
def test_load_task(task_name: str, tokenizer):
    load_task(task_name, tokenizer, seed=0)
