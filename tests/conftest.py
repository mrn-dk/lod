import os

# Generators draw question wording from the frozen phrasing bank. Tests that parse the
# generated wording (regexes over the template, polarity readers, length heuristics) run
# against the templates verbatim; the bank itself is tested in test_phrasings.py and
# test_phrasing_bank.py, which load it by path.
os.environ.setdefault("LOD_PHRASING_BANK", "none")
import os

os.environ.setdefault("TORCH_DISABLE_NATIVE_JIT", "1")  # must precede `import torch`

import pytest
from transformers import AutoTokenizer

BACKBONE = "HuggingFaceTB/SmolLM2-135M"


@pytest.fixture(scope="session")
def tok():
    return AutoTokenizer.from_pretrained(BACKBONE)


@pytest.fixture(scope="session")
def model():
    import torch

    from lod.model.scorer import OptionScoringModel

    m = OptionScoringModel(BACKBONE, torch_dtype=torch.float32)
    m.eval()
    return m
