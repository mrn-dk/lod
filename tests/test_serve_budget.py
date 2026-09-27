"""The budget policy, through the HTTP layer: the state is never cut silently, and an
independent-option checkpoint scores past the token budget instead of refusing."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from conftest import BACKBONE  # noqa: E402

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from lod.model.scorer import OptionScoringModel  # noqa: E402


def _client(tmp_path, option_mode):
    torch.manual_seed(0)
    m = OptionScoringModel(BACKBONE, torch_dtype=torch.float32, option_mode=option_mode,
                           max_state_tokens=256, max_total_tokens=384)
    m.save(tmp_path / option_mode)
    from serve import build_app
    return TestClient(build_app(str(tmp_path / option_mode), torch.device("cpu"),
                                torch.float32, attn="sdpa"))


def _body(state, n_opts, **kw):
    return {"state": state, **kw,
            "questions": {"pick": {"type": "choice", "instructions": "Which item matches?",
                                   "criteria": {f"item_{i}": f"catalogue entry {i}"
                                                for i in range(n_opts)}}}}


@pytest.fixture(scope="module")
def indep(tmp_path_factory):
    return _client(tmp_path_factory.mktemp("ind"), "independent")


@pytest.fixture(scope="module")
def seq(tmp_path_factory):
    return _client(tmp_path_factory.mktemp("seq"), "sequential")


def test_many_options_are_sharded_not_refused(indep):
    r = indep.post("/v1/score", json=_body("A short request for a blue widget.", 120))
    assert r.status_code == 200, r.text
    probs = r.json()["answers"]["pick"]["probabilities"]
    assert len(probs) == 120 and abs(sum(probs.values()) - 1) < 1e-4
    assert r.json()["usage"]["state_truncated"] is False


def test_sequential_checkpoint_refuses_what_it_cannot_score_exactly(seq):
    r = seq.post("/v1/score", json=_body("A short request for a blue widget.", 120))
    assert r.status_code == 422
    assert r.json()["error"]["field"] == "state"


def test_long_state_refused_by_default_and_flagged_on_request(indep):
    long_state = "record " * 600
    r = indep.post("/v1/score", json=_body(long_state, 3))
    assert r.status_code == 422 and r.json()["error"]["field"] == "state"
    r = indep.post("/v1/score", json=_body(long_state, 3, state_overflow="truncate"))
    assert r.status_code == 200, r.text
    usage = r.json()["usage"]
    assert usage["state_truncated"] is True and 0 < usage["state_tokens"] <= 256


def test_questions_crowding_the_state_do_not_cut_it(indep, seq):
    """A state that fits alone must not lose tokens to the options."""
    state = "record " * 200            # ~200 tokens, fits max_state 256 alone
    r = indep.post("/v1/score", json=_body(state, 30))
    assert r.status_code == 200, r.text
    assert r.json()["usage"]["state_truncated"] is False
    r = seq.post("/v1/score", json=_body(state, 30))
    assert r.status_code == 422
