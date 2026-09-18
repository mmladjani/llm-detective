"""Case generator: structure, the hidden boundary, solvability self-check, API flow."""

import json

import pytest
from fastapi.testclient import TestClient

from app import app
from src.generator import build_and_verify, build_case, builder_options, validate_recipe
from src.loop import GameSession

client = TestClient(app)

RECIPE = {"incident_type": "theft", "location": "archive_room",
          "suspects": ["Mara", "Idris", "Lena"], "culprit": "Mara",
          "twist": "none", "red_herring": True, "budget": 9, "seed": 42}


def test_recipe_validation_rejects_bad_input():
    with pytest.raises(ValueError):
        validate_recipe({"incident_type": "arson"})
    with pytest.raises(ValueError):
        validate_recipe({"suspects": ["A", "B"]})                  # premalo
    with pytest.raises(ValueError):
        validate_recipe({"suspects": ["A", "B", "C"], "culprit": "X"})
    with pytest.raises(ValueError):
        validate_recipe({"suspects": ["A", "B", "C"], "budget": 99})


def test_generated_case_keeps_truth_out_of_briefing():
    case = build_case(RECIPE)
    from src.methods import METHOD_DEFINITIONS
    public = case.public_briefing()
    assert public.pop("method_definitions") == METHOD_DEFINITIONS
    briefing = json.dumps(public).lower()
    truth = case.hidden_truth
    assert "hidden" not in briefing
    for token in (truth["method"], truth["motive"], truth["time"]):
        assert token.lower() not in briefing


def test_solved_case_is_actually_solvable_with_right_culprit():
    case = build_and_verify(RECIPE)
    session = GameSession(case)
    session.run()
    assert session.state.status == "solved"
    assert session.state.final_report.culprit == "mara"


def test_insufficient_case_ends_unresolved():
    case = build_and_verify({**RECIPE, "twist": "insufficient", "culprit": "random"})
    session = GameSession(case)
    session.run()
    assert session.state.status in ("unresolved", "budget_exhausted")
    assert session.state.final_report.culprit is None


def test_lying_witness_is_not_convicted():
    case = build_and_verify({**RECIPE, "twist": "lying_witness", "culprit": "Mara",
                             "suspects": ["Mara", "Idris", "Lena", "Tomas"]})
    session = GameSession(case)
    session.run()
    assert session.state.final_report.culprit == "mara"   # the liar (idris) is not accused


def test_seed_makes_generation_reproducible():
    a = build_case(RECIPE)._data if hasattr(build_case(RECIPE), "_data") else None
    c1, c2 = build_case(RECIPE, case_id="x"), build_case(RECIPE, case_id="x")
    assert c1.hidden_truth == c2.hidden_truth
    assert c1.public_briefing() == c2.public_briefing()


def test_api_custom_case_end_to_end():
    r = client.post("/api/custom_case", json=RECIPE)
    assert r.status_code == 200
    cid = r.json()["case_id"]
    assert "hidden" not in r.text.lower()
    sid = client.post("/api/sessions",
                      json={"case_id": cid, "mode": "rule_based"}).json()["session_id"]
    # Hidden truth is blocked while the investigation is active.
    assert client.get(f"/api/sessions/{sid}/evaluation").status_code == 409
    client.post(f"/api/sessions/{sid}/run")
    ev = client.get(f"/api/sessions/{sid}/evaluation").json()
    assert ev["evaluation"]["total_score"] >= 80
    assert ev["hidden_truth"]["culprit"] == "mara"


def test_api_rejects_invalid_recipe():
    assert client.post("/api/custom_case",
                       json={"suspects": ["A"]}).status_code == 400


def test_builder_options_shape():
    o = builder_options()
    assert set(o) >= {"incident_types", "locations", "roster", "twists", "budget"}
    assert len(o["roster"]) >= 6
