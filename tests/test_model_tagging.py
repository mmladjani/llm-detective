"""Model is recorded in history (so KPI averages are never mixed across models), and
generated cases are NOT shown in the default corpus but are loadable by id."""

from __future__ import annotations

from src import cases as cases_mod
from src.cases import list_cases, list_case_files
from src.loop import GameSession
from src.investigator import RuleBasedInvestigator
from src.cases import load_case
from src.evaluator import evaluate
from src import history


def test_history_index_has_model_field(tmp_path):
    case = load_case("case-001")
    s = GameSession(case, requested_mode="rule_based")
    s.investigator = RuleBasedInvestigator(s.toolbox)
    s.run()
    e = evaluate(case, s.state, s.run_metadata())
    history.record_from_trace(case, s, e, runs_dir=tmp_path / "runs")
    idx = history.load_index(runs_dir=tmp_path / "runs")
    assert "model" in idx[0]        # rule_based -> None, but the key exists for filtering


def test_default_corpus_excludes_generated():
    # list_cases() default = 4 hand-written cases, regardless of cases/generated/.
    base = list_case_files(include_generated=False)
    assert all(p.parent == cases_mod.CASES_DIR for p in base)
    assert len(list_cases()) == len(base)
    # include_generated never shrinks the set
    assert len(list_cases(include_generated=True)) >= len(list_cases())
