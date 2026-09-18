#!/usr/bin/env python3
"""Information-quality (info-gain) report from saved run records.

Measures whether the agent chose INFORMATIVE moves: wasted_action_rate (wasted moves) and
info_efficiency (hypothesis moves per action). Works OFFLINE over existing history
records (both rule_based and llm), so it spends no key.

    python eval/info_quality.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src import history          # noqa: E402
from eval import metrics         # noqa: E402


def main() -> None:
    records = list(history.iter_records())
    if not records:
        print("No history records. Run e.g. `python eval/baseline.py --runs 3 "
              "--baseline-only` (rule_based, no key) then this again.")
        return

    by_mode: dict[str, list[dict]] = {}
    for r in records:
        by_mode.setdefault(r.get("mode", "?"), []).append(r)

    print(f"{'mode':<14}{'runs':<7}{'wasted_rate':<14}{'info_eff':<11}{'avg_actions'}")
    print("-" * 60)
    for mode, recs in sorted(by_mode.items()):
        q = metrics.info_quality_over_records(recs)
        print(f"{mode:<14}{q['runs']:<7}{str(q['avg_wasted_rate']):<14}"
              f"{str(q['avg_info_efficiency']):<11}{q['avg_actions']}")
    print("\nThese metrics count state changes, not semantic correctness. "
          "Compare drivers only on matched cases, inference rules and settings.")


if __name__ == "__main__":
    main()
