#!/usr/bin/env python3
"""Generate COMPLEX cases (Phase 1) in cases/generated/ — where the loop has something
to beat the brute-force baseline (8 suspects, decoy objects, more red herrings,
multi-day). Each is DETERMINISTICALLY self-checked (build_and_verify) before writing, so
a broken case never reaches the agent.

Works OFFLINE (no key). The cases are NOT shown in the default corpus; include
them explicitly: `python eval/baseline.py --include-generated`.

    python eval/generate_cases.py --n 6 --seed 1
    python eval/generate_cases.py --clear        # delete previously generated
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.cases import GENERATED_DIR                # noqa: E402
from src.generator import build_and_verify, INCIDENTS, ROSTER, TWISTS  # noqa: E402

NAMES = [n for n, _ in ROSTER]  # 8 names


def _recipe(rng: random.Random, i: int, deep: bool = False) -> dict:
    # Deep cases keep the culprit reachable within budget (4-5 suspects) because the deciding
    # alibi only comes after interviews; shallow cases can go up to 8.
    n = rng.choice([4, 5]) if deep else rng.choice([6, 7, 8])
    twist = "none" if deep else rng.choice(TWISTS)  # deep needs a solvable (broken alibi) path
    return {
        "incident_type": rng.choice(list(INCIDENTS)),
        "suspects": NAMES[:n],
        "culprit": "random",
        "twist": twist,
        "num_red_herrings": rng.choice([1, 2]),
        "decoy_objects": rng.choice([2, 3, 4]),
        "multi_day": rng.random() < 0.5,
        "deep_evidence": deep,
        "budget": 14,
        "seed": rng.randrange(10**6),
        "title": f"Generated {'Deep' if deep else 'Complex'} Case {i}",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--deep", action="store_true",
                    help="deep cases: weak physical evidence, a deciding broken alibi "
                         "(baseline spends ~11 steps instead of 5 -> the loop has something to beat here)")
    ap.add_argument("--clear", action="store_true", help="delete generated/ then exit")
    args = ap.parse_args()

    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    if args.clear:
        removed = 0
        for f in GENERATED_DIR.glob("case_*.json"):
            f.unlink(); removed += 1
        print(f"Deleted {removed} generated cases.")
        return

    rng = random.Random(args.seed)
    made = 0
    tag = "deep" if args.deep else "gen"
    for i in range(1, args.n + 1):
        recipe = _recipe(rng, i, deep=args.deep)
        cid = f"{tag}-{args.seed:03d}-{i:02d}"
        try:
            case = build_and_verify(recipe, case_id=cid)  # raises if not solvable
        except Exception as exc:  # skip a bad seed, keep trying
            print(f"  skipped {cid}: {exc}")
            continue
        path = GENERATED_DIR / f"case_{cid}.json"
        path.write_text(json.dumps(case._data, indent=2, ensure_ascii=False), encoding="utf-8")
        idx = case.index
        print(f"  {cid}: {len(idx['suspects'])} suspects, "
              f"{len(idx['objects'])} objects, budget {case.budget} -> {path.name}")
        made += 1
    print(f"\nGenerated {made} cases in {GENERATED_DIR}. "
          f"Run: python eval/baseline.py --include-generated")


if __name__ == "__main__":
    main()
