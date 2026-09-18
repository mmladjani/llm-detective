"""Conclusion thresholds — the SINGLE source of truth.

These four numbers define when an accusation is "earned". Two independent places need
them and must never disagree:

  * `loop._check_stop` — decides when the engine stops the investigation, and
  * `judge.check_conclusion` — decides whether to accept the agent's conclusion.

They used to be declared twice, once in each module, with a comment in judge.py noting
that it "mirrors" the engine. A mirror is a copy, and a copy drifts: raise the bar in
one file and the engine will stop a run the judge would still reject (or the reverse),
which reads as a mysterious loop rather than a changed constant. Both modules now
import from here, so a single edit moves the whole system.
"""

from __future__ import annotations

# Minimum confidence the leading hypothesis needs before anyone can be accused.
CONFIDENCE_THRESHOLD = 0.70

# How far ahead of the runner-up the lead must be. A near-tie is not a conclusion.
LEAD_MARGIN = 0.15

# How many genuinely independent lines of evidence must support the accusation.
# Independence is by source group, not by tool — see EvidenceItem.independence_key.
MIN_INDEPENDENT = 2

# At exhausted budget: a lead at or above this is reported as "so close"
# (budget_exhausted); below it the honest answer is "unresolved".
WEAK_LEAD_FLOOR = 0.60
