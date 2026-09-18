"""Live semantic reviewer controls, separate from autonomous investigator runs.

These hand-written proposals exercise acceptance AND rejection on collected records.
They are evaluation fixtures, never an investigator policy or a hidden-truth prompt.
Run explicitly (uses the configured Anthropic account):
    .venv/bin/python eval/reviewer_controls.py --runs 2
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from src.agent import MODEL
from src.cases import load_case
from src.investigator import Decision, Investigator, NextAction
from src.judge import _audit_findings, _current_claims, _evidence_digest, make_llm_conclusion_judge
from src.loop import GameSession


class _RecordCollector(Investigator):
    name = "reviewer_fixture"
    concludes_explicitly = True


def collected(case_id, actions, culprit, rationales, facts):
    s = GameSession(load_case(case_id), investigator=_RecordCollector())
    for tool, args in [("read_incident_report", {}), *actions]:
        entry = s.execute_decision(Decision(action=NextAction("unspecified", "Fixture collection", tool, args)))
        assert entry.tool_ok, entry.observation
    for evidence in s.state.evidence_collected:
        rationale = rationales.get(evidence.id)
        entry = s.apply_assessment({"evidence_id": evidence.id, "supports": culprit,
            "weight": 0.2 if rationale else 0,
            "rationale": rationale or "Not used to implicate the accused."}, Decision(action=None))
        assert entry.tool_ok, entry.observation
    for field, (value, evidence_id, rationale) in facts.items():
        entry = s.establish_fact(dict(field=field, value=value, evidence_id=evidence_id,
                                     rationale=rationale), Decision(action=None))
        assert entry.tool_ok, entry.observation
    return s.state


def controls():
    archive = collected("case-011", [
        ("check_access_log", {"location": "archive_room"}),
        ("review_camera", {"location": "archive_room"}),
        ("inspect_location", {"location": "archive_room"}),
        ("analyze_object", {"object": "access_card"}),
        ("analyze_object", {"object": "manuscript_case"}),
    ], "mara", {
        "cam_mara": "The camera identifies Mara carrying a case at the archive door at 21:16.",
        "obj_case": "The case used for the forgery carries fibers matching Mara; corroborates the camera, not proof by itself.",
        "al_mara_entry": "The credential entry follows Mara's camera-observed approach. Attribution is joint, not from card ownership alone.",
        "obj_card": "Identifies the duplicate credential used, not its holder; camera and fibers provide the personal link.",
    }, {
        "action": ("replaced_manuscript", "loc_forgery", "Modern ink in the replacement and a forgery placed using a case establish substitution."),
        "method": ("copied_access_card", "obj_card", "The archive reader identifies the credential used as a duplicate."),
    })
    sound = {"verdict": "solved", "culprit": "mara", "reasoning":
        "Mara is the best-supported explanation of the combined record: cam_mara identifies her "
        "at the archive door at 21:16 carrying a flat case, al_mara_entry records the sole entry "
        "at 21:17, obj_card identifies the duplicate credential, and obj_case links the forgery's "
        "case to her glove fibers. The camera and physical trace corroborate identity independently "
        "of card ownership. No camera sees the actual swap; fibers alone could reflect ordinary "
        "work, and a duplicate alone does not identify its user. Together they favor Mara. "
        "Lena's previous-day slip does not place her there during the window; Idris's recorded "
        "card activity was elsewhere. Framing is conceivable but has no affirmative support in "
        "the collected record. I do not claim to know who manufactured the duplicate."}
    invented = copy.deepcopy(sound)
    invented["reasoning"] = ("Mara confessed to the swap during an interview, and an inside-room "
                              "video at 21:18 directly records her replacing the manuscript. "
                              "These two records are my decisive evidence.")
    # Reproduce a live false acceptance: redefine a candidate as ordinary access,
    # with no collected object analysis to establish the actual mechanism.
    mislabeled_state = copy.deepcopy(archive)
    mislabeled_state.evidence_collected = [e for e in mislabeled_state.evidence_collected if e.id != "obj_card"]
    mislabeled_state.observations = [o for o in mislabeled_state.observations
                                     if "obj_card" not in o.get("evidence_ids", [])]
    mislabeled_state.facts_revealed["method"] = "borrowed_key"
    mislabeled_state.fact_assertions["method"] = {
        "value": "borrowed_key", "evidence_id": "al_mara_entry",
        "rationale": "Her own authorized access card counts as borrowed_key because it was legitimate access."}
    mislabeled = copy.deepcopy(sound)
    mislabeled["reasoning"] = ("The camera and case fibers support Mara. The access log "
        "records her card. Therefore the method is borrowed_key, by which I mean her "
        "own ordinary authorized credential, not an actual borrowed key. There is no "
        "evidence about whether the credential was copied or borrowed.")
    lab = collected("case-013", [
        ("check_access_log", {"location": "lab"}),
        ("review_camera", {"location": "lab"}),
        ("inspect_location", {"location": "lab"}),
        ("analyze_object", {"object": "badge_reader"}),
        ("analyze_object", {"object": "delivery_crate"}),
        ("interview_character", {"suspect": "alex"}),
        ("verify_alibi", {"suspect": "alex"}),
    ], "blair", {
        "reader_serial": "Desk footage identifies Blair taking and returning Alex's badge.",
        "crate_receipt": "The missing prototype was found in Blair's crate with a timed receipt.",
        "cam_courier": "Jacket and crate corroborate Blair; the face is hidden, so this is not facial identification.",
    }, {
        "action": ("stole_prototype", "empty_mount", "The prototype is absent from its mount and recovered in the delivery crate."),
        "method": ("stolen_badge", "reader_serial", "Blair took Alex's badge; its credential opened the lab."),
    })
    framed = {"verdict": "solved", "culprit": "blair", "reasoning":
        "Blair removed the prototype using Alex's badge. reader_serial shows Blair taking "
        "the badge and returning it; crate_receipt recovers the matching serial-numbered "
        "prototype from Blair's crate. cam_courier's jacket corroborates but its hidden face "
        "does not identify Blair on its own. Alex's continuous video alibi and the observed "
        "badge borrowing explain why the badge owner is not the user. The combined desk "
        "video and recovered object support Blair, not merely the badge display."}
    wrong_owner = {"verdict": "solved", "culprit": "alex", "reasoning":
        "Alex's badge opened the lab, so Alex must have used it and stolen the prototype. "
        "Badge ownership is conclusive; Blair's possession and Alex's video alibi are irrelevant."}
    sparse_lab = collected("case-013", [
        ("check_access_log", {"location": "lab"}),
        ("review_camera", {"location": "lab"}),
        ("inspect_location", {"location": "lab"}),
        ("analyze_object", {"object": "delivery_crate"}),
    ], "blair", {
        "crate_receipt": "The recovered prototype and signed receipt support Blair.",
        "cam_courier": "The matching jacket corroborates Blair's identity, not the badge's acquisition.",
    }, {
        "action": ("stole_prototype", "crate_receipt", "The missing prototype was recovered in Blair's crate."),
        "method": ("stolen_badge", "badge_alex", "The badge belongs to Alex but Blair used it, so it must have been stolen without permission."),
    })
    guessed_acquisition = {"verdict": "solved", "culprit": "blair", "reasoning":
        "The matching jacket, recovered prototype and signed collection receipt support Blair. "
        "Alex's badge opened the lab while the camera shows Blair entering, so the method "
        "is stolen_badge: using a non-owned badge proves it was taken without permission. "
        "I have no collected record of how Blair obtained the badge."}
    # Reproduce the user's failure: fixing the narrative alone leaves a bad stored
    # assessment. Reassessing it must remove that claim from the next review.
    repair = GameSession(load_case("case-011"), investigator=_RecordCollector())
    repair.state = copy.deepcopy(archive)
    repair.apply_assessment({"evidence_id": "obj_card", "supports": "mara", "weight": 0.3,
        "rationale": "The duplication proves Mara personally manufactured the copied card; "
                     "this preparation is decisive evidence of her guilt."}, Decision(action=None))
    overstated = copy.deepcopy(repair.state)
    repair.apply_assessment({"evidence_id": "obj_card", "supports": "mara", "weight": 0.2,
        "rationale": "The record establishes duplicate use, not its maker or holder. "
                     "The camera and glove fibers separately corroborate Mara as the user; "
                     "who made or supplied the copy remains unknown."}, Decision(action=None))
    maker = copy.deepcopy(sound)
    maker["reasoning"] = ("Mara personally manufactured the duplicate access card. "
        "The duplication record proves she made it, and this preparation proves guilt.")
    return [("supported_archive", archive, sound, True),
            ("fabricated_records", archive, invented, False),
            ("redefined_method_candidate", mislabeled_state, mislabeled, False),
            ("supported_borrowed_badge", lab, framed, True),
            ("wrong_badge_owner", lab, wrong_owner, False),
            ("unproven_badge_acquisition", sparse_lab, guessed_acquisition, False),
            ("unsupported_current_assessment", overstated, sound, False),
            ("corrected_current_assessment", repair.state, sound, True),
            ("invented_manufacturer", archive, maker, False)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--control", action="append", default=[],
                        help="run only this named control (repeatable)")
    parser.add_argument("--audit-only", action="store_true",
                        help="challenge the critique auditor with known good/bad objections")
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    import anthropic
    client = anthropic.Anthropic()
    judge = make_llm_conclusion_judge(client, MODEL)
    results = []
    fixtures = controls()
    for name, state, proposal, expected in ([] if args.audit_only else fixtures):
        if args.control and name not in args.control:
            continue
        for run in range(args.runs):
            review = judge(state, proposal)
            matched = review["ok"] == expected and not review.get("error")
            results.append(dict(control=name, run=run+1, expected_accept=expected,
                                matched=bool(matched), review=review))
            print(f"{name} {run+1}: accepted={review['ok']} expected={expected} matched={bool(matched)}", flush=True)
    if args.audit_only:
        challenges = [
            ("recorded_duplicate_denied", fixtures[0],
             "copied_access_card is unsupported because Mara already had legitimate access, "
             "so the record does not show that a duplicate was used at 21:17.", "withdraw"),
            ("invented_film_objection", fixtures[1],
             "The proposed confession and inside-room video at 21:18 were never collected.", "uphold"),
            ("wrong_owner_objection", fixtures[4],
             "Alex is accused from badge ownership while the proposal dismisses evidence "
             "of Blair taking that badge and Alex's continuous video alibi.", "uphold"),
            ("unproven_acquisition_objection", fixtures[5],
             "The stolen_badge fact is inferred solely from a non-owner's badge use. "
             "No collected record establishes how the badge was obtained or whether "
             "the owner gave permission, so that acquisition claim is unsupported.", "uphold"),
            ("redefined_method_objection", fixtures[2],
             "The borrowed_key fact is unsupported: ordinary card access and a display "
             "case opened without force do not establish use of a borrowed key. "
             "That candidate cannot be redefined as generic authorized access.", "uphold"),
            ("manufacturer_required_for_use", fixtures[7],
             "copied_access_card requires proof that Mara manufactured or acquired the "
             "copy; even recorded duplicate use with camera and fiber corroboration "
             "cannot establish that method without its maker being identified.", "withdraw"),
            ("current_manufacturer_overclaim", fixtures[6],
             "The current obj_card assessment claims the duplication proves Mara "
             "personally manufactured the card, but no collected record identifies "
             "its maker. Reassess that record to remove the unsupported attribution.", "uphold"),
        ]
        for name, (_, state, proposal, _), finding, expected in challenges:
            if args.control and name not in args.control:
                continue
            for run in range(args.runs):
                try:
                    audit = _audit_findings(client, MODEL, {
                        "proposed_conclusion": proposal, "evidence_record": _evidence_digest(state),
                        "current_claims": _current_claims(state, proposal)}, [finding])
                    matched = audit[0]["disposition"] == expected
                    result = dict(audit=audit)
                except Exception as exc:
                    matched = False
                    result = dict(error=type(exc).__name__)
                    if isinstance(exc, ValueError):
                        result["detail"] = str(exc)
                results.append(dict(control=name, run=run+1, expected_disposition=expected,
                                    matched=matched, **result))
                print(f"{name} {run+1}: expected={expected} matched={matched}", flush=True)
    if not results:
        parser.error("No matching controls in the selected mode")
    path = BASE / "outputs" / f"reviewer_controls_{time.time_ns()}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"model": MODEL, "results": results}, indent=2))
    print(f"Saved: {path}")
    return 0 if all(r["matched"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
