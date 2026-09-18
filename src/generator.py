"""Case generator — the user assembles a recipe through the UI, the server composes the whole case.

Same pattern as the escape-room builder: a recipe (visible, harmless) -> server-side
composition of the HIDDEN world (truth + tool response tables) -> structural check ->
a deterministic solver CONFIRMS the case is solvable (or deliberately unsolvable) before
any agent gets to play it. The recipe and the truth never go to the browser after creation.

Recipe:
{
  "incident_type": "theft" | "sabotage" | "data_breach" | "vandalism",
  "location": "<id from the pool or a free-form name>",
  "suspects": ["Mara", "Idris", "Lena"],        # 3-5 names (pool or free-form)
  "culprit": "Mara" | "random",
  "twist": "none" | "lying_witness" | "insufficient",
  "red_herring": true|false,
  "budget": 6-14,
  "title": "optional",
  "seed": optional int (reproducibility)
}
"""

from __future__ import annotations

import random
import re
from typing import Any

from .cases import Case

# ---------------------------------------------------------------------------
# Pools (builder options)
# ---------------------------------------------------------------------------
ROSTER = [
    ("Mara", "senior archivist"), ("Idris", "curator"), ("Lena", "conservator"),
    ("Tomas", "night guard"), ("Sara", "sysadmin"), ("Petar", "technician"),
    ("Nina", "intern"), ("Goran", "facilities manager"),
]

LOCATIONS = [
    ("archive_room", "Archive Room"), ("server_room", "Server Room"),
    ("gallery_hall", "Gallery Hall"), ("workshop", "Workshop"),
    ("research_lab", "Research Lab"), ("storage_room", "Storage Room"),
]
DECOY_LOCATION = ("main_hall", "Main Hall")

INCIDENTS: dict[str, dict[str, Any]] = {
    "theft": {
        "noun": "a valuable item", "action": "stole_the_item",
        "method": "copied_access_card", "objects": ["access_card", "display_case"],
        "report": ("a valuable item was taken from {loc} and a convincing substitute "
                   "left in its place. Nothing was forced."),
        "motives": ["gambling_debts", "hide_forgery", "private_collector_deal"],
    },
    "sabotage": {
        "noun": "the main equipment", "action": "sabotaged_the_equipment",
        "method": "tampered_power_relay", "objects": ["fuse_panel", "toolbox"],
        "report": ("the main equipment in {loc} failed after deliberate tampering. "
                   "The failure was staged to look accidental."),
        "motives": ["revenge_over_demotion", "cover_up_own_mistake", "delay_inspection"],
    },
    "data_breach": {
        "noun": "confidential records", "action": "exfiltrated_the_records",
        "method": "planted_usb_drive", "objects": ["usb_drive", "terminal"],
        "report": ("confidential records were copied from a terminal in {loc}. "
                   "No remote intrusion was found — it was done on-site."),
        "motives": ["sold_to_competitor", "blackmail_material", "cover_up_own_mistake"],
    },
    "vandalism": {
        "noun": "the exhibit", "action": "defaced_the_exhibit",
        "method": "used_service_door", "objects": ["service_door", "paint_canister"],
        "report": ("the exhibit in {loc} was deliberately damaged overnight. "
                   "Entry was through a non-public route."),
        "motives": ["grudge_against_management", "distract_from_theft", "revenge_over_demotion"],
    },
}

TWISTS = ["none", "lying_witness", "insufficient"]

# Phase 1: neutral "decoy" objects at the scene — they carry no evidence, they only widen the action space
# (analyze_object). A brute-force baseline can waste steps on them; a smart agent skips them.
DECOY_OBJECTS = ["ventilation_grate", "waste_bin", "supply_cabinet", "light_fixture"]
MAX_DECOY_OBJECTS = 4
MAX_SUSPECTS = 8


def builder_options() -> dict[str, Any]:
    return {
        "incident_types": list(INCIDENTS),
        "locations": [{"id": i, "name": n} for i, n in LOCATIONS],
        "roster": [{"name": n, "role": r} for n, r in ROSTER],
        "twists": TWISTS,
        "budget": {"min": 6, "max": 14, "default": 11},
        "suspects": {"min": 3, "max": MAX_SUSPECTS},
        # Phase 1 — levers of world complexity:
        "num_red_herrings": {"min": 0, "max": 3, "default": 1},
        "decoy_objects": {"min": 0, "max": MAX_DECOY_OBJECTS, "default": 0},
        "multi_day": {"default": False,
                      "help": "a wider, two-day time window — harder timeline reconstruction"},
        "deep_evidence": {"default": False,
                          "help": "weak physical evidence; the deciding link is a broken alibi "
                                  "(a fixed order spends many steps — a measure of the loop advantage)"},
    }


def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    return s or "x"


def _pretty(loc_id: str) -> str:
    known = dict(LOCATIONS + [DECOY_LOCATION])
    return known.get(loc_id, loc_id.replace("_", " ").title())


def validate_recipe(recipe: dict[str, Any]) -> dict[str, Any]:
    """Normalize and validate the recipe; ValueError with a clear message for bad input."""
    if not isinstance(recipe, dict):
        raise ValueError("recipe must be an object")
    r = dict(recipe)
    itype = r.get("incident_type", "theft")
    if itype not in INCIDENTS:
        raise ValueError(f"unknown incident_type '{itype}'")
    names = [str(x).strip() for x in (r.get("suspects") or []) if str(x).strip()]
    if not names:
        names = [n for n, _ in ROSTER[:3]]
    if not (3 <= len(names) <= MAX_SUSPECTS):
        raise ValueError(f"suspects: give 3-{MAX_SUSPECTS} names")
    if len({_slug(n) for n in names}) != len(names):
        raise ValueError("suspects must be distinct")
    culprit = str(r.get("culprit") or "random").strip()
    if culprit != "random" and _slug(culprit) not in {_slug(n) for n in names}:
        raise ValueError("culprit must be one of the suspects (or 'random')")
    twist = r.get("twist", "none")
    if twist not in TWISTS:
        raise ValueError(f"unknown twist '{twist}'")
    budget = int(r.get("budget") or 11)
    if not (6 <= budget <= 14):
        raise ValueError("budget must be 6-14")
    loc = _slug(str(r.get("location") or LOCATIONS[0][0]))

    # Phase 1: number of red herrings. Inherit the old bool if not given explicitly.
    # Herrings are taken from the INNOCENTS (except the one whose alibi is used to resolve),
    # so there are at most len(innocents) - 1 of them.
    max_herrings = max(0, len(names) - 2)
    if "num_red_herrings" in r:
        num_herrings = int(r.get("num_red_herrings") or 0)
    else:
        num_herrings = 1 if bool(r.get("red_herring", True)) else 0
    num_herrings = max(0, min(num_herrings, max_herrings, 3))

    decoy_objects = int(r.get("decoy_objects") or 0)
    decoy_objects = max(0, min(decoy_objects, MAX_DECOY_OBJECTS))

    return {"incident_type": itype, "location": loc, "suspects": names,
            "culprit": culprit, "twist": twist,
            "red_herring": num_herrings > 0,
            "num_red_herrings": num_herrings,
            "decoy_objects": decoy_objects,
            "multi_day": bool(r.get("multi_day", False)),
            "deep_evidence": bool(r.get("deep_evidence", False)),
            "budget": budget, "title": str(r.get("title") or "").strip(),
            "seed": r.get("seed")}


def build_case(recipe: dict[str, Any], case_id: str | None = None) -> Case:
    r = validate_recipe(recipe)
    rng = random.Random(r["seed"])
    spec = INCIDENTS[r["incident_type"]]

    names = r["suspects"]
    roles = dict(ROSTER)
    ids = [_slug(n) for n in names]
    by_id = dict(zip(ids, names))
    culprit = rng.choice(ids) if r["culprit"] == "random" else _slug(r["culprit"])
    innocents = [i for i in ids if i != culprit]
    liar = innocents[0] if r["twist"] == "lying_witness" else None
    # Phase 1: the set of red herrings. innocents[0] is reserved (its alibi resolves the case),
    # so herrings are picked from the END of the innocents list. Works for 0 and for several.
    num_herrings = min(int(r.get("num_red_herrings", 1 if r["red_herring"] else 0)),
                       max(0, len(innocents) - 1))
    herrings = innocents[len(innocents) - num_herrings:] if num_herrings else []
    herring = herrings[-1] if herrings else None  # kept for the old logic below

    loc = r["location"]
    loc_name = _pretty(loc)
    h = rng.randint(19, 22)
    win_from, win_to = f"{h}:10", f"{h}:25"
    t_incident = f"{h}:{rng.randint(14, 21)}"
    t_before = f"{h - 1}:40"
    motive = rng.choice(spec["motives"])
    obj_method, obj_scene = spec["objects"]
    insufficient = r["twist"] == "insufficient"

    # Phase 2 — "deep" case: physical evidence is DELIBERATELY weak (does not cross the threshold early), so
    # the deciding link becomes a broken alibi — which a fixed order only reaches after
    # interviews (many steps). build_and_verify still guarantees solvability within budget;
    # unsolvable combinations (culprit interviewed too late) are rejected. An info-gain agent would
    # shorten the path -> that is where the loop advantage is measured.
    deep = bool(r.get("deep_evidence")) and not insufficient
    W_ACCESS = 0.12 if deep else 0.28
    W_CAM = 0.06 if deep else 0.26
    W_OBJ = 0.10 if deep else 0.20
    W_OBJ2 = 0.08 if deep else 0.18
    W_ALIBI = 0.35 if deep else 0.15

    def ev(id_, supports, weight, related, itype_, desc, reveals=None,
           raises=None, resolves=None):
        e = {"id": id_, "supports": supports, "weight": weight,
             "relates_to_incident": related, "independent_type": itype_,
             "reveals": reveals or {}, "description": desc}
        if raises:
            e["raises_question"] = raises
        if resolves:
            e["resolves_question"] = resolves
        return e

    staff_list = ", ".join(f"{by_id[i]} ({roles.get(by_id[i], 'staff member')})"
                           for i in ids)
    multi_day_note = ""
    if r.get("multi_day"):
        multi_day_note = (" The site was accessible across two consecutive evenings, "
                          "so the exact night must be pinned down from the records.")
    report = (f"Between {win_from} and {win_to} last night, "
              + spec["report"].format(loc=f"the {loc_name.lower()}")
              + multi_day_note
              + f" Staff with reason to be nearby: {staff_list}. "
                "It was discovered this morning.")

    tools: dict[str, Any] = {
        "read_incident_report": {"observation": report, "evidence": []},
        "check_access_log": {}, "review_camera": {}, "inspect_location": {},
        "analyze_object": {}, "interview_character": {}, "verify_alibi": {},
        "compare_testimonies": {},
    }

    # ---- physical trail (access / camera / scene / objects) --------------------
    if insufficient:
        tools["check_access_log"][loc] = {
            "observation": f"The access log for the {loc_name.lower()} was disabled "
                           "for the evening event. No reliable entry records exist.",
            "evidence": [ev("al_off", None, 0.0, True, "access",
                            "Access log was disabled during the window.")]}
        tools["review_camera"][loc] = {
            "observation": "The camera was offline for maintenance during the whole "
                           "window. No usable footage exists.",
            "evidence": [ev("cam_off", None, 0.0, True, "camera",
                            "Camera offline during the incident window.")]}
        tools["inspect_location"][loc] = {
            "observation": f"The scene in the {loc_name.lower()} is contaminated: "
                           "many people passed through before it was secured. Traces "
                           "are ambiguous and cannot be attributed to one person.",
            "evidence": [ev("loc_ambiguous", culprit, 0.15, True, "scene",
                            "Ambiguous traces weakly consistent with more than one person.",
                            reveals={"action": spec["action"]}),
                         ev("loc_ambiguous2", innocents[0], 0.10, True, "scene",
                            "Some traces are also consistent with another suspect.")]}
        tools["analyze_object"][obj_method] = {
            "observation": f"The {obj_method.replace('_', ' ')} yields partial, "
                           "smudged traces. Inconclusive on its own.",
            "evidence": [ev("obj_partial", culprit, 0.10, True, "object",
                            "Partial traces — not attributable beyond doubt.")]}
        tools["analyze_object"][obj_scene] = {
            "observation": f"The {obj_scene.replace('_', ' ')} shows normal wear; "
                           "nothing probative.", "evidence": []}
    else:
        tools["check_access_log"][loc] = {
            "observation": (f"Access log for the {loc_name.lower()}: a single entry at "
                            f"{t_incident} using {by_id[culprit]}'s credentials. "
                            f"{by_id[innocents[0]]}'s credentials were last used at "
                            f"{t_before} in the {DECOY_LOCATION[1].lower()}."),
            "evidence": [
                ev("al_culprit", culprit, W_ACCESS, True, "access",
                   f"{by_id[culprit]}'s credentials used at the {loc_name.lower()} at "
                   f"{t_incident} — inside the incident window.",
                   reveals={"time": t_incident}),
                ev("al_innocent_absent", innocents[0], -0.15, True, "access",
                   f"{by_id[innocents[0]]} never entered the {loc_name.lower()} "
                   "during the window.")]}
        tools["review_camera"][loc] = {
            "observation": (f"Corridor camera at {t_incident} shows {by_id[culprit]} "
                            f"approaching the {loc_name.lower()}. No one else appears "
                            "in frame during the window."),
            "evidence": [ev("cam_culprit", culprit, W_CAM, True, "camera",
                            f"Camera places {by_id[culprit]} at the "
                            f"{loc_name.lower()} at {t_incident}.")]}
        scene_ev = [ev("loc_method_trace", culprit, 0.10, True, "scene",
                       "Scene traces show how it was done and hint at why.",
                       reveals={"action": spec["action"], "motive": motive})]
        scene_obs = (f"The scene shows no forced entry. Traces indicate the act was "
                     f"prepared in advance ({motive.replace('_', ' ')}).")
        for hg in herrings:
            scene_ev.append(ev(f"loc_herring_{hg}", hg, 0.05, False, "scene",
                               f"{by_id[hg]}'s personal item found at the scene — "
                               "but dated before the incident; a coincidence."))
        if herrings:
            names_h = ", ".join(by_id[hg] for hg in herrings)
            scene_obs += (f" Also found: personal items belonging to {names_h}, "
                          "each dated to before the incident.")
        tools["inspect_location"][loc] = {"observation": scene_obs, "evidence": scene_ev}
        tools["analyze_object"][obj_method] = {
            "observation": (f"Analysis of the {obj_method.replace('_', ' ')} shows it "
                            f"was used to carry out the act — traces tie it to "
                            f"{by_id[culprit]}."),
            "evidence": [ev("obj_method", culprit, W_OBJ, True, "object",
                            f"The {obj_method.replace('_', ' ')} ties the method to "
                            f"{by_id[culprit]}.", reveals={"method": spec["method"]})]}
        tools["analyze_object"][obj_scene] = {
            "observation": (f"The {obj_scene.replace('_', ' ')} carries secondary "
                            f"traces matching {by_id[culprit]}."),
            "evidence": [ev("obj_secondary", culprit, W_OBJ2, True, "object",
                            f"Secondary traces on the {obj_scene.replace('_', ' ')} "
                            f"match {by_id[culprit]}.",
                            reveals={"action": spec["action"]})]}

    # ---- Phase 1: neutral decoy objects (widen the action space, carry no evidence) ---
    decoy_names = DECOY_OBJECTS[: r.get("decoy_objects", 0)]
    for dname in decoy_names:
        tools["analyze_object"][dname] = {
            "observation": (f"The {dname.replace('_', ' ')} shows only routine use; "
                            "nothing connects it to the incident."),
            "evidence": []}

    # ---- statements (interview / alibi / comparison) ---------------------------
    for sid in ids:
        name = by_id[sid]
        if sid == culprit and not insufficient:
            tools["interview_character"][sid] = {
                "observation": f"{name} says they left the building at {h}:00 and "
                               "did not return.",
                "evidence": [ev(f"int_{sid}", sid, 0.0, True, "statement",
                                f"{name} claims they left at {h}:00 and never returned.",
                                raises={"suspect": sid,
                                        "note": f"{name} claims they left at {h}:00 — "
                                                "the records may say otherwise. Verify."})]}
            tools["verify_alibi"][sid] = {
                "observation": (f"{name}'s claim is false: the records place them at "
                                f"the {loc_name.lower()} at {t_incident}."),
                "evidence": [ev(f"alibi_{sid}_broken", sid, W_ALIBI, True, "alibi",
                                f"{name}'s alibi is false; they were present at the "
                                "moment of the incident.", resolves=sid)]}
        elif sid == liar:
            tools["interview_character"][sid] = {
                "observation": f"{name} insists they were at a late errand across "
                               "town and had nothing to do with it.",
                "evidence": [ev(f"int_{sid}", sid, 0.0, False, "statement",
                                f"{name}'s story sounds rehearsed.",
                                raises={"suspect": sid,
                                        "note": f"{name}'s account should be verified."})]}
            tools["verify_alibi"][sid] = {
                "observation": (f"{name}'s story is false — but only because they were "
                                "at a private appointment they hid out of embarrassment. "
                                "It is unrelated to the incident and they had no access."),
                "evidence": [ev(f"alibi_{sid}_unrelated", sid, 0.0, False, "alibi",
                                f"{name} lied, but about something unrelated to the "
                                "incident.", resolves=sid)]}
        else:
            tools["interview_character"][sid] = {
                "observation": f"{name} says they were doing routine work elsewhere "
                               "all evening.",
                "evidence": [ev(f"int_{sid}", sid, 0.0, not insufficient, "statement",
                                f"{name} reports a routine evening elsewhere.")]}
            if sid == innocents[0] and not insufficient:
                tools["verify_alibi"][sid] = {
                    "observation": f"{name}'s account checks out; two colleagues "
                                   "confirm it.",
                    "evidence": [ev(f"alibi_{sid}_ok", sid, -0.10, True, "alibi",
                                    f"{name}'s alibi is corroborated.")]}

    if not insufficient:
        pair = "|".join(sorted([culprit, innocents[0]]))
        tools["compare_testimonies"][pair] = {
            "observation": (f"{by_id[culprit]}'s account conflicts with the records; "
                            f"{by_id[innocents[0]]}'s account is consistent."),
            "evidence": [ev("cmp_pair", culprit, 0.05, True, "testimony",
                            f"{by_id[culprit]}'s story is the one that conflicts "
                            "with the record.")]}

    title = r["title"] or f"The {loc_name} Incident"
    cid = case_id or f"custom-{rng.randrange(16**6):06x}"
    expected = "unresolved" if insufficient else "solved"
    data = {
        "case_id": cid,
        "title": title,
        "difficulty": "custom",
        "short_description": (f"{spec['noun'].capitalize()} — {loc_name}. "
                              f"{len(ids)} suspects."
                              + (" Something about the trail is off." if insufficient
                                 else "")),
        "budget": r["budget"],
        "incident_report": report,
        "index": {
            "suspects": ids, "location": loc,
            "time_window": {"from": win_from, "to": win_to},
            "objects": [obj_method, obj_scene] + decoy_names,
            "characters": [{"id": i, "name": by_id[i],
                            "role": roles.get(by_id[i], "staff member")} for i in ids],
            "locations": [{"id": loc, "name": loc_name},
                          {"id": DECOY_LOCATION[0], "name": DECOY_LOCATION[1]}],
        },
        "hidden_truth": {
            "culprit": culprit, "action": spec["action"], "method": spec["method"],
            "location": loc, "time": t_incident, "motive": motive,
        },
        "required_evidence": ([] if insufficient
                              else ["al_culprit", "cam_culprit", "obj_method"]),
        "evaluation": {"expected_status": expected, "solvable": not insufficient},
        "tools": tools,
    }
    return Case(data)


def build_and_verify(recipe: dict[str, Any], case_id: str | None = None) -> Case:
    """Assemble the case and PROVE it with the deterministic solver before use:
    a solved case must be solved with the correct culprit; an insufficient one must end
    as unresolved. A broken case never reaches the agent."""
    case = build_case(recipe, case_id)
    from .loop import GameSession  # local import: avoids an import cycle
    session = GameSession(case)
    session.run()
    expected = case.evaluation["expected_status"]
    status = session.state.status
    if expected == "solved":
        report = session.state.final_report
        if status != "solved" or report.culprit != case.hidden_truth["culprit"]:
            raise ValueError(f"generated case failed self-check: status={status}, "
                             f"culprit={getattr(report, 'culprit', None)}")
    else:
        if status not in ("unresolved", "budget_exhausted"):
            raise ValueError(f"generated 'insufficient' case unexpectedly {status}")
    return case
