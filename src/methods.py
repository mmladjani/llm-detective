"""Public answer vocabulary, NOT evidence or selected answers.

Investigator, reviewer and critique auditor use the same contract. Definitions
contain no case identities, observations or hidden-truth dependencies.
"""

METHOD_DEFINITIONS = {
    "copied_access_card": (
        "Access using a copied/duplicated card credential. Requires evidence that a "
        "duplicate was used, not merely that a copy existed. Does NOT assert who "
        "manufactured or obtained the copy. Link its user separately through the "
        "combined record; direct footage of the swipe is not required."
    ),
    "borrowed_key": (
        "Access using a key lent by another person with permission. Requires evidence "
        "of borrowing, not just authorized or non-forced entry with one's own credential."
    ),
    "stolen_badge": (
        "Access using another person's badge taken without permission. Requires "
        "evidence of unauthorized taking as well as use; non-ownership or absence "
        "of a permission record alone does not establish theft."
    ),
    "shared_key": (
        "Access using a key shared with the user by an authorized holder. Requires "
        "evidence of sharing, not merely that the door was opened without force."
    ),
    "forced_lock": "Access by physically forcing or defeating the lock.",
    "forced_door": "Access by physically forcing the door or its locking mechanism.",
    "remote_login": "Access to the system over a remote connection using a login.",
    "usb_drive": "Data transferred using a USB storage device.",
    "printed_records": "Data taken in the form of printed records.",
    "used_staff_badge": "Entry using a normal staff badge credential.",
    "used_service_override": "Entry using a service override credential rather than a normal staff badge.",
    "used_display_key": "Opening a display case with a display key. Does not assert who owns the key.",
    "discarded_in_bins": "Records placed in waste bins. Destruction and the identity of the person require separate evidence.",
    "burned_records": "Records destroyed by fire.",
    "removed_in_vehicle": "Records carried away in a vehicle.",
}


def method_definitions(candidates=None):
    """Return fresh public definitions; filtering uses OPTIONS, never hidden truth."""
    return {label: definition for label, definition in METHOD_DEFINITIONS.items()
            if candidates is None or label in candidates}
