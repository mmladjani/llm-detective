"""Investigation skills — loaded from skills/*.md.

A skill is a named investigative stance the agent can load through the `get_skill`
tool. The .md files are the source of truth: edit a file and the agent's guidance
changes without touching code. This module
only indexes them: name -> one-line summary (the first non-heading line).
"""

from __future__ import annotations

from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"


def _summary(text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s and not s.startswith("#"):
            return s
    return ""


def _load() -> dict[str, str]:
    out = {}
    for f in sorted(SKILLS_DIR.glob("*.md")):
        out[f.stem] = _summary(f.read_text(encoding="utf-8"))
    return out


SKILLS: dict[str, str] = _load()


def describe(skill: str) -> str:
    return SKILLS.get(skill, "")


def read_skill(name: str) -> dict:
    """Full skill content for the `get_skill` tool. Unknown name -> available list."""
    f = SKILLS_DIR / f"{name}.md"
    if not f.exists():
        return {"ok": False, "error": f"Unknown skill '{name}'.",
                "available": sorted(SKILLS.keys())}
    return {"ok": True, "name": name, "content": f.read_text(encoding="utf-8")}
