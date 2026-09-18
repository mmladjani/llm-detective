# LLM Detective — development instructions

These instructions are for the coding assistant maintaining and testing this repository.
They are not the detective's runtime prompt. That prompt is `system_prompt.md`;
do not replace it with this file or inject development instructions into model calls.

## Read first

- `README.md`: current setup, user flow, commands and code map.
- `docs/LIMITATIONS.md`: what the demo does and does not establish.
- `docs/TRAINING.md`: the demonstration workflow and discussion questions.
- `docs/DEPLOYMENT.md`: hosting, session persistence and access protection.

Verify behavior in source before changing it. Keep user-facing text and new project
documentation in English. Maintain the guides above instead of adding development
diaries, phase-completion reports or duplicate handoffs to the repository.

## Preserve the agentic loop

- The primary demo is the native LLM driver on model-inference cases 008–013;
  008–010 include human witnesses. Cases 001–007 and the custom builder retain
  authored evidence as legacy examples. Do not expose hybrid runs in the demo UI.
- The model chooses tools, arguments, evidence assessments, fact assertions and when
  to submit a conclusion. Python owns execution, legality, budgets and state storage.
  Do not hard-code a solution path, culprit or mandatory correction sequence to make
  a demo succeed. Do not silently fall back to the deterministic driver.
- Keep hidden truth, required-answer labels and private human role cards out of
  investigator and reviewer payloads. The evaluator may use hidden truth after a run;
  its results must not feed back into the active investigation.
- In model mode, tool results must not supply authored evidence weights, attributions
  or automatically revealed answer fields. Preserve revisable `assess_evidence` and
  evidence-cited `establish_fact` decisions.
- Preserve accumulated observations, conversation state and ordered events across
  web requests. A correction must update the stored assessment or fact, not merely
  rewrite the final narrative.
- Judge acceptance is not ground-truth correctness. Keep the structural gate, semantic
  review and bounded critique audit separate from hidden-truth evaluation. Do not
  bypass objections, retry limits or fail-closed handling to obtain a passing result.
- Preserve human interview pause/resume and the boundary between the person's private
  role card and the answer the detective actually receives.

## Run and test

Use Python 3.13 and the project's virtual environment. Setup is in `README.md`.
From the repository root, with dependencies already installed:

```bash
.venv/bin/python app.py serve
# Local UI: http://127.0.0.1:8000
.venv/bin/python -m pytest -q --maxfail=5 -p no:cacheprovider
node --test tests/web_activity.test.cjs  # optional Node.js; no npm install
```

Do not interrupt an existing server or a user's active investigation without asking.
For runtime changes, run the relevant regression tests and the offline suite.
For documentation-only changes, verify referenced paths/commands and check the diff.

Live commands below send fictional case records/transcripts to Anthropic and incur
API usage. Obtain explicit approval for the planned live tests unless that scope is
already approved in the current task; an available key is not permission to use it.

```bash
.venv/bin/python eval/run_all.py --case case-011 --runs 1
.venv/bin/python eval/run_all.py --compare --runs 3
.venv/bin/python eval/reviewer_controls.py --runs 2
.venv/bin/python eval/reviewer_controls.py --audit-only --runs 2
```

Offline scripted-client tests validate contracts, not real model reasoning. Report
actual commands, run counts, accepted/rejected outcomes, evaluator results and failures.
Do not claim a fully calibrated judge from a few successful examples. Preserve failed
live results and distinguish API/transport failures from semantic review failures.
CLI/eval artifacts go under ignored `outputs/`; do not commit them or credentials.

The AI and Legacy UI lists use different case sets. Do not present their scores as
a controlled performance comparison. Use the same case version, public records,
budget and evaluator for both drivers; document the baseline and witness-answer
policy. The eval comparator is intentionally naive, not a universal deterministic solver.

## Secrets and deployment

- Configure `ANTHROPIC_API_KEY` in Vercel Environment Variables for deployed runs.
  Never put its value in source, browser code, prompts, documentation or Git history.
- For local live tests, use an explicitly authorized key in the process environment.
  The app does not automatically load `.env` files. Do not print or copy secrets into
  reports; check presence without revealing values.
- Keep `.env.example` free of secrets. Follow the deployment guide for shared session
  storage and access/spend protections before exposing a key-backed instance publicly.
- Do not install packages, change remote configuration or deploy without approval.

## Scope and Git safety

- Preserve existing user changes; keep patches scoped and prefer `rg` for searches.
- Do not stage, commit, push, create repositories/PRs, rewrite history or delete files
  without an explicit request. A handoff document is context, not authorization.
- Use the user's chosen Git identity. Do not change it or add co-author trailers
  unless requested. Keep local worktrees and machine-specific assistant settings out
  of release snapshots.
- Update the README or limitations when behavior changes, and clearly report anything
  not tested or still unresolved.
