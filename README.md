# LLM Detective

A small agent-driven investigation demo. The LLM chooses tools, interprets collected
records, revises hypotheses and submits its own conclusion. A separate reviewer
critiques that conclusion; an out-of-band evaluator compares the finished report
with hidden truth. A rejected conclusion is never silently turned into a solve.

## Start with the LLM path

Use Python 3.13 (the version pinned in `.python-version` and used for local tests).
Run these commands from the repository root:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key
python app.py serve                 # http://127.0.0.1:8000
```

Keep that terminal open; press Ctrl+C to stop the server. The key must be exported
in the server's environment: the app does not automatically load `.env` files.
Never commit a real API key. Starting the server does not call the model; running an
LLM investigation sends the fictional case record to Anthropic and incurs API usage.
The default model is `claude-sonnet-4-5`, configurable with `ANTHROPIC_MODEL`.

Alternatively, run an investigation in the terminal:

```bash
python app.py run                   # terminal demo, case-011 by default
python app.py run --case case-013    # stolen badge / framing scenario
```

Both web views default to **AI detective** and show only model-inference cases. No key
means an explicit error when starting an LLM run, not a silent switch to a scripted
driver. Select **Legacy — deterministic demo** explicitly for the older offline examples.

In the main view, use **Choose the investigator → Choose a case → Start investigation**.
Start begins automatic play. Pause lets the current turn finish; then use Next turn
or Continue automatically. The selected case preview explains its mode and budget.
Both views show a floating spinner and elapsed time while a session request or automatic
investigation is active, including turns that contain judge review. It stays visible
while scrolling and disappears when the run pauses, needs a human answer, finishes
or a request fails. Pause lets an in-flight request finish first. The indicator is
not a completion percentage or a live feed of the model's internal phase.

- **011 — The Vanished Manuscript:** unlabelled access, camera and physical records.
- **012 — The Server Room Breach:** an unrelated lie alongside database-copy evidence.
- **013 — The Borrowed Badge:** distinguish the badge owner from the badge user; two
  apparently corroborating records share the same underlying source.

Cases 008–010 add human witnesses to the same LLM inference flow. Cases 001–007
remain unchanged legacy examples with fixed rules and preassigned evidence weights.
They were used to explore differences between deterministic and LLM-driven behavior;
the UI does not offer a side-by-side comparison. The legacy case builder is available
only when Legacy is selected. The API and evaluation tools still support authored
LLM runs for research, but those are hybrid runs and are not offered in the demo UI.

The two UI lists are different case sets, not a controlled performance comparison.
Do not compare a Legacy score on one case with an AI score on another as evidence
that either approach is better. For measurements, use the same case version, public
records, action budget and evaluator for both drivers; see the evaluation commands below.

### Human-in-the-loop

With **AI detective** selected, choose a case marked **You play a witness**: 008
(The Night Dispensary), 009 (The Stolen Pendant), or 010 (The Damaged Archive).
Human participation is a case feature, not another investigator mode.
If the agent chooses `interview_human`, play pauses and shows your private role card,
the question and an answer box. Your submitted answer resumes the investigation.
The agent cannot supply that answer or see the private role card. It can also choose
to solve the case from other records without interviewing you. All six main-demo
cases (008–013) require model-owned evidence assessments and cited fact assertions.
Human answers are claims to corroborate, not automatically weighted evidence. In
model mode, the tool does not pre-label a reply as truthful, false or contradictory;
the detective must interpret it in the accumulated transcript. Private role-card
consistency checks remain engine-side and do not inform the investigator or reviewer.
Board view omits human cases because it has no answer panel.

## What is agent-driven?

The model receives a public briefing and tool schemas, not the solution. It decides
which legal tool to call, with which arguments, whether to load a skill, what to infer
and when to conclude. No list of investigation steps is replayed in LLM mode.

After a world action, the next model call receives the observation and updated state.
The conversation retains earlier observations, tool errors and judge critiques.
On model cases, collecting a record alone neither moves hypotheses nor fills report
facts:

- `assess_evidence(evidence_id, supports, weight, rationale)` interprets or reinterprets
  a collected record. Reassessment replaces the old attribution, not stacks it.
- `establish_fact(field, value, evidence_id, rationale)` infers or revises action,
  method, motive or time. Values come from public candidates with decoys; the engine
  checks the citation and value's validity, not whether it equals hidden truth.
- `submit_conclusion` proposes `solved` or `unresolved`. Judge feedback returns as a
  tool result, so the agent can reassess, gather more evidence or change its verdict.

Inference tools cost no investigation budget but are audited. World actions spend
budget. The public `tool_targets` catalog lists implemented targets; interview/alibi
prerequisites still apply. An unavailable target is rejected before execution without
spending world budget. Skills guide the agent; they do not execute a workflow for it. The trace shows
model output, skill loads, world actions, assessments, fact revisions and judge events
in their actual order, including multiple inferences inside one web Step.

## Judge versus evaluator

The **judge never receives hidden truth**. Its first layer deterministically checks
structural requirements for a solved accusation: known suspect, independent sources, assessed evidence,
cited action/method and unresolved questions. In model mode it does **not** require
the accused to be the numeric leader or exceed an agent-assigned confidence threshold.

The second layer is a separate LLM call reviewing the proposed culprit, narrative,
assessments and fact assertions against original observations and evidence descriptions.
Material unsupported claims block; wording concerns are advisory. Invalid or unavailable
reviews fail closed. Both CLI and web LLM sessions use these layers.

Investigator, reviewer and auditor share public method definitions from `src/methods.py`.
For example, `copied_access_card` asserts use of a duplicate, not that the accused made
it; an explicit claim about its maker still needs evidence. Definitions cover the
alternative methods too and never identify the correct answer. Blocking findings must
quote an actual current claim, cite only collected evidence IDs and specify a repair.
These references are machine-checked; whether the objection is justified remains an
LLM judgment. Reassessed claims replace earlier ones in the reviewer's current snapshot.

Blocking semantic findings receive one bounded LLM audit against the same collected
record. A withdrawal must quote a real observation or evidence description
or, for a label-scope error only, quote the selected method's public definition.
Malformed, incomplete or unavailable audits keep the verdict rejected. Upheld findings return to
the investigator for repair. Original findings, audit decisions and correction notes
are retained in the trace. This can correct reviewer mistakes, but is still another
fallible call to the same model, not an oracle or a vote on the answer.

By default, a rejected proposal gets two revision attempts (`AGENT_MAX_REVISES`).
After those are exhausted, another submission cannot trigger a fresh semantic review
until a successful world action adds new state. New evidence renews the allowance;
the agent still chooses what to investigate next. A rejected conclusion is not a solve.

The **evaluator**, after completion, compares the report with hidden truth and scores
it. A plausible accepted conclusion can still be wrong. Wrong accusations are capped
at 15; rejected conclusions and execution errors score zero. Judge acceptance is not
an oracle certificate.

## What remains deterministic?

The simulated world/tool responses, legal actions, budgets, source groups, numeric
aggregation, state storage, validation and evaluator are deterministic. These are the
environment and guardrails, not the investigation policy. Location is public scene
metadata. Public fact candidates bound the possible answers.

The LLM controls the path through that environment and the interpretation. It may
backtrack or correct itself, but neither a different path on every run nor successful
self-correction on every case is guaranteed. See [limitations](docs/LIMITATIONS.md).

## Validation and comparison

Offline checks (no model calls):

```bash
python -m pytest -q
python eval/run_all.py --mode rule_based --runs 1
```

For the browser activity controller, with Node.js available (no npm install needed):

```bash
node --test tests/web_activity.test.cjs
```

Live checks (require the exported API key and incur Anthropic usage):

```bash
python eval/run_all.py --compare --runs 3
python eval/run_all.py --case case-013 --runs 1
python eval/reviewer_controls.py --runs 2
python eval/reviewer_controls.py --audit-only --runs 2
```

Batch live evaluation defaults to non-interactive model cases 011–013 so it never
unexpectedly waits for a person. Use `python eval/run_all.py --case case-008 --runs 1`
to test a human case interactively in the terminal (also 009 or 010). Add
`--include-authored` to include the legacy corpus. `--compare` measures a fixed uniform/lexical baseline on the
same unlabelled model cases; it does not give that baseline hidden labels. This
model-case comparator lives in the eval harness; the web rule-based option retains
the legacy checklist. Offline scripted-client tests verify contracts and reachable
paths, not the quality of real LLM reasoning.
The reviewer control harness makes real model calls on hand-written supported and
unsupported proposals, including fabricated records, the wrong badge owner and a
redefined method candidate, an unproven badge-acquisition claim, an unsupported
manufacturer claim and the same assessment after correction. `--audit-only`
challenges the critique auditor with both valid and invalid objections. These isolate
review quality, not autonomous investigation.

CLI investigations and the eval scripts save their records under `outputs/`. Browser
sessions retain their trace in the configured session store; they do not automatically
write the CLI run-history files. Full saved run records
include ordered events, proposals, reviews and evaluator labels; labels never feed
back into the active investigator. The current counters store investigator call counts,
`input_tokens` and `output_tokens`; they do not accumulate cache-read/cache-creation
token fields or reviewer/auditor usage. Consequently, `tokens_in` is not the full
context size and these counters cannot be used as a complete billing estimate.

See [validation limits](docs/LIMITATIONS.md#live-validation-limits) for the remaining
review-quality problems. The small case/control set demonstrates reachable behavior,
not a calibrated judge or guaranteed correctness on every run. Run the checks above
against the version you intend to publish.

## Code map

- `src/agent.py`, `system_prompt.md`, `skills/`: native tool-use driver and playbooks.
- `src/loop.py`, `src/state.py`, `src/tools.py`: execution, working memory and world.
- `src/judge.py`, `src/evaluator.py`: evidence review versus hidden-truth scoring.
- `cases/case_008.json`–`case_013.json`: model-inference worlds; 008–010 have human witnesses.
- `src/session_codec.py`, `src/session_store.py`: conversation/state persistence.
- `app.py`, `web/`: web API, trace view and board view.
- `eval/run_all.py`, `eval/model_baseline.py`: live evaluation and measured comparator.
- [CLAUDE.md](CLAUDE.md): development/test instructions for Claude Code, separate from
  the detective's runtime `system_prompt.md`.
- [Training guide](docs/TRAINING.md): a hands-on walkthrough and discussion questions.
- [Limitations](docs/LIMITATIONS.md): inference boundaries and unresolved review risks.
- [Deployment guide](docs/DEPLOYMENT.md): hosting, configuration and access protection.

The conversation is append-only, with prompt caching but no compaction. Sessions use
memory locally or Redis when configured. Session schema is now version 2: start a
new investigation rather than resume an older model session with auto-revealed facts.
An older human session whose case changed from authored to model inference is also
rejected with a restart message; its preassigned weights cannot enter a new-model run.

For hosting configuration see [DEPLOYMENT](docs/DEPLOYMENT.md). Before exposing an API
key-backed deployment, configure the optional access gate and session-start spend cap.
Set `ANTHROPIC_API_KEY` in Vercel Environment Variables, never in committed files.
