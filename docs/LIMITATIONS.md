# What LLM Detective demonstrates — and what it does not

The primary demo is the native LLM driver on model-inference cases 008–013.
Cases 008–010 include human witnesses. Authored cases 001–007 remain legacy
fixtures, not evidence of model inference; the web UI keeps them in Legacy mode.

## Model-controlled decisions inside a deterministic environment

The model chooses tool order, arguments, skill loads, evidence attribution, fact
assertions, revisions and conclusion. Python executes and validates those decisions.
It still owns deterministic tool tables, legal actions, budget, source grouping,
numeric aggregation, state persistence and final scoring. The world is small and
closed; public candidate answers restrict the inference space. Location is already
given as the scene, not independently inferred.

Model-mode tools suppress authored `supports`, `weight`, `relates_to_incident` and
`reveals`. `assess_evidence` and `establish_fact` are required to populate the relevant
working state. However, prose evidence still contains author-written observations,
sometimes quite explicit identifications. This is not open-world research.

Numeric confidence is a heuristic derived from model-supplied weights, **not a
calibrated probability**. Log-odds accumulation preserves updates better than the old
linear clamp, but rounding can still display 1.0 and correlated evidence can inflate
the score. Source groups constrain the independent-source count. In model mode the
judge does not use the score or rank as a truth test.

## Judge acceptance is not correctness

The judge has a structural gate followed by an LLM reviewer. Neither receives hidden
truth. The reviewer gets raw observations, descriptions, assessment rationales, fact
assertions and the proposed narrative. It can still accept a bad inference or reject
a good one; using a separate call to the same model is not epistemic independence.

Public method definitions align the investigator, reviewer and auditor on what each
label actually asserts. Blocking findings must quote a current claim, reference valid
collected evidence IDs and name a repair. These checks prevent references to invented
or withdrawn claims from being treated as valid reviews; they do not prove that an
objection or its proposed repair is semantically correct. Malformed findings fail
closed, so review-format failures can still stop a valid conclusion from being accepted.

Blocking semantic findings get at most one critique-audit call. It can withdraw a
reviewer error only with a literal quote from a collected observation or description,
or from the selected method's public definition for a label-scope error only;
the original objection and disposition remain visible in history. Quote existence is
machine-checked, but whether it actually supports the withdrawal is still a semantic
judgment. The auditor can also err. It adds latency and API cost, and uses the same
model, so correlated mistakes remain possible. Investigator usage counters exclude
both reviewer and critique-audit calls.

Blocking findings return to the agent for revision. New evidence replenishes its
revision allowance. API/parse failures fail closed. An exhausted rejected proposal
ends as `conclusion_rejected`, never `solved`; ordinary prose cannot bypass submission.
Budget and call/iteration limits can stop the run without a verdict.
The review policy distinguishes reasonable combined inference from imagined
alternatives, and material answer-field errors from incidental wording. The live
control harness tests both false rejection and false acceptance. It is a small,
hand-written regression set, not an independent or comprehensive benchmark.

The out-of-band evaluator knows the solution and checks literal fields. Its prose
screen catches unknown identifiers and unsupported clock times, not every semantic
error. A high score on one small case is not general reasoning evidence.

## Measured deterministic comparison

The AI and Legacy dropdowns expose different case sets for demonstration. Scores
from different cases are not a controlled LLM-versus-deterministic comparison:
the evidence, difficulty and inference rules differ. A measurement must hold the
case version, available public information, world-action budget and evaluator
constant. Human-interview comparisons also need a controlled answer policy.
Report the chosen baseline and repeated LLM outcomes; a gain over a naive baseline
does not establish a gain over every deterministic approach.

Verified with `python eval/run_all.py --mode rule_based --runs 1` (2026-09-18):

| Cases | Policy | Score |
|---|---|---|
| 001–005 | authored checklist | 90 / 89 / 90 / 91 / 85 |
| 006–007 | authored checklist | 5 / 5 |
| 011 | uniform/lexical | 68 |
| 012 | uniform/lexical | 71 |
| 013 | uniform/lexical | 52 |

The model-case comparator follows the fixed collection policy, assigns uniform
positive weight to the first named suspect, and uses lexical matching/first-candidate
fallback for facts. It does not interpret exculpatory language or read hidden truth.
It is intentionally naive, not the strongest possible deterministic solver. Its
`solved` status means its own heuristic stopped; inspect correctness and score too.
The web Legacy mode uses the old checklist on cases 001–007; this comparator is
eval-only. The earlier authored scores for 008–010 do not describe their converted
model-inference versions. No web side-by-side comparison is promised.

Use `--compare --runs N` for actual LLM-versus-baseline measurements. Offline tests
with scripted clients prove that legal solution and revision paths exist, not that
a model discovers them. Small samples do not establish reliability.

## Live validation limits

After converting human cases to model inference (2026-09-19), 559 offline tests
passed and one was skipped. The browser activity controller has 18 offline Node.js
checks. Three real `claude-sonnet-4-5` investigations ran through
the web API's start, step, answer and evaluation handlers, restoring the persisted
session between requests. Each case ran once; human replies were scripted fixtures,
not an actual participant or another LLM. The investigator chose whether to interview.

| Case | Judge result | Correct answer fields | Human replies | Score |
|---|---|---|---|---|
| 008 | accepted | 6/6 | 2 cooperative replies | 85 |
| 009 | accepted | 6/6 | 1 refusal | 83 |
| 010 | accepted | 5/6: time wrong | none requested | 83 |

None used deterministic fallback or needed a rejected-proposal revision. Thus these
runs exercised autonomous collection and human pause/resume, but did not demonstrate
self-correction. Case 008 omitted one evaluator-required record (`obj_override`);
009 omitted two (`al_davis_alone`, `obj_key_davis`). These labels affect only the
post-run evaluator; the judge does not receive them.

**Known unresolved review failure:** in case 010 the agent retained `17:30`, when
Carmen selected the boxes, as the incident time after collecting the bin-opening
record at `18:00`. The final narrative mentioned both events, but the stored time
remained `17:30`. The judge accepted it without an objection; the evaluator marked
time wrong against `18:00`. This is not a passing full-answer test or evidence of a
calibrated judge. A follow-up should clarify event-time semantics and test supported
and unsupported time assertions without revealing the answer to the reviewer.

Validation on 2026-09-18 with `claude-sonnet-4-5`:

- Before the human-case conversion: 542 offline tests passed, one skipped. Scripted tests check plumbing,
  not semantic quality.
- Before the human-case conversion: nine reviewer controls run twice matched all 18 expected outcomes;
  seven auditor controls run twice matched all 14. These include accepting corrected
  assessments while rejecting explicit unsupported manufacture/acquisition claims.
- Before the human-case conversion: case 011 solved on its first review, scored 90, used 6/11 world actions,
  matched all six answer fields and collected every required evidence record.
- Two earlier case-011 trials in this fix also solved at 90; case 012 solved at 80
  but omitted the optional motive; case 013 solved at 86 with all answer fields and
  required records. None used deterministic fallback. These four trials preceded
  the final auditor quote-format correction. Case 013 gathered another record and
  revised its method after rejection; some earlier reviews also had format failures.

This is a small regression sample, not a calibrated reliability benchmark. Exact
claim and quote validation fails closed when a review is malformed, and semantic
false approvals or false rejections remain possible. Raw records and failed trials
remain in local, ignored `outputs/`; they are not bundled with the repository.
Re-run the live checks in the README against the version you intend to publish.
Further calibration needs held-out supported and unsupported conclusions with human
review of errors—not only more tuning on these cases.

## Context and presentation limits

- The full model conversation accumulates observations, errors and critiques. State
  snapshots include facts, hypotheses and open questions after tool execution.
  This supplies context at every model call; it does not guarantee thoughtful analysis.
- Prompt caching is enabled. There is no context compaction, summarization or pruning.
- Model reasoning output, actions and judge events are persisted in order. A web Step
  may execute several free inference/review calls before returning a world action or
  conclusion; this is not token-by-token streaming.
- One tool call per response is enforced for a readable demo. Extra calls return an
  error. This is not a general recommendation against parallel tool use.
- Self-correction is available, not scripted as a mandatory phase. Some runs need no
  correction; others do not successfully repair their mistakes.
- Browser controls prevent concurrent turns from one view. Server-side distributed
  per-session locking is not implemented; concurrent API clients remain a risk.
- Board view excludes human-interview cases because it has no answer panel. The main
  view includes 008–010 in the AI case list with a witness-participation label.
  They use model-owned interpretation, including noticing changes in a human's story.
  Engine-side role-card checks are heuristic and are not supplied to the model.

## Deployment boundary

Optional Basic authentication and a session-start rate cap exist, but are inactive
unless configured. A session-start cap is not a token-level billing cap. Redis/backend
deployment behavior needs validation on the target deployment. Old schema-1 sessions are
rejected with a restart message because they lack the new inference/review contract.
