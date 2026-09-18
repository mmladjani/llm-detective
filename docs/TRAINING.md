# Training with LLM Detective

Use the native LLM investigator on model-inference cases 008–013 to explore how an
agent chooses actions, interprets evidence and responds to criticism. Start with
the [README](../README.md) for environment setup and read the
[limitations](LIMITATIONS.md) before presenting the demo.

## Run an investigation

1. Start the local server with `python app.py serve` and open
   `http://127.0.0.1:8000`.
2. Leave **AI detective** selected, choose case 011, and click **Start investigation**.
   Live investigations require `ANTHROPIC_API_KEY` and incur Anthropic API usage.
3. Click **Pause** to let the current turn finish. Use **Next turn** to inspect
   decisions or **Continue automatically** to resume automatic play.
4. Follow the evidence from the original observation to `assess_evidence`, then to
   any `establish_fact` assertion. Ask whether the cited record supports the claim.
5. When the agent calls `submit_conclusion`, inspect the judge's response. If the
   proposal is rejected, watch whether the agent changes its stored interpretation,
   gathers another record or submits a different conclusion.
6. After completion, compare the accepted or rejected proposal with the evaluator's
   hidden-truth result. Acceptance and correctness are different measurements.

Repeat with cases 012 and 013. Do not promise that each run will follow a different
path or successfully self-correct: the model chooses its actions within a fixed
environment, and it can still make mistakes.

## Questions to discuss

- Which choices came from the model, and which constraints came from Python?
- Did a newly collected observation change an earlier assessment or fact? Find the
  actual state change, not just a rewritten final explanation.
- What context was available before each decision? The conversation retains prior
  results and critiques; state snapshots include hypotheses, facts and open questions.
- Did the judge identify a material unsupported claim? Did the critique audit
  correctly uphold or withdraw the objection? Neither model call sees hidden truth.
- What caused the investigation to stop: an accepted proposal, an exhausted
  rejected proposal, or an execution/budget limit?
- Which conclusions remain unsupported even if the final culprit happens to match
  the evaluator's answer?

For implementation details, inspect `AgentInvestigator.decide` in `src/agent.py`,
`GameSession.apply_assessment` and `establish_fact` in `src/loop.py`, and the
review payload and gates in `src/judge.py`. `Case.public_briefing` in `src/cases.py`
defines the case-level information sent to the investigator.

## Optional evaluation tools

The demo does not require a comparison run. **Legacy — deterministic demo** retains
cases 001–007 and the custom builder as historical examples of the fixed-rule
approach originally used to explore behavioral differences from an LLM loop.

These dropdowns are not an A/B benchmark: they expose different cases. Comparing
their scores directly mixes case difficulty with driver behavior. Use the same
case version, public records, budget and evaluator for a controlled comparison;
for human cases, keep the witness-answer policy consistent as well.

From the repository root, with the environment configured as in the README:

```bash
python -m pytest -q                                # offline contract tests
python eval/run_all.py --mode rule_based --runs 1   # offline baseline
python eval/run_all.py --compare --runs 3           # paid live comparison
python eval/reviewer_controls.py --runs 2           # paid proposal-review controls
python eval/reviewer_controls.py --audit-only --runs 2  # paid critique-audit controls
```

Batch live comparison defaults to non-interactive model cases 011–013. Its deterministic comparator uses
fixed collection, uniform weights and lexical matching; it is not the web UI's
legacy checklist. Cases 001–007 and the custom builder use authored evidence, so
keep those separate from claims about model-owned inference.

Inspect correctness, missed evidence, rejected proposals and fallback metadata—not
just the final status. Scripted offline tests check contracts, not live reasoning.
Save detailed run artifacts in ignored `outputs/`, not in project-history documents.

## Human-in-the-loop exercise

In the main view, leave **AI detective** selected and choose case 008, 009 or 010,
marked **You play a witness**. This uses the same LLM investigation flow as 011–013.
If the agent chooses to interview the human character, read the private role card
and submit your answer. Only the answer reaches the investigator, not the card.
The agent may choose other evidence instead. Try withholding information or changing
your account, then inspect whether the model notices, follows up or corroborates it.
No preassigned weights or automatic contradiction labels interpret these answers for
the model. Neither questioning you nor successfully correcting a mistake is guaranteed.
Use the main view, since Board view has no human-answer panel.

For an interactive terminal test, run `python eval/run_all.py --case case-008 --runs 1`
(or 009/010). Batch evaluation excludes human cases by default to avoid waiting for input.
