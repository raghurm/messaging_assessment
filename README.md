# Context-aware messaging agent

The agent reads JSONL and writes one decision per record, in the same order. Each output contains the input `task_id`, a nullable `next_message`, a nullable `next_action`, and a short `reason`. The result is a proposal for review and export. The project has no delivery or cadence execution integration.

## Set up your interview IDE

1. Extract the ZIP and open the `messaging_assessment` folder in VS Code, Cursor, or another IDE with your coding assistant enabled.
2. Use Python 3.10 or newer. On Windows, install the time-zone database with `py -m pip install tzdata`. Linux and macOS usually supply the database.
3. Run the rehearsal in the IDE terminal. On Windows, substitute `py` for `python` if needed.

```sh
python -m unittest -v
python agent.py practice.jsonl --expect-count 12 -o rehearsal.jsonl
```

Open `rehearsal.jsonl` in the editor. Each line can be copied independently, or you can copy all 12 lines together. The package includes `practice_outputs.jsonl` from a verified run. Practice rows are synthetic variants of the two supplied examples.

## Run the interview hold-out

For OpenAI mode, set `OPENAI_API_KEY` in the IDE terminal through your usual secret setup before sharing your screen. Keep the key out of files and screen-shared commands. The code uses the OpenAI [Chat Completions API with JSON mode](https://developers.openai.com/api/docs/guides/structured-outputs?api-mode=chat). Choose a compatible model with `--model`. Confirm you may process the provided cases with your chosen model provider before using hosted inference.

```sh
python agent.py holdout.jsonl --provider openai --expect-count 12 -o holdout_outputs.jsonl
```

For local processing with the English tour baseline:

```sh
python agent.py holdout.jsonl --expect-count 12 -o holdout_outputs.jsonl
```

Use the actual filename if it differs from `holdout.jsonl`. Open the output file, review any rows with “Review is required,” and copy or share the file as requested. Diagnostics and measured p95 latency go to stderr, so stdout is also clean JSONL when `-o` is omitted.

## Walk through the implementation

1. **Read the case.** Parse each nonblank JSONL line as an object. Keep a separate set of labeled demonstrations. Strip the current record’s `expected` field and exclude matching demonstrations by task ID or identical sanitized case content. Redact recognized sensitive fields and direct contact details before hosted inference.
2. **Check consent.** Honor explicit outreach blocks and require affirmative consent per supported channel. Rank eligible channels using `channel_preferences`. An outreach block or an empty eligible set ends the decision before model inference.
3. **Propose a decision.** OpenAI mode uses one in-context inference call to select no-send or draft a message, schedule, CTA, next action, and reason. Local mode chooses a comparable labeled case and derives cadence and scheduling offsets from it. The local writer uses fixed English tour templates and current-case details. Neither mode depends on a list of sample task IDs to decide what to send.
4. **Validate the proposal.** Check the message shape, channel, subject, UTC offset, time anchor, explicit quiet hours, opt-out language, CTA, and link provenance. The agent returns a no-send for invalid proposals and continues with the remaining rows. A provider failure also yields a reviewable no-send. The hosted route makes one attempt per eligible record.
5. **Export.** Preserve input order and IDs. Keep `next_action` separate so a downstream scheduler can distinguish starting a cadence from sending one message.

## Assumptions and coverage

- Historical sample timing uses the record’s timezone. The local mode learns the difference in local calendar days from the reference’s last interaction, then applies the reference’s local hour. If supplied, `input.as_of`, `input.current_time`, or `input.now` sets the earliest permissible send time. Otherwise it uses `input.last_interaction`.
- Optional `quiet_hours` in `input` or `assertions.constraints` has the shape `{"start":"20:00","end":"09:00"}` in local time, inclusive at the start. The local baseline skips a proposal inside an explicit quiet period. With no explicit window it uses 09:00–18:00 local time. Those are baseline assumptions rather than a learned policy or a legal determination.
- The supplied labels cover one SMS and one email for prospects. They teach tour intent, channel patterns, timing, opt-out style, and next actions. The local route suppresses personas and CTAs outside its coverage. Its template preserves the intent without claiming confirmed availability, amenity hours, or an active unsubscribe link. The `.example` booking URL is part of the supplied demo data.
- The assertion state names guide interpretation of the sample. Consent is enforced in code. Neutral copy and brand style are guided by the examples and prompt. These checks are not a guarantee of fair-housing compliance or comprehensive PII detection, especially in free-form fields. Do not treat the proposal as approved live outreach.
- With leave-one-out on only the two supplied cases, each case is evaluated using the other lifecycle stage. The local agent still selects an eligible channel and tour CTA but cannot recover exact scheduling and cadence from one contrasting case. The 12-row rehearsal instead tests behavior against variants with both training examples available. It is not an independent accuracy benchmark.
- Ten automated tests passed, including simulated model decisions. A local 12-row rehearsal generated seven proposed messages and five suppressions, with verified count, order, consent fallback and a daylight-saving transition. Live OpenAI inference and the interview hold-out were unavailable for verification.

## Discussion points

| Topic | How to explain it |
| --- | --- |
| “Learns from data” | The hosted agent uses in-context examples and does not update model weights. The baseline uses case matching and derives time offsets and follow-up decisions from reference labels. Consent checks and message templates are explicit implementation choices. |
| Label leakage | Keep training examples separate, exclude the current label and duplicate cases, and score on genuinely new labeled records. |
| Semantic evaluation | Score send versus suppress, channel, timing, CTA and next action structurally. Use a rubric for personalization, intent, tone and unsupported claims. A paraphrase can pass. Required constraints should be checked independently. |
| Thresholds | The CLI reports measured per-record p95 for that run. Measure hosted p95 separately on a representative workload. Personalization requires a scoring rubric and new labels. Reply classification F1 needs labeled inbound replies and a separate classification evaluation, neither of which appears in the provided cases. A safety-violation count requires a labeled safety test set and human review of failure classes. |
| Production delivery | Separate proposal from delivery. An actual sender would recheck current consent and quiet hours, use an idempotency key, and record provider delivery status before advancing cadence. |
