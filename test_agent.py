import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from agent import decide, read_jsonl

EXAMPLES = read_jsonl(Path(__file__).with_name("sample.jsonl"))


def variant(index, task_id):
    row = copy.deepcopy(EXAMPLES[index])
    row.pop("expected")
    row["task_id"] = task_id
    row["input"]["profile"]["first_name"] = "Jordan"
    return row


class DecisionTests(unittest.TestCase):
    def test_short_horizon_sms(self):
        result = decide(variant(0, "short"), EXAMPLES, "offline")
        message = result["next_message"]
        self.assertEqual(message["channel"], "sms")
        self.assertEqual(message["send_at"], "2025-12-09T09:00:00-06:00")
        self.assertEqual(message["cta"]["options"], ["Thu", "Fri"])
        self.assertIn("Jordan", message["body"])
        self.assertIn("Reply STOP", message["body"])
        self.assertEqual(result["next_action"]["type"], "start_cadence")

    def test_long_horizon_email(self):
        result = decide(variant(1, "long"), EXAMPLES, "offline")
        message = result["next_message"]
        self.assertEqual(message["channel"], "email")
        self.assertEqual(message["send_at"], "2025-12-09T10:00:00-06:00")
        self.assertIn("pool", message["body"])
        self.assertIn("February", message["body"])
        self.assertEqual(message["cta"]["link"], "https://oakridge.example/tour")
        self.assertEqual(result["next_action"]["value"], 3)

    def test_all_consent_revoked_before_model(self):
        row = variant(0, "revoked")
        row["consent"] = dict.fromkeys(row["consent"], False)
        with patch("agent.openai_decision", side_effect=AssertionError("unexpected model call")):
            self.assertIsNone(decide(row, EXAMPLES, "openai")["next_message"])

    def test_global_block_before_model(self):
        row = variant(0, "blocked")
        row["input"]["do_not_contact"] = True
        with patch("agent.openai_decision", side_effect=AssertionError("unexpected model call")):
            self.assertIsNone(decide(row, EXAMPLES, "openai")["next_message"])

    def test_channel_fallback(self):
        row = variant(0, "email-fallback")
        row["consent"]["sms_opt_in"] = False
        self.assertEqual(decide(row, EXAMPLES, "offline")["next_message"]["channel"], "email")

    def test_different_property_does_not_inherit_link(self):
        row = variant(1, "different-property")
        row["input"]["property_name"] = "Sunrise Apartments"
        message = decide(row, EXAMPLES, "offline")["next_message"]
        self.assertNotIn("link", message["cta"])
        self.assertNotIn("oakridge", message["body"])

    def test_quiet_hours(self):
        row = variant(0, "quiet")
        row["input"]["quiet_hours"] = {"start": "20:00", "end": "10:00"}
        self.assertIsNone(decide(row, EXAMPLES, "offline")["next_message"])

    def test_model_receives_no_current_label_or_sensitive_profile_keys(self):
        row = copy.deepcopy(EXAMPLES[0])
        row["input"]["profile"]["phone"] = "469-555-0101"
        row["input"]["profile"]["religion"] = "redaction sentinel"
        row["expected"] = {"sentinel_label": "never send to model"}
        decision = {"next_message": None, "next_action": None, "reason": "No outreach needed."}
        with patch("agent.openai_decision", return_value=decision) as mock:
            self.assertEqual(decide(row, EXAMPLES, "openai"), decision)
        payload = mock.call_args.args[0]
        self.assertNotIn("expected", payload["case"])
        self.assertNotIn("sentinel", json.dumps(payload))
        self.assertNotIn("469-555-0101", json.dumps(payload))
        self.assertEqual(len(payload["examples"]), 1)

    def test_model_cannot_use_unconsented_channel(self):
        row = variant(0, "voice")
        message = copy.deepcopy(EXAMPLES[0]["expected"]["next_message"])
        message["channel"] = "voice"
        with patch("agent.openai_decision", return_value={
            "next_message": message, "next_action": None, "reason": "Proposed a call.",
        }):
            self.assertIn("Review is required", decide(row, EXAMPLES, "openai")["reason"])

    def test_model_link_and_contact_checks(self):
        row = variant(1, "unverified")
        message = copy.deepcopy(EXAMPLES[1]["expected"]["next_message"])
        for text in ("Use https://unverified.example/book", "Call 469-555-0101"):
            with self.subTest(text=text):
                invalid = dict(message, body=f"{message['body']} {text}")
                with patch("agent.openai_decision", return_value={
                    "next_message": invalid, "next_action": None, "reason": "Proposed email.",
                }):
                    self.assertIsNone(decide(row, EXAMPLES, "openai")["next_message"])


if __name__ == "__main__":
    unittest.main()
