import unittest

from pydantic import ValidationError

from app.schemas import (
    BatchClassificationRequest,
    BatchClassificationResponse,
    ClassificationResponse,
    Ticket,
)


class SchemaBoundaryTests(unittest.TestCase):
    def test_ticket_fields_are_bounded(self):
        with self.assertRaises(ValidationError):
            Ticket(subject="x" * 301)

        with self.assertRaises(ValidationError):
            Ticket(description="x" * 20_001)

    def test_batch_size_is_bounded_and_non_empty(self):
        with self.assertRaises(ValidationError):
            BatchClassificationRequest(tickets=[])

        with self.assertRaises(ValidationError):
            BatchClassificationRequest(tickets=[Ticket()] * 101)

    def test_field_boundaries_are_accepted_at_the_limit(self):
        ticket = Ticket(
            subject="x" * 300,
            description="x" * 20_000,
            source_channel="x" * 32,
            customer_type="x" * 32,
            language="x" * 16,
        )

        self.assertEqual(len(ticket.subject), 300)

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(ValidationError):
            Ticket(subject="x", injected="y")

    def test_control_and_format_characters_are_rejected(self):
        for field, value in (
            ("subject", "invoice\u202egnp.txt"),
            ("subject", "zero\u200dwidth"),
            ("description", "next\x85line"),
            ("source_channel", "email\u2066"),
            ("customer_type", "enterprise\u2028"),
        ):
            with self.subTest(field=field), self.assertRaises(ValidationError):
                Ticket(**{field: value})

    def test_newlines_and_tabs_remain_valid_in_descriptions(self):
        ticket = Ticket(description="line one\nline two\tend")

        self.assertIn("\n", ticket.description)

    def test_classification_response_reason_is_bounded(self):
        response = ClassificationResponse(
            category="Billing",
            confidence=0.9,
            queue="billing",
            reason="x" * 2_000,
            human_review=False,
        )
        self.assertEqual(len(response.reason), 2_000)

        with self.assertRaises(ValidationError):
            ClassificationResponse(
                category="Billing",
                confidence=0.9,
                queue="billing",
                reason="x" * 2_001,
                human_review=False,
            )

    def test_batch_response_degraded_defaults_to_zero(self):
        response = BatchClassificationResponse(
            results=[], total=0, processing_time_ms=0.0
        )

        self.assertEqual(response.degraded, 0)


if __name__ == "__main__":
    unittest.main()
