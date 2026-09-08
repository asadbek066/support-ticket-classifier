import unittest

from pydantic import ValidationError

from app.schemas import BatchClassificationRequest, Ticket


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


if __name__ == "__main__":
    unittest.main()
