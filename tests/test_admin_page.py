import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app import main
from tests.gate_isolation import GateIsolation


class MissingDashboardTests(GateIsolation, unittest.TestCase):
    def test_missing_admin_page_does_not_point_back_at_itself(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(main, "ROOT", Path(directory)):
                body = TestClient(main.app).get("/").json()
        self.assertNotIn("localhost:8000", str(body))
        self.assertEqual(body, {"message": "Admin dashboard not found"})


if __name__ == "__main__":
    unittest.main()
