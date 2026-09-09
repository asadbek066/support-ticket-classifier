import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class UiSecurityTests(unittest.TestCase):
    def test_admin_dashboard_escapes_stored_values_before_inner_html(self):
        html = (ROOT / "app" / "admin.html").read_text(encoding="utf-8")

        self.assertIn("function escapeHtml", html)
        self.assertIn("escapeHtml(ticket.subject", html)
        self.assertIn("escapeHtml(clf.category)", html)
        self.assertIn("escapeHtml(clf.queue)", html)
        self.assertNotIn("${log.ticket.subject ||", html)
        self.assertNotIn("${clf.category}", html)

    def test_test_page_escapes_model_values(self):
        html = (ROOT / "app" / "test.html").read_text(encoding="utf-8")

        self.assertIn("function escapeHtml", html)
        self.assertIn("escapeHtml(data.category)", html)
        self.assertIn("escapeHtml(data.queue)", html)
        self.assertIn("escapeHtml(data.reason", html)
        self.assertNotIn("${data.category}", html)
        self.assertNotIn("${data.queue}", html)
        self.assertNotIn("${data.reason", html)

    def test_admin_dashboard_uses_in_memory_token_for_protected_requests(self):
        html = (ROOT / "app" / "admin.html").read_text(encoding="utf-8")

        self.assertIn('id="admin-token"', html)
        self.assertIn("let adminToken = ''", html)
        self.assertIn("function adminFetch", html)
        self.assertIn("X-Admin-Token", html)
        self.assertNotIn("localStorage.setItem", html)
        self.assertNotIn("sessionStorage.setItem", html)


if __name__ == "__main__":
    unittest.main()
