"""End-to-end tests of the Vercel entry point (api/index.py) with the Flask test client."""
import importlib.util
import os
import subprocess
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_vercel_app():
    spec = importlib.util.spec_from_file_location("vercel_index", os.path.join(ROOT, "api", "index.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VercelAppTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_vercel_app()
        cls.client = cls.module.app.test_client()

    def test_dashboard_renders_with_embedded_data(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        html = r.get_data(as_text=True)
        self.assertIn("window.INITIAL_DATA", html)
        self.assertIn('"total_tweets"', html)

    def test_predict_json_schema(self):
        r = self.client.post("/api/predict", json={"tweet": "Shelter required for family of 4 in Mumbai"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        for key in ("tweet", "cleaned", "tokens", "emergency_signals", "is_request", "probability",
                    "confidence_percent", "resource", "resources", "locations", "explanation", "status"):
            self.assertIn(key, data)
        self.assertTrue(data["is_request"])
        self.assertEqual(data["locations"][0]["name"], "Mumbai")

    def test_predict_requires_text(self):
        self.assertEqual(self.client.post("/api/predict", json={"tweet": "  "}).status_code, 400)

    def test_form_post_fallback_renders_result(self):
        r = self.client.post("/predict", data={"tweet": "Need clean drinking water and rescue boats, flood"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("is-disaster", r.get_data(as_text=True))

    def test_data_endpoints(self):
        for url in ("/api/stats", "/api/tweets?page=2&filter_type=disaster", "/api/dataset?label=disaster",
                    "/api/model-info", "/api/health", "/export/stats.json"):
            self.assertEqual(self.client.get(url).status_code, 200, url)
        stats = self.client.get("/api/stats").get_json()
        self.assertIn("confidence_tiers", stats)
        self.assertIn("locations", stats)
        info = self.client.get("/api/model-info").get_json()
        self.assertIn("needs", info)
        self.assertGreater(info["metrics"]["accuracy"], 80)

    def test_unknown_export_is_404(self):
        self.assertEqual(self.client.get("/export/secrets.txt").status_code, 404)

    def test_live_endpoint_without_network(self):
        snapshot = {"status": "ok", "posts": [], "counts": {"posts": 0, "requests": 0}, "errors": []}
        with mock.patch.object(self.module.live_feed, "get", return_value=snapshot):
            r = self.client.get("/api/live")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_vercel_rewrite_dispatch(self):
        with mock.patch.object(self.module.live_feed, "get", return_value={"status": "ok", "posts": []}):
            r = self.client.get("/api/index", headers={"x-matched-path": "/api/live"})
        self.assertEqual(r.get_json()["status"], "ok")
        r = self.client.get("/api/index", headers={"x-matched-path": "/api/stats"})
        self.assertIn("total_tweets", r.get_json())


class BundleSyncTest(unittest.TestCase):
    def test_api_copies_match_root(self):
        result = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "sync_vercel_bundle.py"), "--check"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
