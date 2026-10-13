import os
import unittest
from unittest import mock

from flarewatch import live
from flarewatch.inference import Classifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def status(id_, content, bot=False, language="en", reblog=None):
    return {
        "id": id_, "url": f"https://mastodon.social/@a/{id_}", "created_at": f"2026-10-13T1{id_}:00:00Z",
        "content": content, "language": language, "reblog": reblog,
        "account": {"acct": f"user{id_}", "bot": bot},
    }


FAKE_TIMELINE = [
    status(1, "<p>We need clean drinking water in Leogane, please help</p>"),
    status(2, "<p>M4.5 earthquake 20 km S of somewhere</p>", bot=True),
    status(3, "<p>Inondation à Paris</p>", language="fr"),
    status(4, "<p>boosted</p>", reblog={"id": 9}),
    status(5, "<p>Thoughts with everyone affected by the flood&nbsp;tonight</p>"),
]


class HtmlTest(unittest.TestCase):
    def test_html_to_text(self):
        self.assertEqual(live.html_to_text("<p>Need <b>water</b><br>now&amp;here</p>"), "Need water now&here")


class MastodonTest(unittest.TestCase):
    def test_filters_bots_reblogs_and_other_languages(self):
        with mock.patch.object(live, "_get_json", return_value=FAKE_TIMELINE):
            posts = live.fetch_mastodon("mastodon.social", "flood")
        self.assertEqual([p["id"] for p in posts], ["mastodon:1", "mastodon:5"])
        self.assertEqual(posts[0]["text"], "We need clean drinking water in Leogane, please help")

    def test_bots_can_be_included(self):
        with mock.patch.object(live, "_get_json", return_value=FAKE_TIMELINE), \
                mock.patch.dict(os.environ, {"FLAREWATCH_LIVE_INCLUDE_BOTS": "1"}):
            posts = live.fetch_mastodon("mastodon.social", "flood")
        self.assertIn("mastodon:2", [p["id"] for p in posts])


class LiveFeedTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = Classifier([ROOT])

    def setUp(self):
        env = {"FLAREWATCH_LIVE_TAGS": "flood,earthquake", "FLAREWATCH_LIVE_TTL": "300"}
        self.env = mock.patch.dict(os.environ, env)
        self.env.start()
        os.environ.pop("BLUESKY_HANDLE", None)
        os.environ.pop("BLUESKY_APP_PASSWORD", None)

    def tearDown(self):
        self.env.stop()

    def test_classifies_dedupes_and_caches(self):
        with mock.patch.object(live, "_get_json", return_value=FAKE_TIMELINE) as get:
            feed = live.LiveFeed(self.clf)
            first = feed.get()
            calls = get.call_count
            second = feed.get()
        self.assertEqual(calls, 2)                 # one request per tag
        self.assertEqual(get.call_count, calls)    # second call served from cache
        self.assertTrue(second["cached"])
        self.assertEqual(first["status"], "ok")
        self.assertEqual(first["counts"]["posts"], 2)  # same posts under both tags are de-duplicated
        request = next(p for p in first["posts"] if p["id"] == "mastodon:1")
        self.assertTrue(request["is_request"])
        self.assertIn("water", request["resources"])
        self.assertEqual(request["locations"][0]["name"], "Leogane")

    def test_upstream_failure_is_reported_not_raised(self):
        with mock.patch.object(live, "_get_json", side_effect=OSError("down")):
            result = live.LiveFeed(self.clf).get()
        self.assertEqual(result["status"], "unavailable")
        self.assertEqual(result["posts"], [])
        self.assertTrue(result["errors"])

    def test_keeps_last_good_snapshot(self):
        feed = live.LiveFeed(self.clf)
        with mock.patch.object(live, "_get_json", return_value=FAKE_TIMELINE):
            feed.get()
        with mock.patch.object(live, "_get_json", side_effect=OSError("down")):
            stale = feed.get(force=True)
        self.assertEqual(stale["status"], "stale")
        self.assertEqual(stale["counts"]["posts"], 2)


if __name__ == "__main__":
    unittest.main()
