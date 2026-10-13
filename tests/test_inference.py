import json
import os
import unittest

import numpy as np

from flarewatch.inference import Classifier, MAX_LEN, clean_text, extract_resource_keywords, keyword_hits

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class CleanTextTest(unittest.TestCase):
    def test_strips_urls_mentions_digits_and_stopwords(self):
        self.assertEqual(clean_text("Shelter required for family of 4 in Mumbai"), "shelter required family mumbai")
        self.assertEqual(clean_text("@ngo We NEED water!! http://x.co #flood"), "need water")

    def test_keyword_rules_are_unchanged(self):
        self.assertEqual(extract_resource_keywords("need food and water"), "food")  # first match wins, as before
        self.assertEqual(extract_resource_keywords("hello"), "unspecified resource")
        self.assertEqual(keyword_hits("need food and water"), [True, True, False, False, False, False])


class RequestModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = Classifier([ROOT])
        with open(os.path.join(ROOT, "model_metrics.json")) as f:
            cls.metrics = json.load(f)

    def test_matches_training_time_predictions(self):
        # Probes were scored at training time with the Keras-verified NumPy pass (rounded to 3 dp)
        for probe in self.metrics["probes"]:
            got = self.clf.request_model.predict(clean_text(probe["text"]))
            self.assertAlmostEqual(got, probe["probability"], delta=6e-4, msg=probe["text"])

    def test_reported_example_is_a_request(self):
        result = self.clf.classify("Shelter required for family of 4 in Mumbai")
        self.assertTrue(result["is_request"])
        self.assertEqual(result["resource"], "shelter")

    def test_casual_post_is_not_a_request(self):
        result = self.clf.classify("Enjoying the sunshine and an iced latte with friends downtown")
        self.assertFalse(result["is_request"])
        self.assertEqual(result["resource"], "None")
        self.assertEqual(result["resources"], [])

    def test_batch_equals_single(self):
        texts = [clean_text(t) for t in ["need water please", "great match yesterday", "trapped under rubble send rescue"]]
        batch = self.clf.request_model.predict_many(texts)
        single = [self.clf.request_model.predict(t) for t in texts]
        np.testing.assert_allclose(batch, single, atol=1e-6)


class ExplanationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = Classifier([ROOT])

    def test_one_weight_per_token(self):
        result = self.clf.classify("Shelter required for family of 4 in Mumbai")
        self.assertEqual([e["token"] for e in result["explanation"]], result["tokens"])
        self.assertTrue(all(np.isfinite(e["weight"]) for e in result["explanation"]))

    def test_request_words_drive_the_prediction(self):
        result = self.clf.classify("Shelter required for family of 4 in Mumbai")
        top = max(result["explanation"], key=lambda e: e["weight"])
        self.assertIn(top["token"], {"shelter", "required"})
        self.assertGreater(top["weight"], 0)

    def test_tokens_beyond_window_have_no_influence(self):
        text = " ".join(f"word{i}" for i in range(MAX_LEN + 5)).replace("word", "help")
        result = self.clf.classify(text)
        self.assertTrue(all(e["weight"] == 0.0 for e in result["explanation"][MAX_LEN:]))


class NeedsModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.clf = Classifier([ROOT])

    def test_needs_model_is_loaded(self):
        self.assertIsNotNone(self.clf.needs_model)

    def test_multiple_needs_in_one_message(self):
        result = self.clf.classify("We need food and water urgently in Chennai!")
        names = {r["name"] for r in result["resources"]}
        self.assertTrue({"food", "water"} <= names, names)

    def test_need_without_keyword(self):
        # "tents" is not a keyword for shelter; the model has to learn it
        result = self.clf.classify("Please send tents, our house collapsed and we sleep outside")
        self.assertIn("shelter", {r["name"] for r in result["resources"]})


if __name__ == "__main__":
    unittest.main()
