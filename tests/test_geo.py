import unittest

from flarewatch.geo import find_locations


def names(text):
    return [(p["name"], p["country_code"]) for p in find_locations(text)]


class GeoTest(unittest.TestCase):
    def test_city(self):
        self.assertEqual(names("Shelter required for family of 4 in Mumbai"), [("Mumbai", "IN")])

    def test_multi_word_and_hyphenated(self):
        found = names("We are in Leogane, roads to Port-au-Prince are blocked")
        self.assertIn(("Leogane", "HT"), found)
        self.assertIn(("Port-au-Prince", "HT"), found)

    def test_all_caps_text(self):
        self.assertIn(("Carrefour", "HT"), names("PLEASE HELP US IN CARREFOUR WE NEED FOOD"))

    def test_context_disambiguation(self):
        self.assertIn(("Georgia", "US"), names("Tornado warnings in Alabama and Georgia tonight"))
        self.assertIn(("Georgia", "GE"), names("Earthquake in Georgia, Tbilisi shaken"))

    def test_common_words_are_not_places(self):
        self.assertEqual(names("That was a nice reading of the split decision"), [])
        self.assertEqual(names("Help is coming from the West"), [])

    def test_coordinates(self):
        place = find_locations("Floods in Chennai")[0]
        self.assertAlmostEqual(place["lat"], 13.08, delta=0.5)
        self.assertAlmostEqual(place["lon"], 80.27, delta=0.5)


if __name__ == "__main__":
    unittest.main()
