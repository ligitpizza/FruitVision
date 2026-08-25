import unittest

from core_modules.marketability import (
    MARKETABLE_LIFE_DAYS,
    average_member_probabilities,
    estimate_marketability,
)


class MarketabilityConsistencyTests(unittest.TestCase):
    def test_rotten_label_never_returns_positive_marketability(self):
        for fruit in MARKETABLE_LIFE_DAYS:
            with self.subTest(fruit=fruit):
                result = estimate_marketability(
                    fruit,
                    "rotten",
                    45,
                    probabilities={"ripe": 0.9, "rotten": 0.05, "unripe": 0.05},
                    blemish_percentage=0,
                    quality_grade="Grade A",
                )
                self.assertEqual(result["status"], "remove")
                self.assertEqual(result["dispatch_priority"], "remove")
                self.assertEqual(result["min_days"], 0)
                self.assertEqual(result["max_days"], 0)
                self.assertNotIn("fresh", result["action"].lower())

    def test_low_confidence_non_rotten_estimate_is_withheld(self):
        result = estimate_marketability(
            "apple", "ripe", 59.9,
            probabilities={"ripe": 0.55, "unripe": 0.25, "rotten": 0.20},
            blemish_percentage=2,
            quality_grade="Grade A",
        )
        self.assertEqual(result["status"], "inspect")
        self.assertIsNone(result["min_days"])
        self.assertIsNone(result["max_days"])

    def test_one_percent_confidence_is_not_misread_as_one_hundred(self):
        result = estimate_marketability(
            "apple", "ripe", 1,
            probabilities={"ripe": 0.40, "unripe": 0.35, "rotten": 0.25},
            blemish_percentage=2,
            quality_grade="Grade A",
        )
        self.assertEqual(result["reliability"], "low")
        self.assertIsNone(result["window"])

    def test_high_rotten_probability_conflict_is_isolated(self):
        result = estimate_marketability(
            "mango", "ripe", 70,
            probabilities={"ripe": 0.45, "unripe": 0.10, "rotten": 0.45},
            blemish_percentage=2,
            quality_grade="Grade A",
        )
        self.assertEqual(result["status"], "isolate")
        self.assertEqual(result["dispatch_priority"], "urgent")
        self.assertIsNone(result["window"])

    def test_clean_confident_ripe_fruit_gets_positive_dispatch_window(self):
        result = estimate_marketability(
            "banana", "ripe", 92,
            probabilities={"ripe": 0.92, "unripe": 0.06, "rotten": 0.02},
            blemish_percentage=2,
            quality_grade="Grade A",
        )
        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["dispatch_priority"], "high")
        self.assertGreater(result["min_days"], 0)
        self.assertGreaterEqual(result["max_days"], result["min_days"])

    def test_visible_damage_shortens_but_does_not_reverse_the_result(self):
        clean = estimate_marketability(
            "orange", "ripe", 90,
            probabilities={"ripe": 90, "unripe": 8, "rotten": 2},
            blemish_percentage=2,
            quality_grade="Grade A",
        )
        damaged = estimate_marketability(
            "orange", "ripe", 90,
            probabilities={"ripe": 90, "unripe": 8, "rotten": 2},
            blemish_percentage=18,
            quality_grade="Grade C",
        )
        self.assertLess(damaged["max_days"], clean["max_days"])
        self.assertEqual(damaged["dispatch_priority"], "urgent")

    def test_severe_surface_damage_requires_manual_inspection(self):
        result = estimate_marketability(
            "apple", "unripe", 95,
            probabilities={"unripe": 0.95, "ripe": 0.04, "rotten": 0.01},
            blemish_percentage=31,
            quality_grade="Grade C",
        )
        self.assertEqual(result["status"], "inspect")
        self.assertIsNone(result["min_days"])


class EnsembleProbabilityTests(unittest.TestCase):
    def test_member_probabilities_are_averaged_without_changing_member_results(self):
        members = {
            "one": {"proba": {"ripe": 80, "unripe": 10, "rotten": 10}},
            "two": {"proba": {"ripe": 0.60, "unripe": 0.25, "rotten": 0.15}},
            "failed": {"error": "model unavailable"},
        }
        original = {name: dict(value) for name, value in members.items()}
        averaged = average_member_probabilities(members)
        self.assertEqual(averaged, {"unripe": 17.5, "ripe": 70.0, "rotten": 12.5})
        self.assertEqual(members, original)


if __name__ == "__main__":
    unittest.main()
