import unittest

from c2rust_translator.experiment_profiles import (
    model_default_insight_bank_path,
    model_display_name,
    profile_choices,
    resolve_experiment_profile,
)


class ExperimentProfilesTest(unittest.TestCase):
    def test_full_profile_uses_model_json_bank_defaults(self):
        profile = resolve_experiment_profile("full")

        self.assertEqual(profile.name, "full")
        self.assertEqual(profile.display_label, "TRAIL")
        self.assertTrue(profile.insights_enabled)
        self.assertEqual(profile.default_insights_path, "")
        self.assertEqual(
            model_default_insight_bank_path("gpt-5.4-mini-2026-03-17"),
            "snapshots/insights/gpt_final_insight_bank.json",
        )
        self.assertEqual(
            model_default_insight_bank_path("ali-deepseek-v4-flash"),
            "snapshots/insights/deepseek_final_insight_bank.json",
        )
        self.assertEqual(
            model_default_insight_bank_path("kimi-k2.5"),
            "snapshots/insights/kimi_final_insight_bank.json",
        )
        self.assertEqual(profile.insight_stage, "multi")
        self.assertTrue(profile.insight_mining)
        self.assertTrue(profile.single_insight_adversarial_learning)
        self.assertTrue(profile.compositional_insight_adversarial_learning)
        self.assertEqual(profile.default_adversarial_rounds, 5)

    def test_profile_aliases_match_profile_ablation_rows(self):
        expected = {
            "multi": "full",
            "no-compositional": "without_compositional",
            "single": "without_compositional",
            "without-compositional": "without_compositional",
            "no-single": "without_single",
            "default": "without_single",
            "without-single": "without_single",
            "no-mining": "without_insight_mining",
            "baseline": "without_insight_mining",
            "off": "without_insight_mining",
            "without-insight-mining": "without_insight_mining",
        }

        for alias, canonical in expected.items():
            with self.subTest(alias=alias):
                self.assertEqual(resolve_experiment_profile(alias).name, canonical)

    def test_without_single_profile_uses_default_mined_snapshot(self):
        profile = resolve_experiment_profile("no-single")

        self.assertEqual(profile.display_label, "w/o Single Insight Adversarial Learning")
        self.assertEqual(profile.default_insights_path, "")
        self.assertEqual(profile.insight_stage, "default")
        self.assertTrue(profile.insight_mining)
        self.assertFalse(profile.single_insight_adversarial_learning)
        self.assertFalse(profile.compositional_insight_adversarial_learning)
        self.assertEqual(profile.default_adversarial_rounds, 0)

    def test_without_mining_profile_disables_insights(self):
        profile = resolve_experiment_profile("baseline")

        self.assertEqual(profile.display_label, "w/o Insight Mining")
        self.assertFalse(profile.insights_enabled)
        self.assertEqual(profile.default_insights_path, "")
        self.assertEqual(profile.insight_stage, "off")
        self.assertFalse(profile.insight_mining)
        self.assertFalse(profile.single_insight_adversarial_learning)
        self.assertFalse(profile.compositional_insight_adversarial_learning)

    def test_profile_choices_include_aliases_for_argparse(self):
        choices = profile_choices()

        self.assertIn("full", choices)
        self.assertIn("no-compositional", choices)
        self.assertIn("no-single", choices)
        self.assertIn("no-mining", choices)
        self.assertIn("baseline", choices)

    def test_model_display_names_match_profile(self):
        self.assertEqual(model_display_name("gpt-5.4-mini-2026-03-17"), "GPT-5.4-mini")
        self.assertEqual(model_display_name("gpt-5.4-mini"), "GPT-5.4-mini")
        self.assertEqual(model_display_name("kimi-k2.5"), "Kimi-K2.5")
        self.assertEqual(model_display_name("ali-deepseek-v4-flash"), "DeepSeek-V4-Flash")
        self.assertEqual(model_display_name("deepseek-v4-flash"), "DeepSeek-V4-Flash")
        self.assertEqual(model_display_name("future-model"), "future-model")


if __name__ == "__main__":
    unittest.main()
