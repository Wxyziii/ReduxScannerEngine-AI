import unittest
from redux_ai.text_config import propose


class TextConfigTests(unittest.TestCase):
    def test_kill_effect_routes_only_to_timecycle_four(self):
        wrong = propose("blue kill effect", "animpostfx.ymt", "contrast 1.0")
        self.assertEqual(wrong["status"], "blocked")
        right = propose("blue kill effect contrast", "common/data/timecycle/timecycle_mods_4.xml", "<contrast>1.0</contrast>")
        self.assertEqual(right["status"], "proposed")
        self.assertFalse(right["sourceModified"])

    def test_sky_proposal_is_structured_and_non_mutating(self):
        source = "fog_density 0.5\nexposure 1.0\nunknown 9.0"
        result = propose("make sky darker and foggy", "visualsettings.dat", source)
        self.assertGreaterEqual(len(result["candidates"]), 2)
        self.assertTrue(result["engineHasFinalAuthority"])
        self.assertEqual(source, "fog_density 0.5\nexposure 1.0\nunknown 9.0")


if __name__ == "__main__": unittest.main()
