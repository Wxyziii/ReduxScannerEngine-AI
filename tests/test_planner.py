import json
import unittest

from redux_ai.planner import classify, plan, plan_json, validate_plan_schema


class PlannerTests(unittest.TestCase):
    def test_required_real_prompts(self):
        cases = {
            "make purple tracers": "core_ypt_tracer",
            "make hit effect red blood spray": "core_ypt_hit_effect",
            "remove smoke and fire particles": "core_ypt_particle_blank",
            "make sky darker and foggy": "timecycle_sky",
            "change kill effect to blue flash": "timecycle_kill_effect",
            "make minimap bars purple with hp numbers": "minimap_editor",
            "build package": "package_builder",
        }
        for prompt, expected in cases.items():
            self.assertIn(expected, classify(prompt))

    def test_strict_json_and_schema(self):
        document = json.loads(plan_json("make purple tracers"))
        validate_plan_schema(document)
        self.assertTrue(document["tasks"][0]["required_user_approval"])
        self.assertEqual(plan("change kill effect to blue flash")["tasks"][0]["target_files"], ["common/data/timecycle/timecycle_mods_4.xml"])


if __name__ == "__main__":
    unittest.main()
