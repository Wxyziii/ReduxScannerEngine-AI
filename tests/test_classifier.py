import tempfile
from pathlib import Path
import unittest

from redux_ai.classifier import classify, load_weights
from training.build_classifier_weights import build


class ClassifierTests(unittest.TestCase):
    def test_trained_router_classifies_required_prompts(self):
        document = build(Path("data/synthetic_training_examples.jsonl"))
        cases = {"make purple tracers": "core_ypt_tracer", "remove smoke and fire particles": "core_ypt_particle_blank", "make sky darker and foggy": "timecycle_sky", "change kill effect to blue flash": "timecycle_kill_effect", "build package": "package_builder"}
        for prompt, expected in cases.items():
            self.assertEqual(classify(prompt, document["labels"])["label"], expected)

    def test_weights_are_external_loadable_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "classifier-weights.json"
            import json
            path.write_text(json.dumps(build(Path("data/synthetic_training_examples.jsonl"))), encoding="utf-8")
            self.assertIn("core_ypt_tracer", load_weights(path))


if __name__ == "__main__":
    unittest.main()
