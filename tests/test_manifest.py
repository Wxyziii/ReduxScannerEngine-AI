import hashlib
import json
from pathlib import Path
import unittest


class ManifestTests(unittest.TestCase):
    def test_all_five_models_have_required_legal_and_runtime_fields(self):
        manifest=json.loads(Path("model_manifest.json").read_text(encoding="utf-8")); self.assertEqual(len(manifest["models"]),5)
        required={"modelName","purpose","sourceUrl","license","redistributionStatus","backendRuntime","hardwareTier","ramGb","vramGb","checksumSha256","downloadUrl","required","fileName"}
        for model in manifest["models"]:
            self.assertFalse(required-model.keys()); self.assertTrue(model["required"]); self.assertIn("/releases/download/",model["downloadUrl"])
            artifact=Path("dist/models-v1.0.0")/model["fileName"]; self.assertTrue(artifact.is_file()); self.assertEqual(hashlib.sha256(artifact.read_bytes()).hexdigest(),model["checksumSha256"])


if __name__ == "__main__": unittest.main()
