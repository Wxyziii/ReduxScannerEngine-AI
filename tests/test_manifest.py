import json
from pathlib import Path
import unittest


class ManifestTests(unittest.TestCase):
    def test_all_five_models_have_required_legal_runtime_and_adapter_fields(self):
        manifest=json.loads(Path("model_manifest.json").read_text(encoding="utf-8"));self.assertEqual(manifest["schemaVersion"],"redux-maker.model-manifest.v2");self.assertEqual(len(manifest["models"]),5)
        required={"id","modelName","purpose","sourceUrl","license","redistributionStatus","adapter","backendRuntime","hardwareTier","ramGb","vramGb","version","checksum","source","required"}
        for model in manifest["models"]:
            self.assertFalse(required-model.keys());self.assertTrue(model["required"]);self.assertIn(model["adapter"],{"gguf","onnx","redux_custom","diffusers","transformers"});self.assertTrue(model["checksum"])
        self.assertEqual({model["id"] for model in manifest["models"]},{"planner-router-v1","classifier-router-v1","text-config-v1","image-worker-v1","vision-validator-v1"})

    def test_classifier_release_artifact_matches_manifest(self):
        import hashlib
        manifest=json.loads(Path("model_manifest.json").read_text(encoding="utf-8"));model=next(item for item in manifest["models"] if item["id"]=="classifier-router-v1")
        artifact=Path("dist/models-v1.1.0")/model["fileName"]
        if artifact.is_file():self.assertEqual("sha256:"+hashlib.sha256(artifact.read_bytes()).hexdigest(),model["checksum"])


if __name__=="__main__":unittest.main()
