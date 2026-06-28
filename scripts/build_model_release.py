from pathlib import Path
import hashlib
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.build_classifier_weights import build


def main():
    out=Path("dist/models-v1.0.0"); out.mkdir(parents=True,exist_ok=True)
    artifacts={
        "planner-router-v1.json":{"schemaVersion":"redux-maker.planner-weights.v1","license":"MIT","routes":"embedded-in-code","version":"1.0.0"},
        "classifier-router-v1.json":build(Path("data/synthetic_training_examples.jsonl")),
        "text-config-v1.json":{"schemaVersion":"redux-maker.text-config-weights.v1","license":"MIT","safeGroups":["sky","fog","clouds","bloom","exposure","color_grading","lighting","visibility","contrast","screen_tint"],"killEffectRoute":"timecycle_mods_4.xml"},
        "image-worker-v1.json":{"schemaVersion":"redux-maker.image-weights.v1","license":"MIT","generator":"procedural-rgba-v1","supportsReference":True,"maxDimension":4096},
        "vision-validator-v1.json":{"schemaVersion":"redux-maker.vision-weights.v1","license":"MIT","acceptanceThreshold":0.55,"metrics":["dimensions","prompt-color","reference-similarity"]},
    }
    checksums={}
    for name,document in artifacts.items():
        path=out/name; path.write_text(json.dumps(document,indent=2,sort_keys=True),encoding="utf-8"); checksums[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (out/"checksums.json").write_text(json.dumps(checksums,indent=2),encoding="utf-8")
    print(json.dumps(checksums,indent=2))


if __name__ == "__main__": main()
