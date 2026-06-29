"""Validate custom Redux artifacts for release; upstream model weights are never repackaged."""
from pathlib import Path
import hashlib
import json


def main():
    root=Path("dist/models-v1.1.0");required=[root/"redux-router-head-v1.onnx",root/"redux-router-labels-v1.json"]
    missing=[str(path) for path in required if not path.is_file()]
    if missing:raise SystemExit(f"Train the classifier before release; missing: {missing}")
    checksums={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in required}
    (root/"checksums.json").write_text(json.dumps(checksums,indent=2),encoding="utf-8")
    print(json.dumps(checksums,indent=2))


if __name__=="__main__":main()
