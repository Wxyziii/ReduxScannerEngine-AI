from __future__ import annotations

import json
from pathlib import Path
import re
from collections import Counter, defaultdict


def build(dataset: Path) -> dict:
    labels: dict[str, Counter] = defaultdict(Counter)
    for line in dataset.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        for token in set(re.findall(r"[a-z0-9_]+", row["prompt"].casefold())):
            labels[row["label"]][token] += 1
    return {"schemaVersion": "redux-maker.classifier-weights.v1", "license": "MIT", "labels": {label: dict(tokens) for label, tokens in sorted(labels.items())}}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/synthetic_training_examples.jsonl"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.write_text(json.dumps(build(args.dataset), indent=2), encoding="utf-8")
