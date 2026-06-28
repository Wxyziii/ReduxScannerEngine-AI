from __future__ import annotations

import json
from pathlib import Path
import re


def load_weights(path: Path) -> dict[str, dict[str, float]]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schemaVersion") != "redux-maker.classifier-weights.v1":
        raise ValueError("unsupported classifier weights")
    return document["labels"]


def classify(prompt: str, weights: dict[str, dict[str, float]]) -> dict:
    tokens = re.findall(r"[a-z0-9_]+", prompt.casefold())
    scores = {label: sum(features.get(token, 0.0) for token in tokens) for label, features in weights.items()}
    label, score = max(scores.items(), key=lambda pair: (pair[1], pair[0]))
    if score <= 0:
        label, score = "validation", 0.1
    total = sum(max(value, 0.0) for value in scores.values()) or score
    return {"schemaVersion": "redux-maker.classifier.v1", "label": label, "confidence": round(min(1.0, score / max(total, score)), 4), "scores": scores}
