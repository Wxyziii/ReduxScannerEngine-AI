from __future__ import annotations

import json
from pathlib import Path
import re


def classify_onnx(prompt: str, encoder_path: str, head_path: Path, labels_path: Path) -> dict:
    """Run BGE embeddings plus the trained Redux Maker ONNX classifier head."""
    try:
        import numpy as np
        import onnxruntime as ort
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("onnx_classifier_runtime_missing: repair ONNX Runtime and Transformers") from error
    tokenizer=AutoTokenizer.from_pretrained(encoder_path,local_files_only=True)
    encoder=AutoModel.from_pretrained(encoder_path,local_files_only=True);encoder.eval()
    tokens=tokenizer([prompt],padding=True,truncation=True,max_length=128,return_tensors="pt")
    with torch.no_grad():embedding=torch.nn.functional.normalize(encoder(**tokens).last_hidden_state[:,0],p=2,dim=1).cpu().numpy().astype(np.float32)
    session=ort.InferenceSession(str(head_path),providers=["CPUExecutionProvider"]);outputs=session.run(None,{session.get_inputs()[0].name:embedding})
    raw_label=outputs[0][0];probabilities=outputs[1][0].tolist();classes=json.loads(labels_path.read_text(encoding="utf-8"));label=str(raw_label) if isinstance(raw_label,str) else classes[int(raw_label)]
    return {"schemaVersion":"redux-maker.classifier.v2","backend":"BGE_ONNX","label":label,"confidence":round(float(max(probabilities)),4),"scores":dict(zip(classes,map(float,probabilities)))}


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
