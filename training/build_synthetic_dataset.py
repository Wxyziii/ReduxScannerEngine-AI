from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

SEED = 20260629

MODULES = {
    "core_ypt_tracer": {
        "subjects": ["tracer", "bullet tracer", "tracer trail", "projectile streak"],
        "actions": ["make", "change", "generate", "edit", "replace"],
        "styles": ["purple", "cyan", "thin blue", "bright vertical", "neon pink", "low glow"],
        "targets": ["ptfx.rpf/core.ypt"], "worker": "image",
    },
    "core_ypt_hit_effect": {
        "subjects": ["Hit Effect", "hit feedback", "blood spray impact", "impact spray"],
        "actions": ["make", "change", "generate", "edit", "replace"],
        "styles": ["red", "purple", "sharp crimson", "subtle dark red", "bright magenta"],
        "targets": ["ptfx.rpf/core.ypt:ptfx_blood_spray"], "worker": "image",
    },
    "core_ypt_particle_blank": {
        "subjects": ["smoke particles", "fire particles", "heavy particle visuals", "smoke and fire textures"],
        "actions": ["blank", "hide", "make invisible", "reduce visually", "replace with transparent textures"],
        "styles": ["without deleting entries", "preserve names", "keep dimensions", "low alpha", "safe black replacement"],
        "targets": ["ptfx.rpf/core.ypt"], "worker": "image",
    },
    "timecycle_sky": {
        "subjects": ["sky", "weather mood", "fog and exposure", "cloud lighting", "timecycle visuals"],
        "actions": ["make", "change", "edit", "tune", "adjust"],
        "styles": ["darker and foggy", "cold blue", "clear with low bloom", "overcast", "warm sunset", "high visibility"],
        "targets": ["visualsettings.dat", "common/data/timecycle/*.xml"], "worker": "text_config",
    },
    "timecycle_kill_effect": {
        "subjects": ["kill effect", "kill flash", "timecycle 4 kill feedback", "death confirmation tint"],
        "actions": ["make", "change", "edit", "tune", "set"],
        "styles": ["blue flash", "purple tint", "short red pulse", "low-opacity cyan", "dark desaturation"],
        "targets": ["common/data/timecycle/timecycle_mods_4.xml"], "worker": "text_config",
    },
    "minimap_editor": {
        "subjects": ["minimap", "health and armor bars", "minimap damage flash", "HP numbers", "minimap icon"],
        "actions": ["make", "change", "edit", "configure", "design"],
        "styles": ["purple with HP numbers", "blue gradient", "glowing green", "damage flash disabled", "custom icon", "numbers above bars"],
        "targets": ["minimap.gfx", "swf.xml"], "worker": "text_config",
    },
    "package_builder": {
        "subjects": ["package", "ReduxMaker output", "rollback bundle", "accepted changes"],
        "actions": ["build", "create", "export", "validate", "assemble"],
        "styles": ["with checksums", "without local backups", "with rollback notes", "from staged files", "after acceptance"],
        "targets": [], "worker": "planner",
    },
    "validation": {
        "subjects": ["generated texture", "staged core.ypt", "edited swf.xml", "package checksums", "timecycle patch"],
        "actions": ["validate", "check", "verify", "inspect", "reject if invalid"],
        "styles": ["dimensions and format", "without writing originals", "structure preserved", "checksum integrity", "safe intended diff only"],
        "targets": [], "worker": "vision",
    },
}

PREFIXES = ["please", "for this project", "in my Redux", "locally", "using the imported baseline", "after preview"]
SUFFIXES = ["and show me before accepting", "then validate it", "but do not edit the original", "for the staged package", "using the reference image", ""]


def examples_per_label(count: int) -> list[dict]:
    rows: list[dict] = []
    for label, spec in MODULES.items():
        combinations = [
            f"{prefix} {action} the {subject} {style} {suffix}".replace("  ", " ").strip()
            for prefix in PREFIXES
            for action in spec["actions"]
            for subject in spec["subjects"]
            for style in spec["styles"]
            for suffix in SUFFIXES
        ]
        rng = random.Random(f"{SEED}:{label}")
        rng.shuffle(combinations)
        for prompt in combinations[:count]:
            rows.append({
                "id": hashlib.sha256(f"{label}:{prompt}".encode()).hexdigest()[:16],
                "prompt": prompt,
                "label": label,
                "targetFiles": spec["targets"],
                "worker": spec["worker"],
                "approvalRequired": True,
                "provenance": "synthetic-from-cited-factual-knowledgebase",
            })
    random.Random(SEED).shuffle(rows)
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("data/generated"))
    parser.add_argument("--per-label", type=int, default=240)
    args = parser.parse_args()
    rows = examples_per_label(args.per_label)
    by_label: dict[str, list[dict]] = {label: [] for label in MODULES}
    for row in rows: by_label[row["label"]].append(row)
    train: list[dict] = []; validation: list[dict] = []; test: list[dict] = []
    for label, label_rows in by_label.items():
        # Split independently per label to preserve balance and prevent duplicate leakage.
        cut_train=int(len(label_rows)*.8); cut_validation=int(len(label_rows)*.9)
        train.extend(label_rows[:cut_train]);validation.extend(label_rows[cut_train:cut_validation]);test.extend(label_rows[cut_validation:])
    for split, split_rows in [("train",train),("validation",validation),("test",test)]:
        random.Random(f"{SEED}:{split}").shuffle(split_rows);write_jsonl(args.out_dir/f"router_{split}.jsonl",split_rows)
    ids=[{row["id"] for row in split_rows} for split_rows in (train,validation,test)]
    assert not ids[0]&ids[1] and not ids[0]&ids[2] and not ids[1]&ids[2]
    report={"schemaVersion":"redux-maker.dataset-report.v1","seed":SEED,"source":"synthetic-from-cited-factual-knowledgebase","rawCommunityTextIncluded":False,"counts":{"train":len(train),"validation":len(validation),"test":len(test),"total":len(rows)},"labels":{label:len(label_rows) for label,label_rows in by_label.items()},"duplicateIdLeakage":False}
    (args.out_dir/"dataset_report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__ == "__main__": main()
