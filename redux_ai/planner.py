from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


ROUTES = {
    "core_ypt_tracer": (["tracer", "bullet trail"], ["ptfx.rpf/core.ypt"], "image", "texture_compare"),
    "core_ypt_hit_effect": (["hit effect", "blood spray"], ["ptfx.rpf/core.ypt:ptfx_blood_spray"], "image", "texture_compare"),
    "core_ypt_particle_blank": (["remove smoke", "remove fire", "blank particle", "no smoke", "no fire"], ["ptfx.rpf/core.ypt"], "image", "texture_compare"),
    "timecycle_kill_effect": (["kill effect", "kill flash"], ["common/data/timecycle/timecycle_mods_4.xml"], "text_config", "sky_timecycle"),
    "timecycle_sky": (["sky", "fog", "cloud", "weather", "bloom", "exposure"], ["visualsettings.dat", "common/data/timecycle/*.xml"], "text_config", "sky_timecycle"),
    "minimap_editor": (["minimap", "health bar", "armor bar", "armour bar", "hp number"], ["minimap.gfx", "swf.xml"], "text_config", "minimap"),
    "package_builder": (["build package", "export package"], [], "planner", "package_summary"),
    "validation": (["validate", "check output"], [], "vision_validation", "validation"),
}


@dataclass(frozen=True)
class Task:
    module: str
    target_files: list[str]
    required_model_worker: str
    validation_steps: list[str]
    risk_level: str
    preview_type: str
    required_user_approval: bool
    output_artifacts: list[str]


def classify(prompt: str) -> list[str]:
    text = prompt.casefold()
    matched = [label for label, (terms, *_rest) in ROUTES.items() if any(term in text for term in terms)]
    return matched or ["validation"]


def plan(prompt: str) -> dict:
    tasks = []
    for label in classify(prompt):
        _terms, files, worker, preview = ROUTES[label]
        tasks.append(asdict(Task(
            module=label,
            target_files=list(files),
            required_model_worker=worker,
            validation_steps=["validate proposal schema", "engine validates target constraints", "user reviews preview", "re-read staged output"],
            risk_level="high" if label != "validation" else "low",
            preview_type=preview,
            required_user_approval=True,
            output_artifacts=[f"{label}_proposal.json", f"{label}_validation.json"],
        )))
    return {"schemaVersion": "redux-maker.planner.v1", "prompt": prompt, "tasks": tasks}


def plan_json(prompt: str) -> str:
    return json.dumps(plan(prompt), separators=(",", ":"), ensure_ascii=False)


def validate_plan_schema(document: dict) -> None:
    if document.get("schemaVersion") != "redux-maker.planner.v1" or not isinstance(document.get("tasks"), list):
        raise ValueError("invalid planner schema")
    required = {"module", "target_files", "required_model_worker", "validation_steps", "risk_level", "preview_type", "required_user_approval", "output_artifacts"}
    for task in document["tasks"]:
        missing = required - task.keys()
        if missing:
            raise ValueError(f"planner task missing fields: {sorted(missing)}")


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = plan_json(args.prompt)
    if args.out:
        args.out.write_text(result, encoding="utf-8")
    else:
        print(result)


if __name__ == "__main__":
    main()
