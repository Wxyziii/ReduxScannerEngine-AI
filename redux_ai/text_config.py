from __future__ import annotations

import re


def propose(prompt: str, target_path: str, text: str) -> dict:
    lower = prompt.casefold()
    path = target_path.replace("\\", "/").casefold()
    if "kill effect" in lower and not path.endswith("timecycle_mods_4.xml"):
        return _blocked(target_path, "v1 kill effect must target timecycle_mods_4.xml")
    candidates = []
    groups = []
    if "blue" in lower: groups.extend(["screen_tint", "color_grading"])
    if "dark" in lower: groups.extend(["exposure", "lighting"])
    if "fog" in lower: groups.append("fog")
    if "bloom" in lower: groups.append("bloom")
    if "contrast" in lower or "kill effect" in lower: groups.append("contrast")
    for index, line in enumerate(text.splitlines()):
        line_lower = line.casefold()
        group = next((group for group in groups if group.replace("_", "") in line_lower.replace("_", "")), None)
        match = re.search(r"(?<![A-Za-z0-9_.-])(-?\d+(?:\.\d+)?)(?![A-Za-z0-9_.-])", line)
        if group and match:
            old = float(match.group(1))
            factor = 1.1 if group == "contrast" else 0.85
            candidates.append({"lineIndex": index, "group": group, "oldValue": old, "proposedValue": round(old * factor, 6), "status": "proposal_requires_engine_validation"})
    return {"schemaVersion": "redux-maker.text-config.v1", "targetFile": target_path, "sourceModified": False, "candidates": candidates, "status": "proposed" if candidates else "blocked_no_known_safe_fields", "engineHasFinalAuthority": True}


def _blocked(path: str, reason: str) -> dict:
    return {"schemaVersion": "redux-maker.text-config.v1", "targetFile": path, "sourceModified": False, "candidates": [], "status": "blocked", "blocker": reason, "engineHasFinalAuthority": True}
