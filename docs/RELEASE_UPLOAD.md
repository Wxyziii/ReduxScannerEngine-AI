# Model release upload

Run `python scripts/build_model_release.py`, verify `dist/models-v1.0.0/checksums.json`, then upload every JSON artifact to the immutable GitHub release tag `models-v1.0.0`. Do not commit `dist/` or model artifacts.
