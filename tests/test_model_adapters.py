from pathlib import Path
from redux_ai.model_adapters import manage


def test_adapter_rejects_unknown_and_requires_permission(tmp_path: Path):
    unknown={"id":"x","version":"1","adapter":"unknown"}
    assert manage(unknown,"install",root=tmp_path)["error"]=="unsupported_model_adapter"
    direct={"id":"x","version":"1","adapter":"gguf","source":"https://example.invalid/x","fileName":"x.gguf","checksumSha256":"0"*64}
    assert manage(direct,"install",root=tmp_path)["error"]=="explicit_download_permission_required"
