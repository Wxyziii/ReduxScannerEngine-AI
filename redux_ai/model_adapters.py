from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request


@dataclass
class AdapterResult:
    success: bool
    adapter: str
    action: str
    model_id: str
    path: str | None = None
    checksum: str | None = None
    error: str | None = None
    recovery_action: str | None = None


def default_model_root() -> Path:
    local=os.environ.get("LOCALAPPDATA")
    if not local: raise RuntimeError("LOCALAPPDATA_is_not_available")
    return Path(local)/"ReduxMaker"/"models"


class ModelAdapter:
    name="base"
    def install(self, model: dict, root: Path, permission: bool, token: str | None = None) -> AdapterResult: raise NotImplementedError
    def remove(self, model: dict, root: Path) -> AdapterResult:
        path=root/model["id"]/model["version"]
        try: shutil.rmtree(path,ignore_errors=False)
        except FileNotFoundError: pass
        except OSError as error:return AdapterResult(False,self.name,"remove",model["id"],error=str(error),recovery_action="Stop active workers and retry")
        return AdapterResult(True,self.name,"remove",model["id"])


class DirectArtifactAdapter(ModelAdapter):
    def __init__(self,name: str):self.name=name
    def install(self,model:dict,root:Path,permission:bool,token:str|None=None)->AdapterResult:
        if not permission:return AdapterResult(False,self.name,"install",model["id"],error="explicit_download_permission_required",recovery_action="Enable downloads and retry")
        expected=model.get("checksumSha256","");name=model.get("fileName")
        if len(expected)!=64 or not name:return AdapterResult(False,self.name,"install",model["id"],error="pinned_artifact_checksum_required",recovery_action="Repair the model manifest")
        target=root/model["id"]/model["version"];target.mkdir(parents=True,exist_ok=True);destination=target/name;partial=destination.with_suffix(destination.suffix+".partial")
        request=urllib.request.Request(model["source"]);token and request.add_header("Authorization",f"Bearer {token}")
        try:
            with urllib.request.urlopen(request,timeout=300) as response,partial.open("wb") as output:shutil.copyfileobj(response,output)
            actual=_sha256(partial)
            if actual.lower()!=expected.lower():raise ValueError(f"model_checksum_failed: expected {expected}, got {actual}")
            previous=destination.with_suffix(destination.suffix+".previous");previous.unlink(missing_ok=True)
            if destination.exists():destination.replace(previous)
            partial.replace(destination)
            return AdapterResult(True,self.name,"install",model["id"],str(destination),actual)
        except Exception as error:
            partial.unlink(missing_ok=True);return AdapterResult(False,self.name,"install",model["id"],error=str(error),recovery_action="Check network, credentials, disk space, and retry")


class OllamaAdapter(ModelAdapter):
    name="ollama"
    def install(self,model:dict,root:Path,permission:bool,token:str|None=None)->AdapterResult:
        if not permission:return AdapterResult(False,self.name,"install",model["id"],error="explicit_download_permission_required",recovery_action="Enable downloads and retry")
        if not shutil.which("ollama"):return AdapterResult(False,self.name,"install",model["id"],error="ollama_not_installed",recovery_action="Install Ollama or select the built-in backend")
        source=model.get("ollamaSource")
        pull=subprocess.run(["ollama","pull",source],text=True,capture_output=True)
        if pull.returncode:return AdapterResult(False,self.name,"install",model["id"],error=pull.stderr,recovery_action="Review Ollama output and retry")
        listing=subprocess.run(["ollama","list"],text=True,capture_output=True)
        if listing.returncode or source.split(":")[0] not in listing.stdout:return AdapterResult(False,self.name,"install",model["id"],error="ollama_install_verification_failed",recovery_action="Start Ollama and repair the model")
        return AdapterResult(True,self.name,"install",model["id"],path=source)
    def remove(self,model:dict,root:Path)->AdapterResult:
        process=subprocess.run(["ollama","rm",model["ollamaSource"]],text=True,capture_output=True)
        return AdapterResult(process.returncode==0,self.name,"remove",model["id"],error=process.stderr or None)


class HuggingFaceSnapshotAdapter(ModelAdapter):
    name="huggingface"
    def install(self,model:dict,root:Path,permission:bool,token:str|None=None)->AdapterResult:
        if not permission:return AdapterResult(False,self.name,"install",model["id"],error="explicit_download_permission_required",recovery_action="Enable downloads and retry")
        executable=shutil.which("hf") or shutil.which("huggingface-cli")
        if not executable:return AdapterResult(False,self.name,"install",model["id"],error="hugging_face_cli_missing",recovery_action="Repair the model-manager runtime, then retry")
        target=root/model["id"]/model["version"];target.mkdir(parents=True,exist_ok=True)
        command=[executable,"download",model["source"],"--revision",model["version"],"--local-dir",str(target)]
        environment=dict(os.environ);token and environment.update({"HF_TOKEN":token})
        process=subprocess.run(command,text=True,capture_output=True,env=environment)
        if process.returncode:return AdapterResult(False,self.name,"install",model["id"],error=process.stderr,recovery_action="Check model access, license acceptance, disk space, and retry")
        receipt=target/"redux-maker-install.json";receipt.write_text(json.dumps({"modelId":model["id"],"source":model["source"],"revision":model["version"],"treeChecksum":_tree_checksum(target)},indent=2),encoding="utf-8")
        return AdapterResult(True,self.name,"install",model["id"],str(target),_tree_checksum(target))


ADAPTERS={"gguf":DirectArtifactAdapter("gguf"),"onnx":DirectArtifactAdapter("onnx"),"redux_custom":DirectArtifactAdapter("redux_custom"),"ollama":OllamaAdapter(),"diffusers":HuggingFaceSnapshotAdapter(),"transformers":HuggingFaceSnapshotAdapter()}


def manage(model:dict,action:str,permission:bool=False,token:str|None=None,root:Path|None=None)->dict:
    root=root or default_model_root();adapter=ADAPTERS.get(model.get("adapter"))
    if not adapter:return asdict(AdapterResult(False,str(model.get("adapter")),action,model.get("id","unknown"),error="unsupported_model_adapter",recovery_action="Use a supported manifest adapter"))
    if action in{"install","update","repair"}:result=adapter.install(model,root,permission,token)
    elif action=="remove":result=adapter.remove(model,root)
    elif action=="verify":result=verify(model,root)
    else:result=AdapterResult(False,adapter.name,action,model["id"],error="unsupported_model_action",recovery_action="Choose install, update, repair, remove, or verify")
    return asdict(result)


def verify(model:dict,root:Path)->AdapterResult:
    adapter=model["adapter"];target=root/model["id"]/model["version"]
    if adapter in{"diffusers","transformers"}:
        receipt=target/"redux-maker-install.json"
        if not receipt.is_file():return AdapterResult(False,adapter,"verify",model["id"],error="model_receipt_missing",recovery_action="Repair the model")
        document=json.loads(receipt.read_text(encoding="utf-8"));actual=_tree_checksum(target,exclude={receipt.name})
        ok=actual==document.get("treeChecksum");return AdapterResult(ok,adapter,"verify",model["id"],str(target),actual,error=None if ok else "model_checksum_failed",recovery_action=None if ok else "Repair or redownload the model")
    path=target/model["fileName"]
    if not path.is_file():return AdapterResult(False,adapter,"verify",model["id"],error="model_file_missing",recovery_action="Repair the model")
    actual=_sha256(path);ok=actual.lower()==model["checksumSha256"].lower();return AdapterResult(ok,adapter,"verify",model["id"],str(path),actual,error=None if ok else "model_checksum_failed",recovery_action=None if ok else "Repair or redownload the model")


def _sha256(path:Path)->str:
    digest=hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda:file.read(1024*1024),b""):digest.update(chunk)
    return digest.hexdigest()


def _tree_checksum(root:Path,exclude:set[str]|None=None)->str:
    exclude=exclude or set();digest=hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file() and item.name not in exclude and ".cache" not in item.parts):
        digest.update(path.relative_to(root).as_posix().encode());digest.update(b"\0");digest.update(_sha256(path).encode());digest.update(b"\n")
    return digest.hexdigest()
