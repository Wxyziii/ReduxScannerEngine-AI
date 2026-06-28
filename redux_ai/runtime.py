from __future__ import annotations

from dataclasses import dataclass, asdict
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import urllib.request


@dataclass
class Hardware:
    cpu_model: str
    logical_cores: int
    physical_cores: int | None
    ram_gb: float
    gpu_model: str | None
    vram_gb: float | None
    disk_free_gb: float
    cuda: bool
    directml: bool
    vulkan: bool
    avx2: bool
    avx512: bool


def detect_hardware(workspace: Path) -> Hardware:
    memory = _memory_gb()
    gpu, vram = _gpu()
    flags = _cpu_flags()
    return Hardware(
        cpu_model=platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", "unknown"),
        logical_cores=os.cpu_count() or 1,
        physical_cores=_physical_cores(),
        ram_gb=memory,
        gpu_model=gpu,
        vram_gb=vram,
        disk_free_gb=round(shutil.disk_usage(workspace).free / 2**30, 2),
        cuda=shutil.which("nvidia-smi") is not None,
        directml=platform.system() == "Windows",
        vulkan=shutil.which("vulkaninfo") is not None,
        avx2="AVX2" in flags,
        avx512="AVX512" in flags,
    )


@dataclass
class ResourceBudget:
    max_cpu_percent: int = 90
    max_ram_gb: float | None = None
    max_vram_gb: float | None = None
    max_parallel_models: int = 2
    queue_mode: bool = True


class Scheduler:
    def __init__(self, hardware: Hardware, budget: ResourceBudget):
        self.hardware, self.budget, self.active, self.queue = hardware, budget, [], []

    def submit(self, task: dict) -> str:
        ram = float(task.get("ramGb", 0)); vram = float(task.get("vramGb", 0))
        active_ram = sum(float(item.get("ramGb", 0)) for item in self.active)
        active_vram = sum(float(item.get("vramGb", 0)) for item in self.active)
        ram_limit = self.budget.max_ram_gb or self.hardware.ram_gb * self.budget.max_cpu_percent / 100
        vram_limit = self.budget.max_vram_gb or (self.hardware.vram_gb or 0) * self.budget.max_cpu_percent / 100
        fits = len(self.active) < self.budget.max_parallel_models and active_ram + ram <= ram_limit and (vram == 0 or active_vram + vram <= vram_limit)
        if fits: self.active.append(task); return "active"
        if self.budget.queue_mode: self.queue.append(task); return "queued"
        return "rejected_resource_budget"


class ModelManager:
    def __init__(self, manifest_path: Path, model_dir: Path, github_token: str | None = None):
        self.manifest_path, self.model_dir = manifest_path, model_dir
        self.github_token = github_token
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def install(self, model_id: str, permission: bool) -> Path:
        if not permission: raise PermissionError("model download requires explicit user permission")
        model = next((item for item in self.manifest["models"] if item["id"] == model_id), None)
        if not model: raise KeyError(model_id)
        self.model_dir.mkdir(parents=True, exist_ok=True); destination=self.model_dir/model["fileName"]; partial=destination.with_suffix(destination.suffix+".partial")
        request = urllib.request.Request(model["downloadUrl"])
        if self.github_token:
            request.add_header("Authorization", f"Bearer {self.github_token}")
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output: shutil.copyfileobj(response, output)
        actual=hashlib.sha256(partial.read_bytes()).hexdigest()
        if actual != model["checksumSha256"]: partial.unlink(missing_ok=True); raise ValueError("model checksum mismatch")
        partial.replace(destination); return destination

    def status(self) -> list[dict]:
        result=[]
        for model in self.manifest["models"]:
            path=self.model_dir/model["fileName"]; actual=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            result.append({"id":model["id"],"installed":path.is_file(),"checksumValid":actual==model["checksumSha256"] if actual else False,"required":model["required"]})
        return result


def recommend_pack(hardware: Hardware) -> str:
    if (hardware.vram_gb or 0) >= 8 and hardware.ram_gb >= 16: return "full"
    if hardware.ram_gb >= 8: return "balanced"
    return "cpu-lite"


def _memory_gb() -> float:
    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_=[("dwLength",ctypes.c_ulong),("dwMemoryLoad",ctypes.c_ulong),("ullTotalPhys",ctypes.c_ulonglong),("ullAvailPhys",ctypes.c_ulonglong),("ullTotalPageFile",ctypes.c_ulonglong),("ullAvailPageFile",ctypes.c_ulonglong),("ullTotalVirtual",ctypes.c_ulonglong),("ullAvailVirtual",ctypes.c_ulonglong),("sullAvailExtendedVirtual",ctypes.c_ulonglong)]
    status=MEMORYSTATUSEX(); status.dwLength=ctypes.sizeof(status)
    return round(status.ullTotalPhys / 2**30, 2) if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)) else 0.0


def _gpu() -> tuple[str | None, float | None]:
    command=shutil.which("nvidia-smi")
    if not command: return None, None
    try:
        line=subprocess.check_output([command,"--query-gpu=name,memory.total","--format=csv,noheader,nounits"],text=True,timeout=5).splitlines()[0]
        name, mib=line.rsplit(",",1); return name.strip(), round(float(mib.strip())/1024,2)
    except Exception: return None, None


def _physical_cores() -> int | None:
    try:
        output=subprocess.check_output(["powershell.exe","-NoProfile","-Command","(Get-CimInstance Win32_Processor | Measure-Object NumberOfCores -Sum).Sum"],text=True,timeout=5)
        return int(output.strip())
    except Exception: return None


def _cpu_flags() -> str:
    try: return subprocess.check_output(["powershell.exe","-NoProfile","-Command","[System.Runtime.Intrinsics.X86.Avx2]::IsSupported; [System.Runtime.Intrinsics.X86.Avx512F]::IsSupported"],text=True,timeout=5).upper().replace("TRUE"," AVX2 ",1).replace("TRUE"," AVX512 ",1)
    except Exception: return ""
