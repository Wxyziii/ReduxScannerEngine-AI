from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import zlib


def generate(prompt: str, width: int, height: int, output: Path, reference: Path | None = None, transparent: bool = True) -> dict:
    if not (1 <= width <= 4096 and 1 <= height <= 4096):
        raise ValueError("dimensions outside safe range")
    seed = hashlib.sha256(prompt.encode("utf-8") + (reference.read_bytes() if reference else b"")).digest()
    color = _prompt_color(prompt, seed)
    rgba = bytearray(width * height * 4)
    for y in range(height):
        for x in range(width):
            index = (y * width + x) * 4
            center = 1.0 - abs((x + 0.5) / width - 0.5) * 2.0
            progress = y / max(height - 1, 1)
            alpha = int(255 * max(0.0, center) * (0.2 + 0.8 * progress)) if transparent else 255
            rgba[index:index+4] = bytes((*color, alpha))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encode_png(width, height, bytes(rgba)))
    return {"schemaVersion": "redux-maker.image.v1", "prompt": prompt, "referenceUsed": reference is not None, "width": width, "height": height, "format": "RGBA8_PNG", "output": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def generate_blank(width: int, height: int, output: Path, alpha_supported: bool = True) -> dict:
    output.parent.mkdir(parents=True, exist_ok=True)
    pixel = b"\x00\x00\x00\x00" if alpha_supported else b"\x00\x00\x00\xff"
    output.write_bytes(encode_png(width, height, pixel * width * height))
    return {"width": width, "height": height, "entryPreservationRequired": True, "mode": "transparent" if alpha_supported else "black_low_energy", "output": str(output)}


def encode_png(width: int, height: int, rgba: bytes) -> bytes:
    if len(rgba) != width * height * 4: raise ValueError("RGBA length mismatch")
    raw = b"".join(b"\x00" + rgba[y*width*4:(y+1)*width*4] for y in range(height))
    signature = b"\x89PNG\r\n\x1a\n"
    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xffffffff)
    return signature + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def png_dimensions(data: bytes) -> tuple[int, int]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or data[12:16] != b"IHDR": raise ValueError("not a PNG")
    return struct.unpack(">II", data[16:24])


def _prompt_color(prompt: str, seed: bytes) -> tuple[int, int, int]:
    colors = {"purple": (160, 40, 255), "red": (255, 35, 35), "blue": (40, 100, 255), "green": (40, 255, 100), "white": (255, 255, 255)}
    lower = prompt.casefold()
    return next((value for name, value in colors.items() if name in lower), (seed[0], seed[1], seed[2]))
