from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import zlib
import math


def diffusion_dimensions(width: int, height: int, target_pixels: int = 512 * 512) -> tuple[int, int]:
    scale = math.sqrt(target_pixels / max(width * height, 1))
    return max(64, round(width * scale / 8) * 8), max(64, round(height * scale / 8) * 8)


def remove_border_background(image):
    """Convert a near-uniform generated border background to soft alpha."""
    from PIL import Image
    rgba=image.convert("RGBA");width,height=rgba.size;pixels=rgba.load();corners=[pixels[0,0],pixels[width-1,0],pixels[0,height-1],pixels[width-1,height-1]]
    background=tuple(sum(pixel[channel] for pixel in corners)//4 for channel in range(3));alpha=Image.new("L",rgba.size);mask=alpha.load()
    for y in range(height):
        for x in range(width):
            pixel=pixels[x,y];distance=sum(abs(pixel[channel]-background[channel]) for channel in range(3));mask[x,y]=max(0,min(255,(distance-24)*5))
    rgba.putalpha(alpha);return rgba


def generate_with_diffusers(
    prompt: str,
    width: int,
    height: int,
    output: Path,
    model_path: str,
    reference: Path | None = None,
    negative_prompt: str = "",
    strength: float = 0.65,
    seed: int = 0,
    transparent_background: bool = True,
) -> dict:
    """Run a real local Diffusers model; never silently fall back to procedural output."""
    if not (1 <= width <= 4096 and 1 <= height <= 4096):
        raise ValueError("dimensions outside safe range")
    try:
        import torch
        from diffusers import AutoPipelineForImage2Image, DiffusionPipeline
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("diffusers_runtime_missing: install the selected image-model runtime or repair it in Model Manager") from error
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device="cpu").manual_seed(seed)
    generation_width, generation_height = diffusion_dimensions(width, height)
    if reference:
        pipeline = AutoPipelineForImage2Image.from_pretrained(model_path, torch_dtype=dtype, variant="fp16" if torch.cuda.is_available() else None)
        if torch.cuda.is_available(): pipeline.enable_model_cpu_offload()
        else: pipeline = pipeline.to(device)
        source = Image.open(reference).convert("RGB").resize((generation_width, generation_height), Image.Resampling.LANCZOS)
        image = pipeline(prompt=prompt, negative_prompt=negative_prompt or None, image=source, strength=max(0.05, min(1.0, strength)), generator=generator, width=generation_width, height=generation_height).images[0]
        mode = "reference_edit"
    else:
        pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=dtype, variant="fp16" if torch.cuda.is_available() else None)
        if torch.cuda.is_available(): pipeline.enable_model_cpu_offload()
        else: pipeline = pipeline.to(device)
        image = pipeline(prompt=prompt, negative_prompt=negative_prompt or None, generator=generator, width=generation_width, height=generation_height).images[0]
        mode = "text_to_image"
    image = image.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS)
    if transparent_background: image=remove_border_background(image)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    image.save(temporary, format="PNG")
    if png_dimensions(temporary.read_bytes()) != (width, height):
        temporary.unlink(missing_ok=True)
        raise RuntimeError("generated_texture_wrong_size")
    temporary.replace(output)
    alpha=image.getchannel("A");return {"schemaVersion":"redux-maker.image.v2","backend":"diffusers","model":model_path,"mode":mode,"prompt":prompt,"negativePrompt":negative_prompt,"referenceUsed":reference is not None,"generationWidth":generation_width,"generationHeight":generation_height,"width":width,"height":height,"format":"RGBA8_PNG","transparentBackground":transparent_background,"alphaExtrema":list(alpha.getextrema()),"output":str(output),"sha256":hashlib.sha256(output.read_bytes()).hexdigest()}


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


if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(); parser.add_argument("--prompt",default=""); parser.add_argument("--width",type=int,required=True); parser.add_argument("--height",type=int,required=True); parser.add_argument("--out",type=Path,required=True); parser.add_argument("--reference",type=Path); parser.add_argument("--blank",action="store_true"); parser.add_argument("--no-alpha",action="store_true"); parser.add_argument("--opaque-background",action="store_true"); parser.add_argument("--model-path"); parser.add_argument("--negative-prompt",default=""); parser.add_argument("--strength",type=float,default=.65); parser.add_argument("--seed",type=int,default=0); args=parser.parse_args()
    if args.blank: result=generate_blank(args.width,args.height,args.out,not args.no_alpha)
    elif args.model_path: result=generate_with_diffusers(args.prompt,args.width,args.height,args.out,args.model_path,args.reference,args.negative_prompt,args.strength,args.seed,not args.opaque_background)
    else: raise SystemExit("image_model_required: configure --model-path; procedural generation is test/fallback-only")
    import json
    print(json.dumps(result))
