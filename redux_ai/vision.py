from __future__ import annotations

from pathlib import Path
import struct
import zlib


COLORS = {"purple": (160, 40, 255), "red": (255, 35, 35), "blue": (40, 100, 255), "green": (40, 255, 100), "white": (255, 255, 255)}


def validate_with_smolvlm(image: Path, prompt: str, model_path: str, reference: Path | None = None) -> dict:
    """Use the installed local SmolVLM2 checkpoint for semantic validation."""
    try:
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForImageTextToText
    except ImportError as error:
        raise RuntimeError("vision_runtime_missing: repair the SmolVLM2/Transformers installation in Model Manager") from error
    processor=AutoProcessor.from_pretrained(model_path)
    model=AutoModelForImageTextToText.from_pretrained(model_path,torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,device_map="auto" if torch.cuda.is_available() else None)
    images=[Image.open(image).convert("RGB")]
    request=f"Assess whether this generated Redux Maker visual matches the request: {prompt!r}. Reply with strict JSON containing match (boolean), score (0 to 1), and concise reason. Do not assess game-file dimensions or binary format."
    if reference:
        images.append(Image.open(reference).convert("RGB")); request += " The second image is the user reference; compare style and color consistency."
    messages=[{"role":"user","content":[*[{"type":"image"} for _ in images],{"type":"text","text":request}]}]
    text=processor.apply_chat_template(messages,add_generation_prompt=True)
    inputs=processor(text=text,images=images,return_tensors="pt").to(model.device)
    generated=model.generate(**inputs,max_new_tokens=160,do_sample=False)
    response=processor.batch_decode(generated[:,inputs["input_ids"].shape[1]:],skip_special_tokens=True)[0].strip()
    import json
    try: decision=json.loads(response)
    except json.JSONDecodeError as error: raise RuntimeError(f"vision_model_invalid_json: {response[:200]}") from error
    if not isinstance(decision.get("match"),bool) or not isinstance(decision.get("score"),(int,float)): raise RuntimeError("vision_model_invalid_schema")
    return {"schemaVersion":"redux-maker.vision.v2","backend":"SmolVLM2","model":model_path,"status":"passed" if decision["match"] and float(decision["score"])>=.55 else "rejected","score":round(float(decision["score"]),4),"reason":str(decision.get("reason","")),"referenceUsed":reference is not None,"engineMetadataValidationStillRequired":True}


def validate(image: Path, prompt: str, expected_width: int, expected_height: int, reference: Path | None = None) -> dict:
    width, height, rgba = decode_rgba_png(image.read_bytes())
    opaque = [(rgba[i], rgba[i+1], rgba[i+2]) for i in range(0, len(rgba), 4) if rgba[i+3] > 16]
    average = tuple(round(sum(pixel[channel] for pixel in opaque) / max(len(opaque), 1), 2) for channel in range(3))
    requested = next((rgb for name, rgb in COLORS.items() if name in prompt.casefold()), None)
    color_score = 1.0 if requested is None else max(0.0, 1.0 - sum(abs(average[i] - requested[i]) for i in range(3)) / 765.0)
    dimension_ok = (width, height) == (expected_width, expected_height)
    reference_score = None
    if reference:
        rw, rh, rrgba = decode_rgba_png(reference.read_bytes())
        reference_score = 0.0 if (rw, rh) != (width, height) else _byte_similarity(rgba, rrgba)
    score = color_score * (1.0 if dimension_ok else 0.0) * (reference_score if reference_score is not None else 1.0)
    return {"schemaVersion": "redux-maker.vision.v1", "backend":"deterministic_low_end_fallback", "status": "passed" if score >= 0.55 else "rejected", "score": round(score, 4), "dimensionOk": dimension_ok, "promptColorScore": round(color_score, 4), "referenceScore": None if reference_score is None else round(reference_score, 4), "averageRgb": average, "engineMetadataValidationStillRequired": True}


def decode_rgba_png(data: bytes) -> tuple[int, int, bytes]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"): raise ValueError("not PNG")
    pos, width, height, compressed = 8, 0, 0, bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos+4])[0]; kind=data[pos+4:pos+8]; payload=data[pos+8:pos+8+length]; pos += 12 + length
        if kind == b"IHDR": width, height, depth, color, *_ = struct.unpack(">IIBBBBB", payload); 
        elif kind == b"IDAT": compressed.extend(payload)
        elif kind == b"IEND": break
    if depth != 8 or color != 6: raise ValueError("only RGBA8 PNG supported")
    raw=zlib.decompress(bytes(compressed)); stride=width*4; rows=[]; cursor=0; previous=bytearray(stride)
    for _ in range(height):
        filter_type=raw[cursor]; cursor+=1; scan=bytearray(raw[cursor:cursor+stride]); cursor+=stride
        if filter_type != 0: raise ValueError("vision worker requires filter type 0")
        rows.append(bytes(scan)); previous=scan
    return width, height, b"".join(rows)


def _byte_similarity(left: bytes, right: bytes) -> float:
    return max(0.0, 1.0 - sum(abs(a-b) for a,b in zip(left,right)) / (255.0 * max(len(left), 1)))


def main() -> None:
    import argparse, json
    parser=argparse.ArgumentParser();parser.add_argument("--image",type=Path,required=True);parser.add_argument("--prompt",required=True);parser.add_argument("--model-path");parser.add_argument("--reference",type=Path);parser.add_argument("--expected-width",type=int);parser.add_argument("--expected-height",type=int);parser.add_argument("--deterministic-fallback",action="store_true");args=parser.parse_args()
    if args.model_path and not args.deterministic_fallback: result=validate_with_smolvlm(args.image,args.prompt,args.model_path,args.reference)
    elif args.expected_width and args.expected_height: result=validate(args.image,args.prompt,args.expected_width,args.expected_height,args.reference)
    else: raise SystemExit("vision_model_or_explicit_fallback_dimensions_required")
    print(json.dumps(result))


if __name__ == "__main__": main()
