"""Generate synthetic images with Stable Diffusion + ControlNet (Canny)."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

# Allow running as: python scripts/generate_synthetic.py
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from PIL import Image
from tqdm import tqdm

from src.hw35_config import (
    CLASS_GEN_PARAMS,
    CLASS_NEGATIVE_PROMPTS,
    CONTROLNET_MODEL,
    DEFAULT_CONTROL_SCALE,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_IMAGES_PER_CLASS,
    DEFAULT_SD_STEPS,
    MIN_REF_CROP_AREA,
    NEGATIVE_PROMPT,
    PROMPT_VARIANTS,
    PROMPTS,
    SD_IMAGE_SIZE,
    SD_MODEL,
    SYNTHETIC_CLASSES,
    SYNTHETIC_DIR,
)
from src.utils import ensure_dirs


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


from src.synthetic_utils import iter_reference_paths, prepare_reference_image


def load_reference_images(
    crops_dir: Path,
    class_name: str,
    max_refs: int = 20,
    min_area: int = MIN_REF_CROP_AREA,
) -> list[Image.Image]:
    selected_paths = iter_reference_paths(crops_dir, class_name, min_area=min_area)
    random.shuffle(selected_paths)

    refs: list[Image.Image] = []
    for path in selected_paths[:max_refs]:
        refs.append(prepare_reference_image(Image.open(path).convert("RGB")))
    return refs


def build_pipeline(device: torch.device):
    from controlnet_aux import CannyDetector
    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline

    dtype = torch.float16 if device.type == "cuda" else torch.float32
    controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_MODEL,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    if device.type == "cuda":
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)
    canny = CannyDetector()
    return pipe, canny


def generate_for_class(
    pipe,
    canny,
    class_name: str,
    refs: list[Image.Image],
    output_dir: Path,
    num_images: int,
    steps: int,
    guidance_scale: float,
    control_scale: float,
    seed: int,
) -> list[str]:
    class_out = output_dir / class_name
    ensure_dirs(class_out)
    prompt_variants = PROMPT_VARIANTS.get(class_name, [PROMPTS[class_name]])
    class_negative = CLASS_NEGATIVE_PROMPTS.get(class_name, "")
    negative_prompt = f"{NEGATIVE_PROMPT}, {class_negative}" if class_negative else NEGATIVE_PROMPT
    class_params = CLASS_GEN_PARAMS.get(class_name, {})
    class_guidance = class_params.get("guidance_scale", guidance_scale)
    class_control = class_params.get("control_scale", control_scale)
    saved: list[str] = []

    for i in tqdm(range(num_images), desc=f"generate {class_name}"):
        ref = refs[i % len(refs)]
        control = canny(ref, low_threshold=100, high_threshold=200)
        gen_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=gen_device).manual_seed(seed + i)
        prompt = prompt_variants[i % len(prompt_variants)]

        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control,
            num_inference_steps=steps,
            guidance_scale=class_guidance,
            controlnet_conditioning_scale=class_control,
            generator=generator,
        )
        image = result.images[0]
        out_path = class_out / f"synthetic_{i:04d}.jpg"
        image.save(out_path, quality=95)
        saved.append(str(out_path))

    return saved


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic images with SD + ControlNet.")
    parser.add_argument("--crops-dir", type=Path, default=Path("data/cls_crops"))
    parser.add_argument("--output-dir", type=Path, default=SYNTHETIC_DIR)
    parser.add_argument("--classes", nargs="+", default=list(SYNTHETIC_CLASSES))
    parser.add_argument("--num-images", type=int, default=DEFAULT_IMAGES_PER_CLASS)
    parser.add_argument("--steps", type=int, default=DEFAULT_SD_STEPS)
    parser.add_argument("--guidance-scale", type=float, default=DEFAULT_GUIDANCE_SCALE)
    parser.add_argument("--control-scale", type=float, default=DEFAULT_CONTROL_SCALE)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")
    if device.type == "cpu":
        print("Warning: SD generation on CPU is very slow. GPU is strongly recommended.")

    ensure_dirs(args.output_dir)
    pipe, canny = build_pipeline(device)

    manifest: dict = {
        "sd_model": SD_MODEL,
        "controlnet_model": CONTROLNET_MODEL,
        "classes": {},
        "device": str(device),
        "num_images_per_class": args.num_images,
    }

    for class_name in args.classes:
        refs = load_reference_images(args.crops_dir, class_name)
        saved = generate_for_class(
            pipe=pipe,
            canny=canny,
            class_name=class_name,
            refs=refs,
            output_dir=args.output_dir,
            num_images=args.num_images,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            control_scale=args.control_scale,
            seed=args.seed,
        )
        manifest["classes"][class_name] = {
            "num_generated": len(saved),
            "prompt": PROMPTS[class_name],
            "prompt_variants": PROMPT_VARIANTS.get(class_name, [PROMPTS[class_name]]),
            "negative_prompt": (
                f"{NEGATIVE_PROMPT}, {CLASS_NEGATIVE_PROMPTS[class_name]}"
                if class_name in CLASS_NEGATIVE_PROMPTS
                else NEGATIVE_PROMPT
            ),
            "guidance_scale": CLASS_GEN_PARAMS.get(class_name, {}).get(
                "guidance_scale", args.guidance_scale
            ),
            "control_scale": CLASS_GEN_PARAMS.get(class_name, {}).get(
                "control_scale", args.control_scale
            ),
            "examples": saved[:5],
        }

    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
