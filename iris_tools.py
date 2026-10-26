"""Standalone Iris image generation, experimental img2img, relative depth and restoration."""

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import time

from PIL import Image, ImageOps

SIZES = ["1024x1024", "1344x768", "1280x832", "1152x896", "896x1152", "832x1280", "768x1344"]
TASKS = {"Image generation": "generate", "Image-to-image (experimental)": "img2img",
         "Relative depth": "depth", "Restore / upscale": "restore"}
from depth_lab import DEPTH_TASKS,add_depth_args,validate_depth_args
TASKS.update(DEPTH_TASKS)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=list(TASKS.values()))
    parser.add_argument("--name", default=None)
    parser.add_argument("--save_dir", default="results/iris-tools")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--size", choices=SIZES, default="1024x1024")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance", type=float)
    parser.add_argument("--shift", type=float, default=4.)
    parser.add_argument("--order", type=int, choices=[1, 2], default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--strength", type=float, default=.5)
    parser.add_argument("--turbo", action="store_true", help="Optional community Turbo LoRA for text-to-image only")
    parser.add_argument("--turbo_adapter", choices=["r2-1000", "r1-1500"], default="r2-1000")
    parser.add_argument("--max_side", type=int, default=1024, help="Depth inference long-side cap; 0 = native")
    parser.add_argument("--scale", type=int, choices=[1, 2, 4], default=4, help="Restoration output scale; model always restores at 4x")
    parser.add_argument("--no_budget", action="store_true", help="Disable restoration input size limit")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--text_encoder_device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model_path", help="Local task-specific INT8 folder or parent containing depth/upscaler")
    parser.add_argument("--text_encoder_path")
    return add_depth_args(parser)


def validate_args(args):
    if args.name is not None and (not args.name.strip() or args.name in (".", "..") or any(c in args.name for c in '<>:"/\\|?*')):
        raise ValueError("Choose a run name without folder separators or special filename characters")
    args.steps = args.steps if args.steps is not None else (4 if args.turbo else 30)
    args.guidance = args.guidance if args.guidance is not None else (1. if args.turbo else 3.)
    if args.steps < 1 or args.num_samples < 1 or args.seed < 0 or args.seed + args.num_samples > 2**63:
        raise ValueError("Use positive steps/sample count and seeds in [0, 2**63)")
    if not math.isfinite(args.guidance) or not math.isfinite(args.shift) or args.shift <= 0:
        raise ValueError("Guidance must be finite and shift must be positive and finite")
    if args.task in ("generate", "img2img") and not args.prompt.strip():
        raise ValueError("Write an image prompt")
    if args.task != "generate" and (args.input is None or not args.input.is_file()):
        raise ValueError('Choose an existing input video' if args.task=='parallax_video' else 'Choose an existing input image')
    if args.turbo and (args.task != "generate" or args.guidance != 1 or args.negative_prompt.strip()):
        raise ValueError("Turbo is for text-to-image only, with guidance 1 and no negative prompt")
    if args.task == "img2img" and (not math.isfinite(args.strength) or not 0 < args.strength <= 1):
        raise ValueError("Image-to-image strength must be in (0, 1]")
    if args.max_side < 0 or (args.max_side and args.max_side < 16):
        raise ValueError("Depth max side must be at least 16, or 0 for native resolution")
    if args.task in ("depth", "restore") and args.num_samples != 1:
        raise ValueError("Depth and restoration process one input image per run")
    validate_depth_args(args)
    return args


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    source = None
    if args.task=='parallax_video':
        from visual_anagrams.video_parallax import probe_video,validate_video_info
        try:
            source=probe_video(args.input)
            validate_video_info(args,source)
        except ValueError as exc:parser.error(str(exc))
    elif args.task != "generate":
        try:
            with Image.open(args.input) as image:
                source = ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, ValueError) as exc:
            parser.error(f"Cannot open input image: {exc}")
    if args.task in DEPTH_TASKS.values():
        from depth_lab import run_depth_task
        run_depth_task(args,source)
        return
    import torch
    from visual_anagrams.iris import CODE_REVISION, MODEL_ID, MODEL_REVISION, TEXT_ID, TEXT_REVISION

    if args.device == "cuda" or (args.task in ("generate", "img2img") and args.text_encoder_device == "cuda"):
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            parser.error("CUDA Iris requires a BF16-capable GPU; select CPU or install CUDA PyTorch")
    name = args.name or f"{args.task}-{datetime.now():%Y%m%d-%H%M%S-%f}"
    output = Path(args.save_dir) / name
    if output.exists():
        parser.error(f"Output folder already exists: {output}; choose a new --name")
    torch.set_num_threads(min(6, torch.get_num_threads()))
    width, height = (int(x) for x in args.size.split("x"))
    records = dict(task=args.task, settings={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                   iris_code_revision=CODE_REVISION, model_id=MODEL_ID,
                   input_size=list(source.size) if source is not None else None)
    print(f"Preparing Iris {args.task}", flush=True)
    if args.task in ("depth", "restore"):
        from visual_anagrams.iris_tasks import DOWNSTREAM_REVISION, load_task, resolve_task_model

        directory = resolve_task_model(args.task, args.model_path)
        print(f"Loading Iris INT8 {args.task} (no text encoder)", flush=True)
        runner = load_task(args.task, directory, args.device)
        records["model_revision"] = DOWNSTREAM_REVISION if args.model_path is None else None
    else:
        from visual_anagrams.iris import encode_prompts, load_model, read_config, resolve_model
        directory = resolve_model(args.model_path)
        cfg = read_config(directory)
        if cfg.flow.prediction != "v" or cfg.flow.num_train_timesteps != 1000:
            parser.error("Iris Tools expects the standard Iris velocity model and 1000 training timesteps")
        print("Preparing Qwen text encoder (released before Iris)", flush=True)
        positive, negative = encode_prompts(cfg, [args.prompt], args.negative_prompt,
                    args.text_encoder_device, args.text_encoder_path, encode_negative=not args.turbo)
        turbo_path = None
        if args.turbo:
            from visual_anagrams.turbo import TURBO_ID, TURBO_REVISION, resolve_turbo_adapter
            turbo_path = resolve_turbo_adapter(args.turbo_adapter)
            records["turbo"] = dict(repo=TURBO_ID, revision=TURBO_REVISION, adapter=args.turbo_adapter)
        print("Loading Iris INT8 generation", flush=True)
        model = load_model(directory, cfg, args.device, turbo_path=turbo_path)
        records.update(model_revision=MODEL_REVISION if args.model_path is None else None,
                       text_encoder_id=TEXT_ID, text_encoder_revision=TEXT_REVISION if args.text_encoder_path is None else None)
    records["model_manifest"] = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "metadata.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    for index in range(args.num_samples):
        seed = args.seed + index
        destination = output / f"{seed:04}"
        destination.mkdir()
        if source is not None:
            source.save(destination / "input.png")
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        details = dict(task=args.task, seed=seed)
        if args.task == "depth":
            from visual_anagrams.iris_tasks import save_depth
            depth = runner(source, max_side=args.max_side)
            details.update(save_depth(depth, destination))
        elif args.task == "restore":
            from iris3b.downstream.restoration import fit_budget
            work = source if args.no_budget else fit_budget(source)
            print(f"Restoring {work.width}x{work.height}; output {work.width*args.scale}x{work.height*args.scale}", flush=True)
            result = runner(work, scale=4.)
            if args.scale != 4:
                result = result.resize((work.width * args.scale, work.height * args.scale), Image.Resampling.LANCZOS)
            result.save(destination / "result.png")
            details.update(processed_input_size=list(work.size), output_size=list(result.size), tiles=runner.completed_tiles)
        else:
            from visual_anagrams.iris import sample_iris
            from visual_anagrams.views import get_views
            print(f"Generating sample {index+1}/{args.num_samples}, seed {seed}", flush=True)
            generator = torch.Generator(device=args.device).manual_seed(seed)
            if args.task == "img2img":
                import numpy as np
                from visual_anagrams.iris_tasks import sample_img2img
                fitted = ImageOps.fit(source, (width, height), method=Image.Resampling.LANCZOS)
                fitted.save(destination / "input.png")
                initial = torch.from_numpy(np.array(fitted)).permute(2,0,1)[None].float() / 127.5 - 1
                pixels = sample_img2img(model, positive, negative, initial, strength=args.strength,
                    steps=args.steps, guidance=args.guidance, shift=args.shift, order=args.order,
                    generator=generator, device=args.device)
                details["experimental"] = "Flow noise-and-denoise, not trained instruction editing"
            elif args.turbo:
                from visual_anagrams.turbo import sample_turbo
                pixels = sample_turbo(model, positive, get_views(["identity"]), size=(height,width),
                    num_inference_steps=args.steps, shift=args.shift, generator=generator, device=args.device)
            else:
                pixels = sample_iris(model, positive, negative, get_views(["identity"]), size=(height,width),
                    num_inference_steps=args.steps, guidance_scale=args.guidance, shift=args.shift,
                    order=args.order, generator=generator, device=args.device)
            array = ((pixels[0].cpu().permute(1,2,0).float() + 1) * 127.5).round().byte().numpy()
            Image.fromarray(array).save(destination / "result.png")
            details["output_size"] = [width, height]
        if args.device == "cuda":
            torch.cuda.synchronize()
        details.update(seconds=time.perf_counter()-start,
                       peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30 if args.device == "cuda" else None)
        (destination / "run.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
        print(f"Saved {destination / 'result.png'}", flush=True)
        print(json.dumps(details), flush=True)


if __name__ == "__main__":
    main()
