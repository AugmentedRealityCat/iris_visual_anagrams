"""Generate visual anagrams with Aikimi/iris-3b-int8."""

import argparse
import json
import math
from pathlib import Path


def build_parser():
    from visual_anagrams.export import add_export_args

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--save_dir", default="results")
    parser.add_argument("--prompts", required=True, nargs="+")
    parser.add_argument("--views", required=True, nargs="+")
    parser.add_argument("--view_args", nargs="+")
    parser.add_argument("--style", default="")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_inference_steps", type=int, default=None, help="Default: 4, or 100 with --standard")
    parser.add_argument("--guidance_scale", type=float, default=None, help="Default: 1, or 3 with --standard")
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument("--turbo", dest="turbo", action="store_true", help="Use Turbo LoRA (default)")
    sampling.add_argument("--standard", dest="turbo", action="store_false", help="Use standard Iris with DPM sampling")
    parser.set_defaults(turbo=True)
    parser.add_argument("--turbo_adapter", choices=["r2-1000", "r1-1500"], default="r2-1000")
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--reduction", choices=["mean", "sum", "alternate"], default="mean")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--text_encoder_device", default="cuda", help="Use cpu to encode prompts without GPU memory")
    parser.add_argument("--model_path", help="Local INT8 export with config.yaml, model.safetensors and manifest.json")
    parser.add_argument("--text_encoder_path", help="Local Qwen3-VL-4B-Instruct directory")
    parser.add_argument("--shift", type=float, default=4.0)
    parser.add_argument("--solver_order", type=int, choices=[1, 2], default=2)
    # Kept for existing command lines; Iris has only one generation stage.
    parser.add_argument("--generate_1024", action="store_true", help="Compatibility option; Iris already generates at 1024")
    parser.add_argument("--save_metadata", action="store_true", help="Compatibility option; metadata is always saved")
    return add_export_args(parser)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.num_inference_steps = args.num_inference_steps if args.num_inference_steps is not None else (4 if args.turbo else 100)
    args.guidance_scale = args.guidance_scale if args.guidance_scale is not None else (1.0 if args.turbo else 3.0)
    if args.turbo and (args.guidance_scale != 1.0 or args.negative_prompt.strip()):
        parser.error("Turbo has built-in guidance: use --guidance_scale 1 and an empty negative prompt")
    if len(args.prompts) != len(args.views):
        parser.error("--prompts and --views must have the same number of entries")
    if args.view_args is not None and len(args.view_args) != len(args.views):
        parser.error("--view_args must have one entry per view")
    if args.num_samples < 1 or args.num_inference_steps < 1:
        parser.error("--num_samples and --num_inference_steps must be positive")
    if not math.isfinite(args.guidance_scale) or not math.isfinite(args.shift) or args.shift <= 0:
        parser.error("guidance must be finite and --shift must be positive and finite")
    if args.seed < 0 or args.seed + args.num_samples > 2**63:
        parser.error("sample seeds must be in [0, 2**63)")
    from visual_anagrams.export import animation_targets, options_from_args

    try:
        export_options = options_from_args(args)
        if args.animation_formats:
            animation_targets(args.views)
    except ValueError as exc:
        parser.error(str(exc))

    import torch
    from visual_anagrams.iris import (
        CODE_REVISION, MODEL_ID, MODEL_REVISION, TEXT_ID, TEXT_REVISION,
        encode_prompts, load_model, read_config, resolve_model, sample_iris,
    )
    from visual_anagrams.utils import save_illusion, save_metadata
    from visual_anagrams.views import VIEW_MAP, get_views

    for name in args.views:
        if name not in VIEW_MAP:
            parser.error(f"unknown view {name!r}; choose from: {', '.join(VIEW_MAP)}")
    for option in ("device", "text_encoder_device"):
        try:
            device = torch.device(getattr(args, option))
        except (RuntimeError, ValueError):
            parser.error(f"invalid --{option}")
        if device.type not in ("cuda", "cpu"):
            parser.error(f"--{option} must be cuda, cuda:N, or cpu")
        if device.type == "cuda" and not torch.cuda.is_available():
            parser.error("CUDA is unavailable; install CUDA PyTorch or select --device cpu --text_encoder_device cpu")
        if device.type == "cuda":
            try:
                with torch.cuda.device(device):
                    bf16_supported = torch.cuda.is_bf16_supported()
            except (RuntimeError, ValueError):
                parser.error(f"unavailable CUDA device for --{option}: {device}")
            if not bf16_supported:
                parser.error("Iris CUDA inference requires a GPU with BF16 support")

    torch.manual_seed(args.seed)
    views = get_views(args.views, args.view_args)
    prompts = [f"{args.style} {prompt}".strip() for prompt in args.prompts]
    print(f"Preparing {MODEL_ID} (1024x1024)", flush=True)
    directory = resolve_model(args.model_path)
    cfg = read_config(directory)
    turbo_path, turbo_metadata = None, None
    if args.turbo:
        from visual_anagrams.turbo import (
            TURBO_CODE_REVISION, TURBO_ID, TURBO_REVISION,
            read_adapter_metadata, resolve_turbo_adapter,
        )
        if cfg.flow.prediction != "v" or cfg.flow.num_train_timesteps != 1000:
            parser.error("The Turbo LoRA needs the standard Iris velocity prediction and 1000 training timesteps")
        print(f"Preparing Turbo LoRA: {args.turbo_adapter}", flush=True)
        turbo_path = resolve_turbo_adapter(args.turbo_adapter)
        turbo_metadata = dict(
            repo=TURBO_ID, revision=TURBO_REVISION, adapter=args.turbo_adapter,
            file=f"{args.turbo_adapter}/student.safetensors", sampler_code_revision=TURBO_CODE_REVISION,
            settings=read_adapter_metadata(turbo_path),
        )
    print("Preparing Qwen text encoder (released before Iris inference)", flush=True)
    positive, negative = encode_prompts(
        cfg, prompts, args.negative_prompt, args.text_encoder_device, args.text_encoder_path,
        encode_negative=not args.turbo,
    )
    print("Loading Iris INT8", flush=True)
    model = load_model(directory, cfg, args.device, turbo_path=turbo_path)

    save_dir = Path(args.save_dir) / args.name
    save_dir.mkdir(parents=True, exist_ok=True)
    save_metadata(views, args, save_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    metadata = dict(vars(args), model_id=MODEL_ID,
                    model_revision=MODEL_REVISION if args.model_path is None else None,
                    model_path_resolved=str(directory.resolve()), model_manifest=manifest,
                    iris_code_revision=CODE_REVISION, text_encoder_id=TEXT_ID,
                    text_encoder_revision=TEXT_REVISION if args.text_encoder_path is None else None,
                    size=1024, turbo_lora=turbo_metadata)
    (save_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for index in range(args.num_samples):
        seed = args.seed + index
        print(f"Generating sample {index + 1}/{args.num_samples}, seed {seed}", flush=True)
        generator = torch.Generator(device=args.device).manual_seed(seed)
        if args.turbo:
            from visual_anagrams.turbo import sample_turbo

            image = sample_turbo(
                model, positive, views, generator=generator, device=args.device,
                num_inference_steps=args.num_inference_steps, reduction=args.reduction, shift=args.shift,
            )
        else:
            image = sample_iris(
                model, positive, negative, views, generator=generator, device=args.device,
                num_inference_steps=args.num_inference_steps, guidance_scale=args.guidance_scale,
                reduction=args.reduction, shift=args.shift, order=args.solver_order,
                num_train_timesteps=cfg.flow.num_train_timesteps, prediction=cfg.flow.prediction,
            )
        sample_dir = save_dir / f"{seed:04}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        save_illusion(image.cpu(), views, sample_dir)
        print(f"Saved {sample_dir / 'sample_1024.png'}", flush=True)
        if args.label_outputs:
            from visual_anagrams.export import label_saved_images

            for path in label_saved_images(sample_dir / "sample_1024.png", args.views, args.prompts):
                print(f"Saved labels {path}", flush=True)
        if args.animation_formats:
            from visual_anagrams.export import export_animations

            export_animations(sample_dir / "sample_1024.png", views, args.views,
                              args.prompts, args.animation_formats, export_options)


if __name__ == "__main__":
    main()
