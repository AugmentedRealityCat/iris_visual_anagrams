"""Publish the four small Iris Tools GPU checks, without additional generation."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "results/iris-tools-check")
    parser.add_argument("--output", type=Path, default=ROOT / "examples/iris-tools")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = {"generate": "landscape-smoke", "img2img": "img2img-smoke", "depth": "depth-smoke", "restore": "restore-smoke"}
    records = {}
    font_path = ROOT / "visual_anagrams/assets/CourierPrime-Regular.ttf"
    font = ImageFont.truetype(str(font_path), 20)
    for task, case in cases.items():
        folder = args.source / case
        sample = folder / "0000"
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        record = json.loads((sample / "run.json").read_text(encoding="utf-8"))
        result = Image.open(sample / "result.png").convert("RGB")
        record.update(metadata=metadata, actual_output_size=list(result.size),
                      result_sha256=hashlib.sha256((sample / "result.png").read_bytes()).hexdigest(),
                      local_result=str(sample / "result.png"))
        if task == "depth":
            values = np.load(sample / "depth.npy", allow_pickle=False)
            assert values.shape == (result.height,result.width) and np.isfinite(values).all()
            with Image.open(sample / "depth16.png") as image:
                assert image.size == result.size
            record["depth_validation"] = dict(shape=list(values.shape), finite=True, raw_dtype=str(values.dtype))
        records[task] = record
        if task == "generate":
            result.save(args.output / "generation.jpg", quality=92)
        else:
            with Image.open(sample / "input.png") as image:
                before = image.convert("RGB")
            sheet = Image.new("RGB", (1024,550), "#eef0f4")
            draw = ImageDraw.Draw(sheet)
            draw.text((12,8), f"Input {before.width}x{before.height}", font=font, fill="#182337")
            draw.text((524,8), {"depth":"Relative depth (near = bright)", "restore":"Restored 4x", "img2img":"Img2img / strength 0.35"}[task], font=font, fill="#182337")
            for index, viewed in enumerate((before,result)):
                viewed = viewed.resize((512,512), Image.Resampling.BICUBIC if index == 0 else Image.Resampling.LANCZOS)
                sheet.paste(viewed,(index*512,38))
            sheet.save(args.output / f"{task}.jpg",quality=92)
    sources = ["iris_tools.py","visual_anagrams/iris_tasks.py","visual_anagrams/tools_gui.py",
               "visual_anagrams/gui.py","visual_anagrams/iris.py","visual_anagrams/turbo.py",
               "scripts/check_iris_tools_generation.py","scripts/report_iris_tools.py"]
    def portable(value):
        if isinstance(value, dict):
            return {key: portable(item) for key,item in value.items()}
        if isinstance(value, list):
            return [portable(item) for item in value]
        if isinstance(value, str):
            normalized = value.replace("\\", "/")
            if normalized.startswith(ROOT.as_posix() + "/"):
                return normalized[len(ROOT.as_posix())+1:]
        return value

    report = dict(hardware="RTX 3060 12 GB / Ryzen 5 5500 / 64 GB RAM", runs=portable(records),
                  source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                  note="One example per task, not a quality benchmark. Generation/img2img reused matching Qwen embeddings from the earlier pilot. Timings include processing and saving, exclude downloads, model loading and text encoding. VRAM is PyTorch peak allocated memory.")
    (args.output/"measurements.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    lines = ["# Iris Tools checks", "", "One real-model check per new task on the RTX 3060 12 GB. These checks "
             "verify execution and output formats; they do not measure depth accuracy, restoration fidelity or general image quality.", "",
             "| Task | Settings | Output | Processing seconds | Peak allocated GiB |",
             "| --- | --- | --- | ---: | ---: |"]
    settings = {"generate":"Standard, 30 steps, CFG 3, seed 0", "img2img":"15 steps, CFG 3, strength 0.35, seed 0",
                "depth":"1024-pixel long-side cap, one forward", "restore":"256x256 input, 4x, one tile"}
    for task, record in records.items():
        width,height = record["actual_output_size"]
        lines.append(f"| {task} | {settings[task]} | {width}x{height} | {record['seconds']:.2f} | {record['peak_allocated_gib']:.2f} |")
    lines += ["", "Generation and image-to-image reuse the exact prompt `a watercolor painting of an owl` and "
              "its cached Qwen embeddings from the earlier quality pilot. They run the real image model through the "
              "public CLI entry point, with no sampler mocks. Depth and restoration load their separate INT8 weights "
              "and saved empty-prompt embeddings, without Qwen.", "", "## Ordinary generation", "",
              "![Landscape generation](generation.jpg)", "", "## Experimental image-to-image", "",
              "![Image-to-image input and output](img2img.jpg)", "",
              "The input is an earlier anagram. At strength 0.35, much of its structure remains, with changes in texture. "
              "This is noise-and-denoise, not trained instruction editing; the check does not establish an image-quality improvement.", "",
              "## Relative depth", "", "![Relative depth](depth.jpg)", "",
              "The example uses a generated painting, not a photograph with ground-truth depth. Raw float32 NPY "
              "values were finite and matched the input resolution. A near-white 16-bit PNG and colour preview were "
              "also saved. Depth is relative log depth, not distance in metres.", "", "## Restoration and upscaling", "",
              "![Restoration](restore.jpg)", "", "The input is the earlier anagram reduced to 256x256. The model "
              "produced 1024x1024 in one tile. The input above is displayed enlarged using bicubic interpolation. "
              "The restorer reconstructs detail; it cannot establish what detail was present in a real original.", "",
              "[Measurements, pinned manifests, source hashes and run settings](measurements.json). "
              "Timings include preprocessing, inference and saving, and exclude downloads, model loading and text encoding. "
              "VRAM is PyTorch peak allocated memory; desktop applications and the CUDA context consume additional VRAM.", "",
              "Reproduce the generation checks with `scripts/check_iris_tools_generation.py` if the earlier pilot's "
              "prompt cache is present. For ordinary use, run `iris_tools.py` as documented in the main README. "
              "`scripts/report_iris_tools.py` rebuilds this report from saved results, without running the models.", ""]
    (args.output/"README.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({k:{"seconds":v["seconds"],"peak_allocated_gib":v["peak_allocated_gib"]} for k,v in records.items()},indent=2))


if __name__ == "__main__":
    main()
