"""Small real-model Iris Tools check, reusing the earlier pilot's prompt embeddings.

Runs one landscape generation and one experimental img2img through the public
CLI entry point. Reusing the exact matching Qwen encodings avoids reloading the
text encoder; it does not replace the image model or sampler.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    from iris_tools import main as run
    from visual_anagrams.iris import PromptEncoding

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/iris-tools-check")
    parser.add_argument("--pilot", type=Path, default=ROOT / "results/quality_pilot_20261024")
    args = parser.parse_args()
    plan = json.loads((args.pilot / "plan.json").read_text(encoding="utf-8"))
    saved = torch.load(args.pilot / "prompt-encodings.pt", map_location="cpu", weights_only=True)
    assert saved["plan_hash"] == hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    case = plan["cases"][0]
    prompt = f"{case['style']} {case['prompts'][0]}"
    positive = PromptEncoding(saved["embeddings"][:1], saved["mask"][:1])
    negative = PromptEncoding(saved["negative_embeddings"], saved["negative_mask"])

    def encode(_cfg, prompts, negative_prompt, *_args, **_kwargs):
        assert prompts == [prompt] and negative_prompt == ""
        return positive, negative

    trials = [
        ("landscape-smoke", ["--task", "generate", "--size", "1344x768", "--steps", "30"]),
        ("img2img-smoke", ["--task", "img2img", "--input", str(args.pilot / "standard-g3/owl-dog/0000/sample_1024.png"),
                           "--steps", "15", "--strength", "0.35"]),
    ]
    for name, options in trials:
        folder = args.output / name
        if (folder / "0000/run.json").is_file():
            print("Already complete", name, flush=True)
            continue
        with patch("visual_anagrams.iris.encode_prompts", encode):
            run(["--name", name, "--save_dir", str(args.output), "--prompt", prompt,
                 "--text_encoder_device", "cpu", "--seed", "0", *options])
        path = folder / "metadata.json"
        metadata = json.loads(path.read_text(encoding="utf-8"))
        metadata["test_encoding_reused_from"] = str(args.pilot / "prompt-encodings.pt")
        path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
