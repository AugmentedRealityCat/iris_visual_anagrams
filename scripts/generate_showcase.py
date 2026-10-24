"""Generate reproducible candidates for the GitHub/Reddit showcase."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--plan", type=Path, default=ROOT / "examples/showcase/plan.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/showcase_candidates")
    args = parser.parse_args()
    cases = json.loads(args.plan.read_text(encoding="utf-8"))
    for index, case in enumerate(cases, 1):
        complete = all((args.output / case["id"] / f"{case['seed'] + offset:04}" / "sample_1024.png").is_file()
                       for offset in range(args.samples))
        if complete:
            print(f"Already generated: {case['id']}", flush=True)
            continue
        command = [sys.executable, "-u", str(ROOT / "generate.py"), "--name", case["id"],
                   "--save_dir", str(args.output), "--style", case["style"],
                   "--prompts", *case["prompts"], "--views", *case["views"],
                   "--turbo", "--turbo_adapter", "r2-1000", "--seed", str(case["seed"]),
                   "--num_samples", str(args.samples), "--num_inference_steps", "4", "--guidance_scale", "1"]
        if case.get("view_args"):
            command.extend(["--view_args", *case["view_args"]])
        print(f"\nShowcase {index}/{len(cases)}: {case['method']} / {case['style']}", flush=True)
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
