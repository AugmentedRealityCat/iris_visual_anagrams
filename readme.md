# Visual Anagrams with Iris 3B INT8
<img width="2558" height="1851" alt="GUI_VA" src="https://github.com/user-attachments/assets/a9252156-b5cb-412d-8e1a-04ce054b48e2" />



`generate.py` now uses [Aikimi/iris-3b-int8](https://huggingface.co/Aikimi/iris-3b-int8), replacing DeepFloyd IF's two-stage generation with one **1024 x 1024 pixel-space** generation pass. Every denoising step predicts each prompted view, applies its inverse transformation, and combines the aligned predictions. There is no VAE or separate upscaler in this path.

**Turbo r2 is the default:** four steps with built-in guidance. Choose Standard
Iris in the GUI, or pass `--standard`, for the original 100-step Iris sampler.

## Example gallery and showcase





https://github.com/user-attachments/assets/88020cdb-0cd1-42b6-b5cc-13d67ae80aea





[Prompts, seeds and settings](examples/showcase/manifest.json)

This is an experimental adaptation of [Visual Anagrams](https://dangeng.github.io/visual_anagrams/) by Daniel Geng, Aaron Park and Andrew Owens. Iris has different training and sampling behavior from DeepFloyd; illusion quality needs experimentation with prompts, guidance and seeds.

## Windows setup

**Verified on an NVIDIA GeForce RTX 3060 with 12 GB VRAM.** The example gallery
was generated on this card at 1024 × 1024 using Iris INT8 and Turbo r2.

For the supplied Windows setup script, install Git, Python 3.11 or newer (the
script defaults to 3.12), [uv](https://docs.astral.sh/uv/), and an NVIDIA driver
compatible with the script's CUDA 12.8 PyTorch build. Install uv if needed with
`python -m pip install uv`, then run in PowerShell:

```powershell
.\setup-iris.ps1
```

The script creates `.venv`, installs CUDA PyTorch and the inference dependencies, and installs Iris's upstream inference code at a fixed commit. It skips the upstream Linux training dependencies. It does not install the Forge UI.

The first generation downloads approximately **3.23 GB of Iris weights plus 8.88 GB of Qwen text-encoder weights** to the Hugging Face cache. Downloads are reused on later runs. No DeepFloyd license acceptance is needed for Iris. Allow additional disk space for Python/PyTorch and download caches.

The INT8 model uses compressed weights with BF16 computation on CUDA. Views
are evaluated separately with batch size one, and the Qwen text encoder is
released before Iris is loaded onto the GPU. The downloaded weight sizes above
describe disk usage; the two models do not occupy VRAM together. This setup has
completed multi-view generation on the RTX 3060 12 GB. `--text_encoder_device cpu`
also provides an option to encode prompts using system RAM.



https://github.com/user-attachments/assets/1f574ac2-1c45-40f7-b166-2d4942e1d605



## Generate an illusion

### Desktop GUI

Double-click **`launch-gui.bat`**, or run:

```powershell
.\.venv\Scripts\python.exe gui.py
```

The desktop window opens without loading or downloading models. Enter a prompt
and transformation for each view, adjust the style, steps, guidance and seed,
then click **Generate illusion**. Use **Add another view** for more perspectives.
The first generation downloads the required weights if they are not cached.

The GUI shows progress and logs, lets you cancel generation, and previews each
completed sample in **Original image** and **All views** tabs. Existing run names
receive a numeric suffix to preserve previous results. **Open results** opens the
output folder; **Load result** previews an image saved earlier. Advanced settings
include CPU text encoding and local Iris/Qwen model folders. No additional GUI
packages are needed on Windows; Linux may require its `python3-tk` package.

### Moving images and prompt labels

The **Animation & Labels** panel offers **Animated GIF**, **MP4 video**, or
**GIF + MP4**, with frame rate, output size, transition time and pauses per view.
Enable **Automatically animate new images** to export after each generation.
Enable **Animate back to the original view** for a smooth return; GIFs loop.
Rotations turn the actual image, while vertical flips turn it around a
horizontal axis. For runs with several transformed views, each receives its
own animation.

Enable **Overlay prompt labels** to show the matching prompt on the image
(for example, “a horse” at the rotated view). Labels fade out during movement
and reappear at the endpoints. This also saves `.labeled.png` and
`.views.labeled.png` copies; the clean PNGs remain available for future exports.

To animate an existing result without generating again, use **Load result**,
choose the settings, then click **Export animation** beside the preview.
**Play animation** opens the finished file. Its run's `metadata.pkl` must be
present so random transformations can be reproduced. Repeated animation
exports use a numbered suffix to keep earlier files.

Command-line examples:

```powershell
# Export an existing result as GIF and MP4 with prompt overlays:
.\.venv\Scripts\python.exe export_animation.py --im_path results/iris-rotation-fix/0000/sample_1024.png --animation_formats gif mp4 --label_outputs

# Generate and automatically export:
.\.venv\Scripts\python.exe generate.py --name animated-illusion --prompts "a snowy mountain village" "a horse" --style "an oil painting of" --views identity rotate_cw --turbo --animation_formats gif mp4 --label_outputs
```

Use `--animation_fps`, `--animation_size`, `--animation_transition`, and
`--animation_hold` to change timing and size. Add `--animation_one_way` to omit
the return movement. Exports run on CPU using the already-installed imageio
and FFmpeg packages; no model loading or downloads are needed for saved images.

### Turbo LoRA (default)

Choose **Turbo · r2-1000 (recommended)** under **Generation mode** in the GUI.
**Turbo · r1-1500** selects the earlier release. Switching back to **Standard
Iris** restores your previous steps, guidance, negative prompt and flow shift.

The command line uses Turbo r2 automatically (`--turbo` remains accepted):

```powershell
.\.venv\Scripts\python.exe generate.py --name iris-turbo --prompts "a snowy mountain village" "a horse" --style "an oil painting of" --views identity rotate_cw --turbo
```

This downloads only the selected adapter from
[pentacoxian-dev/iris-3b-turbo-lora](https://huggingface.co/pentacoxian-dev/iris-3b-turbo-lora)
(about 460 MB), at pinned revision `fe16b3496eb440585993c92d95641c08c6b08263`.
`r2-1000` is the default; use `--turbo_adapter r1-1500` for the earlier adapter.

Turbo defaults to **4 steps** and **guidance 1** because CFG 3 is already
distilled into the adapter. Negative prompts and DPM solver order are unused;
the CLI rejects extra guidance or nonempty negative prompts in Turbo mode.
The GUI disables those controls while Turbo is selected. The step count stays
editable for experiments; the released adapters were trained for four steps.

The Turbo sampler follows the author's shifted re-noising schedule. It aligns
the velocity predictions from all views before adding fresh noise to the shared
image between steps. INT8 base weights stay packed, with low-rank residuals
evaluated alongside them. FP32 linear updates, including shared modulation
cores, are merged in place. Adapter identity and settings are saved in metadata.
This INT8, multi-view adaptation has automated CPU/CUDA checks and a completed
1024-square Turbo smoke run. Illusion quality still depends on prompts and seeds.

### Standard generation

```powershell
.\.venv\Scripts\python.exe generate.py --name iris.village.horse --prompts "a snowy mountain village" "a horse" --style "an oil painting of" --views identity rotate_cw --standard --num_samples 1 --num_inference_steps 100 --guidance_scale 3.0
```

Output:

- `results/iris.village.horse/0000/sample_1024.png`: the illusion.
- `results/iris.village.horse/0000/sample_1024.views.png`: the prompted views side by side.
- `metadata.pkl`: view objects for the existing animation tool.
- `metadata.json`: arguments and pinned model/code revisions.

Try `identity flip`, `identity rotate_180`, `identity negate`, `identity inner_circle`, `identity jigsaw`, or `identity patch_permute` for other transformations. `--prompts` and `--views` must contain the same number of entries. Random views are seeded before construction and shared across samples in a run.

Useful options:

- `--negative_prompt "..."`: one negative prompt shared by all views.
- `--seed 42`: first sample seed; later samples increment it.
- `--reduction mean`: average predictions, the default for anagrams. `alternate` cycles views between steps; `sum` adds components for factorized diffusion.
- `--view_args`: one argument per view, following the existing view definitions.
- `--shift 4.0` and `--solver_order 2`: upstream flow sampling defaults.
- `--model_path PATH`: a local Iris INT8 generation export containing `model.safetensors`, `config.yaml`, and `manifest.json`.
- `--text_encoder_path PATH`: a local Qwen3-VL-4B-Instruct directory.

The command intentionally uses 1024-square images: the existing permutation views support that resolution, and the model publisher reports artifacts at 256/512. `--generate_1024` remains accepted for old commands and is redundant. Iris does not produce the old 64/256 intermediate files. Reference-image inverse problems (`--ref_im_path`) and stage-two noise (`--noise_level`) remain available through the legacy DeepFloyd command, not the Iris command.

Animate a two-view result with the existing tool:

```powershell
.\.venv\Scripts\python.exe animate.py --im_path results/iris.village.horse/0000/sample_1024.png --metadata_path results/iris.village.horse/metadata.pkl
```

## Other platforms / manual installation

Install an appropriate PyTorch/torchvision build first, then use Python 3.11+:

```bash
python -m pip install -r requirements-iris.txt
python -m pip install --no-deps 'iris-3b @ git+https://github.com/speridlabs/iris-3b.git@a8d15239dea469aba042cfa56ca3bb4e450d5ebc'
python -m pip install --no-deps -e .
python -m pytest tests/test_iris.py -q
```

CPU inference is supported with `--device cpu --text_encoder_device cpu` and is very slow at this resolution. No model weights are downloaded by the automated tests.

## Implementation and sources

- `visual_anagrams/iris.py`: checkpoint verification, rowwise INT8 loader, sequential text encoding, and multi-view integration using upstream `FlowDPMSolver` with FP32 state.
- The 1024 inverse permutation now uses the matching 1024 permutation; a duplicate destination caused by a jigsaw mask boundary pixel is repaired into a bijection.
- Right-angle rotations use exact pixel permutations. Affine-grid rotations under BF16 autocast rounded coordinates at 1024px, causing a small central image surrounded by noise. Full-resolution CUDA regression tests cover this failure in both RGB and six-channel views.
- [Iris inference source](https://github.com/speridlabs/iris-3b/tree/a8d15239dea469aba042cfa56ca3bb4e450d5ebc) (Apache-2.0).
- [INT8 checkpoint format and benchmarks](https://huggingface.co/Aikimi/iris-3b-int8).
- [Aikimi's reference INT8 loader](https://github.com/AiWithYou/aikimi-forge-neo/blob/neo/modules_forge/iris/quantization.py).

The original DeepFloyd generation script is retained as `generate_deepfloyd.py`. See [the original documentation](readme_deepfloyd.md) and `environment_deepfloyd.yml` for its separate legacy environment. The original notebooks still use DeepFloyd. [Factorized diffusion documentation](readme_factorized_diffusion.md) describes the legacy path; its commands now explicitly target that script.
