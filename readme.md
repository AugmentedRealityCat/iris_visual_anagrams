# Visual Anagrams with Iris 3B INT8
<img width="3823" height="2051" alt="GUI_VA" src="https://github.com/user-attachments/assets/0818f5c2-ac48-43e1-b737-6658daedcde0" />


`generate.py` now uses [Aikimi/iris-3b-int8](https://huggingface.co/Aikimi/iris-3b-int8), replacing DeepFloyd IF's two-stage generation with one **1024 x 1024 pixel-space** generation pass. Every denoising step predicts each prompted view, applies its inverse transformation, and combines the aligned predictions. There is no VAE or separate upscaler in this path.

**Standard Iris is the default:** 30 steps, guidance 3, Auto view combining,
flow shift **8**, solver order 2, and **adaptive balancing for two geometric
views**, now with **Qwen cross-view negatives at strength 0.5**. The
[latest quality pilot](examples/quality-experiments/README.md) also enables
adaptive component guidance for Standard low/high-frequency and grayscale/colour
pairs. These are starting points from a small test, not universally optimal settings.
Cross negatives add roughly 50% more model evaluations; choose **Off** for speed.
Turbo r2 remains an
optional fast sampler for previews, seed searches, and puzzle experiments.
Turbo starts at four steps, guidance 1 and shift 8, with adaptive balancing for
two geometric Auto/Mean views. Component groups and other view counts use Classic.

## Example gallery and showcase

https://github.com/user-attachments/assets/49d2e602-4bef-40a6-b81d-c5f48c77cda7


This is an experimental adaptation of [Visual Anagrams](https://dangeng.github.io/visual_anagrams/) by Daniel Geng, Aaron Park and Andrew Owens. Iris has different training and sampling behavior from DeepFloyd; illusion quality needs experimentation with prompts, guidance and seeds.

## Windows setup

**Verified on an NVIDIA GeForce RTX 3060 with 12 GB VRAM.** The example gallery
was generated on this card at 1024 × 1024 using Iris INT8 and Turbo r2.
The Standard/Turbo quality pilot also ran on this RTX 3060.

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
The editable **Shared style** dropdown offers 29 presets, including gouache,
charcoal, pen and ink, woodcut, engraving, linocut, comic and graphic novel art,
art nouveau, art deco, stained glass, mosaic, paper cut, origami, silhouettes,
vector illustration, screenprint, pixel art and Japanese woodblock prints.
You can also type your own style or clear the field.
The first generation downloads the required weights if they are not cached.
The initial settings are **Standard Iris**, **30 steps**, **guidance 3**,
**flow shift 8**, **Auto** combining and **adaptive balancing** for the initial
two geometric views. Selecting a Turbo mode sets four steps, shift 8 and built-in guidance;
switching back restores your previous Standard settings.
Turbo starts with adaptive balancing when the current run has two geometric
views and Auto/Mean combining. You can select Classic instead; switching
between generation modes remembers your choice for each mode.

The GUI shows progress and logs, lets you cancel generation, and previews each
completed sample in **Original image** and **All views** tabs. Existing run names
receive a numeric suffix to preserve previous results. **Open results** opens the
output folder; **Load result** previews an image saved earlier. Advanced settings
include CPU text encoding and local Iris/Qwen model folders. No additional GUI
packages are needed on Windows; Linux may require its `python3-tk` package.

Use **Dark mode** at the top right to switch the appearance of both pages.
Your choice is remembered for the next launch. It changes controls, dropdowns,
logs and preview backgrounds; saved images keep their original colours.

### Iris Tools page

The separate **Iris Tools** tab uses Iris for tasks beyond visual anagrams.
The **Visual Anagrams** tab keeps its own prompts and settings. Only one model
task runs at a time; weights are released when its worker exits.

| Task | Use | Weights |
| --- | --- | --- |
| **Image generation** | One prompt → an ordinary image, with square, portrait or landscape sizes | Existing generation INT8 model and Qwen; Standard or optional Turbo LoRA |
| **Image-to-image (experimental)** | Add flow noise to an input, then denoise with a prompt; useful for variations and restyling | Existing generation INT8 model and Qwen; Standard only |
| **Relative depth** | Estimate near/far structure from one image | Separate depth INT8 model; no Qwen |
| **Restore / upscale** | Restore a low-quality image and enlarge it up to 4× | Separate restoration INT8 model; no Qwen |
| **Depth parallax (experimental)** | Small 2.5D camera movement, exported as borderless MP4/GIF with GUI playback | Depth INT8, or cached raw depth |
| **Video parallax (experimental)** | Apply a depth-driven camera movement to an existing video or selected clip | Depth INT8 evaluated on each output frame; optional source audio in MP4 |
| **Magic Eye stereogram (experimental)** | Hide a predicted depth relief in random dots or an optional repeating texture | Depth INT8, or cached raw depth |
| **Depth focus sweep (experimental)** | Animate focus across depth layers of one fixed image | Depth INT8, or cached raw depth |
| **Colour / depth anagram (experimental)** | Edit one RGB image so its estimated depth approaches a second reference's relief | Depth INT8 with frozen-weight input gradients; no generator or Qwen |

Ordinary generation, monocular depth, and restoration/upscaling are released
Iris tasks. The depth and restoration models are task-specific fine-tunes;
the generation checkpoint cannot substitute for them.
[Official Iris documentation](https://github.com/speridlabs/iris-3b#depth-image-restoration-and-upscaling),
[INT8 model card](https://huggingface.co/Aikimi/iris-3b-int8).

Choose a task and, where needed, browse for an input image. The page shows
**Input** and **Result** previews, progress, logs, cancellation and an output
folder button. Existing GUI run names receive a numeric suffix. Results go to
`results/iris-tools/` by default, with `metadata.json` for the run and
`run.json` beside each result.

Depth and restoration each download approximately **3.23 GB of weights plus
37 MB of empty-prompt embeddings** on first use. Only the selected task is
downloaded. Files are checked against the pinned export's SHA-256 manifest.
A local task folder can be selected with **Local INT8 task model**; for depth
or restoration this must be the matching export or its parent folder.

- **Generation:** size presets follow the upstream model's approximately
  one-megapixel buckets. Standard starts at 30 steps and guidance 3. Turbo
  starts at four steps and guidance 1; switching back restores the Standard
  values. The optional adapters are the existing community LoRAs, rather than
  the newer, separate official full-model Turbo checkpoint.
- **Image-to-image:** strength is in `(0, 1]`, default 0.5. Lower values retain
  more of the input. The input is resized and centre-cropped to the selected
  output shape. This is an experimental noise-and-denoise adaptation, not a
  model trained to follow editing instructions or preserve identity.
- **Depth:** returns affine-invariant relative log depth, **not metres**.
  `depth.npy` retains raw float32 values; `depth16.png` maps nearer regions to
  white using the output's 2nd–98th percentiles. `result.png` is a colour preview.
  Inference defaults to a 1024-pixel long-side cap, then returns depth at the
  original input resolution.
- **Restoration:** runs the upstream 4× one-step recipe with overlapping
  1024-pixel tiles and wavelet colour correction. **1×** and **2×** resize the
  restored 4× result. By default large inputs are first limited to 512 pixels
  on the short side and 1024 on the long side, so the output size is based on
  that processed input. Uncheck **Limit large restoration inputs** to keep
  native input dimensions; time and memory grow with output size. Generated
  details may differ from the original scene.

The same tasks are available without the GUI:

```powershell
# Ordinary landscape generation (existing generation weights):
.\.venv\Scripts\python.exe iris_tools.py --task generate --prompt "a watercolor painting of a lighthouse" --size 1344x768

# Experimental image-to-image:
.\.venv\Scripts\python.exe iris_tools.py --task img2img --input photo.jpg --prompt "an oil painting of a coastal village" --strength 0.35

# Relative depth, including raw NPY and 16-bit PNG:
.\.venv\Scripts\python.exe iris_tools.py --task depth --input photo.jpg

# Restore and enlarge 4x, or use --scale 1 to retain the processed input size:
.\.venv\Scripts\python.exe iris_tools.py --task restore --input photo.jpg --scale 4
```

Use `--text_encoder_device cpu` for generation without Qwen on the GPU.
`--model_path` selects local task weights and `--text_encoder_path` selects a
local Qwen folder. CLI runs get timestamped names unless `--name` is given;
an existing CLI output folder is rejected to preserve previous results.

**Original Iris Tools check on the RTX 3060 12 GB:** all four original task paths completed. Landscape
generation at 1344×768 took 188 seconds (30 steps), image-to-image took
96 seconds (15 steps), and single-pass depth/restoration took about 4 seconds
each. Peak allocated VRAM was **6.78–6.87 GiB** for these examples. Model
loading and text encoding are excluded from those processing times.
[Example outputs, exact settings and test limitations](examples/iris-tools/README.md).
That earlier regression suite passed 162 checks.

### Depth experiments

The experimental depth tasks above are available in **Iris Tools → Task**.
[Examples, MP4 loops and measured tradeoffs](examples/depth-lab/README.md).
Parallax uses a depth buffer and local filling for newly exposed pixels; small
movements work best. The focus sweep renders one fixed image with depth-dependent
blur. It does not crossfade subjects or guarantee a second readable image.
Magic Eye outputs use parallel viewing and include a decoded-depth diagnostic.
The optional texture colours the repeating pattern rather than preserving an
independently readable second portrait.

The colour/depth pilot edits Joker toward a Batman depth relief. The gentler
384-pixel trial improved same-resolution depth correlation from 0.61 to 0.92,
while retaining visible colour artifacts. It survived JPEG and resize checks at
that resolution; the report also includes a separate 512-pixel inference check.
This is estimator-driven input optimization, not a trained depth-conditioned
generator or a proven human-visible illusion. It starts at **384 pixels,
24 iterations and RGB edit bound 0.06**. The target is centre-fitted to the input
aspect ratio; saved pixels are rounded to 8-bit. The other anagram defaults
remain unchanged.

```powershell
python iris_tools.py --task parallax --input batman.png --depth_formats gif mp4
python iris_tools.py --task parallax_video --input video.mp4 --video_start 0 --video_seconds 1
python iris_tools.py --task stereogram --input batman.png --texture joker.png
python iris_tools.py --task focus --input hybrid.png
python iris_tools.py --task depth_anagram --input joker.png --target batman.png
```

Parallax/focus settings include **Animation output** (MP4 video, Animated GIF,
or GIF + MP4), render size, duration, FPS, camera movement or
defocus radius. Stereograms expose the repeat period and random seed. Renderers
can reuse `--depth_map path/to/depth.npy` matching the original input dimensions;
those runs load no model weights. The optimizer exposes iteration count, work
resolution and edit bound. It saves before/after depth, a target relief, pixel
change measurements, per-iteration scores and JPEG/resize checks. Parallax and
focus animations play and loop directly in the GUI's **Result** preview. Use
**Play / Pause** to control playback; **Load result…** accepts earlier MP4s,
GIFs and images. Loading a saved result PNG also discovers its accompanying
animation. MP4 remains the default export. CLI users select `--depth_formats
gif`, `mp4`, or `gif mp4`; both formats share the same render pass.

**Video input:** choose **Video parallax (experimental)** and browse for an
MP4, MOV, MKV, AVI or WebM file. **Start time** and **Clip length** select the
part to process. Start **0**, length **1** exports only the first second;
start **10**, length **2** selects seconds 10–12. A length of **0** processes
the rest of the video. These times are in seconds, including fractions.
The default output frame rate follows the source, capped at 60; select a lower
rate for faster processing. **Camera loop** controls the added movement's
period, independently of clip length. Output is constant-frame-rate MP4, GIF,
or both, and plays in the GUI.

**Keep source audio in MP4** preserves the selected clip's first audio track,
re-encoded as AAC. GIFs and the embedded GUI preview are silent. **Depth
smoothing** blends recent depth only where RGB changes are small and resets
at detected scene cuts. Set it to 0 to disable history blending; lower values
are preferable for fast motion. This is a heuristic using the existing
single-image Iris depth model, not motion-compensated or video-trained depth.
Each output frame requires inference, and large movements can expose halos or
stretched background. A single cached `depth.npy` is not used for video input.
Clip timing, actual frame count/FPS, depth settings and audio retention are
recorded in `run.json`. [Short real-model check](examples/video-parallax/README.md).

### Moving images and prompt labels

The **Animation & Labels** panel offers **Animated GIF**, **MP4 video**, or
**GIF + MP4**, with frame rate, output size, transition time and pauses per view.
Enable **Automatically animate new images** to export after each generation.
Enable **Animate back to the original view** for a smooth return; GIFs loop.
Rotations turn the actual image, while vertical flips turn it around a
horizontal axis. For runs with several transformed views, each receives its
own animation.

Enable **Borderless / fill frame** to remove the outer margin from GIF and MP4
exports. Rotating corners and moving pieces are cropped to the image viewport;
edge-on flips use a blurred image background instead of blank padding. Leave it
unchecked to keep space around the image for the complete motion.

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
.\.venv\Scripts\python.exe generate.py --name animated-illusion --prompts "a snowy mountain village" "a horse" --style "an oil painting of" --views identity rotate_cw --animation_formats gif mp4 --label_outputs
```

Use `--animation_fps`, `--animation_size`, `--animation_transition`, and
`--animation_hold` to change timing and size. Add `--animation_one_way` to omit
the return movement. Add `--animation_borderless` to fill the frame without
outer padding. Exports run on CPU using the already-installed imageio
and FFmpeg packages; no model loading or downloads are needed for saved images.

### Quality findings and starting settings

Our pilot generated **18 images**: two rotation prompt pairs, seed 0, and
1024 × 1024 INT8 inference on the RTX 3060. The table shows the mean of each
image's weaker-view CLIP score, using ViT-L/14, which did not guide generation.
Higher scores indicate stronger prompt alignment, not necessarily a better
optical illusion.

| Settings | Weaker-view score ↑ | Generation time per image |
| --- | ---: | ---: |
| **Standard, 30 steps, guidance 3** | **0.268** | 403 s |
| Turbo r2, 4 steps | 0.204 | 28 s |
| Turbo r2, 8 steps | 0.244 | 56 s |
| Turbo r2, 16 steps | 0.201 | 113 s |
| Turbo r2, 4 steps, experimental adaptive weighting | 0.240 | 28 s |

Standard guidance 3–4 gave stronger overall alignment than the four-step Turbo
baseline; increasing guidance to 5 did not improve the mean score. Eight Turbo
steps were a useful experiment, while sixteen were unreliable. Adaptive
weighting improved both tested pairs at little extra sampling time and is now
available as a standalone GUI option. Bounded variance correction
had little practical effect.

This pilot used only one seed per pair and did not test puzzles, all other
transformations, or 100-step Standard sampling. Some Standard images still
separate the two subjects into different areas despite high scores. Inspect
both views, try different seeds, and treat these settings as a starting point.
The two CLIP evaluators also disagreed on some rankings.

[Full results, both evaluators, comparison images and reproduction commands](examples/quality-pilot/README.md).

### Adaptive balancing

Select **Quality method → Adaptive balancing (experimental)** in the Visual
Anagrams page. It works with **Standard Iris or Turbo**, with **two geometric
views** and **Auto/Mean** combining. The gradient-optimization fields are hidden
because this method adjusts prediction weights without attention gradients or
variance correction. Factorized component groups and Alternate combining use
Classic.

The method scores clean previews with a pinned **CLIP ViT-B/32 on CPU** every
two steps below noise level 0.95. The weaker view receives more weight, with
desired weights bounded to 0.25–0.75 and smoothed halfway toward the previous
weights. Each new sample starts at equal weights. The first use downloads
about 600 MB of CLIP weights if they are not cached. The GUI itself loads no
models when it opens.

```powershell
.\.venv\Scripts\python.exe generate.py --name adaptive-owl-dog --prompts "an owl" "a dog" --style "a watercolor painting of" --views identity rotate_180 --turbo --quality_method adaptive
```

Omit `--turbo` for Standard sampling. Adaptive balancing uses the selected flow
shift, step count and Standard solver order. Its per-step scores, weights and
scorer revision are saved in `quality.json` beside each output.

**Default policy:** adaptive balancing is selected by default for two geometric
Auto/Mean views in both Standard and Turbo. It improved weaker-view alignment
in the earlier Turbo pilot, three subsequent Turbo comparisons, and both
30-step Standard comparisons below. The anagram flow-shift default is 8.
Other view counts, component groups and Alternate combining use Classic.
Select **Classic** in the GUI, or pass `--quality_method plain`, to use equal
weights explicitly. The lower-level sampler API still uses Classic unless a
balancer is passed.

Standard paired Adaptive views also use Qwen cross negatives following the newer
quality pilot below. Pass `--cross_negative off` to reproduce the earlier
Adaptive-only comparisons in this section.

[Production-method comparison and images](examples/adaptive-balancing/README.md):
held-out weaker-view scores changed from **0.225 to 0.230** (four steps / shift 4),
**0.245 to 0.273** (four steps / shift 8), and **0.230 to 0.256** (eight steps /
shift 4), adding roughly 1–2 seconds per Turbo image. The images still separate the
subjects, so this is evidence for better alignment, not a convincing illusion
or a general quality guarantee.

A [30-step Standard adaptive check](examples/standard-adaptive/README.md)
compared both shifts 4 and 8 with their Classic controls. Adaptive raised the
weaker-view score from 0.239 to 0.250 at shift 4 and from 0.255 to 0.263 at
shift 8, adding about 18 seconds per image. Both faces still occupy separate
areas. **Standard now starts at 30 steps, guidance 3, shift 8 and adaptive
balancing** for paired geometric views. This chooses the strongest tested
Standard alignment profile while keeping Classic available as an override.

### Quality experiments and tested defaults

The Visual Anagrams page has a **Quality experiments** card. The options below
can be controlled independently. The [full-size quality pilot](examples/quality-experiments/README.md)
contains nine new 1024-pixel, 30-step trials against four verified cached controls.
**Qwen cross negatives at strength 0.5 are now the default for Standard paired
geometric Auto/Mean views with Adaptive balancing**: weaker-view alignment and
prompt separation improved on both Venom/Spawn warp and Hulk/Loki jigsaw.
The gains are modest and do not guarantee a convincing illusion. This uses six
instead of four Iris forwards per guided step, roughly 50% more sampling work.
Classic geometric views retain shared negatives unless explicitly changed.

The pilot also supports **component guidance by default for Standard low/high-frequency and
grayscale/colour pairs**, using Classic and Sum. The rose-view score was almost
unchanged, while the frequency hybrid's weaker-view score improved about 14%.
Both component trials used fewer model evaluations and ran faster in this check.
Untested motion groups and Turbo retain their previous settings.
The base defaults remain Standard, 30 steps, guidance 3, shift 8 and Adaptive
balancing for paired geometric views.
Use **Show / hide experiment parameters** to adjust their strengths and timing.

| GUI option | CLI switch | Compatibility and behaviour |
| --- | --- | --- |
| Contrastive adaptive balancing | `--contrastive_balancing` | Two geometric Mean/Auto views, Adaptive; Standard or Turbo. Balances own-prompt minus other-prompt similarity, retaining the existing bounded, smoothed weights. |
| Cross-view negative prompts | `--cross_negative subjects`, `qwen`, or `manual` | Standard, two geometric Mean/Auto views, Classic or Adaptive. Uses a different subject negative for each view. |
| Adaptive component guidance | `--component_guidance` | Standard, Classic, complete low/high, grayscale/colour, or motion/residual groups with Sum/Auto. Scores the observable filtered/full views and adjusts each branch's extra CFG. |
| Schedule CFG in middle noise levels | `--guidance_schedule middle` | Standard, Classic or Adaptive. Smoothly ramps extra CFG inside a flow-noise interval; outside it, guidance is 1, retaining text conditioning. |
| Conflict-aware prompt directions | `--conflict_projection` | Standard, two geometric Mean/Auto views, Classic or Adaptive. Dampens conflicting conditional directions while preserving their shared-reference predictions. |

The GUI disables incompatible controls and restores Standard experiment settings
when returning from Turbo. Defaults are shared with the CLI and explicit choices
override them. To disable automatic component guidance, uncheck its GUI box or
use `--no_component_guidance`. Other boolean options have equivalent
`--no_contrastive_balancing` and `--no_conflict_projection` overrides;
use `--cross_negative off`, `--guidance_schedule constant` or
`--balance_freeze_below 0` for the remaining options. These options do not combine with Anagram-MTL or
SyncVC. New experiments are saved under `quality_experiments` in `metadata.json`.
When enabled, each sample's `quality.json` contains `balancing` and `experiments`
with the actual margins, weights, guidance scales, component gains or conflict
measurements. Controller state starts fresh for every sample.

**Automatic negative prompts:** choose **Qwen subject extraction** to reuse the
bundled Qwen3-VL-4B-Instruct text decoder. It extracts short, distinct subject
phrases from the two prompts and swaps them: Venom's negative is Spawn, and
Spawn's negative is Venom. Shared style words are excluded. This runs while Qwen
is already loaded for encoding; its tied token embeddings supply the language
head, so no second LLM or additional model weights are needed. Qwen is released
before Iris loads. The resolved negatives and raw planning response are printed
and saved as `resolved_negative_plan` in `metadata.json`.

The extracted phrases must occur in their own prompt and distinguish it from
the other prompt. If Qwen subject extraction fails under the automatic CLI default
or in the GUI, the run logs the reason and falls back to shared negatives only;
`resolved_negative_plan` records that fallback and the actual mode is saved as
`off`. An explicit CLI `--cross_negative qwen` stops on invalid output unless
`--negative_fallback` is supplied. Encoding and model errors still stop the run.
**Other subject (automatic)**
uses a simple shared-prefix/suffix removal instead of LLM generation. For full
control, choose **Manual per-view negatives** and separate them with `|`.
The prompts you enter remain unchanged.

For example, to enable only Qwen-planned cross negatives on a warp:

```powershell
python generate.py --name venom-spawn-negative --standard --prompts "Venom, a black symbiote with sharp teeth" "Spawn, a masked antihero with green eyes and a red cape" --style "a dark comic book portrait of" --views identity skew --view_args 0 0.75 --cross_negative qwen
```

Manual CLI negatives use `--cross_negative manual --view_negative_prompts
"Spawn" "Venom"`. `--cross_negative_strength` blends the shared negative and
other-subject reference, from 0 to 1 (initial experimental value: 0.5). Between
0 and 1 this adds one model forward per view at guided steps. At 1 it can replace
the shared negative forward, unless conflict projection also needs that reference.
Cross negatives can suppress useful shared features, so stronger is not assumed
to be better.

For component guidance, keep the complete partition and Classic balancing:

```powershell
python generate.py --name vader-castle-component --standard --prompts "Darth Vader's helmet" "a haunted castle" --style "a monochrome engraving of" --views low_pass high_pass --view_args 2 2 --quality_method plain --component_guidance
```

Component guidance keeps a single shared reference `u` and combines
`u + sum_i s_i * P_i(c_i - u)`, where the component operators sum to identity.
It adapts extra CFG gains, bounded to 0.5–1.5 with 50% smoothing, from CPU CLIP
scores every two steps below noise 0.95. It does not reweight the component
partition itself. The shared reference needs only one model evaluation per step.

Scheduled CFG parameters are `--guidance_low 0.1 --guidance_high 0.9
--guidance_ramp 0.05`. These bounds refer to **flow noise**, not percentages of
steps: 1 is noise and 0 is clean. Smooth ramps avoid an abrupt CFG switch.
`--balance_freeze_below 0.6` holds Adaptive weights or component gains fixed near
the end of the tested 30-step / shift-8 grid; its default remains 0, disabled.
At these settings the lowest model-evaluation noise is about 0.216, and the
last two balancing updates occur around 0.551 and 0.363. A freeze threshold of
0.1 therefore has no effect on this grid; 0.6 skips those two updates. The
freeze trial did not improve the warp result, so it is not enabled by default.

Conflict projection uses the original two conditional directions symmetrically,
only between noise 0.05 and 0.95. `--conflict_strength` is in [0,1], initially
0.5 when the option is enabled. It is a PCGrad-inspired inference heuristic,
not a claim that flow velocities satisfy gradient-descent guarantees. All five
options run without backpropagation through Iris.

Implementation checks cover the math, neutral sampler equivalence, GUI/CLI routing,
metadata and Qwen's cached-model subject extraction. Full-size comparisons,
images, scores and the final default decision are in the
[quality-pilot report](examples/quality-experiments/README.md). This is a small
matched pilot, not a universal optimum. Combinations were not validated; the
selected defaults use only the individually tested option for each context.

Research background: [Visual Anagrams](https://arxiv.org/abs/2311.17919),
[Factorized Diffusion](https://arxiv.org/abs/2404.11615),
[guidance intervals](https://arxiv.org/abs/2404.07724), and
[PCGrad](https://arxiv.org/abs/2001.06782).

### Optional Anagram-MTL and SyncVC methods

The GUI's **Quality method** menu offers **Classic**,
**Adaptive balancing (experimental)**, **Anagram-MTL · Iris (experimental)**,
and **SyncVC · Iris (experimental)**. Anagram-MTL and SyncVC require
**Standard Iris** and **two geometric views** with Auto/Mean combining. Turbo
offers Classic or standalone adaptive balancing; factorized components use Classic.

Use the default budget of **one optimization pass every five steps**, strength
**0.05**, with 30 generation steps and guidance 3. Increasing the passes or
optimizing more often costs more time. CPU activation offloading is enabled
by default to reduce GPU memory; the methods still use additional system RAM
and more VRAM than Classic. They keep the image model's weights frozen.

For Anagram-MTL, enter the main subject words separated by `|`, such as
`owl | dog`. Leave this field empty to use each view's full prompt phrase.
Subject words must appear in their corresponding prompts.

```powershell
# Anagram-MTL adaptation: balancing, variance correction, attention overlap
.\.venv\Scripts\python.exe generate.py --name mtl-owl-dog --prompts "an owl" "a dog" --style "a watercolor painting of" --views identity rotate_180 --quality_method anagram_mtl --subjects owl dog

# SyncVC adaptation: optimize controls coupling two denoising trajectories
.\.venv\Scripts\python.exe generate.py --name syncvc-owl-dog --prompts "an owl" "a dog" --style "a watercolor painting of" --views identity rotate_180 --quality_method syncvc
```

`--optimization_interval`, `--optimization_steps`, and `--method_strength`
control the optimization budget. `--no_gradient_offload` keeps saved activations
on the GPU for machines with more free VRAM. Per-step diagnostics and the actual
solver are saved as `quality.json` beside each generated PNG.

These are **Iris adaptations**, not reproductions of the DeepFloyd pipelines:

- **Anagram-MTL:** scores clean previews with CPU CLIP, balances the predicted
  velocities, applies bounded variance correction in noise space, and takes
  gradient steps to increase overlap of subject attention maps. It reads real
  image-to-text probabilities from Iris's first single-stream joint-attention
  block, using queries pooled to at most 32 × 32. Attention edits are scaled by
  `t * (1 - t)` to limit their effect near pure noise and the clean endpoint.
  The original uses a noise-aware CLIP encoder and U-Net cross-attention maps.
- **SyncVC:** maintains two trajectories and optimizes image-space control
  variables for clean-image agreement plus prior regularization. It uses a
  **first-order flow Euler solver**, rather than the Classic DPM order setting.
  The original DDIM coefficients are replaced by Iris's `alpha = 1 - t` and
  `sigma = t`; optimization skips the near-pure-noise endpoint. The one-pass,
  five-step interval is a smaller optimization budget than the reference code.

Both use block checkpointing and a memory-saving backward pass that recreates
frozen INT8 linear weights instead of retaining dequantized weight copies.
Their quality depends on prompts, seed, and guidance strength. Keep Classic
as the baseline when comparing results.

[Small RTX 3060 comparison, outputs and diagnostics](examples/advanced-methods/README.md)
tests both options with one prompt pair and seed, reusing the earlier Classic
baseline. It also preserves the failed first MTL trial and its corrective test.
Neither adaptation improved that pair: held-out weaker-view CLIP scores were
**0.305 Classic**, **0.293 MTL**, and **0.188 SyncVC**. Corrected MTL took
about **11 minutes** and SyncVC **16 minutes**, both using about **9.45 GiB**
of allocated VRAM on the RTX 3060. MTL and SyncVC remain optional; the paired
anagram default uses standalone adaptive balancing. One pair cannot establish
whether these larger adaptations help other prompts.

Sources: [Anagram-MTL](https://github.com/Pixtella/Anagram-MTL),
[SyncVC](https://github.com/mandt-lab/SyncVC).

### Turbo LoRA (optional fast mode)

Choose **Turbo · r2-1000 (fast)** under **Generation mode**, or pass `--turbo`.
**Turbo · r1-1500** selects the earlier adapter. Turbo is useful when exploring
many prompts and seeds quickly, including patch-shuffle and jigsaw puzzles.
The existing showcase includes Turbo puzzle examples; the quality pilot did
not establish whether Standard or Turbo is better for puzzles.

```powershell
.\.venv\Scripts\python.exe generate.py --name iris-turbo --prompts "a snowy mountain village" "a horse" --style "an oil painting of" --views identity rotate_cw --turbo

# A fast 4-by-4 patch puzzle experiment:
.\.venv\Scripts\python.exe generate.py --name turbo-puzzle --prompts "an elephant" "a rabbit" --style "a pencil sketch of" --views identity patch_permute --view_args 0 4 --turbo
```

This downloads only the selected adapter from
[pentacoxian-dev/iris-3b-turbo-lora](https://huggingface.co/pentacoxian-dev/iris-3b-turbo-lora)
(about 460 MB), at pinned revision `fe16b3496eb440585993c92d95641c08c6b08263`.
`r2-1000` is the default adapter when Turbo is selected; use
`--turbo_adapter r1-1500` for the earlier adapter. Standard does not load a LoRA.

Turbo defaults to **4 steps** and **guidance 1** because CFG 3 is already
distilled into the adapter. Negative prompts and DPM solver order are unused;
the CLI rejects extra guidance or nonempty negative prompts in Turbo mode.
The GUI disables those controls while Turbo is selected. The step count stays
editable for experiments; the released adapters were trained for four steps.
Use `--turbo --num_inference_steps 8` to try the longer Turbo setting from the
pilot. More steps do not guarantee a stronger illusion.

The Turbo sampler follows the author's shifted re-noising schedule. It aligns
the velocity predictions from all views before adding fresh noise to the shared
image between steps. INT8 base weights stay packed, with low-rank residuals
evaluated alongside them. FP32 linear updates, including shared modulation
cores, are merged in place. Adapter identity and settings are saved in metadata.
This INT8, multi-view adaptation has automated CPU/CUDA checks and a completed
1024-square Turbo smoke run. Illusion quality still depends on prompts and seeds.

A small [four-step flow-shift comparison](examples/flow-shift/README.md)
tested shifts 2, 4 and 6 with Joker upright and Batman after a 180° rotation.
Shifts 2 and 4 were close; 6 scored lowest for Batman. All three remained
Joker-dominated, so this check does not justify changing the default shift of 4.

The [extended shift/step-count check](examples/flow-shift-extended/README.md)
adds shifts 1, 3 and 8 at four steps, plus eight-step runs at shifts 2, 4 and 6.
Four steps / shift 8 had the strongest weaker-view score for this pair, but
none made a convincing Joker-to-Batman transformation. Eight steps roughly
doubled time without improving the mean score across matched shifts.

With the current adaptive default, a [higher-shift check](examples/high-flow-shift/README.md)
added shifts 12, 16 and 24 at four steps. None beat shift 8's weaker-view score
for this pair. At shift 24, the four-step schedule never triggers adaptive
scoring, so weights remain equal; the report records this interaction.

A [30-step Standard comparison](examples/standard-flow-shift/README.md), with
CFG 3 and Classic combining, found stronger Batman alignment at shift 8 than
4 (weaker-view score 0.255 versus 0.239). Both still place the two faces in
separate areas; this single pair does not establish a new default.

### Standard generation (default)

The CLI uses 30 steps, guidance 3, shift 8 and adaptive balancing for paired
geometric views without extra sampling flags. `--standard`
can also be supplied explicitly. Override steps or guidance when experimenting.

```powershell
.\.venv\Scripts\python.exe generate.py --name iris.village.horse --prompts "a snowy mountain village" "a horse" --style "an oil painting of" --views identity rotate_cw
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
- `--reduction auto` (default): average geometric view predictions and sum factorized components. `alternate` cycles geometric views between steps; `mean` and `sum` remain explicit options. Component methods require a complete matching set and `auto` or `sum`: low/high frequencies with the same sigma, grayscale/color, all three frequency bands, motion/residual, or scale weights that sum to 1.
- For rotations and other geometric views, use **Auto/Mean**. The pilot's **Sum** controls produced severely overexposed images.
- `--view_args`: one argument per view, following the existing view definitions.
- `--shift 8.0` and `--solver_order 2`: current anagram defaults. Pass `--shift 4`
  for the earlier/upstream schedule, or `--quality_method plain` for Classic combining.
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
