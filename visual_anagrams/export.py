"""CPU-only animation and prompt-label exports for saved visual anagrams."""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
import math
import os
from pathlib import Path
import pickle
import tempfile

from PIL import GifImagePlugin, Image, ImageDraw, ImageFont, ImageOps


class _GifStreamWriter:
    """Write bounded-memory GIF frames with correct per-frame palettes.

    Pillow emits both the local palette and its matching descriptor. This
    also handles flat midpoint frames (color inversion / edge-on flips),
    which the old imageio legacy writer could encode with a wrong table size.
    """

    def __init__(self, path, fps):
        self.stream = Path(path).open("wb")
        self.fps, self.count = fps, 0

    def __enter__(self):
        return self

    def append_data(self, pixels):
        image = Image.fromarray(pixels).convert("RGB").quantize(colors=256, method=Image.Quantize.MEDIANCUT)
        if self.count == 0:
            header, _ = GifImagePlugin.getheader(image, info={"loop": 0, "optimize": False})
            for data in header:
                self.stream.write(data)
        duration = (round((self.count + 1) * 100 / self.fps) - round(self.count * 100 / self.fps)) * 10
        for data in GifImagePlugin.getdata(image, duration=duration, disposal=2, include_color_table=True):
            self.stream.write(data)
        self.count += 1

    def __exit__(self, *_exception):
        try:
            if self.count:
                self.stream.write(b";")
        finally:
            self.stream.close()


@dataclass
class AnimationOptions:
    fps: int = 24
    size: int = 768
    transition: float = 1.5
    hold: float = 1.0
    boomerang: bool = True
    labels: bool = False

    def validate(self):
        if not 1 <= self.fps <= 60:
            raise ValueError("Animation FPS must be between 1 and 60.")
        if not 128 <= self.size <= 2048 or self.size % 16:
            raise ValueError("Animation size must be a multiple of 16 between 128 and 2048.")
        if not math.isfinite(self.transition) or not 0.1 <= self.transition <= 10:
            raise ValueError("Transition time must be between 0.1 and 10 seconds.")
        if not math.isfinite(self.hold) or not 0 <= self.hold <= 10:
            raise ValueError("Hold time must be between 0 and 10 seconds.")
        return self


def add_export_args(parser):
    parser.add_argument("--animation_formats", choices=["gif", "mp4"], nargs="+", default=[])
    parser.add_argument("--animation_fps", type=int, default=24)
    parser.add_argument("--animation_size", type=int, default=768, help="GIF/video canvas width and height")
    parser.add_argument("--animation_transition", type=float, default=1.5, help="Seconds per transition")
    parser.add_argument("--animation_hold", type=float, default=1.0, help="Seconds to pause at each view")
    parser.add_argument("--animation_one_way", action="store_true", help="Do not animate back to the original view")
    parser.add_argument("--label_outputs", action="store_true", help="Overlay prompt labels on animations and save labeled PNG copies")
    return parser


def options_from_args(args):
    return AnimationOptions(args.animation_fps, args.animation_size, args.animation_transition,
                            args.animation_hold, not args.animation_one_way, args.label_outputs).validate()


def animation_targets(view_names):
    targets = [index for index, name in enumerate(view_names) if name != "identity"]
    if not targets:
        raise ValueError("Animation needs at least one transformed view, such as Rotate 90 degrees.")
    return targets


def _font(size):
    path = Path(__file__).parent / "assets" / "CourierPrime-Regular.ttf"
    return ImageFont.truetype(str(path), size) if path.is_file() else ImageFont.load_default(size=size)


def prompt_overlay(image, text, *, opacity=1.0, bottom=None, max_width=None, font_size=None):
    """Readable, wrapped text over the image, without changing its dimensions."""
    if not text.strip() or opacity <= 0:
        return image.copy()
    image = image.convert("RGBA")
    width, height = image.size
    font = _font(font_size or max(12, round(width * 0.033)))
    max_width = max_width or round(width * 0.88)
    words = text.split()
    lines, current = [], ""
    for word in words:
        candidate = (current + " " + word).strip()
        if current and font.getlength(candidate) > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > 3:
        lines = lines[:3]
        lines[-1] += "…"
    for index, line in enumerate(lines):
        while font.getlength(line) > max_width and len(line) > 2:
            line = line[:-2].rstrip("…") + "…"
        lines[index] = line
    text = "\n".join(lines)
    overlay = Image.new("RGBA", image.size)
    draw = ImageDraw.Draw(overlay)
    spacing = max(3, font.size // 5)
    box = draw.multiline_textbbox((0, 0), text, font=font, spacing=spacing, align="center")
    tw, th = box[2] - box[0], box[3] - box[1]
    pad = max(6, round(width * 0.014))
    bottom = bottom if bottom is not None else height - pad
    left, top = (width - tw) / 2, bottom - th - 2 * pad
    alpha = max(0, min(1, opacity))
    draw.rounded_rectangle((left - pad, top, left + tw + pad, bottom), radius=pad,
                           fill=(12, 18, 28, round(185 * alpha)))
    draw.multiline_text((left - box[0], top + pad - box[1]), text, font=font,
                         fill=(255, 255, 255, round(255 * alpha)), spacing=spacing, align="center")
    return Image.alpha_composite(image, overlay).convert("RGB")


def label_saved_images(image_path, view_names, prompts):
    """Create labeled copies, retaining clean images as animation sources."""
    image_path = Path(image_path)
    identity = view_names.index("identity") if "identity" in view_names else None
    with Image.open(image_path) as image:
        label = prompts[identity] if identity is not None else "Original image"
        result = prompt_overlay(image, label)
        labeled = image_path.with_name(image_path.stem + ".labeled.png")
        result.save(labeled)
    sheet_path = image_path.with_name(image_path.stem + ".views.png")
    paths = [labeled]
    if sheet_path.is_file():
        with Image.open(sheet_path) as sheet:
            sheet = sheet.convert("RGB")
            side = sheet.height
            for index, prompt in enumerate(prompts):
                tile = sheet.crop((index * side, 0, (index + 1) * side, side))
                sheet.paste(prompt_overlay(tile, prompt), (index * side, 0))
            target = image_path.with_name(image_path.stem + ".views.labeled.png")
            sheet.save(target)
            paths.append(target)
    return paths


def animation_timeline(options):
    """Normalized view positions for holds, eased movement, and the return."""
    options.validate()
    hold = round(options.hold * options.fps)
    transition = max(2, round(options.transition * options.fps))
    smooth = lambda t: t * t * (3 - 2 * t)
    forward = [smooth(index / (transition - 1)) for index in range(transition)]
    timeline = [0.0] * hold + forward + [1.0] * hold
    if options.boomerang:
        timeline += list(reversed(forward))
    return timeline


def _art_frame(image, view_name, view, t, side):
    """Render actual rotation/flip geometry, without clipping the corners."""
    if view_name in ("rotate_cw", "rotate_ccw", "rotate_180", "flip", "negate", "identity"):
        source = image.resize((side, side), Image.Resampling.LANCZOS)
        if view_name.startswith("rotate"):
            angle = {"rotate_cw": -90, "rotate_ccw": 90, "rotate_180": 180}[view_name]
            return source.convert("RGBA").rotate(angle * t, resample=Image.Resampling.BICUBIC, expand=True)
        if view_name == "flip":
            # A vertical flip turns around the horizontal axis; it is not a
            # 180-degree in-plane rotation. Show the edge at the halfway point.
            if t > 0.5:
                source = ImageOps.flip(source)
            height = max(1, round(side * abs(math.cos(math.pi * t))))
            return source.resize((side, height), Image.Resampling.LANCZOS).convert("RGBA")
        if view_name == "negate":
            return Image.blend(source, ImageOps.invert(source), t).convert("RGBA")
        return source.convert("RGBA")
    try:
        # Existing custom views retain their saved random permutations and
        # their own animation implementations (jigsaw, patch shuffle, etc.).
        frame = view.make_frame(image, t).convert("RGBA")
        frame.thumbnail((round(side * 1.5), round(side * 1.5)), Image.Resampling.LANCZOS)
        return frame
    except NotImplementedError:
        return image.resize((side, side), Image.Resampling.LANCZOS).convert("RGBA")


def render_frame(image, view_name, view, t, options, prompt_1="", prompt_2=""):
    side = round(options.size * 0.625)
    art = _art_frame(image, view_name, view, t, side)
    frame = Image.new("RGBA", (options.size, options.size), (245, 246, 249, 255))
    frame.alpha_composite(art, ((options.size - art.width) // 2, (options.size - art.height) // 2))
    frame = frame.convert("RGB")
    if options.labels:
        label = prompt_1 if t < 0.5 else prompt_2
        opacity = max(0.0, 1 - t * 5) if t < 0.5 else max(0.0, (t - 0.8) * 5)
        frame = prompt_overlay(frame, label, opacity=opacity,
                               bottom=(options.size + side) // 2 - 8,
                               max_width=round(side * 0.9), font_size=max(12, round(options.size * 0.03)))
    return frame


def _available_paths(prefix, formats):
    paths = [Path(str(prefix) + "." + fmt) for fmt in formats]
    number = 2
    while any(path.exists() for path in paths):
        paths = [Path(str(prefix) + f"_{number}." + fmt) for fmt in formats]
        number += 1
    return paths


def export_animations(image_path, views, view_names, prompts, formats, options=None):
    """Stream frames to GIF/MP4 encoders and publish only completed files."""
    import imageio.v2 as imageio
    import numpy as np
    from tqdm.auto import tqdm

    options = (options or AnimationOptions()).validate()
    formats = list(dict.fromkeys(formats))
    if not formats or any(fmt not in ("gif", "mp4") for fmt in formats):
        raise ValueError("Choose GIF, MP4, or both for animation export.")
    if len(views) != len(view_names) or len(prompts) != len(views):
        raise ValueError("Animation metadata must have one prompt and view name per saved view.")
    targets = animation_targets(view_names)
    image_path = Path(image_path)
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    timeline = animation_timeline(options)
    identity = view_names.index("identity") if "identity" in view_names else None
    original_prompt = prompts[identity] if identity is not None else "Original image"
    output = []
    for index in targets:
        suffix = "" if len(targets) == 1 else f".view_{index + 1}_{view_names[index]}"
        # Keep dotted run/sample names intact when appending extensions.
        prefix = image_path.with_name(image_path.stem + suffix)
        paths = _available_paths(prefix, formats)
        temporary = []
        try:
            with ExitStack() as stack:
                writers = []
                for path in paths:
                    handle = tempfile.NamedTemporaryFile(prefix="." + path.stem + ".", suffix=path.suffix,
                                                         dir=path.parent, delete=False)
                    handle.close()
                    temp = Path(handle.name)
                    temporary.append(temp)
                    if path.suffix == ".gif":
                        writer = _GifStreamWriter(temp, options.fps)
                    else:
                        writer = imageio.get_writer(temp, format="FFMPEG", mode="I", fps=options.fps,
                                                     codec="libx264", quality=8, pixelformat="yuv420p",
                                                     macro_block_size=16, output_params=["-movflags", "+faststart"],
                                                     ffmpeg_log_level="error")
                    writers.append(stack.enter_context(writer))
                print(f"Exporting animation: view {index + 1} ({view_names[index]})", flush=True)
                for t in tqdm(timeline, desc="Animation export"):
                    frame = np.asarray(render_frame(image, view_names[index], views[index], t, options,
                                                     original_prompt, prompts[index]))
                    for writer in writers:
                        writer.append_data(frame)
            for temp, path in zip(temporary, paths):
                os.replace(temp, path)
                output.append(path)
                print(f"Saved animation {path}", flush=True)
        finally:
            for temp in temporary:
                try:
                    temp.unlink(missing_ok=True)
                except PermissionError:
                    # An encoder may still be releasing its Windows handle
                    # after an error. Keep the original encoding exception.
                    pass
    return output


def export_saved(image_path, metadata_path, formats, options=None):
    options = (options or AnimationOptions()).validate()
    with Path(metadata_path).open("rb") as stream:
        metadata = pickle.load(stream)
    args, views = metadata["args"], metadata["views"]
    if options.labels:
        for path in label_saved_images(image_path, args.views, args.prompts):
            print(f"Saved labels {path}", flush=True)
    return export_animations(image_path, views, args.views, args.prompts, formats, options)
