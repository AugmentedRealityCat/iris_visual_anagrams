"""Real GIF/MP4 writing, transform endpoints, labels and export UI checks."""

import pickle
import sys
import time
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageOps
import pytest

from generate import build_parser, main
from visual_anagrams.export import (
    AnimationOptions, animation_timeline, export_animations, label_saved_images, render_frame,
)
from visual_anagrams.gui import AnagramApp, build_command, build_export_command


def colored_image(size=80):
    pixels = np.zeros((size, size, 3), dtype=np.uint8)
    half = size // 2
    pixels[:half, :half] = (230, 25, 25)
    pixels[:half, half:] = (25, 220, 25)
    pixels[half:, :half] = (25, 25, 220)
    pixels[half:, half:] = (220, 220, 25)
    return Image.fromarray(pixels)


@pytest.mark.parametrize("view,operation", [
    ("rotate_cw", Image.Transpose.ROTATE_270), ("rotate_ccw", Image.Transpose.ROTATE_90),
    ("rotate_180", Image.Transpose.ROTATE_180), ("flip", Image.Transpose.FLIP_TOP_BOTTOM),
])
def test_animation_endpoint_matches_actual_transform(view, operation):
    image = colored_image()
    frame = render_frame(image, view, None, 1, AnimationOptions(size=128))
    actual = np.asarray(frame.crop((24, 24, 104, 104)))
    np.testing.assert_array_equal(actual, np.asarray(image.transpose(operation)))


@pytest.mark.parametrize("view,operation", [
    ("rotate_cw", Image.Transpose.ROTATE_270), ("rotate_ccw", Image.Transpose.ROTATE_90),
    ("rotate_180", Image.Transpose.ROTATE_180), ("flip", Image.Transpose.FLIP_TOP_BOTTOM),
])
def test_borderless_endpoints_fill_canvas_with_complete_transformed_image(view, operation):
    image = colored_image(128)
    options = AnimationOptions(size=128, borderless=True)
    first = render_frame(image, view, None, 0, options)
    last = render_frame(image, view, None, 1, options)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(image))
    np.testing.assert_array_equal(np.asarray(last), np.asarray(image.transpose(operation)))


@pytest.mark.parametrize("view", ["rotate_cw", "rotate_ccw", "rotate_180"])
def test_borderless_rotations_cover_corners_throughout_movement(view):
    from visual_anagrams.export import _art_frame

    image = colored_image(128)
    for t in np.linspace(0, 1, 21):
        art = _art_frame(image, view, None, float(t), 128, borderless=True)
        assert art.size == (128, 128)
        assert np.asarray(art.getchannel("A")).min() == 255


def test_borderless_custom_view_crops_canvas_without_detecting_white_artwork():
    class PaddedView:
        def make_frame(self, image, t):
            frame = Image.new("RGB", (192, 192), "white")
            frame.paste(image, (32, 32))
            return frame

    image = colored_image(128)
    # White content inside the image must be retained, not auto-trimmed.
    image.paste("white", (0, 0, 40, 128))
    actual = render_frame(image, "custom", PaddedView(), 0, AnimationOptions(size=128, borderless=True))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(image))


def test_borderless_flip_keeps_image_backed_background_at_edge_on_midpoint():
    frame = render_frame(colored_image(128), "flip", None, 0.5, AnimationOptions(size=128, borderless=True))
    assert frame.size == (128, 128)
    assert np.asarray(frame).max() < 240
    assert np.asarray(frame).std() > 20


def test_negation_endpoint_and_loop_timing():
    image = colored_image()
    options = AnimationOptions(fps=10, size=128, transition=0.4, hold=0.2)
    actual = render_frame(image, "negate", None, 1, options).crop((24, 24, 104, 104))
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(ImageOps.invert(image)))
    timeline = animation_timeline(options)
    assert len(timeline) == 12
    assert timeline[:2] == [0, 0]
    assert timeline[6:8] == [1, 1]
    assert timeline[-1] == 0


def test_labels_create_copies_without_modifying_clean_pngs(tmp_path):
    image = colored_image(128)
    path = tmp_path / "sample_1024.png"
    sheet_path = tmp_path / "sample_1024.views.png"
    image.save(path)
    sheet = Image.new("RGB", (256, 128))
    sheet.paste(image)
    sheet.paste(image.transpose(Image.Transpose.ROTATE_270), (128, 0))
    sheet.save(sheet_path)
    before = [path.read_bytes(), sheet_path.read_bytes()]
    outputs = label_saved_images(path, ["identity", "rotate_cw"], ["a village", "a horse"])
    assert len(outputs) == 2
    assert path.read_bytes() == before[0] and sheet_path.read_bytes() == before[1]
    with Image.open(outputs[0]) as labeled:
        assert labeled.size == image.size
        assert not np.array_equal(np.asarray(labeled), np.asarray(image))
        np.testing.assert_array_equal(np.asarray(labeled)[:20], np.asarray(image)[:20])


def test_animation_labels_fade_during_movement():
    image = colored_image()
    plain = AnimationOptions(size=128, labels=False)
    labeled = AnimationOptions(size=128, labels=True)
    midpoint = render_frame(image, "rotate_cw", None, 0.5, plain)
    midpoint_labeled = render_frame(image, "rotate_cw", None, 0.5, labeled, "a village", "a horse")
    np.testing.assert_array_equal(np.asarray(midpoint), np.asarray(midpoint_labeled))
    start = render_frame(image, "rotate_cw", None, 0, labeled, "a village", "a horse")
    assert not np.array_equal(np.asarray(start), np.asarray(render_frame(image, "rotate_cw", None, 0, plain)))


@pytest.mark.parametrize("borderless", [False, True])
def test_real_gif_and_mp4_have_frames_correct_timing_and_unique_names(tmp_path, borderless):
    source = tmp_path / "sample.with.dots.png"
    colored_image().save(source)
    options = AnimationOptions(fps=10, size=128, transition=0.2, hold=0.1, labels=True, borderless=borderless)
    kwargs = dict(views=[None, None], view_names=["identity", "rotate_cw"],
                  prompts=["a village", "a horse"], formats=["gif", "mp4"], options=options)
    paths = export_animations(source, **kwargs)
    assert [path.name for path in paths] == ["sample.with.dots.gif", "sample.with.dots.mp4"]
    frames = len(animation_timeline(options))
    with Image.open(paths[0]) as gif:
        assert gif.is_animated and gif.n_frames == frames
        assert gif.info["loop"] == 0
        expected_corner = (230, 25, 25) if borderless else (245, 246, 249)
        np.testing.assert_allclose(np.asarray(gif.convert("RGB"))[:8, :8].mean(axis=(0, 1)),
                                   expected_corner, atol=5)
        duration = 0
        for index in range(gif.n_frames):
            gif.seek(index)
            duration += gif.info["duration"]
        assert duration == round(frames / options.fps * 1000)
    with imageio.get_reader(paths[1], format="FFMPEG") as movie:
        assert movie.count_frames() == frames
        assert movie.get_meta_data()["fps"] == options.fps
        assert movie.get_meta_data()["size"] == (128, 128)
        np.testing.assert_allclose(movie.get_data(0)[:8, :8].mean(axis=(0, 1)),
                                   expected_corner, atol=12)
        assert not np.array_equal(movie.get_data(0), movie.get_data(3))
    original_bytes = [path.read_bytes() for path in paths]
    again = export_animations(source, **kwargs)
    assert all("_2." in path.name for path in again)
    assert original_bytes == [path.read_bytes() for path in paths]


def test_multi_view_exports_each_transformation(tmp_path):
    source = tmp_path / "sample.png"
    colored_image().save(source)
    paths = export_animations(source, [None] * 3, ["identity", "rotate_cw", "negate"],
                               ["village", "horse", "flowers"], ["gif"],
                               AnimationOptions(fps=5, size=128, transition=0.2, hold=0))
    assert [path.name for path in paths] == ["sample.view_2_rotate_cw.gif", "sample.view_3_negate.gif"]


@pytest.mark.parametrize("view", ["flip", "negate"])
def test_gif_survives_flat_midpoint_frames(view, tmp_path):
    source = tmp_path / "sample.png"
    colored_image().save(source)
    options = AnimationOptions(fps=10, size=128, transition=0.5, hold=0.1)
    outputs = export_animations(source, [None, None], ["identity", view], ["first", "second"], ["gif"], options)
    expected_frames = len(animation_timeline(options))
    with Image.open(outputs[0]) as gif:
        assert gif.n_frames == expected_frames
        assert gif.info["loop"] == 0
        for index in range(expected_frames):
            gif.seek(index)
            gif.load()
        assert np.asarray(gif.convert("RGB")).std() > 10


@pytest.mark.parametrize("flag,value", [("--animation_fps", "0"), ("--animation_size", "129"),
                                       ("--animation_transition", "nan")])
def test_invalid_animation_options_fail_before_model_download(flag, value):
    with pytest.raises(SystemExit) as exc:
        main(["--name", "test", "--prompts", "village", "--views", "identity", flag, value])
    assert exc.value.code == 2


def test_gui_forwards_animation_and_label_options(tmp_path):
    settings = dict(name="test", output=str(tmp_path), style="", negative="", steps="4", samples="1", seed="0",
                    guidance="1", shift="4", order="2", reduction="mean", device="cuda", text_device="cpu",
                    animation_auto="yes", animation_format="GIF + MP4", labels="yes", animation_loop="no",
                    animation_borderless="yes")
    command, _ = build_command(settings, [{"prompt": "village", "view": "identity"},
                                         {"prompt": "horse", "view": "rotate_cw"}])
    args = build_parser().parse_args(command[3:])
    assert args.animation_formats == ["gif", "mp4"] and args.label_outputs
    assert args.animation_one_way and args.animation_fps == 24
    assert args.animation_borderless


def test_desktop_exports_existing_image_without_loading_models(tmp_path):
    sample = tmp_path / "run" / "0000"
    sample.mkdir(parents=True)
    source = sample / "sample_1024.png"
    colored_image().save(source)
    metadata = dict(args=SimpleNamespace(views=["identity", "rotate_cw"], prompts=["village", "a horse"]), views=[None, None])
    (sample.parent / "metadata.pkl").write_bytes(pickle.dumps(metadata))
    app = AnagramApp()
    app.withdraw()
    try:
        app.images = [source]
        app.sample_picker.configure(values=["Seed 0"])
        app.sample_picker.current(0)
        for key, value in dict(animation_format="GIF + MP4", animation_fps="5", animation_size="128",
                               animation_transition="0.2", animation_hold="0", labels="yes",
                               animation_borderless="yes").items():
            app.form[key].set(value)
        command = build_export_command({key: value.get() for key, value in app.form.items()}, source)
        assert "export_animation.py" in command[2] and "generate.py" not in command[2]
        assert "--animation_borderless" in command
        app.export_animation()
        deadline = time.monotonic() + 15
        while app.running and time.monotonic() < deadline:
            app.update()
            time.sleep(0.02)
        assert not app.running
        assert "export finished" in app.status.get()
        assert source.with_suffix(".gif").is_file() and source.with_suffix(".mp4").is_file()
        assert source.with_name("sample_1024.labeled.png").is_file()
        assert app.play_button.instate(["!disabled"])
    finally:
        app.job.cancel()
        app.destroy()
