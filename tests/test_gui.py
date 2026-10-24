"""UI integration tests use a tiny subprocess; no model downloads or inference."""

import queue
import sys
import time

import pytest

from generate import build_parser
from visual_anagrams.gui import AnagramApp, GenerationJob, build_command


@pytest.fixture
def settings(tmp_path):
    return dict(name="test illusion", output=str(tmp_path), style="an oil painting of",
                negative="blurry", steps="10", samples="2", seed="42", guidance="3.0",
                shift="4", order="2", reduction="mean", device="cuda", text_device="cpu")


def test_form_builds_cli_arguments_without_shell_parsing(settings, tmp_path):
    rows = [{"prompt": 'a village with "snow" & trees', "view": "identity"},
            {"prompt": "a horse", "view": "patch_permute", "argument": "16"}]
    command, destination = build_command(settings, rows)
    args = build_parser().parse_args(command[3:])
    assert args.prompts == [row["prompt"] for row in rows]
    assert args.views == ["identity", "patch_permute"]
    assert args.view_args == ["0", "16"]
    assert args.text_encoder_device == "cpu"
    assert destination == tmp_path / "test illusion"
    destination.mkdir()
    _, new_destination = build_command(settings, rows)
    assert new_destination.name == "test illusion_2"


@pytest.mark.parametrize("field,value", [
    ("name", "../outside"), ("steps", "0"), ("samples", "1.5"),
    ("guidance", "nan"), ("shift", "0"), ("seed", "-1"),
])
def test_invalid_settings_are_rejected(settings, field, value):
    settings[field] = value
    with pytest.raises(ValueError):
        build_command(settings, [{"prompt": "a horse", "view": "identity"}])


def test_invalid_view_parameter_is_rejected(settings):
    with pytest.raises(ValueError, match="Invalid parameter"):
        build_command(settings, [{"prompt": "a horse", "view": "patch_permute", "argument": "7"}])


def test_turbo_choice_is_forwarded_to_cli(settings):
    settings.update(mode="Turbo · r1-1500", steps="4", guidance="1", negative="")
    command, _ = build_command(settings, [{"prompt": "a horse", "view": "identity"}])
    args = build_parser().parse_args(command[3:])
    assert args.turbo and args.turbo_adapter == "r1-1500"
    assert args.num_inference_steps == 4 and args.guidance_scale == 1


def test_turbo_toggle_restores_standard_settings():
    app = AnagramApp()
    app.withdraw()
    try:
        app.form["mode"].set("Standard Iris")
        app.mode_changed()
        app.form["steps"].set("40")
        app.form["negative"].set("blurry")
        app.form["mode"].set("Turbo · r2-1000 (recommended)")
        app.mode_changed()
        assert app.form["steps"].get() == "4"
        assert app.form["negative"].get() == ""
        assert app.form["guidance"].get() == "1.0"
        assert app.guidance_entry.instate(["disabled"])
        assert app.negative_entry.instate(["disabled"])
        assert app.order_combo.instate(["disabled"])
        app.form["mode"].set("Standard Iris")
        app.mode_changed()
        assert app.form["steps"].get() == "40"
        assert app.form["negative"].get() == "blurry"
        assert app.form["guidance"].get() == "3.0"
        assert app.order_combo.instate(["readonly", "!disabled"])
    finally:
        app.destroy()


def test_turbo_r2_defaults_and_explicit_standard_command(tmp_path):
    app = AnagramApp()
    app.withdraw()
    try:
        assert app.form["mode"].get() == "Turbo · r2-1000 (recommended)"
        assert app.form["steps"].get() == "4" and app.form["guidance"].get() == "1.0"
        assert app.guidance_entry.instate(["disabled"])
        app.form["output"].set(str(tmp_path))
        command, _ = build_command({key: value.get() for key, value in app.form.items()}, [row.values() for row in app.rows])
        args = build_parser().parse_args(command[3:])
        assert args.turbo and args.turbo_adapter == "r2-1000"
        app.form["mode"].set("Standard Iris")
        app.mode_changed()
        command, _ = build_command({key: value.get() for key, value in app.form.items()}, [row.values() for row in app.rows])
        args = build_parser().parse_args(command[3:])
        assert not args.turbo and args.guidance_scale == 3.0 and args.num_inference_steps == 100
    finally:
        app.destroy()


def wait_for_exit(job, timeout=10):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            event = job.events.get(timeout=0.1)
        except queue.Empty:
            continue
        events.append(event)
        if event[0] == "exit":
            return events
    job.cancel()
    pytest.fail("Worker did not exit")


def test_worker_streams_unicode_progress_and_failure():
    job = GenerationJob()
    job.start([sys.executable, "-u", "-c",
               "import sys; print('snow: 雪', flush=True); print('50%| 1/2', end='\\r', flush=True); sys.exit(3)"])
    events = wait_for_exit(job)
    text = "".join(value for kind, value in events if kind == "log")
    assert "snow: 雪" in text
    assert "1/2\r" in text
    assert events[-1] == ("exit", 3)


def test_cancel_terminates_worker():
    job = GenerationJob()
    job.start([sys.executable, "-u", "-c", "import time; print('started', flush=True); time.sleep(30)"])
    kind, value = job.events.get(timeout=10)
    assert kind == "log" and "started" in value
    job.cancel()
    events = wait_for_exit(job)
    assert job.cancelled
    assert events[-1][1] != 0
    assert job.process.poll() is not None


def test_desktop_runs_worker_and_loads_saved_previews(tmp_path, monkeypatch):
    app = AnagramApp()
    app.withdraw()
    try:
        app.form["output"].set(str(tmp_path))
        destination = tmp_path / "iris-illusion"
        script = tmp_path / "fake_generate.py"
        script.write_text(
            "from pathlib import Path\n"
            "from PIL import Image\n"
            f"folder = Path({str(destination)!r}) / '0000'\n"
            "folder.mkdir(parents=True)\n"
            "print('Generating sample 1/1, seed 0', flush=True)\n"
            "print('Iris multi-view denoising: 100%|##| 2/2', flush=True)\n"
            "Image.new('RGB', (1024,1024), 'blue').save(folder / 'sample_1024.png')\n"
            "Image.new('RGB', (2048,1024), 'green').save(folder / 'sample_1024.views.png')\n"
            "print('Saved image', flush=True)\n", encoding="utf-8",
        )
        original = app.job.start
        monkeypatch.setattr(app.job, "start", lambda _: original([sys.executable, "-u", str(script)]))
        app.generate()
        deadline = time.monotonic() + 10
        while app.running and time.monotonic() < deadline:
            app.update()
            time.sleep(0.02)
        assert not app.running
        assert app.status.get() == "Finished · 1 image(s) saved"
        assert app.preview_images[0].size == (1024, 1024)
        assert app.preview_images[1].size == (2048, 1024)
        assert len(app.images) == 1
        assert app.generate_button.instate(["!disabled"])
    finally:
        app.job.cancel()
        app.destroy()
