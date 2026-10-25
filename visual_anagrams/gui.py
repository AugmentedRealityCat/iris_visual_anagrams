"""Tk desktop UI; generation runs in a cancellable, isolated subprocess.

No model imports or downloads happen when the window opens. Tk widgets are
only touched on the main thread; the worker forwards subprocess output.
"""

from __future__ import annotations

import codecs
import math
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk
from visual_anagrams.method_options import QUALITY_METHODS, MethodOptions, default_quality_method
from visual_anagrams.experiment_options import GUI_DEFAULTS, NEGATIVE_MODES, options_from_gui, context_options, gui_values
from visual_anagrams.styles import SHARED_STYLES
from visual_anagrams.theme import DARK, LIGHT, load_preferences, preferences_path, save_theme, style_log_scrollbar

ROOT = Path(__file__).resolve().parents[1]
VIEWS = {
    "Original": "identity", "Rotate 90° right": "rotate_cw",
    "Rotate 90° left": "rotate_ccw", "Rotate 180°": "rotate_180",
    "Flip vertically": "flip", "Invert colors": "negate",
    "Inner circle": "inner_circle", "Jigsaw": "jigsaw",
    "Shuffle patches": "patch_permute", "Shuffle pixel blocks": "pixel_permute",
    "Square hinge": "square_hinge", "Skew": "skew",
    "Low frequencies": "low_pass", "High frequencies": "high_pass",
    "Grayscale component": "grayscale", "Color component": "color", "Scale component": "scale",
}
VIEW_DEFAULTS = {"patch_permute": "8", "pixel_permute": "64", "skew": "1.5",
                 "low_pass": "2.0", "high_pass": "2.0", "scale": "0.5"}
VIEW_HINTS = {"patch_permute": "Patches per side: 1, 2, 4, 8, 16, 32, or 64.",
              "pixel_permute": "Blocks per side: 1, 2, 4, 8, 16, 32, or 64.",
              "skew": "Parameter controls the skew strength.",
              "low_pass": "Pair with High frequencies using the same sigma. Auto sums components.",
              "high_pass": "Pair with Low frequencies using the same sigma. Auto sums components.",
              "scale": "Component weights must sum to 1. Auto sums their contributions.",
              "grayscale": "Pair with Color component. Auto sums components.",
              "color": "Pair with Grayscale component. Auto sums components."}
BG, PANEL, INK, MUTED, ACCENT = "#f3f5f8", "#ffffff", "#202b3c", "#637086", "#5146d9"
GENERATION_MODES = {"Standard Iris": None, "Turbo · r2-1000 (fast)": "r2-1000", "Turbo · r1-1500": "r1-1500"}
ANIMATION_FORMATS = {"MP4 video": ["mp4"], "Animated GIF": ["gif"], "GIF + MP4": ["gif", "mp4"]}


def animation_arguments(settings):
    from visual_anagrams.export import AnimationOptions

    try:
        options = AnimationOptions(
            fps=int(settings.get("animation_fps", "24")), size=int(settings.get("animation_size", "768")),
            transition=float(settings.get("animation_transition", "1.5")), hold=float(settings.get("animation_hold", "1.0")),
            boomerang=settings.get("animation_loop", "yes") == "yes", labels=settings.get("labels", "no") == "yes",
            borderless=settings.get("animation_borderless", "no") == "yes",
        ).validate()
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid animation settings: {exc}") from None
    choice = settings.get("animation_format", "MP4 video")
    if choice not in ANIMATION_FORMATS:
        raise ValueError("Choose MP4, GIF, or both for animation export.")
    arguments = ["--animation_formats", *ANIMATION_FORMATS[choice], "--animation_fps", str(options.fps),
                 "--animation_size", str(options.size), "--animation_transition", str(options.transition),
                 "--animation_hold", str(options.hold)]
    if not options.boomerang:
        arguments.append("--animation_one_way")
    if options.borderless:
        arguments.append("--animation_borderless")
    if options.labels:
        arguments.append("--label_outputs")
    return arguments


def build_export_command(settings, image_path, *, python=None, root=ROOT):
    image = Path(image_path).resolve()
    metadata = image.parent.parent / "metadata.pkl"
    if not image.is_file() or not metadata.is_file():
        raise ValueError("Choose a saved result with its run's metadata.pkl file so the original transformations can be animated.")
    return [python or sys.executable, "-u", str(Path(root) / "export_animation.py"),
            "--im_path", str(image), "--metadata_path", str(metadata), *animation_arguments(settings)]


def build_command(settings, rows, *, python=None, root=ROOT):
    """Validate form values and build an argument list, never a shell string."""
    name = settings["name"].strip()
    if not name or name in (".", "..") or any(char in name for char in '<>:"/\\|?*'):
        raise ValueError("Give the run a name without folder separators or special filename characters.")
    if not rows or any(not row["prompt"].strip() for row in rows):
        raise ValueError("Write a prompt for every view.")
    integers = {}
    for key, label, minimum in (("steps", "Steps", 1), ("samples", "Samples", 1), ("seed", "Seed", 0)):
        try:
            integers[key] = int(str(settings[key]))
        except ValueError:
            raise ValueError(f"{label} must be a whole number.") from None
        if integers[key] < minimum:
            raise ValueError(f"{label} must be at least {minimum}.")
    if integers["seed"] + integers["samples"] > 2**63:
        raise ValueError("Sample seeds must be smaller than 2**63.")
    numbers = {}
    for key in ("guidance", "shift"):
        try:
            numbers[key] = float(settings[key])
        except ValueError:
            raise ValueError(f"{key.capitalize()} must be a number.") from None
        if not math.isfinite(numbers[key]) or (key == "shift" and numbers[key] <= 0):
            raise ValueError(f"{key.capitalize()} must be finite" + (" and positive." if key == "shift" else "."))
    if settings["reduction"] not in ("auto", "mean", "sum", "alternate"):
        raise ValueError("Choose Auto, Mean, Sum, or Alternate for combining views.")
    if settings["device"] not in ("cuda", "cpu") or settings["text_device"] not in ("cuda", "cpu"):
        raise ValueError("Choose a valid generation and text encoding device.")
    if str(settings["order"]) not in ("1", "2"):
        raise ValueError("Solver order must be 1 or 2.")
    if not settings["output"].strip():
        raise ValueError("Choose an output folder.")
    mode = settings.get("mode", "Standard Iris")
    if mode not in GENERATION_MODES:
        raise ValueError("Choose a valid generation mode.")
    adapter = GENERATION_MODES[mode]
    if adapter and (numbers["guidance"] != 1.0 or settings["negative"].strip()):
        raise ValueError("Turbo has built-in guidance. Set Guidance to 1 and leave the negative prompt empty.")

    output = Path(settings["output"]).expanduser().resolve()
    destination = output / name
    suffix = 2
    while destination.exists():
        destination = output / f"{name}_{suffix}"
        suffix += 1

    view_names, view_args = [], []
    for row in rows:
        view = row["view"]
        if view not in VIEWS.values():
            raise ValueError(f"Unknown transformation: {view}")
        arg = row.get("argument", "").strip() or VIEW_DEFAULTS.get(view, "0")
        if view in VIEW_DEFAULTS:
            try:
                numeric = float(arg)
                if not math.isfinite(numeric):
                    raise ValueError
                if view in ("patch_permute", "pixel_permute"):
                    if arg not in ("1", "2", "4", "8", "16", "32", "64"):
                        raise ValueError
                if view in ("low_pass", "high_pass") and numeric <= 0:
                    raise ValueError
            except ValueError:
                raise ValueError(f"Invalid parameter for {view}. {VIEW_HINTS[view]}") from None
        view_names.append(view)
        view_args.append(arg)

    from visual_anagrams.composition import resolve_reduction

    reduction = resolve_reduction(settings["reduction"], view_names, view_args)
    method_label = settings.get("quality_method", "Classic")
    if method_label == "Classic (default)":  # older saved form dictionaries
        method_label = "Classic"
    if method_label not in QUALITY_METHODS:
        raise ValueError("Choose a valid quality method")
    method = QUALITY_METHODS[method_label]
    method_args = ["--quality_method", "plain"]
    if method == "adaptive":
        MethodOptions(method).validate(view_names, turbo=bool(adapter), reduction=reduction, steps=integers["steps"])
        method_args = ["--quality_method", "adaptive"]
    elif method != "plain":
        try:
            strength = settings.get("method_strength", "").strip()
            options = MethodOptions(method, int(settings.get("optimization_interval", "5")),
                                    int(settings.get("optimization_steps", "1")), float(strength) if strength else None)
        except ValueError:
            raise ValueError("Optimization interval/passes must be integers and strength must be a number") from None
        options.validate(view_names, turbo=bool(adapter), reduction=reduction, steps=integers["steps"])
        method_args = ["--quality_method", method, "--optimization_interval", str(options.interval),
                       "--optimization_steps", str(options.iterations), "--method_strength", str(options.strength)]
        if method == "anagram_mtl" and settings.get("subjects", "").strip():
            subjects = [subject.strip() for subject in settings["subjects"].split("|")]
            if len(subjects) != len(rows) or any(not s for s in subjects):
                raise ValueError("Enter one subject phrase per view, separated by |")
            for row, subject in zip(rows, subjects):
                if subject.casefold() not in f"{settings['style']} {row['prompt']}".casefold():
                    raise ValueError("Each subject phrase must occur in its corresponding prompt")
            method_args += ["--subjects", *subjects]
    command = [python or sys.executable, "-u", str(Path(root) / "generate.py"),
               "--name", destination.name, "--save_dir", str(output),
               "--prompts", *[row["prompt"].strip() for row in rows],
               "--views", *view_names, "--view_args", *view_args,
               "--style", settings["style"].strip(), "--negative_prompt", settings["negative"].strip(),
               "--num_inference_steps", str(integers["steps"]), "--num_samples", str(integers["samples"]),
               "--seed", str(integers["seed"]), "--guidance_scale", str(numbers["guidance"]),
               "--shift", str(numbers["shift"]), "--solver_order", str(settings["order"]),
               "--reduction", reduction, "--device", settings["device"],
               "--text_encoder_device", settings["text_device"]]
    command.extend(method_args)
    experiments=options_from_gui(settings,context_options(turbo=bool(adapter),views=view_names,reduction=reduction,quality_method=method))
    experiments.validate(view_names,reduction=reduction,turbo=bool(adapter),quality_method=method)
    command.extend(experiments.cli_args(explicit=True))
    if experiments.cross_negative=='qwen':command.append('--negative_fallback')
    if experiments.cross_negative == 'manual':
        negatives=[p.strip() for p in settings.get('exp_manual_negatives','').split('|')]
        if len(negatives) != len(rows) or any(not p for p in negatives):
            raise ValueError('Enter one nonempty negative per view, separated by |')
        command += ['--view_negative_prompts',*negatives]
    if adapter:
        command.extend(["--turbo", "--turbo_adapter", adapter])
    else:
        command.append("--standard")
    if settings.get("animation_auto", "no") == "yes":
        from visual_anagrams.export import animation_targets

        animation_targets(view_names)
        command.extend(animation_arguments(settings))
    elif settings.get("labels", "no") == "yes":
        command.append("--label_outputs")
    for field, flag in (("model_path", "--model_path"), ("text_path", "--text_encoder_path")):
        if settings.get(field, "").strip():
            folder = Path(settings[field]).expanduser().resolve()
            if not folder.is_dir():
                raise ValueError(f"Local model folder does not exist: {folder}")
            command.extend([flag, str(folder)])
    return command, destination


def worker_python():
    # pythonw launches the GUI without a console, while python guarantees
    # stdout/stderr handles for the generation worker on Windows.
    executable = Path(sys.executable)
    return str(executable.with_name("python.exe")) if executable.name.lower() == "pythonw.exe" else str(executable)


class GenerationJob:
    """One subprocess at a time, with output events consumed by Tk."""

    def __init__(self):
        self.events = queue.Queue()
        self.process = None
        self.cancelled = False

    def start(self, command, cwd=ROOT):
        if self.process is not None and self.process.poll() is None:
            raise RuntimeError("Generation is already running.")
        self.cancelled = False
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", TOKENIZERS_PARALLELISM="false")
        self.process = subprocess.Popen(
            command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, env=env,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        threading.Thread(target=self._read, args=(self.process,), daemon=True).start()

    def _read(self, process):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            with process.stdout:
                while chunk := process.stdout.read1(4096):
                    self.events.put(("log", decoder.decode(chunk)))
            tail = decoder.decode(b"", final=True)
            if tail:
                self.events.put(("log", tail))
        finally:
            self.events.put(("exit", process.wait()))

    def cancel(self):
        if self.process is not None and self.process.poll() is None:
            self.cancelled = True
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass


class ViewRow:
    def __init__(self, parent, index, remove, prompt="", label="Original", colors=None):
        colors = colors or LIGHT
        self.frame = ttk.Frame(parent, style="Card.TFrame", padding=12)
        self.frame.columnconfigure(0, weight=1)
        self.title = ttk.Label(self.frame, text=f"VIEW {index}", style="Eyebrow.TLabel")
        self.title.grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Button(self.frame, text="Remove", command=lambda: remove(self), style="Small.TButton").grid(row=0, column=1)
        self.prompt = tk.Text(self.frame, height=2, wrap="word", font=("Segoe UI", 11),
                              bg=colors["field"], fg=colors["ink"], insertbackground=colors["ink"],
                              selectbackground=colors["selection"], selectforeground="white",
                              relief="flat", padx=10, pady=8, undo=True,
                              highlightthickness=1, highlightbackground=colors["border"], highlightcolor=colors["accent"])
        self.prompt.insert("1.0", prompt)
        self.prompt.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 9))
        options = ttk.Frame(self.frame, style="Card.TFrame")
        options.grid(row=2, column=0, columnspan=2, sticky="ew")
        options.columnconfigure(0, weight=1)
        self.view = tk.StringVar(value=label)
        combo = ttk.Combobox(options, textvariable=self.view, values=list(VIEWS), state="readonly", width=22)
        combo.grid(row=0, column=0, sticky="ew")
        combo.bind("<<ComboboxSelected>>", self._changed)
        self.argument = tk.StringVar()
        self.arg_entry = ttk.Entry(options, textvariable=self.argument, width=7)
        self.arg_entry.grid(row=0, column=1, padx=(8, 0))
        self.hint = ttk.Label(self.frame, style="Hint.TLabel", wraplength=390)
        self.hint.grid(row=3, column=0, columnspan=2, sticky="w", pady=(5, 0))
        self._changed()

    def _changed(self, _event=None):
        key = VIEWS[self.view.get()]
        self.argument.set(VIEW_DEFAULTS.get(key, ""))
        if key in VIEW_DEFAULTS:
            self.arg_entry.grid()
        else:
            self.arg_entry.grid_remove()
        self.hint.configure(text=VIEW_HINTS.get(key, ""))
        if hasattr(self, "on_change"):
            self.on_change()

    def values(self):
        return {"prompt": self.prompt.get("1.0", "end").strip(),
                "view": VIEWS[self.view.get()], "argument": self.argument.get()}


class AnagramApp(tk.Tk):
    def __init__(self, *, settings_path=None):
        if sys.platform == "win32":
            import ctypes
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except (AttributeError, OSError):
                pass
        super().__init__()
        self.settings_path = Path(settings_path) if settings_path is not None else preferences_path()
        self.dark_mode = tk.BooleanVar(value=load_preferences(self.settings_path).get("theme") == "dark")
        self.colors = DARK if self.dark_mode.get() else LIGHT
        self.title("Visual Anagrams · Iris Studio")
        self.ui_scale = max(1.0, self.winfo_fpixels("1i") / 96)
        width = min(round(1280 * self.ui_scale), self.winfo_screenwidth() - 80)
        height = min(round(900 * self.ui_scale), self.winfo_screenheight() - 120)
        self.geometry(f"{width}x{height}")
        self.minsize(min(round(1000 * self.ui_scale), width), min(round(720 * self.ui_scale), height))
        self.configure(bg=self.colors["bg"])
        self.protocol("WM_DELETE_WINDOW", self.close)
        self.job = GenerationJob()
        self.running = False
        self.task = "generation"
        self.output_dir = None
        self.images = []
        self.rows = []
        self._experiment_context=None
        self._experiment_profiles={}
        self._quality_ready = False
        self.log_buffer = ""
        self.current_sample = 1
        self.form = {key: tk.StringVar(value=value) for key, value in {
            **GUI_DEFAULTS,
            "name": "iris-illusion", "output": str(ROOT / "results"), "style": "an oil painting of",
            "negative": "", "steps": "30", "samples": "1", "guidance": "3.0", "seed": "0",
            "reduction": "auto", "shift": "8.0", "order": "2", "device": "cuda",
            "text_device": "cuda", "model_path": "", "text_path": "",
            "mode": "Standard Iris",
            "quality_method": "Adaptive balancing (experimental)", "optimization_interval": "5", "optimization_steps": "1",
            "method_strength": "0.05", "subjects": "",
            "animation_auto": "no", "animation_format": "MP4 video", "animation_fps": "24",
            "animation_size": "768", "animation_transition": "1.5", "animation_hold": "1.0",
            "animation_loop": "yes", "labels": "no", "animation_borderless": "no",
        }.items()}
        self.standard_settings = None
        self.turbo_quality_method = None
        self._styles()
        appearance = ttk.Frame(self, padding=(20, 6, 20, 2))
        appearance.pack(fill="x")
        ttk.Checkbutton(appearance, text="Dark mode", variable=self.dark_mode,
                        style="Toolbar.TCheckbutton", command=self.apply_theme).pack(side="right")
        self.pages = ttk.Notebook(self)
        self.pages.pack(fill="both", expand=True)
        self.anagram_page = ttk.Frame(self.pages)
        self.pages.add(self.anagram_page, text="Visual Anagrams")
        self._layout()
        from .tools_gui import IrisToolsPage

        self.tools_page = IrisToolsPage(self.pages, self)
        self.pages.add(self.tools_page, text="Iris Tools")
        for log in (self.log, self.tools_page.log):
            style_log_scrollbar(log)
        self.mode_changed()
        self.add_view("a snowy mountain village", "Original")
        self.add_view("a horse", "Rotate 90° right")
        self._quality_ready = True
        self._update_experiments()
        self.apply_theme(save=False)
        self.after(100, self.poll)

    def _styles(self):
        p = self.colors
        style = ttk.Style(self)
        if style.theme_use() != "clam":
            style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), foreground=p["ink"], background=p["bg"],
                        bordercolor=p["border"], lightcolor=p["border"], darkcolor=p["border"],
                        selectbackground=p["selection"], selectforeground="white", focuscolor=p["accent"])
        style.configure("TFrame", background=p["bg"])
        style.configure("Card.TFrame", background=p["panel"])
        style.configure("TLabel", background=p["panel"], padding=0)
        style.configure("Title.TLabel", background=p["bg"], font=("Segoe UI", 23, "bold"))
        style.configure("Subtitle.TLabel", background=p["bg"], foreground=p["muted"])
        style.configure("Eyebrow.TLabel", foreground=p["accent"], font=("Segoe UI", 9, "bold"))
        style.configure("Hint.TLabel", foreground=p["muted"], font=("Segoe UI", 9))
        style.configure("TButton", padding=(12, 7), background=p["button"], borderwidth=0)
        style.map("TButton", background=[("disabled", p["disabled_bg"]), ("pressed", p["pressed"]), ("active", p["hover"])],
                  foreground=[("disabled", p["disabled"]), ("!disabled", p["ink"])])
        style.configure("Accent.TButton", background=p["selection"], foreground="white", font=("Segoe UI", 11, "bold"))
        style.map("Accent.TButton", background=[("disabled", p["accent_disabled"]), ("active", p["accent_hover"])],
                  foreground=[("disabled", p["muted"]), ("!disabled", "white")])
        style.configure("Small.TButton", padding=(7, 3), font=("Segoe UI", 9))
        for name, padding in (("TEntry", 6), ("TCombobox", 5)):
            style.configure(name, fieldbackground=p["field"], foreground=p["ink"], background=p["button"],
                            insertcolor=p["ink"], arrowcolor=p["ink"], padding=padding)
            style.map(name, fieldbackground=[("disabled", p["disabled_bg"]), ("readonly", p["field"])],
                      foreground=[("disabled", p["disabled"]), ("!disabled", p["ink"])],
                      selectbackground=[("!disabled", p["selection"])], selectforeground=[("!disabled", "white")],
                      arrowcolor=[("disabled", p["disabled"]), ("!disabled", p["ink"])])
        style.configure("TCheckbutton", background=p["panel"], padding=(0, 5), indicatorbackground=p["field"],
                        indicatorforeground=p["ink"],
                        indicatorsize=round(22 * self.ui_scale),
                        indicatormargin=(0, 0, round(9 * self.ui_scale), 0))
        style.map("TCheckbutton", background=[("active", p["panel"])],
                  foreground=[("disabled", p["disabled"]), ("!disabled", p["ink"])],
                  indicatorbackground=[("selected", p["selection"]), ("!selected", p["field"])],
                  indicatorforeground=[("selected", "white"), ("!selected", p["ink"])])
        style.configure("Toolbar.TCheckbutton", background=p["bg"])
        style.map("Toolbar.TCheckbutton", background=[("active", p["bg"])])
        style.configure("TNotebook", background=p["bg"], borderwidth=0)
        style.configure("TNotebook.Tab", padding=(14, 8), background=p["button"])
        style.map("TNotebook.Tab", background=[("selected", p["panel"]), ("active", p["hover"])],
                  foreground=[("selected", p["ink"]), ("!selected", p["muted"])])
        style.configure("Vertical.TScrollbar", background=p["button"], troughcolor=p["bg"], arrowcolor=p["ink"])
        style.map("Vertical.TScrollbar", background=[("active", p["hover"]), ("pressed", p["pressed"])])
        style.configure("Horizontal.TProgressbar", background=p["selection"], troughcolor=p["trough"], borderwidth=0)
        for option, value in {"background": p["field"], "foreground": p["ink"],
                              "selectBackground": p["selection"], "selectForeground": "white"}.items():
            self.option_add(f"*TCombobox*Listbox.{option}", value)

    def apply_theme(self, *, save=True):
        self.colors = DARK if self.dark_mode.get() else LIGHT
        self._styles()
        p = self.colors
        self.configure(bg=p["bg"])
        previews = set(self.preview_canvases + self.tools_page.previews)
        logs = {self.log, self.tools_page.log}

        def recolor(widget):
            if isinstance(widget, tk.Canvas):
                widget.configure(background=p["preview"] if widget in previews else p["bg"])
            elif isinstance(widget, tk.Text):
                widget.configure(background=p["log_bg"] if widget in logs else p["field"],
                                 foreground=p["log_ink"] if widget in logs else p["ink"],
                                 insertbackground=p["ink"], selectbackground=p["selection"], selectforeground="white",
                                 highlightbackground=p["border"], highlightcolor=p["accent"])
            elif isinstance(widget, tk.Frame) and not isinstance(widget, ttk.Frame):
                widget.configure(background=p["bg"])
            elif isinstance(widget, tk.Scrollbar) and not isinstance(widget, ttk.Scrollbar):
                widget.configure(background=p["button"], activebackground=p["hover"], troughcolor=p["bg"])
            elif isinstance(widget, ttk.Combobox):
                # Update a dropdown that was already opened before switching.
                popup = str(widget) + ".popdown.f.l"
                if widget.tk.call("winfo", "exists", popup):
                    widget.tk.call(popup, "configure", "-background", p["field"], "-foreground", p["ink"],
                                   "-selectbackground", p["selection"], "-selectforeground", "white")
            for child in widget.winfo_children():
                recolor(child)

        recolor(self)
        self.render_previews()
        self.tools_page.render()
        self.after_idle(self._titlebar_theme)
        if save:
            save_theme(self.settings_path, self.dark_mode.get())

    def _titlebar_theme(self):
        if os.name == "nt":
            try:
                import ctypes
                from ctypes import wintypes

                get_parent = ctypes.windll.user32.GetParent
                get_parent.argtypes, get_parent.restype = [wintypes.HWND], wintypes.HWND
                handle = get_parent(self.winfo_id())
                setter = ctypes.windll.dwmapi.DwmSetWindowAttribute
                setter.argtypes = [wintypes.HWND, wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD]
                dark = ctypes.c_int(self.dark_mode.get())
                if setter(handle, 20, ctypes.byref(dark), ctypes.sizeof(dark)):
                    setter(handle, 19, ctypes.byref(dark), ctypes.sizeof(dark))
            except (AttributeError, OSError, tk.TclError):
                pass

    def _layout(self):
        header = ttk.Frame(self.anagram_page, padding=(24, 18))
        header.pack(fill="x")
        ttk.Label(header, text="Visual Anagrams", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="IRIS 3B INT8  /  1024 × 1024", style="Subtitle.TLabel").pack(side="right")
        ttk.Label(self.anagram_page, text="One image. Different perspectives. Describe what each view should reveal.",
                  style="Subtitle.TLabel").pack(anchor="w", padx=25, pady=(0, 16))

        body = ttk.Frame(self.anagram_page, padding=(20, 0, 20, 16))
        body.pack(fill="both", expand=True)
        panel_width = round(440 * self.ui_scale)
        body.columnconfigure(0, weight=0, minsize=panel_width)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)
        left = ttk.Frame(body, width=panel_width)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.form_canvas = tk.Canvas(left, width=panel_width, bg=BG, highlightthickness=0)
        self.form_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.form_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.form_canvas.configure(yscrollcommand=scrollbar.set)
        self.controls = ttk.Frame(self.form_canvas)
        window = self.form_canvas.create_window((0, 0), window=self.controls, anchor="nw")
        self.controls.bind("<Configure>", lambda _: self.form_canvas.configure(scrollregion=self.form_canvas.bbox("all")))
        self.form_canvas.bind("<Configure>", lambda event: self.form_canvas.itemconfigure(window, width=event.width))
        self.bind_all("<MouseWheel>", self._scroll)

        self.view_frame = ttk.Frame(self.controls)
        self.view_frame.pack(fill="x")
        ttk.Button(self.controls, text="+ Add another view", command=self.add_view).pack(fill="x", pady=(0, 12))
        settings = self._card(self.controls, "GENERATION")
        settings.columnconfigure((0, 1), weight=1)
        mode_frame = ttk.Frame(settings, style="Card.TFrame")
        mode_frame.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        mode_frame.columnconfigure(0, weight=1)
        ttk.Label(mode_frame, text="Generation mode").grid(row=0, column=0, sticky="w")
        mode_combo = ttk.Combobox(mode_frame, textvariable=self.form["mode"], values=list(GENERATION_MODES), state="readonly")
        mode_combo.grid(row=1, column=0, sticky="ew", pady=(4, 4))
        mode_combo.bind("<<ComboboxSelected>>", self.mode_changed)
        self.mode_hint = ttk.Label(mode_frame, text="Standard sampling with adjustable guidance.", style="Hint.TLabel", wraplength=round(380 * self.ui_scale))
        self.mode_hint.grid(row=2, column=0, sticky="w")
        ttk.Label(settings, text="Shared style").grid(row=2, column=0, columnspan=2, sticky="w")
        ttk.Combobox(settings, textvariable=self.form["style"], values=SHARED_STYLES,height=16).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(4, 10))
        for index, (key, label) in enumerate((("steps", "Steps"), ("guidance", "Guidance"), ("samples", "Samples"), ("seed", "Seed"))):
            row, col = 4 + (index // 2) * 2, index % 2
            ttk.Label(settings, text=label).grid(row=row, column=col, sticky="w", padx=(0, 8))
            entry = ttk.Entry(settings, textvariable=self.form[key], width=12)
            entry.grid(row=row + 1, column=col, sticky="ew", padx=(0, 8), pady=(4, 10))
            if key == "guidance":
                self.guidance_entry = entry
        ttk.Label(settings, text="Negative prompt").grid(row=8, column=0, columnspan=2, sticky="w")
        self.negative_entry = ttk.Entry(settings, textvariable=self.form["negative"])
        self.negative_entry.grid(row=9, column=0, columnspan=2, sticky="ew", pady=(4, 10))
        ttk.Label(settings, text="Combine views").grid(row=10, column=0, sticky="w")
        self.reduction_combo = ttk.Combobox(settings, textvariable=self.form["reduction"], values=["auto", "mean", "alternate", "sum"], state="readonly", width=16)
        self.reduction_combo.grid(row=10, column=1, sticky="ew")
        self.reduction_combo.bind("<<ComboboxSelected>>", self._views_changed)

        quality = self._card(self.controls, "QUALITY METHOD")
        quality.columnconfigure(0, weight=1)
        self.quality_combo = ttk.Combobox(quality, textvariable=self.form["quality_method"],
                                          values=list(QUALITY_METHODS), state="readonly")
        self.quality_combo.grid(row=1, column=0, sticky="ew")
        self.quality_combo.bind("<<ComboboxSelected>>", self.quality_changed)
        self.quality_hint = ttk.Label(quality, style="Hint.TLabel", wraplength=round(370 * self.ui_scale))
        self.quality_hint.grid(row=2, column=0, sticky="w", pady=6)
        self.method_entries = []
        self.method_labels = []
        for row, (label, key) in enumerate((("Optimize every N steps", "optimization_interval"),
                                          ("Optimization passes", "optimization_steps"),
                                          ("Optimization strength", "method_strength"),
                                          ("Subject words (MTL, optional; separate with |)", "subjects"))):
            field_label = ttk.Label(quality, text=label)
            field_label.grid(row=3 + row * 2, column=0, sticky="w", pady=(5, 0))
            entry = ttk.Entry(quality, textvariable=self.form[key])
            entry.grid(row=4 + row * 2, column=0, sticky="ew", pady=(3, 5))
            self.method_entries.append(entry)
            self.method_labels.append(field_label)

        self._experiment_panel()
        output = self._card(self.controls, "SAVE RESULTS")
        output.columnconfigure(0, weight=1)
        self._entry(output, "Run name", "name", 1)
        self._folder(output, "Output folder", "output", 3)
        animation = self._card(self.controls, "ANIMATION & LABELS")
        animation.columnconfigure((0, 1), weight=1)
        ttk.Checkbutton(animation, text="Automatically animate new images", variable=self.form["animation_auto"],
                        onvalue="yes", offvalue="no").grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 8))
        ttk.Label(animation, text="Format").grid(row=2, column=0, sticky="w")
        ttk.Combobox(animation, textvariable=self.form["animation_format"], values=list(ANIMATION_FORMATS), state="readonly",
                     width=16).grid(row=2, column=1, sticky="ew", pady=4)
        for index, (key, label) in enumerate((("animation_size", "Animation size (px)"), ("animation_fps", "Frames per second"),
                                               ("animation_transition", "Transition (seconds)"), ("animation_hold", "Pause per view (seconds)"))):
            row, col = 3 + (index // 2) * 2, index % 2
            ttk.Label(animation, text=label).grid(row=row, column=col, sticky="w", pady=(6, 0))
            ttk.Entry(animation, textvariable=self.form[key], width=10).grid(row=row + 1, column=col, sticky="ew", padx=(0, 8), pady=(4, 6))
        ttk.Checkbutton(animation, text="Animate back to the original view", variable=self.form["animation_loop"],
                        onvalue="yes", offvalue="no").grid(row=7, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Checkbutton(animation, text="Overlay prompt labels", variable=self.form["labels"],
                        onvalue="yes", offvalue="no", command=self.show_sample).grid(row=8, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Checkbutton(animation, text="Borderless / fill frame", variable=self.form["animation_borderless"],
                        onvalue="yes", offvalue="no").grid(row=9, column=0, columnspan=2, sticky="w", pady=5)
        ttk.Label(animation, text="Borderless removes outer padding and crops moving edges. Labels appear on GIF/MP4 and labeled PNG copies. Export saved images beside the preview.",
                  style="Hint.TLabel", wraplength=round(370 * self.ui_scale)).grid(row=10, column=0, columnspan=2, sticky="w", pady=(5, 0))
        advanced = self._card(self.controls, "ADVANCED")
        advanced.columnconfigure(0, weight=1)
        for row, (label, key, choices) in enumerate((
            ("Generation device", "device", ["cuda", "cpu"]),
            ("Text encoding device", "text_device", ["cuda", "cpu"]),
            ("Solver order", "order", ["2", "1"]),
        ), start=1):
            ttk.Label(advanced, text=label).grid(row=row, column=0, sticky="w", pady=5)
            combo = ttk.Combobox(advanced, textvariable=self.form[key], values=choices, state="readonly", width=10)
            combo.grid(row=row, column=1, sticky="e")
            if key == "order":
                self.order_combo = combo
        self._entry(advanced, "Flow shift", "shift", 4)
        self._folder(advanced, "Local Iris model (optional)", "model_path", 6)
        self._folder(advanced, "Local Qwen encoder (optional)", "text_path", 8)
        ttk.Label(advanced, text="Choose CPU text encoding if GPU memory is tight.", style="Hint.TLabel", wraplength=380).grid(row=10, column=0, columnspan=2, sticky="w", pady=(10, 0))

        actions = ttk.Frame(left, padding=(0, 12, 12, 0))
        actions.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.generate_button = ttk.Button(actions, text="Generate illusion", style="Accent.TButton", command=self.generate)
        self.generate_button.pack(side="left", fill="x", expand=True)
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="right", padx=(8, 0))

        right = ttk.Frame(body)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(2, weight=1, minsize=100)
        toolbar = ttk.Frame(right)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(toolbar, text="PREVIEW", style="Subtitle.TLabel").pack(side="left")
        ttk.Button(toolbar, text="Open results", command=self.open_results, style="Small.TButton").pack(side="right")
        ttk.Button(toolbar, text="Load result…", command=self.load_result, style="Small.TButton").pack(side="right", padx=6)
        self.play_button = ttk.Button(toolbar, text="Play animation", command=self.play_animation, style="Small.TButton", state="disabled")
        self.play_button.pack(side="right", padx=6)
        self.export_button = ttk.Button(toolbar, text="Export animation", command=self.export_animation, style="Small.TButton")
        self.export_button.pack(side="right")
        self.selection = tk.StringVar()
        self.sample_picker = ttk.Combobox(right, textvariable=self.selection, state="readonly")
        self.sample_picker.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        self.sample_picker.bind("<<ComboboxSelected>>", lambda _: self.show_sample())
        self.notebook = ttk.Notebook(right)
        self.notebook.grid(row=2, column=0, sticky="nsew")
        self.preview_canvases = []
        self.preview_images = [None, None]
        self.preview_photos = [None, None]
        for title in ("Original image", "All views"):
            canvas = tk.Canvas(self.notebook, background="#e8ecf3", highlightthickness=0)
            self.notebook.add(canvas, text=title)
            self.preview_canvases.append(canvas)
            canvas.bind("<Configure>", lambda _event: self.render_previews())
        self.notebook.bind("<<NotebookTabChanged>>", lambda _: self.after_idle(self.render_previews))
        self.status = tk.StringVar(value="Ready to create")
        ttk.Label(right, textvariable=self.status, style="Subtitle.TLabel", wraplength=round(600 * self.ui_scale)).grid(row=3, column=0, sticky="w", pady=(12, 6))
        self.progress = ttk.Progressbar(right, mode="indeterminate")
        self.progress.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.log = scrolledtext.ScrolledText(right, height=7, state="disabled", wrap="word",
                                            font=("Consolas", 9), bg="#202b3c", fg="#d7deed",
                                            relief="flat", padx=12, pady=10)
        self.log.grid(row=5, column=0, sticky="ew")
        ttk.Label(right, text="The first run downloads the Iris and Qwen weights. Saved images appear here when each sample finishes.",
                  style="Subtitle.TLabel", wraplength=round(600 * self.ui_scale)).grid(row=6, column=0, sticky="w", pady=(8, 0))
        self.after_idle(self.render_previews)

    def _card(self, parent, title):
        frame = ttk.Frame(parent, style="Card.TFrame", padding=12)
        frame.pack(fill="x", pady=(0, 12))
        ttk.Label(frame, text=title, style="Eyebrow.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 12))
        return frame

    def mode_changed(self, _event=None):
        if self._experiment_context is not None:
            self._experiment_profiles[self._experiment_context]={key:self.form[key].get() for key in GUI_DEFAULTS}
        self._mode_experiment_transition=True
        turbo = GENERATION_MODES[self.form["mode"].get()] is not None
        if turbo and self.standard_settings is None:
            self.standard_settings = {key: self.form[key].get() for key in ("steps", "guidance", "negative", "shift", "quality_method", *GUI_DEFAULTS)}
            from visual_anagrams.composition import resolve_reduction

            view_rows = [row.values() for row in self.rows]
            names = [row["view"] for row in view_rows]
            try:
                reduction = resolve_reduction(self.form["reduction"].get(), names, [row["argument"] for row in view_rows])
            except ValueError:
                reduction = "sum"
            default_method = default_quality_method(turbo=True, views=names, reduction=reduction)
            default_label = next(label for label, method in QUALITY_METHODS.items() if method == default_method)
            turbo_quality = self.turbo_quality_method or default_label
            for key, value in {"steps": "4", "guidance": "1.0", "negative": "", "shift": "8.0", "quality_method": turbo_quality}.items():
                self.form[key].set(value)
        elif not turbo and self.standard_settings is not None:
            self.turbo_quality_method = self.form["quality_method"].get()
            for key, value in self.standard_settings.items():
                self.form[key].set(value)
            self.standard_settings = None
        self.guidance_entry.configure(state="disabled" if turbo else "normal")
        self.negative_entry.configure(state="disabled" if turbo else "normal")
        self.order_combo.configure(state="disabled" if turbo else "readonly")
        self.mode_hint.configure(text=("4 steps, built-in guidance, shift 8. Two geometric views use adaptive balancing by default."
                                       if turbo else "Defaults: 30 steps, guidance 3, shift 8, adaptive balancing for two geometric views."))
        self.quality_combo.configure(state="readonly", values=[label for label, method in QUALITY_METHODS.items()
                                      if not turbo or method in ("plain", "adaptive")])
        self.quality_changed()
        self._views_changed()
        self._mode_experiment_transition=False

    def quality_changed(self, _event=None):
        method = QUALITY_METHODS[self.form["quality_method"].get()]
        previous = getattr(self, "previous_quality_method", "plain")
        previous_default = MethodOptions(previous).strength
        try:
            if float(self.form["method_strength"].get()) == previous_default:
                self.form["method_strength"].set(str(MethodOptions(method).strength))
        except ValueError:
            pass
        self.previous_quality_method = method
        turbo = GENERATION_MODES[self.form["mode"].get()] is not None
        hints = {"plain": "Classic view averaging. Adaptive balancing also supports Turbo; MTL and SyncVC require Standard Iris.",
                 "adaptive": "Scores clean previews with CPU CLIP every two steps and gives weaker views more weight. Standard or Turbo; two geometric views.",
                 "anagram_mtl": "Balances views and encourages subject attention to overlap. Uses CPU CLIP and gradient optimization.",
                 "syncvc": "Optimizes agreement between two image trajectories. Uses a flow Euler solver and gradient optimization."}
        self.quality_hint.configure(text=hints[method])
        for index, (label, entry) in enumerate(zip(self.method_labels, self.method_entries)):
            if method in ("plain", "adaptive") or turbo:
                label.grid_remove()
                entry.grid_remove()
            else:
                label.grid()
                entry.grid()
            entry.configure(state="disabled" if method in ("plain", "adaptive") or turbo or (index == 3 and method != "anagram_mtl") else "normal")
        self.order_combo.configure(state="disabled" if turbo or method == "syncvc" else "readonly")
        self._update_experiments()

    def _experiment_panel(self):
        panel=self._card(self.controls,'QUALITY EXPERIMENTS')
        panel.columnconfigure(0,weight=1)
        self.experiment_widgets={}
        def check(row,text,key,onvalue='yes',offvalue='no'):
            widget=ttk.Checkbutton(panel,text=text,variable=self.form[key],onvalue=onvalue,offvalue=offvalue,
                command=self._update_experiments)
            widget.grid(row=row,column=0,sticky='w',pady=5)
            self.experiment_widgets[key]=widget
        check(1,'Contrastive adaptive balancing','exp_contrastive')
        ttk.Label(panel,text='Cross-view negative prompts').grid(row=2,column=0,sticky='w',pady=(7,3))
        combo=ttk.Combobox(panel,textvariable=self.form['exp_negative'],values=list(NEGATIVE_MODES),state='readonly')
        combo.grid(row=3,column=0,sticky='ew')
        combo.bind('<<ComboboxSelected>>',self._update_experiments)
        self.experiment_widgets['exp_negative']=combo
        self.manual_negative_label=ttk.Label(panel,text='Manual negatives: view 1 | view 2',style='Hint.TLabel')
        self.manual_negative_label.grid(row=4,column=0,sticky='w',pady=(6,2))
        self.manual_negative_entry=ttk.Entry(panel,textvariable=self.form['exp_manual_negatives'])
        self.manual_negative_entry.grid(row=5,column=0,sticky='ew',pady=(0,5))
        check(6,'Adaptive component guidance','exp_component')
        check(7,'Schedule CFG in middle noise levels','exp_schedule','middle','constant')
        check(8,'Conflict-aware prompt directions','exp_conflict')
        self.experiment_hint=ttk.Label(panel,style='Hint.TLabel',wraplength=round(370*self.ui_scale))
        self.experiment_hint.grid(row=9,column=0,sticky='w',pady=6)
        details=ttk.Frame(panel,style='Card.TFrame')
        details.grid(row=11,column=0,sticky='ew')
        details.columnconfigure(0,weight=1)
        def toggle():
            if details.winfo_manager():details.grid_remove()
            else:details.grid()
        ttk.Button(panel,text='Show / hide experiment parameters',style='Small.TButton',command=toggle).grid(row=10,column=0,sticky='ew',pady=5)
        for row,(key,label) in enumerate([
            ('exp_cross_strength','Cross-negative strength (0–1)'),
            ('exp_low','CFG lower noise bound'),('exp_high','CFG upper noise bound'),
            ('exp_ramp','CFG smooth ramp width'),('exp_conflict_strength','Conflict projection strength (0–1)'),
            ('exp_freeze','Freeze balancing below noise (0 = off)')]):
            ttk.Label(details,text=label).grid(row=row*2,column=0,sticky='w',pady=(5,0))
            entry=ttk.Entry(details,textvariable=self.form[key])
            entry.grid(row=row*2+1,column=0,sticky='ew',pady=(3,5))
            self.experiment_widgets[key]=entry
        details.grid_remove()

    def _update_experiments(self,_event=None):
        if not hasattr(self,'experiment_widgets'):return
        from visual_anagrams.composition import resolve_reduction
        rows=[row.values() for row in self.rows]
        names=[row['view'] for row in rows]
        try:reduction=resolve_reduction(self.form['reduction'].get(),names,[row['argument'] for row in rows])
        except ValueError:reduction=None
        method=QUALITY_METHODS[self.form['quality_method'].get()]
        ordinary=method in ('plain','adaptive')
        standard=GENERATION_MODES[self.form['mode'].get()] is None
        context=(standard,tuple(names),reduction,method)
        if context!=self._experiment_context:
            if self._experiment_context is not None and not getattr(self,'_mode_experiment_transition',False):
                self._experiment_profiles[self._experiment_context]={key:self.form[key].get() for key in GUI_DEFAULTS}
            defaults=gui_values(context_options(turbo=not standard,views=names,reduction=reduction,quality_method=method))
            chosen=self._experiment_profiles.get(context,defaults)
            for key,value in chosen.items():self.form[key].set(value)
            self._experiment_context=context
        geometric=len(names)==2 and reduction=='mean'
        component=(reduction=='sum' and len(names)==2 and all(n in
            ('low_pass','high_pass','grayscale','color','motion','motion_res') for n in names))
        allowed=dict(exp_contrastive=ordinary and geometric and method=='adaptive',
            exp_negative=ordinary and standard and geometric,exp_component=ordinary and standard and component and method=='plain',
            exp_schedule=ordinary and standard,exp_conflict=ordinary and standard and geometric)
        for key,enabled in allowed.items():
            if not enabled:self.form[key].set(GUI_DEFAULTS[key])
            self.experiment_widgets[key].configure(state=('readonly' if key=='exp_negative' else 'normal') if enabled else 'disabled')
        manual=self.form['exp_negative'].get()=='Manual per-view negatives'
        for widget in (self.manual_negative_label,self.manual_negative_entry):
            if manual:widget.grid()
            else:widget.grid_remove()
        states=dict(exp_cross_strength=allowed['exp_negative'] and self.form['exp_negative'].get()!='Off',
            exp_low=allowed['exp_schedule'] and self.form['exp_schedule'].get()=='middle',
            exp_high=allowed['exp_schedule'] and self.form['exp_schedule'].get()=='middle',
            exp_ramp=allowed['exp_schedule'] and self.form['exp_schedule'].get()=='middle',
            exp_conflict_strength=allowed['exp_conflict'] and self.form['exp_conflict'].get()=='yes',
            exp_freeze=ordinary and (method=='adaptive' or self.form['exp_component'].get()=='yes'))
        for key,enabled in states.items():self.experiment_widgets[key].configure(state='normal' if enabled else 'disabled')
        if not states['exp_freeze']:self.form['exp_freeze'].set('0')
        hint=('Select Classic or Adaptive to use these experiments.' if not ordinary else
              'Turbo supports contrastive balancing and balance freeze; the other experiments require Standard.' if not standard else
              'Component guidance adapts prompt strength while preserving Sum. Geometric-only controls are disabled.' if component else
              'Options follow the tested profile and can be overridden. Contrastive needs Adaptive. Qwen cross negatives add about 50% sampling work at strength 0.5; plans are logged and saved.')
        self.experiment_hint.configure(text=hint)

    def _entry(self, parent, label, key, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, columnspan=2, sticky="w")
        ttk.Entry(parent, textvariable=self.form[key]).grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(4, 10))

    def _folder(self, parent, label, key, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, columnspan=2, sticky="w")
        ttk.Entry(parent, textvariable=self.form[key]).grid(row=row + 1, column=0, sticky="ew", pady=(4, 10))
        ttk.Button(parent, text="Browse", style="Small.TButton", command=lambda: self.browse(key)).grid(row=row + 1, column=1, padx=(6, 0), pady=(4, 10))

    def _scroll(self, event):
        x, y = self.winfo_pointerxy()
        widget = self.winfo_containing(x, y)
        canvases = [self.form_canvas]
        if hasattr(self, "tools_page"):
            canvases.append(self.tools_page.canvas)
        if widget is not None and not isinstance(widget, tk.Text):
            for canvas in canvases:
                if str(widget).startswith(str(canvas)):
                    canvas.yview_scroll(int(-event.delta / 120), "units")
                    break

    def add_view(self, prompt="", label="Original"):
        row = ViewRow(self.view_frame, len(self.rows) + 1, self.remove_view, prompt, label, self.colors)
        row.on_change = self._views_changed
        self.rows.append(row)
        row.frame.pack(fill="x", pady=(0, 10))
        self._views_changed()

    def remove_view(self, row):
        if len(self.rows) == 1:
            return
        self.rows.remove(row)
        row.frame.destroy()
        for index, remaining in enumerate(self.rows, start=1):
            remaining.title.configure(text=f"VIEW {index}")
        self._views_changed()

    def _views_changed(self, _event=None):
        if not self._quality_ready:
            return
        from visual_anagrams.composition import resolve_reduction

        rows = [row.values() for row in self.rows]
        try:
            reduction = resolve_reduction(self.form["reduction"].get(), [row["view"] for row in rows],
                                          [row["argument"] for row in rows])
        except ValueError:
            reduction = "sum"
        if QUALITY_METHODS[self.form['quality_method'].get()]=='adaptive' and (len(rows) != 2 or reduction != "mean"):
            self.form["quality_method"].set("Classic")
            self.quality_changed()
        self._update_experiments()

    def browse(self, key):
        path = filedialog.askdirectory(parent=self, title="Choose folder")
        if path:
            self.form[key].set(path)

    def generate(self):
        if self.running or self.tools_page.running:
            return
        try:
            command, destination = build_command(
                {key: value.get() for key, value in self.form.items()},
                [row.values() for row in self.rows], python=worker_python(),
            )
            self.job.start(command)
        except (ValueError, OSError, RuntimeError) as exc:
            messagebox.showerror("Could not start generation", str(exc), parent=self)
            return
        self.output_dir = destination
        self.task = "generation"
        self.images = []
        self.sample_picker.configure(values=[])
        self.selection.set("")
        self.preview_images = [None, None]
        self.render_previews()
        self.play_button.configure(state="disabled")
        self.current_sample = 1
        self.log_buffer = ""
        self.running = True
        self.generate_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.status.set("Preparing Iris and model downloads…")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.append_log(f"Saving to {destination}\n")

    def export_animation(self):
        if self.running or self.tools_page.running:
            return
        index = self.sample_picker.current()
        if index < 0 or index >= len(self.images):
            messagebox.showerror("Choose an image", "Generate an image or load a saved result first.", parent=self)
            return
        try:
            command = build_export_command({key: value.get() for key, value in self.form.items()},
                                            self.images[index], python=worker_python())
            self.job.start(command)
        except (ValueError, OSError, RuntimeError) as exc:
            messagebox.showerror("Could not export animation", str(exc), parent=self)
            return
        self.task = "animation"
        self.output_dir = self.images[index].parent.parent
        self.running = True
        self.log_buffer = ""
        self.generate_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.status.set("Preparing animation export…")
        self.append_log(f"Animating {self.images[index]}\n")

    def animation_files(self):
        index = self.sample_picker.current()
        if index < 0 or index >= len(self.images):
            return []
        image = self.images[index]
        return sorted([*image.parent.glob(image.stem + "*.mp4"), *image.parent.glob(image.stem + "*.gif")])

    def play_animation(self):
        files = self.animation_files()
        if not files:
            return
        path = max(files, key=lambda item: item.stat().st_mtime)
        try:
            if os.name == "nt":
                os.startfile(path)
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])
        except OSError as exc:
            messagebox.showerror("Could not open animation", str(exc), parent=self)

    def append_log(self, text):
        self.log.configure(state="normal")
        self.log.insert("end", text + "\n")
        # Bound log memory during long downloads and large batches.
        if int(self.log.index("end-1c").split(".")[0]) > 800:
            self.log.delete("1.0", "200.0")
        self.log.see("end")
        self.log.configure(state="disabled")

    def handle_line(self, line):
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()
        if not line:
            return
        self.append_log(line)
        match = re.search(r"(Iris multi-view denoising|Iris anagram_mtl|Iris syncvc|Animation export):.*?(\d+)/(\d+)", line)
        if match:
            stage = match[1]
            current, total = int(match[2]), int(match[3])
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=total, value=current)
            self.status.set(f"Animation · Frame {current}/{total}" if stage == "Animation export" else f"Sample {self.current_sample} · Denoising {current}/{total}")
        elif line.startswith("Generating sample "):
            match = re.search(r"sample (\d+)", line)
            self.current_sample = int(match[1]) if match else 1
            self.status.set(line)
            self.progress.configure(value=0)
        elif line.startswith("Preparing") or line.startswith("Loading Iris"):
            self.status.set(line)
        elif line.startswith("Exporting animation"):
            self.status.set(line)
        elif line.startswith(("Saved animation ", "Saved labels ")):
            self.show_sample()
        elif line.startswith("Saved "):
            self.refresh_results()

    def poll(self):
        try:
            while True:
                kind, value = self.job.events.get_nowait()
                if kind == "log":
                    self.log_buffer += value
                    lines = re.split(r"[\r\n]", self.log_buffer)
                    self.log_buffer = lines.pop()
                    for line in lines:
                        self.handle_line(line)
                elif kind == "exit":
                    if self.log_buffer:
                        self.handle_line(self.log_buffer)
                        self.log_buffer = ""
                    self.running = False
                    self.progress.stop()
                    self.generate_button.configure(state="normal")
                    self.export_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    if self.task == "generation":
                        self.refresh_results()
                    else:
                        self.show_sample()
                    if self.job.cancelled:
                        self.status.set("Cancelled. Completed outputs have been kept.")
                    elif value == 0:
                        self.progress.configure(mode="determinate", maximum=1, value=1)
                        self.status.set("Animation export finished. Use Play animation to view it." if self.task == "animation" else f"Finished · {len(self.images)} image(s) saved")
                    else:
                        self.status.set("Export failed. See the log below for details." if self.task == "animation" else "Generation failed. See the log below for details.")
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def cancel(self):
        self.job.cancel()
        self.cancel_button.configure(state="disabled")
        self.status.set("Stopping generation…")

    def refresh_results(self):
        if self.output_dir is None or not self.output_dir.is_dir():
            return
        self.images = sorted(self.output_dir.glob("*/sample_1024.png"))
        labels = [f"Seed {image.parent.name}" for image in self.images]
        self.sample_picker.configure(values=labels)
        if labels:
            self.sample_picker.current(len(labels) - 1)
            self.show_sample()

    def show_sample(self):
        index = self.sample_picker.current()
        if index < 0 or index >= len(self.images):
            return
        path = self.images[index]
        paths = [path, path.with_name(path.stem + ".views.png")]
        for index, candidate in enumerate(paths):
            labeled = candidate.with_name(candidate.stem + ".labeled.png")
            if self.form["labels"].get() == "yes" and labeled.is_file():
                candidate = labeled
            try:
                with Image.open(candidate) as image:
                    self.preview_images[index] = image.convert("RGB")
            except (OSError, ValueError):
                self.preview_images[index] = None
        self.render_previews()
        self.play_button.configure(state="normal" if self.animation_files() else "disabled")

    def render_previews(self):
        for index, canvas in enumerate(self.preview_canvases):
            canvas.delete("all")
            width, height = canvas.winfo_width(), canvas.winfo_height()
            if width < 4 or height < 4:
                continue
            image = self.preview_images[index]
            if image is None:
                canvas.create_text(width / 2, height / 2 - 20 * self.ui_scale, text="A new perspective awaits", fill=self.colors["placeholder"], font=("Segoe UI", 17), width=max(1, width - 30))
                canvas.create_text(width / 2, height / 2 + 18 * self.ui_scale, text="Generate an illusion or load a saved result.", fill=self.colors["placeholder"], font=("Segoe UI", 10), width=max(1, width - 30))
            else:
                preview = image.copy()
                preview.thumbnail((max(1, width - 24), max(1, height - 24)), Image.Resampling.LANCZOS)
                self.preview_photos[index] = ImageTk.PhotoImage(preview, master=self)
                canvas.create_image(width / 2, height / 2, image=self.preview_photos[index])

    def load_result(self):
        path = filedialog.askopenfilename(parent=self, title="Open a generated image", filetypes=[("PNG images", "*.png")])
        if path:
            candidate = Path(path)
            if candidate.name.endswith(".labeled.png"):
                candidate = candidate.with_name(candidate.name.replace(".labeled.png", ".png"))
            if candidate.name.endswith(".views.png"):
                candidate = candidate.with_name(candidate.name.replace(".views.png", ".png"))
            self.images = [candidate]
            self.sample_picker.configure(values=[f"Loaded: {candidate.parent.name}"])
            self.sample_picker.current(0)
            self.show_sample()

    def open_results(self):
        folder = self.output_dir if self.output_dir and self.output_dir.is_dir() else Path(self.form["output"].get()).expanduser()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(folder)
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(folder)])
        except OSError as exc:
            messagebox.showerror("Could not open folder", str(exc), parent=self)

    def close(self):
        if self.running or self.tools_page.running:
            if not messagebox.askyesno("Stop current task?", "Stop the current task and close the window? Completed outputs will be kept.", parent=self):
                return
            self.job.cancel()
            self.tools_page.job.cancel()
        self.destroy()


def main():
    AnagramApp().mainloop()
