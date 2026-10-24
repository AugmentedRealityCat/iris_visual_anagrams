"""Separate Iris Tools desktop page. Models only load in a worker subprocess."""

from pathlib import Path
import queue
import re
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageOps, ImageTk

from iris_tools import SIZES, TASKS, build_parser, validate_args

ROOT = Path(__file__).resolve().parents[1]
MODES = {"Standard Iris": None, "Turbo r2-1000 (fast)": "r2-1000", "Turbo r1-1500": "r1-1500"}
HINTS = {
    "generate": "Generate an ordinary image from one prompt. Reuses your existing Iris/Qwen weights; portrait and landscape sizes are available.",
    "img2img": "Experimental noise-and-denoise with the generation model. Lower strength retains more input structure. The input is centre-cropped to the selected size. This is not instruction-based editing.",
    "depth": "Photo to relative log depth, not distance in metres. Saves a colour preview, a near-white 16-bit PNG and raw .npy values. Separate INT8 weights; no Qwen download.",
    "restore": "Restore details with the separate one-step model, using overlapping tiles. 4× is the trained scale; 1×/2× resize the restored 4× result. The input size limit reduces large photos before processing. No Qwen needed.",
}


def build_tools_command(settings, *, python=None, root=ROOT):
    from .gui import worker_python

    task = TASKS.get(settings.get("task"))
    if task is None:
        raise ValueError("Choose an Iris task")
    name = settings.get("name", "").strip()
    if not name or name in (".", "..") or any(c in name for c in '<>:"/\\|?*'):
        raise ValueError("Give the run a valid name")
    if not settings.get("output", "").strip():
        raise ValueError("Choose an output folder")
    output = Path(settings["output"]).expanduser().resolve()
    destination = output / name
    suffix = 2
    while destination.exists():
        destination = output / f"{name}_{suffix}"
        suffix += 1
    arguments = ["--task", task, "--name", destination.name, "--save_dir", str(output),
                 "--device", settings.get("device", "cuda")]
    if task != "generate":
        arguments += ["--input", settings.get("input", "")]
    if task in ("generate", "img2img"):
        mode = settings.get("mode", "Standard Iris")
        if mode not in MODES:
            raise ValueError("Choose a generation mode")
        arguments += ["--prompt", settings.get("prompt", "").strip(), "--negative_prompt", settings.get("negative", ""),
                      "--size", settings.get("size", "1024x1024"), "--text_encoder_device", settings.get("text_device", "cuda")]
        for key, flag, cast in [("steps", "--steps", int), ("guidance", "--guidance", float),
                                ("seed", "--seed", int), ("samples", "--num_samples", int),
                                ("strength", "--strength", float)]:
            if key == "strength" and task != "img2img":
                continue
            try:
                value = cast(settings[key])
            except (ValueError, KeyError):
                raise ValueError(f"Enter a valid number for {key}") from None
            arguments += [flag, str(value)]
        if MODES[mode]:
            arguments += ["--turbo", "--turbo_adapter", MODES[mode]]
        if settings.get("text_path", "").strip():
            arguments += ["--text_encoder_path", settings["text_path"].strip()]
    elif task == "depth":
        try:
            arguments += ["--max_side", str(int(settings.get("max_side", "1024")))]
        except ValueError:
            raise ValueError("Depth size must be a whole number") from None
    else:
        if settings.get("scale", "4") not in ("1", "2", "4"):
            raise ValueError("Choose restoration scale 1, 2 or 4")
        arguments += ["--scale", settings.get("scale", "4")]
        if settings.get("budget", "yes") != "yes":
            arguments += ["--no_budget"]
    if settings.get("model_path", "").strip():
        arguments += ["--model_path", settings["model_path"].strip()]
    try:
        validate_args(build_parser().parse_args(arguments))
    except SystemExit:
        raise ValueError("Choose valid task, size and device settings") from None
    return [python or worker_python(), "-u", str(Path(root) / "iris_tools.py"), *arguments], destination


class IrisToolsPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, padding=18)
        from .gui import GenerationJob

        self.app, self.job = app, GenerationJob()
        self.running = False
        self.output_dir = None
        self.images, self.photos = [], [None, None]
        self.preview_images = [None, None]
        self.log_buffer = ""
        self.standard_settings = None
        self.form = {key: tk.StringVar(value=value) for key, value in {
            "task": "Image generation", "name": "iris-image", "output": str(ROOT / "results/iris-tools"),
            "input": "", "prompt": "a watercolor painting of a lighthouse on a rocky coast",
            "negative": "", "mode": "Standard Iris", "size": "1024x1024", "steps": "30", "guidance": "3",
            "seed": "0", "samples": "1", "strength": "0.5", "max_side": "1024", "scale": "4",
            "budget": "yes", "device": "cuda", "text_device": "cuda", "model_path": "", "text_path": "",
        }.items()}
        self.columnconfigure(0, weight=0, minsize=round(420 * app.ui_scale))
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)
        ttk.Label(self, text="Iris Tools", style="Title.TLabel").grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 14))
        left = ttk.Frame(self)
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 18))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(left, background="#f3f5f8", highlightthickness=0, width=round(415 * app.ui_scale))
        self.canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=self.canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.canvas.configure(yscrollcommand=scrollbar.set)
        controls = ttk.Frame(self.canvas, padding=10, style="Card.TFrame")
        controls.columnconfigure(0, weight=1)
        window = self.canvas.create_window((0, 0), window=controls, anchor="nw")
        controls.bind("<Configure>", lambda _: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(window, width=event.width))
        self.fields, self.entries = {}, {}
        choices = {"task": list(TASKS), "mode": list(MODES), "size": SIZES, "scale": ["4", "2", "1"],
                   "max_side": ["1024", "768", "512"], "device": ["cuda", "cpu"], "text_device": ["cuda", "cpu"]}
        labels = [("task", "Task"), ("input", "Input image"), ("prompt", "Prompt"),
                  ("mode", "Generation mode"), ("size", "Output size (width × height)"),
                  ("steps", "Steps"), ("guidance", "Guidance"), ("negative", "Negative prompt"),
                  ("seed", "Seed"), ("samples", "Samples"), ("strength", "Image-to-image strength (0–1)"),
                  ("max_side", "Depth inference long side"), ("scale", "Restoration output scale"),
                  ("budget", "Limit large restoration inputs"), ("device", "Device"),
                  ("text_device", "Text encoding device"), ("model_path", "Local INT8 task model (optional)"),
                  ("text_path", "Local Qwen encoder (optional)"), ("name", "Run name"), ("output", "Output folder")]
        for row, (key, label) in enumerate(labels):
            frame = ttk.Frame(controls, style="Card.TFrame")
            frame.grid(row=row+1, column=0, sticky="ew", pady=(0, 9))
            frame.columnconfigure(0, weight=1)
            self.fields[key] = frame
            ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")
            if key == "budget":
                entry = ttk.Checkbutton(frame, text="At most 512 px short side / 1024 px long side", variable=self.form[key], onvalue="yes", offvalue="no")
            elif key in choices:
                entry = ttk.Combobox(frame, textvariable=self.form[key], values=choices[key], state="readonly")
            else:
                entry = ttk.Entry(frame, textvariable=self.form[key])
            entry.grid(row=1, column=0, sticky="ew", pady=(4, 0))
            self.entries[key] = entry
            if key in ("input", "output", "model_path", "text_path"):
                ttk.Button(frame, text="Browse", command=lambda k=key: self.browse(k), style="Small.TButton").grid(row=1, column=1, padx=(5, 0))
        self.entries["task"].bind("<<ComboboxSelected>>", self.task_changed)
        self.entries["mode"].bind("<<ComboboxSelected>>", self.mode_changed)
        actions = ttk.Frame(left, padding=(0, 12, 0, 0))
        actions.grid(row=1, column=0, columnspan=2, sticky="ew")
        self.run_button = ttk.Button(actions, text="Generate image", command=self.run, style="Accent.TButton")
        self.run_button.pack(side="left", fill="x", expand=True)
        self.cancel_button = ttk.Button(actions, text="Cancel", command=self.cancel, state="disabled")
        self.cancel_button.pack(side="right", padx=(8, 0))
        right = ttk.Frame(self)
        right.grid(row=1, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(3, weight=1)
        self.hint = ttk.Label(right, style="Subtitle.TLabel", wraplength=round(650 * app.ui_scale))
        self.hint.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        toolbar = ttk.Frame(right)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(0, 9))
        ttk.Button(toolbar, text="Open results", command=self.open_results).pack(side="right")
        ttk.Button(toolbar, text="Load result…", command=self.load_result).pack(side="right", padx=6)
        self.selection = tk.StringVar()
        self.picker = ttk.Combobox(right, textvariable=self.selection, state="readonly")
        self.picker.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.picker.bind("<<ComboboxSelected>>", lambda _: self.show_result())
        notebook = ttk.Notebook(right)
        notebook.grid(row=3, column=0, sticky="nsew")
        self.previews = []
        for title in ("Input", "Result"):
            preview = tk.Canvas(notebook, background="#e8ecf3", highlightthickness=0)
            notebook.add(preview, text=title)
            preview.bind("<Configure>", lambda _: self.render())
            self.previews.append(preview)
        notebook.select(1)
        self.progress = ttk.Progressbar(right, mode="indeterminate")
        self.progress.grid(row=4, column=0, sticky="ew", pady=(10, 5))
        self.status = tk.StringVar(value="Ready")
        ttk.Label(right, textvariable=self.status, wraplength=600).grid(row=5, column=0, sticky="ew", pady=6)
        self.log = scrolledtext.ScrolledText(right, height=6, state="disabled", wrap="word", font=("Consolas", 9))
        self.log.grid(row=6, column=0, sticky="ew")
        self.task_changed()
        self.after(100, self.poll)

    def task_changed(self, _event=None):
        task = TASKS[self.form["task"].get()]
        if task == "img2img" and MODES[self.form["mode"].get()]:
            self.form["mode"].set("Standard Iris")
            self.mode_changed()
        common = {"task", "device", "model_path", "name", "output"}
        generation = {"prompt", "size", "steps", "guidance", "negative", "seed", "samples", "text_device", "text_path"}
        visible = common | (generation if task in ("generate", "img2img") else set())
        visible |= {"mode"} if task == "generate" else {"input"}
        visible |= {"strength"} if task == "img2img" else {"max_side"} if task == "depth" else {"scale", "budget"} if task == "restore" else set()
        for key, field in self.fields.items():
            field.grid() if key in visible else field.grid_remove()
        self.hint.configure(text=HINTS[task])
        self.run_button.configure(text={"generate": "Generate image", "img2img": "Transform image", "depth": "Estimate depth", "restore": "Restore image"}[task])

    def mode_changed(self, _event=None):
        turbo = MODES[self.form["mode"].get()] is not None
        if turbo and self.standard_settings is None:
            self.standard_settings = {k: self.form[k].get() for k in ("steps", "guidance", "negative")}
            for key, value in dict(steps="4", guidance="1", negative="").items():
                self.form[key].set(value)
        elif not turbo and self.standard_settings is not None:
            for key, value in self.standard_settings.items():
                self.form[key].set(value)
            self.standard_settings = None
        for key in ("guidance", "negative"):
            self.entries[key].configure(state="disabled" if turbo else "normal")

    def browse(self, key):
        path = (filedialog.askopenfilename(parent=self, filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff")])
                if key == "input" else filedialog.askdirectory(parent=self))
        if path:
            self.form[key].set(path)
            if key == "input":
                try:
                    with Image.open(path) as image:
                        self.preview_images[0] = ImageOps.exif_transpose(image).convert("RGB")
                    self.render()
                except (OSError, ValueError) as exc:
                    messagebox.showerror("Cannot open image", str(exc), parent=self)

    def run(self):
        if self.running or self.app.running:
            messagebox.showinfo("Iris is busy", "Wait for the current task or cancel it first.", parent=self)
            return
        try:
            command, destination = build_tools_command({k:v.get() for k,v in self.form.items()})
            if TASKS[self.form["task"].get()] == "generate":
                self.preview_images[0] = None
            else:
                with Image.open(self.form["input"].get()) as image:
                    self.preview_images[0] = ImageOps.exif_transpose(image).convert("RGB")
            self.job.start(command)
        except (ValueError, OSError, RuntimeError) as exc:
            messagebox.showerror("Could not start Iris", str(exc), parent=self)
            return
        self.output_dir, self.running = destination, True
        self.images = []
        self.picker.configure(values=[])
        self.selection.set("")
        self.preview_images[1] = None
        self.render()
        self.log_buffer = ""
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.run_button.configure(state="disabled")
        self.app.generate_button.configure(state="disabled")
        self.app.export_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.progress.configure(mode="indeterminate")
        self.progress.start(12)
        self.status.set("Preparing Iris and task weights…")

    def handle_line(self, line):
        line = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", line).strip()
        if not line:
            return
        self.log.configure(state="normal")
        self.log.insert("end", line + "\n")
        if int(self.log.index("end-1c").split(".")[0]) > 800:
            self.log.delete("1.0", "200.0")
        self.log.see("end")
        self.log.configure(state="disabled")
        match = re.search(r"Iris (?:multi-view denoising|image-to-image):.*?(\d+)/(\d+)", line)
        if match:
            current, total = int(match[1]), int(match[2])
            self.progress.stop()
            self.progress.configure(mode="determinate", maximum=total, value=current)
            self.status.set(f"Denoising {current}/{total}")
        elif line.startswith(("Preparing", "Loading", "Restoring", "Restoration tile", "Generating")):
            self.status.set(line)
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
                    self.run_button.configure(state="normal")
                    self.app.generate_button.configure(state="normal")
                    self.app.export_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    self.refresh_results()
                    if self.job.cancelled:
                        self.status.set("Cancelled. Completed results have been kept.")
                    elif value:
                        self.status.set("Iris task failed. See the log for details.")
                    else:
                        self.progress.configure(mode="determinate", maximum=1, value=1)
                        self.status.set(f"Finished · {len(self.images)} result(s) saved")
        except queue.Empty:
            pass
        self.after(100, self.poll)

    def cancel(self):
        self.job.cancel()
        self.cancel_button.configure(state="disabled")
        self.status.set("Stopping Iris…")

    def refresh_results(self):
        if self.output_dir is not None:
            self.images = sorted(self.output_dir.glob("*/result.png"))
            self.picker.configure(values=[f"Result {p.parent.name}" for p in self.images])
            if self.images:
                self.picker.current(len(self.images)-1)
                self.show_result()

    def show_result(self):
        index = self.picker.current()
        if index < 0 or index >= len(self.images):
            return
        path = self.images[index]
        for index, candidate in enumerate((path.parent / "input.png", path)):
            try:
                with Image.open(candidate) as image:
                    self.preview_images[index] = image.convert("RGB")
            except (OSError, ValueError):
                self.preview_images[index] = None
        self.render()

    def render(self):
        for index, canvas in enumerate(self.previews):
            canvas.delete("all")
            width, height = canvas.winfo_width(), canvas.winfo_height()
            if width < 4 or height < 4:
                continue
            image = self.preview_images[index]
            if image is None:
                canvas.create_text(width/2, height/2, text="Choose an input or run an Iris task", fill=self.app.colors["placeholder"], width=max(1,width-30))
            else:
                preview = image.copy()
                preview.thumbnail((max(1,width-24), max(1,height-24)), Image.Resampling.LANCZOS)
                self.photos[index] = ImageTk.PhotoImage(preview, master=self)
                canvas.create_image(width/2, height/2, image=self.photos[index])

    def open_results(self):
        import os
        import subprocess
        import sys
        folder = (self.output_dir if self.output_dir is not None and self.output_dir.is_dir()
                  else Path(self.form["output"].get()).expanduser())
        try:
            folder.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                os.startfile(folder)
            else:
                subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(folder)])
        except OSError as exc:
            messagebox.showerror("Could not open results", str(exc), parent=self)

    def load_result(self):
        path = filedialog.askopenfilename(parent=self, filetypes=[("PNG images", "*.png")])
        if path:
            self.images = [Path(path)]
            self.picker.configure(values=[Path(path).name])
            self.picker.current(0)
            self.show_result()
