"""Shared desktop colours and a small per-user appearance preference."""

import json
import os
from pathlib import Path

LIGHT = dict(bg="#f3f5f8", panel="#ffffff", ink="#202b3c", muted="#637086", accent="#5146d9",
             field="#f7f8fc", border="#dce1eb", button="#e6e9f0", hover="#d8ddeb", pressed="#cbd2df",
             accent_hover="#4035c5", accent_disabled="#b9b6df", disabled="#8790a1",
             disabled_bg="#e8ebf1", preview="#e8ecf3", placeholder="#657185", trough="#e5e8f0",
             log_bg="#202b3c", log_ink="#d7deed", selection="#5146d9")
DARK = dict(bg="#151922", panel="#1e2430", ink="#e6eaf2", muted="#a6b0c3", accent="#a79aff",
            field="#141a24", border="#3b4558", button="#30394a", hover="#404c61", pressed="#4b5970",
            accent_hover="#7562e0", accent_disabled="#49445f", disabled="#8590a6",
            disabled_bg="#262d3a", preview="#10151e", placeholder="#a0acc1", trough="#252d3b",
            log_bg="#0e131c", log_ink="#d5dfef", selection="#6756c5")


def preferences_path():
    if os.name == "nt":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "visual-anagrams" / "ui.json"


def style_log_scrollbar(text):
    """Use ttk because Windows' native Tk scrollbar ignores dark colours."""
    from tkinter import ttk

    text.vbar.destroy()
    text.vbar = ttk.Scrollbar(text.frame, command=text.yview)
    text.vbar.pack(side="right", fill="y")
    text.configure(yscrollcommand=text.vbar.set)


def load_preferences(path):
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def save_theme(path, dark):
    path = Path(path)
    values = load_preferences(path)
    values["theme"] = "dark" if dark else "light"
    temporary = path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(values, indent=2), encoding="utf-8")
        temporary.replace(path)
    except OSError:
        # Appearance still changes if the preference folder is read-only.
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
