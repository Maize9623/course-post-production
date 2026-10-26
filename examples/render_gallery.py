#!/usr/bin/env python3
"""Render synthetic layout examples with a local Chrome/Chromium, without GPU."""
from pathlib import Path
import argparse
import os
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCENES = ("presenter", "diagram", "practice", "prompt", "montage", "playback")

def chrome_path(explicit):
    candidates = [explicit, os.environ.get("CHROME_BIN"), shutil.which("google-chrome"),
                  shutil.which("chromium"), shutil.which("chromium-browser"),
                  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    program_files = os.environ.get("PROGRAMFILES")
    if program_files:
        candidates.append(str(Path(program_files) / "Google/Chrome/Application/chrome.exe"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return str(candidate)
    raise SystemExit("Chrome/Chromium required only for this optional gallery. Use --chrome or CHROME_BIN.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chrome")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "assets/examples")
    parser.add_argument("--scene", choices=SCENES, action="append")
    args = parser.parse_args()
    executable = chrome_path(args.chrome)
    if not (ROOT / "assets/demo-presenter.png").is_file():
        raise SystemExit("Missing fictional demo-presenter.png asset.")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="course-gallery-browser-") as profile:
        for scene in args.scene or SCENES:
            destination = (args.out_dir / (scene + ".png")).resolve()
            cmd = [executable, "--headless=new", "--disable-gpu", "--no-first-run",
                   "--no-default-browser-check", "--disable-background-networking",
                   "--disable-sync", "--disable-extensions", "--hide-scrollbars",
                   "--force-device-scale-factor=1", "--window-size=2560,1440",
                   "--virtual-time-budget=1800", "--allow-file-access-from-files",
                   "--user-data-dir=" + profile, "--screenshot=" + str(destination),
                   (ROOT / "examples/gallery.html").as_uri() + "?scene=" + scene]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode or not destination.is_file():
                raise SystemExit("Gallery render failed: " + result.stderr[-2000:])
            print(scene + ": " + str(destination), flush=True)

if __name__ == "__main__":
    main()
