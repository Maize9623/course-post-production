#!/usr/bin/env python3
"""Register read-only course inputs and audit a 60 fps project structure.

init creates project.json and input-manifest.json without copying source media.
It also freezes the mode-selected brand profile as project-style.json; explicit
--style-overrides apply only to that lesson, never to the installed skill.
audit reads shots.json (or --shots), checks hashes, image usage, montage source
reuse, continuity, and chapter-core visibility. It never validates final pixels,
AAC, speech transcription or lip synchronization. Missing shot plans are NOT_READY.

Source intervals are integer frames on a 60 fps adapted source timeline:
source:{asset_id,start_frame,end_frame}, or asset_id plus source_start_frame /
source_end_frame. speed/crop do not create a new source identity.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
import mimetypes
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

FPS = 60
ROLES = ("narration", "script", "operation", "image", "video", "bgm")
FOLDERS = ("private", "transcript", "edit", "assets", "render", "qa", "delivery")
SHOT_KINDS = {"diagram", "figure", "reading", "operation", "montage", "process", "presenter", "cover", "transition"}
STYLE_PATH = Path(__file__).resolve().parents[1] / "assets/style.json"
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.mkv', '.m4v', '.webm', '.avi'}
AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg', '.aiff', '.aif'}
ROLE_EXTENSIONS = {
    'narration': VIDEO_EXTENSIONS | AUDIO_EXTENSIONS,
    'operation': VIDEO_EXTENSIONS, 'video': VIDEO_EXTENSIONS,
    'image': {'.png', '.jpg', '.jpeg', '.webp', '.gif', '.svg', '.pdf', '.tif', '.tiff'},
    'script': {'.pdf', '.docx', '.txt', '.md', '.rtf', '.html', '.odt', '.ppt', '.pptx'},
    'bgm': AUDIO_EXTENSIONS,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def merge_style(base, overrides):
    """Recursively override this lesson's settings without editing the skill."""
    result = copy.deepcopy(base)
    for key, value in overrides.items():
        result[key] = merge_style(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else copy.deepcopy(value)
    return result


def project_style(mode, overrides_path=None):
    style = read_json(STYLE_PATH)
    profile = style.get("mode_profiles", {}).get(mode, mode)
    settings = style.get("profiles", {}).get(profile)
    if not settings:
        raise ValueError(f"No brand profile for course mode: {mode}")
    resolved = merge_style(style, settings)
    resolved.pop("profiles", None)
    resolved.pop("mode_profiles", None)
    if overrides_path:
        overrides = read_json(overrides_path)
        if not isinstance(overrides, dict):
            raise ValueError("Style overrides must be a JSON object.")
        resolved = merge_style(resolved, overrides)
    # Profile identity is derived from the course mode, not an override label.
    resolved.update(profile=profile, course_mode=mode, style_source=str(STYLE_PATH),
                    style_source_sha256=sha256(STYLE_PATH))
    return resolved


def dump_json(path, value, exclusive=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    else:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(data)
    return value.hexdigest()


def find_ffprobe():
    for candidate in (os.environ.get("FFPROBE") or "ffprobe",):
        found = shutil.which(candidate)
        if found:
            return found
    raise ValueError("ffprobe is required to register audio/video. Add it to PATH.")


def probe(path):
    fields = ("format=duration,format_name:stream=index,codec_type,codec_name,width,height,"
              "r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames,"
              "sample_rate,sample_fmt,channels,channel_layout,bits_per_sample")
    completed = subprocess.run([find_ffprobe(), "-v", "error", "-show_entries", fields,
                                "-of", "json", str(path)], capture_output=True, text=True)
    if completed.returncode:
        raise ValueError(f"ffprobe failed for {path.name}: {completed.stderr.strip()[:1000]}")
    result = json.loads(completed.stdout)
    for stream in result.get("streams", []):
        if stream.get("codec_type") == "video":
            value = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
            try:
                stream["fps"] = float(Fraction(value))
            except (ValueError, ZeroDivisionError, TypeError):
                stream["fps"] = None
    return result


def image_metadata(path):
    # Lightweight header metadata only; no pixel decoding or external image tools.
    with path.open("rb") as stream:
        head = stream.read(32)
    if head.startswith(b"\x89PNG\r\n\x1a\n") and len(head) >= 24:
        width, height = struct.unpack(">II", head[16:24])
        return dict(width=width, height=height, header_format="PNG")
    if head[:6] in (b"GIF87a", b"GIF89a") and len(head) >= 10:
        width, height = struct.unpack("<HH", head[6:10])
        return dict(width=width, height=height, header_format="GIF")
    return {}


def expand_inputs(values, role):
    """Discover role-compatible files in explicit folders; never follow links."""
    for value in values:
        source = Path(value).expanduser()
        if not source.is_dir():
            yield source
            continue
        found = [p for p in sorted(source.rglob('*')) if p.is_file() and not p.is_symlink()
                 and not any(part.startswith('.') for part in p.relative_to(source).parts)
                 and p.suffix.lower() in ROLE_EXTENSIONS[role]]
        if not found:
            raise ValueError(f"No supported {role} files found in directory: {source}")
        yield from found


def register_project(args):
    root = args.project.expanduser().resolve()
    if (root / "project.json").exists():
        raise ValueError(f"Project already exists; refusing to overwrite {root / 'project.json'}")
    if (root / "input-manifest.json").exists():
        raise ValueError("An input manifest already exists; refusing to overwrite it.")
    if (root / "project-style.json").exists():
        raise ValueError("A project style already exists; refusing to overwrite it.")
    resolved_style = project_style(args.mode, getattr(args, "style_overrides", None))
    assets, seen_role_paths, cached_paths, first_hash = [], set(), {}, {}
    counters = dict.fromkeys(ROLES, 0)
    for role in ROLES:
        values = getattr(args, role)
        paths = values if role != "bgm" else ([values] if values else [])
        for value in expand_inputs(paths or [], role):
            path = Path(value).expanduser().resolve()
            if (role, path) in seen_role_paths:
                continue
            seen_role_paths.add((role, path))
            counters[role] += 1
            asset = dict(id=f"{role}-{counters[role]:03d}", role=role, path=str(path),
                         read_only=True, optional=role == "bgm", exists=path.is_file())
            if role == "bgm":
                asset["is_default"] = False
            if not path.is_file():
                if role != "bgm":
                    raise ValueError(f"Required {role} file does not exist: {path}")
                asset.update(status="missing", sha256=None, metadata={})
                assets.append(asset)
                continue
            if path not in cached_paths:
                before = path.stat()
                checksum = sha256(path)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError(f"Source changed during registration: {path}")
                cached_paths[path] = dict(size=after.st_size, mtime_ns=after.st_mtime_ns, sha256=checksum)
            asset.update(cached_paths[path], status="registered")
            metadata = dict(name=path.name, extension=path.suffix.lower(), mime_type=mimetypes.guess_type(path)[0])
            if role in {"narration", "operation", "video", "bgm"}:
                metadata["ffprobe"] = probe(path)
            elif role == "image":
                metadata.update(image_metadata(path))
            asset["metadata"] = metadata
            if asset["sha256"] in first_hash:
                asset["duplicate_of"] = first_hash[asset["sha256"]]
            else:
                first_hash[asset["sha256"]] = asset["id"]
            assets.append(asset)
    project = dict(schema_version=1, title=args.title, mode=args.mode,
                   created_at_utc=datetime.now(timezone.utc).isoformat(), fps=FPS, frames=None,
                   manifest="input-manifest.json", shots_file="shots.json", folders=list(FOLDERS),
                   profile=resolved_style["profile"], style_file="project-style.json",
                   source_policy="Register absolute paths and hashes; source files remain in place and read-only.")
    manifest = dict(schema_version=1, assets=assets)
    root.mkdir(parents=True, exist_ok=True)
    for folder in FOLDERS:
        (root / folder).mkdir(exist_ok=True)
    # Exclusive creation protects existing project metadata. Original assets are
    # never copied into the project or the reusable skill.
    dump_json(root / "input-manifest.json", manifest, exclusive=True)
    created_style = False
    try:
        dump_json(root / "project-style.json", resolved_style, exclusive=True)
        created_style = True
        dump_json(root / "project.json", project, exclusive=True)
    except BaseException:
        (root / "input-manifest.json").unlink()
        if created_style:
            (root / "project-style.json").unlink(missing_ok=True)
        raise
    return dict(status="INITIALIZED", project=str(root), assets=len(assets),
                profile=resolved_style["profile"], style_file=str(root / "project-style.json"),
                missing_optional_assets=[a["id"] for a in assets if not a["exists"]],
                content_duplicates=[dict(id=a["id"], duplicate_of=a["duplicate_of"]) for a in assets if "duplicate_of" in a])


def is_frame(value):
    return type(value) is int and value >= 0


def media_duration(asset):
    metadata = asset.get("metadata", {}).get("ffprobe", {})
    value = metadata.get("format", {}).get("duration")
    if value is None:
        values = [float(s["duration"]) for s in metadata.get("streams", []) if s.get("duration") not in (None, "N/A")]
        return max(values) if values else None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except ValueError:
        return None


def audit_project(args):
    root = args.project.expanduser().resolve()
    project = read_json(root / "project.json")
    manifest = read_json(root / project.get("manifest", "input-manifest.json"))
    assets = {a["id"]: a for a in manifest["assets"]}
    errors, warnings, pending = [], [], []
    if len(assets) != len(manifest["assets"]):
        errors.append("Manifest contains duplicate asset IDs.")
    source_checks = []
    checked_paths = {}
    for asset in assets.values():
        path = Path(asset["path"])
        if not path.is_file():
            row = dict(asset_id=asset["id"], status="MISSING", path=str(path))
            if asset.get("optional") and asset.get("status") == "missing":
                warnings.append(f"Optional asset remains missing: {asset['id']}")
            else:
                errors.append(f"Registered source is missing: {asset['id']}")
        else:
            if path not in checked_paths:
                before = path.stat()
                checksum = sha256(path)
                after = path.stat()
                checked_paths[path] = (checksum, (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns))
            checksum, stable = checked_paths[path]
            if not stable:
                errors.append(f"Source changed while auditing: {asset['id']}")
            if not asset.get("sha256"):
                pending.append(f"Previously missing asset now exists; register its hash before use: {asset['id']}")
                row = dict(asset_id=asset["id"], status="UNREGISTERED_CONTENT", sha256=checksum)
            elif checksum != asset["sha256"]:
                errors.append(f"Source content changed: {asset['id']}")
                row = dict(asset_id=asset["id"], status="MODIFIED", expected_sha256=asset["sha256"], actual_sha256=checksum)
            else:
                row = dict(asset_id=asset["id"], status="UNCHANGED", sha256=checksum)
        source_checks.append(row)
    if project.get("fps") != FPS:
        errors.append("Project fps must remain 60.")
    shots_path = args.shots or Path(project.get("shots_file", "shots.json"))
    if not shots_path.is_absolute():
        shots_path = root / shots_path
    image_checks, repeated, transitions = [], [], []
    shots, frames = [], None
    if not shots_path.is_file():
        pending.append(f"Shot plan not present: {shots_path}")
    else:
        document = read_json(shots_path)
        if not isinstance(document, dict):
            errors.append("shots.json must be an object with fps, frames and shots.")
            document = {}
        if document.get("fps") != FPS:
            errors.append("Shot plan must explicitly declare fps: 60.")
        frames = document.get("frames", project.get("frames"))
        if frames is None:
            pending.append("Final frame count has not been declared.")
        elif not is_frame(frames) or frames <= 0:
            errors.append("frames must be a positive integer, derived from this project.")
            frames = None
        elif project.get("frames") is not None and project["frames"] != frames:
            errors.append("Project and shot plan frame counts disagree.")
        shots = document.get("shots", [])
        if not isinstance(shots, list):
            errors.append("shots must be an array.")
            shots = []
        if not shots:
            pending.append("No shots have been planned.")
        ids, used_images, intervals, valid_shots = set(), set(), {}, []

        def source_reference(item, label, montage=False):
            if any(k in item for k in ("source_start", "source_end")):
                errors.append(f"{label}: unlabelled source_start/end units are forbidden; use source frame fields.")
            source = item.get("source", item.get("source_interval"))
            if "source" in item and "source_interval" in item:
                errors.append(f"{label}: declare only one source interval object.")
            if source is not None:
                if not isinstance(source, dict):
                    errors.append(f"{label}: source must be an object.")
                    return
                if any(k in source for k in ("start", "end")):
                    errors.append(f"{label}: source.start/end must be named start_frame/end_frame.")
                asset_id, a, b = source.get("asset_id"), source.get("start_frame"), source.get("end_frame")
                hold = source.get("hold_frame")
            else:
                asset_id = item.get("asset_id")
                a, b = item.get("source_start_frame"), item.get("source_end_frame")
                hold = item.get("hold_frame")
            if asset_id is None:
                if montage:
                    pending.append(f"{label}: montage needs an original asset_id and source frame interval.")
                return
            if asset_id not in assets:
                errors.append(f"{label}: unknown asset_id {asset_id}.")
                return
            asset = assets[asset_id]
            if asset["role"] == "image":
                used_images.add(asset_id)
                if montage:
                    errors.append(f"{label}: a still image cannot count as video montage material.")
            streams = asset.get("metadata", {}).get("ffprobe", {}).get("streams", [])
            dynamic = any(s.get("codec_type") in ("video", "audio") for s in streams)
            if dynamic and not (item.get("clips") and source is None and a is None and b is None and hold is None):
                duration = media_duration(asset)
                if duration is None:
                    pending.append(f"{label}: registered media duration is unavailable; source bounds cannot be verified.")
                # Half-open intervals address the 60 fps adapted source timeline.
                # A hold is an explicit single source frame, not an unbounded interval.
                limit = math.ceil(duration * FPS - 1e-6) if duration is not None else None
                if hold is not None:
                    if a is not None or b is not None or not is_frame(hold):
                        errors.append(f"{label}: hold_frame must be a single nonnegative integer, without interval bounds.")
                    elif limit is not None and hold >= limit:
                        errors.append(f"{label}: source hold_frame exceeds the registered media duration.")
                    if not any(s.get("codec_type") == "video" for s in streams):
                        errors.append(f"{label}: hold_frame requires a video source.")
                elif not is_frame(a) or not is_frame(b) or b <= a:
                    errors.append(f"{label}: dynamic source needs increasing nonnegative integer start_frame/end_frame, or explicit hold_frame.")
                elif limit is not None and b > limit:
                    errors.append(f"{label}: source interval exceeds the registered media duration.")
            if montage:
                if not any(stream.get("codec_type") == "video" for stream in streams):
                    errors.append(f"{label}: montage source has no registered video stream.")
                if hold is not None:
                    errors.append(f"{label}: a held frame cannot count as video montage material.")
                    return
                if not is_frame(a) or not is_frame(b) or b <= a:
                    pending.append(f"{label}: valid source start_frame/end_frame are required for montage reuse audit.")
                    return
                identity = asset.get("sha256") or asset["path"]
                intervals.setdefault(identity, []).append(dict(asset_id=asset_id, shot=label, start_frame=a, end_frame=b))

        cursor = 0
        for index, shot in enumerate(shots):
            if not isinstance(shot, dict):
                errors.append(f"Shot {index + 1} is not an object.")
                continue
            label = shot.get("id", f"shot-{index + 1}")
            if label in ids:
                errors.append(f"Duplicate shot id: {label}")
            ids.add(label)
            a, b = shot.get("start_frame"), shot.get("end_frame")
            kind = shot.get("kind")
            if kind not in SHOT_KINDS:
                errors.append(f"{label}: unsupported or missing shot kind {kind}.")
            if not is_frame(a) or not is_frame(b) or b <= a:
                errors.append(f"{label}: shot bounds must be increasing nonnegative integer frames.")
            else:
                if a != cursor:
                    errors.append(f"{label}: shot continuity gap/overlap, expected {cursor}, got {a}.")
                cursor = b
                valid_shots.append(shot)
            clips = shot.get("clips", [])
            if clips and not isinstance(clips, list):
                errors.append(f"{label}: clips must be an array.")
                clips = []
            source_reference(shot, label, kind == "montage" and not clips)
            for clip_index, clip in enumerate(clips):
                if not isinstance(clip, dict):
                    errors.append(f"{label}: clip {clip_index + 1} is not an object.")
                    continue
                source_reference(clip, f"{label}/clip-{clip_index + 1}", kind == "montage")
            if kind == "transition":
                transitions.append(shot)
        if frames is not None and shots and cursor != frames:
            errors.append(f"Shot coverage ends at {cursor}, expected {frames}.")
        for usage in document.get("image_usage", []):
            if not isinstance(usage, dict) or usage.get("asset_id") not in assets:
                errors.append("image_usage contains an unknown or missing asset_id.")
                continue
            asset_id = usage["asset_id"]
            if assets[asset_id]["role"] != "image":
                errors.append(f"image_usage refers to a non-image asset: {asset_id}")
            elif usage.get("shot_id") in ids or isinstance(usage.get("location"), str) and usage["location"].strip():
                used_images.add(asset_id)
            else:
                errors.append(f"Image {asset_id} usage needs a valid shot_id or an explicit appendix/delivery location.")
        if shots:
            used_hashes = {assets[i].get("sha256") for i in used_images if assets[i].get("sha256")}
            for asset in assets.values():
                if asset["role"] != "image":
                    continue
                covered = asset["id"] in used_images or asset.get("sha256") in used_hashes
                image_checks.append(dict(asset_id=asset["id"], covered=covered,
                                         via_identical_content=covered and asset["id"] not in used_images))
                if not covered:
                    errors.append(f"Original image has no recorded usage: {asset['id']}")
        for content_hash, uses in intervals.items():
            uses.sort(key=lambda x: (x["start_frame"], x["end_frame"]))
            for index, first in enumerate(uses):
                for second in uses[index + 1:]:
                    if second["start_frame"] >= first["end_frame"]:
                        break
                    overlap = min(first["end_frame"], second["end_frame"]) - max(first["start_frame"], second["start_frame"])
                    if overlap > 0:
                        repeated.append(dict(content_identity=content_hash, first=first, second=second, overlapping_source_frames=overlap))
                        errors.append(f"Repeated montage source interval: {first['shot']} and {second['shot']} ({overlap} frames).")
        transitions.extend(document.get("transitions", []))
        seen_transitions = {}
        transition_results = []
        for index, transition in enumerate(transitions):
            if not isinstance(transition, dict):
                errors.append("Transition record is not an object.")
                continue
            label = transition.get("id", f"transition-{index + 1}")
            a, b = transition.get("start_frame"), transition.get("end_frame")
            key = (label, a, b)
            if key in seen_transitions:
                if seen_transitions[key] != (transition.get("presenter_visible"), transition.get("subtitle_visible")):
                    errors.append(f"{label}: duplicate transition declarations disagree about layer visibility.")
                continue
            seen_transitions[key] = (transition.get("presenter_visible"), transition.get("subtitle_visible"))
            bounds_ok = is_frame(a) and is_frame(b) and b > a and (frames is None or b <= frames)
            presenter_hidden = transition.get("presenter_visible") is False
            subtitle_blank = transition.get("subtitle_visible") is False
            if not bounds_ok:
                errors.append(f"{label}: invalid transition frame bounds.")
            if not presenter_hidden or not subtitle_blank:
                errors.append(f"{label}: chapter core requires presenter_visible=false and subtitle_visible=false.")
            # An overlay record may disable both layers even when its enclosing
            # source shot normally shows them. This remains a structural contract.
            transition_results.append(dict(id=label, start_frame=a, end_frame=b,
                                           presenter_hidden=presenter_hidden, subtitle_blank=subtitle_blank))
        transitions = transition_results
    status = "FAIL" if errors else ("NOT_READY" if pending else "STRUCTURE_PASS")
    report = dict(schema_version=1, status=status, scope="STRUCTURE_ONLY", checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  project=str(root), shots_file=str(shots_path), fps=FPS, frames=frames, shots=len(shots),
                  source_checks=source_checks, image_coverage=image_checks, montage_repeated_intervals=repeated,
                  transitions=transitions, errors=errors, pending=pending, warnings=warnings,
                  limitations=["This does not validate rendered video/audio, ASR, lip sync, subtitle pixels or chapter alpha.",
                               "Original-image coverage means registered image assets have declared usage; PDF-embedded figures must be extracted/registered before this check can cover them.",
                               "Montage reuse is checked by original asset SHA256 and 60 fps source intervals. Derivatives must retain the original source asset identity; visually identical re-encodes cannot be inferred from different hashes.",
                               "Operation teaching revisits and figure explanations are intentionally excluded from montage de-duplication."])
    dump_json(root / "qa/structure-audit.json", report)
    return report


def _synthetic_avi(path, frames=120):
    """Small uncompressed 2x2 AVI produced with stdlib only, no real imagery."""
    def chunk(name, data):
        return name + struct.pack("<I", len(data)) + data + (b"\0" if len(data) & 1 else b"")
    def listing(kind, data):
        return chunk(b"LIST", kind + data)
    avih = struct.pack("<14I", 16667, 960, 0, 0, frames, 0, 1, 16, 2, 2, 0, 0, 0, 0)
    strh = struct.pack("<4s4sIHHIIIIIIIIhhhh", b"vids", b"DIB ", 0, 0, 0, 0, 1, 60, 0, frames, 16, 0xFFFFFFFF, 0, 0, 0, 2, 2)
    strf = struct.pack("<IiiHHIIiiII", 40, 2, 2, 1, 24, 0, 16, 0, 0, 0, 0)
    hdrl = listing(b"hdrl", chunk(b"avih", avih) + listing(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf)))
    movi = listing(b"movi", b"".join(chunk(b"00db", bytes([i % 256, 100, 180, 20, 90, 110, 0, 0]) * 2) for i in range(frames)))
    data = b"AVI " + hdrl + movi
    path.write_bytes(b"RIFF" + struct.pack("<I", len(data)) + data)


def self_test():
    with tempfile.TemporaryDirectory(prefix="course-project-synthetic-") as temporary:
        root = Path(temporary)
        video, duplicate = root / "synthetic.avi", root / "same-content.avi"
        _synthetic_avi(video)
        shutil.copyfile(video, duplicate)
        script = root / "lesson.txt"
        script.write_text("Synthetic lesson outline, with no real person or course content.")
        image = root / "diagram.png"
        def png_chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + png_chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)) + png_chunk(b"IDAT", zlib.compress(b"\0\x20\x80\xa0")) + png_chunk(b"IEND", b""))
        project = root / "project"
        args = argparse.Namespace(project=project, title="Synthetic test", mode="mixed", narration=[], script=[script], operation=[], image=[image], video=[video, video, duplicate], bgm=root / "missing-optional.wav")
        initialized = register_project(args)
        assert initialized["assets"] == 5
        resolved_style = read_json(project / "project-style.json")
        assert initialized["profile"] == "practical"
        assert resolved_style["course_mode"] == "mixed"
        assert resolved_style["layout"]["lecturer_circle"] == [2164, 945, 340]
        assert resolved_style["layout"]["operation_view"] == [56, 180, 2128, 1272]
        assert resolved_style["audio"]["music_max_gain"] == .07
        assert resolved_style["audio"]["speech_music_target_gain"] == .035
        assert read_json(project / "project.json")["style_file"] == "project-style.json"
        assert project_style("theory")["layout"]["lecturer_circle"] == [56, 945, 340]
        overrides = root / "style-overrides.json"
        dump_json(overrides, {"audio": {"music_max_gain": .06}, "layout": {"lecturer_circle": [2164, 940, 340]}})
        custom = project_style("practical", overrides)
        assert custom["audio"]["music_max_gain"] == .06
        assert custom["audio"]["speech_music_target_gain"] == .035
        assert custom["layout"]["lecturer_circle"] == [2164, 940, 340]
        assert custom["layout"]["operation_view"] == [56, 180, 2128, 1272]
        manifest = read_json(project / "input-manifest.json")
        assert len([a for a in manifest["assets"] if a["role"] == "video"]) == 2
        assert next(a for a in manifest["assets"] if a["id"] == "video-002")["duplicate_of"] == "video-001"
        assert next(a for a in manifest["assets"] if a["id"] == "video-001")["metadata"]["ffprobe"]["streams"][0]["width"] == 2
        folder=root/'input-folder'; (folder/'nested').mkdir(parents=True)
        shutil.copy2(image,folder/'a.png');shutil.copy2(image,folder/'nested/b.png')
        (folder/'.DS_Store').write_bytes(b'ignored folder metadata')
        folder_args=argparse.Namespace(project=root/'folder-project',title='Folder sources',mode='theory',
            narration=[],script=[],operation=[],image=[folder],video=[],bgm=None)
        folder_result=register_project(folder_args)
        folder_assets=read_json(folder_args.project/'input-manifest.json')['assets']
        assert folder_result['assets']==2 and all(a['role']=='image' for a in folder_assets)
        assert not (folder_args.project/'assets/a.png').exists()
        assert folder_result['missing_optional_assets']==[]
        audit_args = argparse.Namespace(project=project, shots=None)
        assert audit_project(audit_args)["status"] == "NOT_READY"
        planned = dict(fps=60, frames=258, shots=[
            dict(id="m1", kind="montage", start_frame=0, end_frame=60, clips=[dict(source=dict(asset_id="video-001", start_frame=0, end_frame=60))]),
            dict(id="m2", kind="montage", start_frame=60, end_frame=120, source=dict(asset_id="video-002", start_frame=60, end_frame=120), speed=2., crop=[.1,.1,.8,.8]),
            dict(id="figure", kind="figure", start_frame=120, end_frame=180, asset_id="image-001"),
            dict(id="chapter", kind="transition", start_frame=180, end_frame=258, presenter_visible=False, subtitle_visible=False)], transitions=[], image_usage=[])
        dump_json(project / "shots.json", planned)
        assert audit_project(audit_args)["status"] == "STRUCTURE_PASS"
        planned["shots"][2].update(kind="reading", presenter_visible=False, subtitle_visible=False)
        dump_json(project / "shots.json", planned)
        assert audit_project(audit_args)["status"] == "STRUCTURE_PASS"
        planned["shots"][2]["kind"] = "figure"
        planned["shots"][1]["source"].update(start_frame=30, end_frame=90)
        dump_json(project / "shots.json", planned)
        repeat = audit_project(audit_args)
        assert repeat["status"] == "FAIL" and repeat["montage_repeated_intervals"][0]["overlapping_source_frames"] == 30
        planned["shots"][0]["kind"] = planned["shots"][1]["kind"] = "operation"
        dump_json(project / "shots.json", planned)
        assert audit_project(audit_args)["status"] == "STRUCTURE_PASS"
        planned["shots"][1]["source"]["end_frame"] = 999999
        dump_json(project / "shots.json", planned)
        assert any("exceeds the registered media duration" in e for e in audit_project(audit_args)["errors"])
        planned["shots"][1]["source"]["end_frame"] = 90
        planned["shots"][0]["clips"][0]["source"]["end_frame"] = 999999
        dump_json(project / "shots.json", planned)
        assert any("exceeds the registered media duration" in e for e in audit_project(audit_args)["errors"])
        planned["shots"][0]["clips"][0]["source"]["end_frame"] = 60
        saved_source = planned["shots"][1]["source"]
        planned["shots"][1]["source"] = dict(asset_id="video-002", hold_frame=119)
        dump_json(project / "shots.json", planned)
        assert audit_project(audit_args)["status"] == "STRUCTURE_PASS"
        planned["shots"][1]["source"]["hold_frame"] = 120
        dump_json(project / "shots.json", planned)
        assert any("source hold_frame exceeds" in e for e in audit_project(audit_args)["errors"])
        planned["shots"][1]["source"] = saved_source
        planned["shots"][-1]["presenter_visible"] = True
        dump_json(project / "shots.json", planned)
        assert any("chapter core" in e for e in audit_project(audit_args)["errors"])
        planned["shots"][-1]["presenter_visible"] = False
        planned["shots"][2].pop("asset_id")
        dump_json(project / "shots.json", planned)
        assert any("Original image" in e for e in audit_project(audit_args)["errors"])
        planned["shots"][2]["asset_id"] = "image-001"
        planned["shots"][2]["start_frame"] = 121
        dump_json(project / "shots.json", planned)
        assert any("continuity" in e for e in audit_project(audit_args)["errors"])
        planned["shots"][2]["start_frame"] = 120
        dump_json(project / "shots.json", planned)
        script.write_text("Synthetic source modified.")
        assert any("Source content changed" in e for e in audit_project(audit_args)["errors"])
        image.unlink()
        assert any("source is missing" in e for e in audit_project(audit_args)["errors"])
        original_project = (project / "project.json").read_bytes()
        try:
            register_project(args)
        except ValueError as exc:
            assert "already exists" in str(exc)
        else:
            raise AssertionError("Existing project should not be overwritten.")
        assert (project / "project.json").read_bytes() == original_project
        return dict(status="PASS", cases=["Mode-selected practical/mixed and preserved theory profiles", "Project-local resolved style and recursive explicit overrides", "7% music / 3.5% speech defaults", "Reading shots accepted", "Real ffprobe of stdlib-generated synthetic AVI", "Recursive role-compatible directory discovery without hidden metadata or source copying", "Omitted BGM registers no imaginary music", "Same path/role de-duplication", "Identical-content duplicate_of", "Missing optional music accepted", "Missing shots NOT_READY", "Valid variable-length structure", "Repeated montage interval despite speed/crop and copied file", "Operation revisits allowed", "Operation source and nested clip range enforced", "Explicit hold_frame allowed only within video duration", "Hidden presenter/blank subtitle chapter contract", "Original-image usage coverage", "Shot gap detection", "Source modification/missing detection", "Existing project refusal without overwrite"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Register a new project; never copy original inputs.")
    init.add_argument("--project", type=Path, required=True)
    init.add_argument("--title", required=True)
    init.add_argument("--mode", choices=("theory", "practical", "mixed"), required=True)
    init.add_argument("--style-overrides", type=Path, help="Optional JSON object merged into this lesson's project-style.json; never edits the reusable skill.")
    for role in ROLES[:-1]:
        init.add_argument("--" + role, type=Path, action="append", default=[], metavar="PATH")
    init.add_argument("--bgm", type=Path, help="Optional explicit licensed BGM file. No music is assumed when omitted.")
    audit = commands.add_parser("audit", help="Check project structure, not final media or speech.")
    audit.add_argument("--project", type=Path, required=True)
    audit.add_argument("--shots", type=Path, help="Optional shot-plan path; relative paths are project-relative.")
    commands.add_parser("self-test", help="Use only temporary synthetic AVI/PNG/text inputs; requires ffprobe.")
    args = parser.parse_args(argv)
    try:
        result = register_project(args) if args.command == "init" else audit_project(args) if args.command == "audit" else self_test()
        if args.command == "audit":
            print(json.dumps({k: result[k] for k in ("status", "scope", "fps", "frames", "shots", "errors", "pending", "warnings")}, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 1 if result["status"] == "FAIL" else 2 if result["status"] == "NOT_READY" else 0
    except Exception as exc:
        print(json.dumps(dict(status="FAIL", error=f"{type(exc).__name__}: {exc}"), ensure_ascii=False, indent=2))
        return 1


if __name__ == "__main__":
    sys.exit(main())
