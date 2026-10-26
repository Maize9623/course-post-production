#!/usr/bin/env python3
"""CPU-only final-media QA for a variable-length 2560x1440 / 60 fps lesson.

Required: --final MP4 --shots shots.json --reference-audio mixed.wav --out qa.json
Optional: --captions captions-edited.json --srt captions.srt --chapters chapters.json
          --project-style project-style.json --ffmpeg PATH --ffprobe PATH
Run --self-test for temporary synthetic fixtures, with no real speech or images.

shots.json uses fps=60, frames=N and contiguous shots with integer start_frame /
end_frame. Chapter cores come from kind=transition shots or transitions[]. They
have presenter_visible=false, subtitle_visible=false, and normally 78 frames.
--project-style supplies the actual lecturer circle, subtitle band, background
and blank guards. Every core/guard frame is checked. Without explicit complete
configuration only the old left-side core scope is checked, with REVIEW status;
it can never certify the new practical-course standard.
All intervals are half open. Inputs are read-only. Scratch decodes are deleted.
TECHNICAL_PASS is not a judgment about wording, teaching quality, or appearance.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import wave

try:
    import numpy as np
except ImportError:
    np = None

FPS, RATE, MATCH_RATE = 60, 48000, 16000
WIDTH, HEIGHT = 2560, 1440
BACKGROUND = "#F5F7F5"
LIMITATIONS = [
    "TECHNICAL_PASS covers the numerical checks listed here, not language, ASR, teaching accuracy, subjective viewing quality, or visible lip motion.",
    "Audio is compared to the supplied already-edited mixed master. This cannot prove that the master itself contains the correct original words or source edits.",
    "Audio waveform checks use mono 16 kHz CPU decodes. Very quiet blocks (reference RMS <= 0.002 FS) and frequencies above this comparison bandwidth are not certified by correlation.",
    "Video packet PTS/durations are checked for the entire track. Chapter pixel checks cover every declared core/guard frame in the configured lecturer circle and subtitle band, not every pixel of every lesson frame.",
    "Clean configured regions prove only that those regions match the expected flat chapter background. They cannot detect a person elsewhere, all compositing defects, an undeclared chapter, or verify the final file's actual composition against an incorrectly supplied configuration.",
    "Without a complete explicit project-style configuration, chapter checks are marked REVIEW and limited to the reported legacy scope.",
    "Caption checks validate supplied time/text records and optional SRT consistency, without OCR or proof that the rendered subtitles match those records.",
]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def identity(path):
    p = Path(path).resolve()
    s = p.stat()
    h = hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):h.update(block)
    end = p.stat()
    if (s.st_size,s.st_mtime_ns,s.st_ino)!=(end.st_size,end.st_mtime_ns,end.st_ino):
        raise ValueError(f'Input changed while hashing: {p}')
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns, sha256=h.hexdigest())


def atomic_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix=".qa-", suffix=".json", dir=p.parent, delete=False) as f:
        temporary = Path(f.name)
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    try:
        temporary.replace(p)
    finally:
        temporary.unlink(missing_ok=True)


def executable(name, override=None):
    choices = [str(override)] if override else [shutil.which(os.environ.get(name.upper()) or name)]
    for candidate in choices:
        if candidate and Path(candidate).is_file():
            return str(Path(candidate).resolve())
    raise ValueError(f"{name} executable not found. Supply --{name} PATH or add it to PATH.")


def run(command):
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        detail = result.stderr.decode("utf-8", "replace")[-4000:]
        raise RuntimeError(f"{Path(command[0]).name} failed (exit {result.returncode}): {detail}")
    return result.stdout


def probe(path, ffprobe):
    return json.loads(run([ffprobe, "-v", "error", "-threads", "2", "-show_format", "-show_streams", "-show_chapters", "-of", "json", str(path)]))


def result(errors=(), reviews=(), **details):
    return dict(status="FAIL" if errors else "REVIEW" if reviews else "PASS", errors=list(errors), review=list(reviews), **details)


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def chapter_config(path=None):
    """An explicit complete lesson configuration is required for a full PASS."""
    style = read_json(path) if path else {}
    if not isinstance(style, dict):
        raise ValueError("Project style must be a JSON object.")
    layout, colors, transitions = (style.get(key, {}) for key in ("layout", "colors", "transitions"))
    if any(not isinstance(value, dict) for value in (layout, colors, transitions)):
        raise ValueError("Project style layout, colors and transitions must be objects.")
    reviews = []
    required = [(layout, "lecturer_circle"), (layout, "subtitle_band"), (colors, "background"),
                (transitions, "presenter_guard_before_frames"), (transitions, "presenter_guard_after_frames"),
                (transitions, "lecturer_visible_during_guards"), (transitions, "subtitle_visible_during_guards")]
    if not path:
        reviews.append("No --project-style supplied: only the legacy left lecturer circle and chapter cores are certified; new core-plus-guard standard is not certified.")
    elif any(key not in container for container, key in required):
        reviews.append("Incomplete project-style chapter geometry/guard configuration; fallback values are reported and the new standard is not certified.")
    circle = layout.get("lecturer_circle", [56, 945, 340])
    band = layout.get("subtitle_band", [0, 1280, 2560, 160])
    if not isinstance(circle, list) or len(circle) != 3 or not all(integer(x) for x in circle) or not circle[2] or circle[0] + circle[2] > WIDTH or circle[1] + circle[2] > HEIGHT:
        raise ValueError("lecturer_circle must be [x,y,positive diameter] inside the 2560x1440 canvas.")
    if not isinstance(band, list) or len(band) != 4 or not all(integer(x) for x in band) or not band[2] or not band[3] or band[0] + band[2] > WIDTH or band[1] + band[3] > HEIGHT:
        raise ValueError("subtitle_band must be [x,y,positive width,positive height] inside the canvas.")
    background = colors.get("background", BACKGROUND)
    if not isinstance(background, str) or not re.fullmatch(r"#[0-9A-Fa-f]{6}", background):
        raise ValueError("Project chapter background must be an explicit #RRGGBB color.")
    before = transitions.get("presenter_guard_before_frames", 0)
    after = transitions.get("presenter_guard_after_frames", 0)
    core = transitions.get("frames", 78)
    if not integer(before) or not integer(after) or not integer(core) or not core:
        raise ValueError("Chapter core and guard lengths must be integer frames, with a positive core length.")
    for key in ("lecturer_visible_during_guards", "subtitle_visible_during_guards"):
        if key in transitions and transitions[key] is not False:
            raise ValueError(f"{key} must be false for the series chapter standard.")
    canvas = style.get("canvas", {})
    if not isinstance(canvas, dict) or any(canvas.get(key, default) != default for key, default in (("width", WIDTH), ("height", HEIGHT), ("fps", FPS))):
        raise ValueError("Project style canvas must remain 2560x1440 at 60 fps.")
    return dict(profile=style.get("profile", "legacy"), circle=circle, subtitle_band=band,
                background=background.upper(), background_rgb=[int(background[i:i+2], 16) for i in (1, 3, 5)],
                guard_before_frames=before, guard_after_frames=after, core_frames=core,
                explicit_complete=bool(path) and not reviews, legacy_core_only=not path,
                configuration_file=str(Path(path).resolve()) if path else None, review=reviews)


def fraction(value):
    if value is None or isinstance(value, bool):
        raise ValueError(f"Missing or invalid numeric value: {value}")
    return Fraction(str(value))


def interval(item, frames, label, allow_seconds=False):
    if not isinstance(item, dict):
        raise ValueError(f"{label} must be an object.")
    if "start_frame" in item or "end_frame" in item:
        a, b = item.get("start_frame"), item.get("end_frame")
        if not integer(a) or not integer(b):
            raise ValueError(f"{label} start_frame/end_frame must be nonnegative integers.")
        a, b = Fraction(a, FPS), Fraction(b, FPS)
        for name, expected in (("start", a), ("end", b)):
            if name in item and abs(fraction(item[name]) - expected) > Fraction(1, 100000):
                raise ValueError(f"{label} has inconsistent frame and second fields.")
    elif allow_seconds:
        a, b = fraction(item.get("start")), fraction(item.get("end"))
    else:
        raise ValueError(f"{label} requires explicit start_frame/end_frame.")
    if not 0 <= a < b <= Fraction(frames, FPS):
        raise ValueError(f"{label} interval is empty, reversed, or outside the timeline.")
    return a, b


def check_shots(document, config=None):
    config = config or chapter_config()
    errors, reviews, transitions = [], [], []
    if not isinstance(document, dict):
        raise ValueError("shots JSON must be an object.")
    if document.get("fps") != FPS:
        raise ValueError("shots.fps must be 60.")
    frames = document.get("frames")
    if not integer(frames) or not frames:
        raise ValueError("shots.frames must be a positive integer for this lesson.")
    for k, expected in (("width", WIDTH), ("height", HEIGHT)):
        if k in document and document[k] != expected:
            errors.append(f"shots.{k} must be {expected}.")
    shots = document.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("shots.shots must be a nonempty array.")
    cursor, ids = Fraction(0), set()
    chapter_records = []
    for i, item in enumerate(shots):
        label = str(item.get("id", f"shot-{i + 1}")) if isinstance(item, dict) else str(i)
        try:
            a, b = interval(item, frames, label)
            if a != cursor:
                errors.append(f"{label}: shot gap/overlap at frame {float(a * FPS):g}; expected {float(cursor * FPS):g}.")
            cursor = b
            if label in ids:
                errors.append(f"Duplicate shot id: {label}")
            ids.add(label)
            if item.get("kind") == "transition":
                chapter_records.append(item)
        except ValueError as exc:
            errors.append(str(exc))
    if cursor != Fraction(frames, FPS):
        errors.append("Shots do not cover the complete declared frame range.")
    extra = document.get("transitions", [])
    if not isinstance(extra, list):
        errors.append("transitions must be an array.")
    else:
        chapter_records.extend(extra)
    seen = {}
    for i, item in enumerate(chapter_records):
        label = str(item.get("id", f"transition-{i + 1}")) if isinstance(item, dict) else str(i)
        try:
            a, b = interval(item, frames, label)
            af, bf = int(a * FPS), int(b * FPS)
            bg = item.get("background_color", item.get("background", document.get("chapter_background", config["background"])))
            flags = (item.get("presenter_visible"), item.get("subtitle_visible"), bg)
            if (af, bf) in seen:
                if seen[(af, bf)] != flags:
                    errors.append(f"{label}: duplicate chapter declarations disagree.")
                continue
            seen[(af, bf)] = flags
            if flags[:2] != (False, False):
                errors.append(f"{label}: chapter core requires presenter_visible=false and subtitle_visible=false.")
            if bf - af != config["core_frames"]:
                reviews.append(f"{label}: chapter core has {bf - af} frames; configured default is {config['core_frames']}. All declared frames will still be checked.")
            supported = isinstance(bg, str) and bg.upper() == config["background"]
            if not supported:
                reviews.append(f"{label}: background {bg!r} is outside the fixed ROI color-check mode; manual/new-mode review required.")
            lo = max(0, af - config["guard_before_frames"])
            hi = min(frames, bf + config["guard_after_frames"])
            transitions.append(dict(id=label, start_frame=af, end_frame=bf, background=bg, roi_supported=supported,
                                    check_start_frame=lo, check_end_frame=hi,
                                    checked_guard_before_frames=af-lo, checked_guard_after_frames=hi-bf,
                                    guard_clipped_at_timeline_start=af-config["guard_before_frames"] < 0,
                                    guard_clipped_at_timeline_end=bf+config["guard_after_frames"] > frames))
        except ValueError as exc:
            errors.append(str(exc))
    transitions.sort(key=lambda x: x["start_frame"])
    for a, b in zip(transitions, transitions[1:]):
        if b["start_frame"] < a["end_frame"]:
            errors.append(f"Chapter cores overlap: {a['id']} and {b['id']}.")
    return result(errors, reviews, fps=FPS, frames=frames, duration_seconds=frames / FPS, shot_count=len(shots), transitions=transitions)


def check_packets(packets, time_base, frames):
    errors, anomalies = [], []
    tb = fraction(time_base)
    parsed = []
    for i, packet in enumerate(packets):
        try:
            pts, duration = int(packet["pts"]), int(packet["duration"])
            if str(pts) != str(packet["pts"]) or str(duration) != str(packet["duration"]):
                raise ValueError("not integer")
            parsed.append((pts, duration))
        except (KeyError, TypeError, ValueError):
            errors.append(f"Video packet {i} lacks integer PTS/duration.")
    parsed.sort()
    if len(packets) != frames:
        errors.append(f"Video packet count {len(packets)} != timeline frames {frames}.")
    for i, (pts, duration) in enumerate(parsed):
        if pts * tb != Fraction(i, FPS) or duration * tb != Fraction(1, FPS):
            anomalies.append(dict(presentation_index=i, pts=pts, duration=duration, pts_seconds=str(pts * tb), duration_seconds=str(duration * tb)))
    if anomalies:
        errors.append(f"{len(anomalies)} video packets violate the exact 1/60-second PTS/duration grid.")
    return result(errors, packet_count=len(packets), time_base=str(tb), anomaly_count=len(anomalies), anomalies=anomalies[:100], arithmetic="Fraction(integer_ticks) * Fraction(time_base), exact; packets sorted by PTS to allow B-frame decode order.")


def check_media(metadata, frames, final, ffprobe):
    errors = []
    videos = [s for s in metadata.get("streams", []) if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")]
    audios = [s for s in metadata.get("streams", []) if s.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        return result(["Final must contain exactly one video track and one audio track."]), None
    v, a = videos[0], audios[0]
    expected = Fraction(frames, FPS)
    if (v.get("width"), v.get("height")) != (WIDTH, HEIGHT):
        errors.append("Final resolution is not 2560x1440.")
    for field in ("r_frame_rate", "avg_frame_rate"):
        if fraction(v.get(field)) != FPS:
            errors.append(f"Video {field} is not exactly 60 fps.")
    if v.get("nb_frames") not in (None, "N/A") and int(v["nb_frames"]) != frames:
        errors.append("Video stream nb_frames differs from shots.frames.")
    if int(a.get("sample_rate", 0)) != RATE or a.get("channels") != 2 or a.get("codec_name") != "aac":
        errors.append("Final audio must be AAC, 48000 Hz, stereo.")
    for label, stream, tolerance in (("video", v, Fraction(0)), ("audio", a, Fraction(0))):
        tb = fraction(stream.get("time_base"))
        actual = int(stream["duration_ts"]) * tb if stream.get("duration_ts") is not None else fraction(stream.get("duration"))
        if abs(actual - expected) > tolerance:
            errors.append(f"{label} duration {actual} differs from {expected} beyond {tolerance} seconds.")
        start = int(stream["start_pts"]) * tb if stream.get("start_pts") is not None else fraction(stream.get("start_time"))
        if abs(start) > Fraction(1, RATE):
            errors.append(f"{label} presentation start is {start}, not zero.")
    if abs(fraction(metadata.get("format", {}).get("duration")) - expected) > Fraction(1, 1000000):
        errors.append("Container duration differs from the timeline beyond printed microsecond precision.")
    packets = json.loads(run([ffprobe, "-v", "error", "-select_streams", "v:0", "-show_packets", "-show_entries", "packet=pts,duration", "-of", "json", str(final)]))["packets"]
    packet_result = check_packets(packets, v["time_base"], frames)
    return result(errors, expected_frames=frames, expected_duration_seconds=float(expected), video={k: v.get(k) for k in ("codec_name", "width", "height", "r_frame_rate", "avg_frame_rate", "nb_frames", "duration", "time_base", "start_time")}, audio={k: a.get(k) for k in ("codec_name", "sample_rate", "channels", "duration", "duration_ts", "time_base", "start_time")}, audio_duration_tolerance_seconds=0), packet_result


def check_audio_packets(metadata, frames, final, ffprobe):
    """Count presented samples after packet timestamps and declared skip/discard.

    Container packet durations describe presentation, not all physical decoded
    AAC-frame samples. No fixed tail-pad allowance may excuse missing words.
    """
    stream = next((s for s in metadata.get('streams',[]) if s.get('codec_type')=='audio'),None)
    if stream is None:return result(['No audio stream to inspect.'])
    tb = fraction(stream['time_base']); expected = frames * (RATE // FPS)
    packets = json.loads(run([ffprobe,'-v','error','-select_streams','a:0','-show_packets',
        '-show_entries','packet=pts,duration:packet_side_data=skip_samples,discard_padding','-of','json',str(final)])).get('packets',[])
    errors=[]; cursor=0; count=0; skips=0; discards=0
    for i,p in enumerate(packets):
        try:
            start=Fraction(int(p['pts']))*tb*RATE
            duration=Fraction(int(p['duration']))*tb*RATE
            if start.denominator!=1 or duration.denominator!=1 or duration<0:raise ValueError('noninteger samples')
            skip=sum(int(s.get('skip_samples',0)) for s in p.get('side_data_list',[]))
            discard=sum(int(s.get('discard_padding',0)) for s in p.get('side_data_list',[]))
            skips+=skip;discards+=discard
            a=max(0,int(start)+skip); b=int(start+duration)-discard
            if b<=a:continue
            if a!=cursor:errors.append(f'Audio packet{i} begins at sample{a}; expected contiguous sample{cursor}.')
            count+=b-a;cursor=b
        except (ValueError,KeyError,TypeError) as exc:
            errors.append(f'Audio packet{i} cannot establish exact presented samples: {exc}')
    if count!=expected or cursor!=expected:
        errors.append(f'Presented AAC samples/count end {count}/{cursor} != timeline {expected}.')
    return result(errors,packet_count=len(packets),presented_samples=count,presentation_end_sample=cursor,
        expected_samples=expected,declared_skip_samples=skips,declared_discard_padding=discards,
        method='Exact packet PTS/duration in sample units, negative priming clipped at zero and current skip/discard applied; physical AAC decoder tail is not assumed to be speech.')


def check_full_decode(final, frames, ffmpeg):
    progress=run([ffmpeg,'-hide_banner','-nostdin','-v','error','-xerror','-err_detect','explode',
        '-threads','2','-i',str(final),'-map','0:v:0','-map','0:a:0','-filter_threads','2',
        '-fps_mode','passthrough','-progress','pipe:1','-nostats','-f','null','-']).decode('utf-8','replace')
    counts=[int(line.split('=',1)[1].strip()) for line in progress.splitlines() if line.startswith('frame=')]
    actual=counts[-1] if counts else None
    errors=[] if actual==frames and 'progress=end' in progress else [f'Complete strict decode frame count/end {actual} != {frames}.']
    return result(errors,decoded_video_frames=actual,expected_video_frames=frames,audio_and_video_mapped=True,
        strict_decoder_error_exit=True,completed='progress=end' in progress,
        scope='Full audio/video decode; exact valid audio presentation is checked separately from current packet metadata.')


def check_reference(path, frames, ffprobe):
    p = probe(path, ffprobe)
    streams = p.get("streams", [])
    errors, samples = [], None
    if len(streams) != 1:
        errors.append("Reference must contain only one uncompressed PCM audio stream.")
    else:
        s = streams[0]
        if not str(s.get("codec_name", "")).startswith("pcm_") or s.get("codec_type") != "audio":
            errors.append("Reference is not uncompressed PCM audio.")
        if int(s.get("sample_rate", 0)) != RATE or s.get("channels") != 2:
            errors.append("Reference PCM must be 48000 Hz stereo.")
        if s.get("duration_ts") is not None:
            count = int(s["duration_ts"]) * fraction(s["time_base"]) * RATE
            if count.denominator != 1:
                errors.append("Reference does not expose an integer PCM sample count.")
            else:
                samples = int(count)
        else:
            errors.append("Reference PCM exact sample count unavailable.")
    if samples != frames * 800:
        errors.append(f"Reference samples {samples} != timeline frames * 800 ({frames * 800}).")
    return result(errors, samples=samples, expected_samples=frames * 800, sample_rate=RATE, channels=2)


def decode_mono(path, target, ffmpeg):
    # Preserve presentation offsets; never time-stretch the candidate to fit.
    run([ffmpeg, "-v", "error", "-threads", "2", "-copyts", "-i", str(path), "-map", "0:a:0", "-vn", "-filter_threads", "2", "-af", "aresample=16000:async=1:first_pts=0", "-ac", "1", "-c:a", "pcm_s16le", "-y", str(target)])
    with wave.open(str(target), "rb") as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (MATCH_RATE, 1, 2):
            raise ValueError("Internal comparison decode has an unexpected format.")
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.


def measures(reference, candidate):
    r, c = reference.astype(np.float64), candidate.astype(np.float64)
    rr = float(np.sqrt(np.mean(r * r)))
    cr = float(np.sqrt(np.mean(c * c)))
    r -= r.mean(); c -= c.mean()
    energy = float(np.dot(r, r) * np.dot(c, c))
    corr = float(np.dot(r, c) / math.sqrt(energy)) if energy > 1e-25 else None
    gain = 20 * math.log10(max(cr, 1e-15) / max(rr, 1e-15))
    return dict(correlation=corr, gain_db=gain, reference_rms=rr, candidate_rms=cr)


def find_match(reference, search):
    r, c = reference.astype(np.float64), search.astype(np.float64)
    count = len(r)
    r -= r.mean()
    energy = float(np.dot(r, r))
    if len(c) < count or energy < 1e-12:
        raise ValueError("Insufficient non-silent audio for a reliable lag comparison.")
    fft_size = 1 << (len(c) + count - 2).bit_length()
    dots = np.fft.irfft(np.fft.rfft(c, fft_size) * np.conj(np.fft.rfft(r, fft_size)), fft_size)
    sums, squares = np.r_[0., np.cumsum(c)], np.r_[0., np.cumsum(c * c)]
    local_energy = squares[count:] - squares[:-count] - (sums[count:] - sums[:-count]) ** 2 / count
    correlations = dots[:len(c) - count + 1] / np.sqrt(np.maximum(local_energy, 1e-20) * energy)
    offset = int(np.argmax(correlations))
    return offset, float(correlations[offset])


def check_audio(final, reference_path, frames, transitions, temporary, ffmpeg):
    reference = decode_mono(reference_path, temporary / "reference.wav", ffmpeg)
    candidate = decode_mono(final, temporary / "candidate.wav", ffmpeg)
    errors, reviews = [], []
    expected_seconds = frames / FPS
    tolerance = 1024 / RATE + 1 / MATCH_RATE
    if abs(len(reference) / MATCH_RATE - expected_seconds) > 1 / MATCH_RATE:
        errors.append("Reference resample length differs from the timeline.")
    if abs(len(candidate) / MATCH_RATE - expected_seconds) > tolerance:
        errors.append("Decoded final audio duration differs beyond one AAC packet plus one comparison sample.")
    if len(candidate) < len(reference):
        errors.append(f"Candidate is {len(reference) - len(candidate)} comparison samples shorter than reference; full tail cannot be certified.")
    count = min(len(reference), len(candidate))
    rows, quiet, uncertain = [], [], []
    for begin in range(0, count, MATCH_RATE):
        end = min(begin + MATCH_RATE, count)
        m = measures(reference[begin:end], candidate[begin:end])
        row = dict(start_sample=begin, end_sample=end, start_seconds=begin / MATCH_RATE, duration_seconds=(end - begin) / MATCH_RATE, **m)
        if m["reference_rms"] <= .002:
            quiet.append(row)
        elif m["correlation"] is None:
            uncertain.append(row)
        else:
            rows.append(row)
    bad = [x for x in rows if x["correlation"] < .95 or abs(x["gain_db"]) > .5]
    if bad:
        errors.append(f"{len(bad)} whole-track 1-second/tail blocks fail zero-delay correlation >= .95 or absolute gain <= .5 dB.")
    if uncertain:
        reviews.append(f"{len(uncertain)} active blocks have insufficient variance for Pearson correlation.")
    if not rows:
        reviews.append("No active reference blocks; audio identity/synchronization cannot be established.")
    # Include each tail, even below 100 ms; no floor-duration omission.
    scan = dict(block_seconds=1, minimum_reference_rms=.002, examined_blocks=len(rows) + len(quiet) + len(uncertain), active_checked_blocks=len(rows), quiet_unchecked_blocks=len(quiet), checked_duration_seconds=sum(x["duration_seconds"] for x in rows), examined_duration_seconds=count / MATCH_RATE, tail_duration_seconds=(count % MATCH_RATE) / MATCH_RATE, minimum_correlation=min((x["correlation"] for x in rows), default=None), maximum_absolute_gain_db=max((abs(x["gain_db"]) for x in rows), default=None), anomalous_blocks=bad, insufficient_variance_blocks=uncertain, quiet_intervals=[dict(start_seconds=x["start_seconds"], duration_seconds=x["duration_seconds"]) for x in quiet])
    # Fixed distributed targets, plus every chapter before/inside/after. Quiet
    # targets are reported instead of silently moving them to a favorable spot.
    window = min(round(2.5 * MATCH_RATE), count)
    starts = {int(round(x)): [f"distributed-{i + 1}"] for i, x in enumerate(np.linspace(0, max(0, count - window), 14))}
    for t in transitions:
        a, b = t["start_frame"] / FPS, t["end_frame"] / FPS
        for suffix, center in (("before", a - 1.3), ("inside", (a + b) / 2), ("after", b + 1.3)):
            start = max(0, min(count - window, round(center * MATCH_RATE - window / 2)))
            starts.setdefault(start, []).append(t["id"] + "-" + suffix)
    windows, skipped = [], []
    radius = round(.1 * MATCH_RATE)
    for begin, labels in sorted(starts.items()):
        ref = reference[begin:begin + window]
        if not len(ref) or float(np.sqrt(np.mean(ref * ref))) <= .002 or float(np.var(ref)) < 1e-12:
            skipped.append(dict(labels=labels, start_seconds=begin / MATCH_RATE, reason="Reference too quiet/constant to prove lag."))
            continue
        left, right = max(0, begin - radius), min(len(candidate), begin + window + radius)
        offset, correlation = find_match(ref, candidate[left:right])
        lag = left + offset - begin
        m = measures(ref, candidate[left + offset:left + offset + window])
        passed = correlation >= .97 and abs(lag) <= 16 and abs(m["gain_db"]) <= .5
        windows.append(dict(labels=labels, start_seconds=begin / MATCH_RATE, duration_seconds=window / MATCH_RATE, lag_samples=lag, lag_ms=lag / 16., correlation=correlation, gain_db=m["gain_db"], status="PASS" if passed else "FAIL"))
    if any(w["status"] == "FAIL" for w in windows):
        errors.append("Distributed/chapter audio windows fail correlation, gain, or <= 1 ms absolute lag.")
    lags = [w["lag_samples"] for w in windows]
    spread = (max(lags) - min(lags)) / 16. if lags else None
    if spread is not None and spread > 1:
        errors.append("Lag spread across the lesson exceeds 1 ms.")
    if not windows:
        reviews.append("No reliable distributed window; audio drift cannot be established.")
    return result(errors, reviews, reference_samples_16k=len(reference), final_samples_16k=len(candidate), duration_tolerance_seconds=tolerance, waveform_scan=scan, lag_windows=windows, quiet_lag_windows=skipped, lag_spread_ms=spread, thresholds=dict(full_scan_min_correlation=.95, lag_window_min_correlation=.97, maximum_absolute_gain_db=.5, maximum_absolute_lag_ms=1, maximum_lag_spread_ms=1), comparison="Mean-removed Pearson correlation, with separate RMS gain check. Full scan uses zero lag; distributed windows search +/-100 ms and must resolve to within 1 ms.")


def check_roi(final, transitions, ffmpeg, config=None):
    config = config or chapter_config()
    errors, reviews, rows = [], list(config["review"]), []
    x, y, diameter = config["circle"]
    xx, yy = np.meshgrid(np.arange(diameter) + .5, np.arange(diameter) + .5)
    circle = (xx - diameter / 2) ** 2 + (yy - diameter / 2) ** 2 <= (diameter / 2) ** 2
    expected = np.array(config["background_rgb"], dtype=np.int16)
    regions = [("lecturer", (x, y, diameter, diameter), circle)]
    # Legacy checks remain useful diagnostics, but cannot become a full PASS.
    if not config["legacy_core_only"]:
        regions.append(("subtitle_band", tuple(config["subtitle_band"]), None))
    for transition in transitions:
        a = transition.get("check_start_frame", transition["start_frame"])
        b = transition.get("check_end_frame", transition["end_frame"])
        if not transition["roi_supported"]:
            reviews.append(f"{transition['id']}: unsupported chapter background; regions were not certified.")
            rows.append(dict(id=transition["id"], status="REVIEW", start_frame=a, end_frame=b))
            continue
        region_results, combined = {}, {}
        for name, (rx, ry, width, height), mask in regions:
            # Seek down to microseconds so a fractional start never skips the
            # first requested frame. RGB conversion precedes odd-coordinate crop.
            microseconds = a * 1000000 // FPS
            seek = f"{microseconds // 1000000}.{microseconds % 1000000:06d}"
            raw = run([ffmpeg, "-v", "error", "-threads", "2", "-ss", seek, "-i", str(final), "-an", "-frames:v", str(b-a), "-filter_threads", "2", "-vf", f"format=rgb24,crop={width}:{height}:{rx}:{ry}:exact=1", "-fps_mode", "passthrough", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"])
            size = width * height * 3
            if len(raw) != (b-a) * size:
                errors.append(f"{transition['id']}/{name}: decoded {len(raw)/size:g} frames, expected {b-a}.")
                region_results[name] = dict(status="FAIL", decoded_bytes=len(raw), checked_frames=0)
                continue
            images = np.frombuffer(raw, np.uint8).reshape(b-a, height, width, 3)
            stats = []
            for index, frame in enumerate(images):
                pixels = (frame[mask] if mask is not None else frame.reshape(-1, 3)).astype(np.int16)
                delta = np.abs(pixels - expected)
                maximum, deviation = int(delta.max()), float(pixels.std(axis=0).max())
                absolute = a + index
                phase = "pre_guard" if absolute < transition["start_frame"] else "post_guard" if absolute >= transition["end_frame"] else "core"
                item = dict(frame=absolute, phase=phase, maximum_channel_difference=maximum, maximum_channel_std=deviation, passed=maximum <= 12 and deviation <= 4)
                stats.append(item)
                if not item["passed"]:
                    combined.setdefault(absolute, dict(frame=absolute, phase=phase, regions=[]))["regions"].append(name)
            failed = [row for row in stats if not row["passed"]]
            if failed:
                errors.append(f"{transition['id']}/{name}: {len(failed)} core/guard frames retain non-background pixels.")
            region_results[name] = dict(status="FAIL" if failed else "PASS", checked_frames=b-a,
                                        first_frame=stats[0], last_frame=stats[-1],
                                        maximum_channel_difference=max(row["maximum_channel_difference"] for row in stats),
                                        maximum_channel_std=max(row["maximum_channel_std"] for row in stats), failed_frames=failed)
        failed = [combined[index] for index in sorted(combined)]
        regions_pass = all(row["status"] == "PASS" for row in region_results.values())
        rows.append(dict(id=transition["id"], status="PASS" if regions_pass else "FAIL", start_frame=a, end_frame=b,
                         core_start_frame=transition["start_frame"], core_end_frame=transition["end_frame"],
                         checked_frames=min((row["checked_frames"] for row in region_results.values()), default=0),
                         guard_before_frames=transition.get("checked_guard_before_frames", 0),
                         guard_after_frames=transition.get("checked_guard_after_frames", 0),
                         guard_clipped_at_timeline_start=transition.get("guard_clipped_at_timeline_start", False),
                         guard_clipped_at_timeline_end=transition.get("guard_clipped_at_timeline_end", False),
                         regions=region_results, failed_frames=failed))
    return result(errors, reviews, chapter_count=len(transitions), checked_frames=sum(row.get("checked_frames", 0) for row in rows),
                  roi=dict(x=x, y=y, width=diameter, height=diameter, mask="circle at pixel centers", expected_background=config["background"], maximum_channel_difference=12, maximum_channel_std=4),
                  subtitle_band=config["subtitle_band"] if not config["legacy_core_only"] else None,
                  scope="CONFIGURED_CORE_AND_GUARDS" if config["explicit_complete"] else "INCOMPLETE_OR_LEGACY_SCOPE",
                  chapters=rows, first_and_last_frames_included=True)


def caption_records(path):
    value = read_json(path)
    if isinstance(value, dict):
        if value.get("clock") in ("original_narration", "source", "source_clock"):
            raise ValueError("Captions must already be mapped to the final/edited clock.")
        value = value.get("captions")
    if not isinstance(value, list):
        raise ValueError("Captions JSON must be a list or an object with captions[].")
    return value


def srt_records(path):
    text = Path(path).read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
    records = []
    stamp = r"(\d{2,}):(\d{2}):(\d{2})[,.](\d{3})"
    for block in re.split(r"\n\s*\n", text) if text else []:
        lines = block.splitlines()
        index = 1 if lines[0].strip().isdigit() else 0
        match = re.fullmatch(stamp + r"\s*-->\s*" + stamp, lines[index].strip())
        if not match or len(lines) <= index + 1:
            raise ValueError("Invalid SRT timestamp/text block.")
        def seconds(groups):
            h, m, s, ms = map(int, groups)
            if m >= 60 or s >= 60:
                raise ValueError("Invalid SRT timestamp fields.")
            return Fraction((h * 3600 + m * 60 + s) * 1000 + ms, 1000)
        records.append(dict(start=str(seconds(match.groups()[:4])), end=str(seconds(match.groups()[4:])), text="\n".join(lines[index + 1:])))
    return records


def check_captions(path, srt, frames, transitions):
    errors, reviews, summaries = [], [], []
    sources = []
    if path:
        sources.append(("json", caption_records(path)))
    if srt:
        sources.append(("srt", srt_records(srt)))
    if not sources:
        return dict(status="NOT_REQUESTED", errors=[], review=[], limitation="No subtitle JSON/SRT supplied; no subtitle mapping check performed.")
    normalized = {}
    for label, records in sources:
        previous = Fraction(0)
        normalized[label] = []
        for i, cue in enumerate(records):
            try:
                # SRT rounds the exact video grid to milliseconds. Bounds are
                # permitted 0.5 ms of serialization rounding at the final end.
                copy = dict(cue)
                if label == "srt" and fraction(copy["end"]) > Fraction(frames, FPS) and fraction(copy["end"]) - Fraction(frames, FPS) <= Fraction(1, 2000):
                    copy["end"] = str(Fraction(frames, FPS))
                a, b = interval(copy, frames, f"{label} caption {i + 1}", allow_seconds=True)
                if a < previous:
                    errors.append(f"{label} captions overlap or are not sorted at {i + 1}.")
                previous = b
                if not isinstance(cue.get("text"), str) or not cue["text"].strip():
                    errors.append(f"{label} caption {i + 1} has empty text.")
                if cue.get("needs_text_review"):
                    reviews.append(f"{label} caption {i + 1} is still marked needs_text_review.")
                for t in transitions:
                    lo, hi = max(a, Fraction(t.get("check_start_frame", t["start_frame"]), FPS)), min(b, Fraction(t.get("check_end_frame", t["end_frame"]), FPS))
                    if hi > lo and not (label == "srt" and hi - lo <= Fraction(1, 2000)):
                        errors.append(f"{label} caption {i + 1} overlaps blank chapter core/guard {t['id']}.")
                normalized[label].append((a, b, cue.get("text", "")))
            except (ValueError, TypeError) as exc:
                errors.append(str(exc))
        summaries.append(dict(format=label, count=len(records)))
    if "json" in normalized and "srt" in normalized:
        left, right = normalized["json"], normalized["srt"]
        if len(left) != len(right):
            errors.append("SRT and JSON caption counts differ.")
        for i, (a, b) in enumerate(zip(left, right)):
            if any(abs(a[j] - b[j]) > Fraction(1, 2000) for j in (0, 1)) or a[2].strip() != b[2].strip():
                errors.append(f"SRT and JSON differ at caption {i + 1}.")
    return result(errors, reviews, inputs=summaries, overlap_check="Half-open chapter core-plus-guard intervals when configured; only SRT millisecond serialization allows <=0.5 ms edge rounding.", rendered_pixels_checked=False)


def check_chapters(path, metadata, frames):
    if not path:
        return dict(status="NOT_REQUESTED", errors=[], review=[], limitation="No chapter manifest supplied; embedded chapter metadata not compared.")
    value = read_json(path)
    records = value.get("chapters") if isinstance(value, dict) else value
    if not isinstance(records, list):
        raise ValueError("Chapters must be a list or {chapters:[...]}.")
    errors, normalized = [], []
    previous = Fraction(0)
    for i, item in enumerate(records):
        copy = dict(item)
        if "end_frame" not in copy and "end" not in copy:
            following = records[i + 1] if i + 1 < len(records) else None
            if "start_frame" in copy:
                copy["end_frame"] = following.get("start_frame") if following else frames
            else:
                copy["end"] = following.get("start") if following else str(Fraction(frames, FPS))
        try:
            a, b = interval(copy, frames, f"chapter {i + 1}", allow_seconds=True)
            if a < previous:
                errors.append(f"Chapter {i + 1} overlaps or is out of order.")
            previous = b
            normalized.append((a, b, item.get("title")))
        except ValueError as exc:
            errors.append(str(exc))
    actual = metadata.get("chapters", [])
    if len(actual) != len(records):
        errors.append(f"Embedded chapter count {len(actual)} != supplied chapter count {len(records)}.")
    for i, (expected, embedded) in enumerate(zip(normalized, actual)):
        tb = fraction(embedded["time_base"])
        a, b = int(embedded["start"]) * tb, int(embedded["end"]) * tb
        if abs(a - expected[0]) > Fraction(1, 1000) or abs(b - expected[1]) > Fraction(1, 1000):
            errors.append(f"Embedded chapter {i + 1} differs by over 1 ms.")
        if expected[2] is not None and embedded.get("tags", {}).get("title") != expected[2]:
            errors.append(f"Embedded chapter {i + 1} title differs.")
    return result(errors, supplied_count=len(records), embedded_count=len(actual), tolerance_seconds=.001)


def verify(args):
    if np is None:
        raise ValueError("NumPy is required; run with a Python environment that already provides numpy.")
    ffmpeg, ffprobe = executable("ffmpeg", args.ffmpeg), executable("ffprobe", args.ffprobe)
    style_path = getattr(args, "project_style", None)
    inputs = [args.final, args.shots, args.reference_audio] + [p for p in (args.captions, args.srt, args.chapters, style_path) if p]
    if any(args.out.resolve() == p.resolve() for p in inputs):
        raise ValueError("QA output must not overwrite a media or metadata input.")
    identities = [identity(p) for p in inputs]
    checks = {}
    config = chapter_config(style_path)
    checks["chapter_scope"] = result(reviews=config["review"], configuration={key: value for key, value in config.items() if key != "review"})
    shots_result = check_shots(read_json(args.shots), config)
    checks["timeline"] = shots_result
    frames, transitions = shots_result["frames"], shots_result["transitions"]
    metadata = probe(args.final, ffprobe)
    checks["media"], packets = check_media(metadata, frames, args.final, ffprobe)
    if packets is not None:
        checks["video_packets"] = packets
    for label,function in [('audio_packets',lambda:check_audio_packets(metadata,frames,args.final,ffprobe)),
                           ('full_decode',lambda:check_full_decode(args.final,frames,ffmpeg))]:
        try:checks[label]=function()
        except (ValueError,RuntimeError,KeyError,TypeError) as exc:checks[label]=result([str(exc)])
    checks["reference_pcm"] = check_reference(args.reference_audio, frames, ffprobe)
    for label, function in (("captions", lambda: check_captions(args.captions, args.srt, frames, transitions)), ("chapters", lambda: check_chapters(args.chapters, metadata, frames))):
        try:
            checks[label] = function()
        except (ValueError, KeyError, TypeError) as exc:
            checks[label] = result([str(exc)])
    with tempfile.TemporaryDirectory(prefix="course-media-qa-") as tmp:
        for label, function in (("audio_waveform", lambda: check_audio(args.final, args.reference_audio, frames, transitions, Path(tmp), ffmpeg)), ("chapter_roi", lambda: check_roi(args.final, transitions, ffmpeg, config))):
            try:
                checks[label] = function()
            except (ValueError, RuntimeError, KeyError, TypeError, wave.Error) as exc:
                checks[label] = result([str(exc)])
    changed = []
    for before, path in zip(identities, inputs):
        try:
            if identity(path) != before:
                changed.append(str(path))
        except OSError:
            changed.append(str(path))
    checks["input_stability"] = result(["Inputs changed during QA: " + ", ".join(changed)] if changed else [], identities=identities)
    statuses = [v["status"] for v in checks.values()]
    status = "FAIL" if "FAIL" in statuses else "REVIEW" if "REVIEW" in statuses else "TECHNICAL_PASS"
    return dict(schema_version=1, status=status, checked_at_utc=datetime.now(timezone.utc).isoformat(), scope="FINAL_MEDIA_TECHNICAL_ONLY", frames=frames, fps=FPS, duration_seconds=frames / FPS, tools=dict(ffmpeg=ffmpeg, ffprobe=ffprobe), checks=checks, limitations=LIMITATIONS)


def self_test(ffmpeg, ffprobe):
    """Only temporary synthetic RGB video, tones/noise, and fictional captions."""
    if np is None:
        raise ValueError("NumPy is required for the synthetic self-test.")
    cases = []
    with tempfile.TemporaryDirectory(prefix="course-media-synthetic-") as tmp:
        root = Path(tmp)
        # 4 + 1/60 seconds exercises a real partial final comparison block.
        frames = 241
        n = frames * 800
        t = np.arange(n) / RATE
        rng = np.random.default_rng(12345)
        noise = np.convolve(rng.normal(0, .06, n), np.ones(17) / 17, mode="same")
        signal = .11 * np.sin(2 * np.pi * (190 * t + 29 * t * t)) + .05 * np.sin(2 * np.pi * (490 * t + 11 * t ** 3)) + noise
        signal *= np.minimum(1, t / .03) * np.minimum(1, (frames / FPS - t) / .03)
        pcm = np.round(np.stack((signal, signal * .97), axis=1) * 32767).astype("<i2")
        reference = root / "reference.wav"
        with wave.open(str(reference), "wb") as w:
            w.setparams((2, 2, RATE, n, "NONE", "not compressed")); w.writeframes(pcm.tobytes())
        shots = root / "shots.json"
        plan = dict(fps=60, frames=frames, shots=[dict(id="a", kind="diagram", start_frame=0, end_frame=90), dict(id="core", kind="transition", start_frame=90, end_frame=168, presenter_visible=False, subtitle_visible=False), dict(id="b", kind="diagram", start_frame=168, end_frame=frames)])
        atomic_json(shots, plan)
        captions = root / "captions.json"
        clean_captions = [dict(start_frame=0, end_frame=72, text="合成素材，无真实人物"), dict(start_frame=186, end_frame=240, text="合成测试字幕")]
        atomic_json(captions, clean_captions)
        project_style = root / "project-style.json"
        atomic_json(project_style, dict(profile="practical", canvas=dict(width=WIDTH, height=HEIGHT, fps=FPS),
                                       layout=dict(lecturer_circle=[2164,945,340], subtitle_band=[0,1280,2560,160]),
                                       colors=dict(background=BACKGROUND),
                                       transitions=dict(frames=78, presenter_guard_before_frames=18, presenter_guard_after_frames=18,
                                                        lecturer_visible_during_guards=False, subtitle_visible_during_guards=False)))
        good = root / "good.mp4"
        def encode(destination, audio_filter=None, video_filter=None, rate=60):
            command = [ffmpeg, "-v", "error", "-threads", "2", "-f", "lavfi", "-i", f"color=c=0xF5F7F5:s=2560x1440:r={rate}", "-i", str(reference), "-map", "0:v:0", "-map", "1:a:0", "-frames:v", str(frames), "-filter_threads", "2"]
            if video_filter:
                command += ["-vf", video_filter]
            if audio_filter:
                command += ["-af", audio_filter]
            command += ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p", "-threads", "2", "-c:a", "aac", "-b:a", "256k", "-movie_timescale", str(RATE), "-movflags", "+faststart", "-y", str(destination)]
            run(command)
        encode(good)
        args = argparse.Namespace(final=good, shots=shots, reference_audio=reference, out=root / "qa.json", captions=captions, srt=None, chapters=None, project_style=project_style, ffmpeg=ffmpeg, ffprobe=ffprobe)
        report = verify(args)
        if report["status"] != "TECHNICAL_PASS":
            raise AssertionError("Synthetic valid fixture failed: " + json.dumps(report, ensure_ascii=False))
        assert report["checks"]["chapter_roi"]["checked_frames"] == 114
        assert report["checks"]["chapter_roi"]["roi"]["x"] == 2164
        assert report["checks"]["chapter_roi"]["chapters"][0]["regions"]["subtitle_band"]["checked_frames"] == 114
        assert report["checks"]["audio_waveform"]["waveform_scan"]["tail_duration_seconds"] > 0
        assert report['checks']['audio_packets']['presented_samples']==n
        assert report['checks']['full_decode']['decoded_video_frames']==frames
        cases += ["Variable-length 241-frame 2K60 AAC/reference fixture with explicit practical profile TECHNICAL_PASS", "Every one-second block plus 1/60-second tail checked", "All 78 core plus 36 guard frames in right-side lecturer circle and full subtitle band, including first and last"]
        short_audio=root/'short-audio.mp4'
        run([ffmpeg,'-v','error','-threads','2','-i',str(good),'-i',str(reference),'-map','0:v:0',
             '-map','1:a:0','-c:v','copy','-af',f'atrim=end_sample={n-32}','-c:a','aac','-b:a','256k',
             '-movie_timescale',str(RATE),'-y',str(short_audio)])
        args.final=short_audio
        short_report=verify(args)
        assert short_report['status']=='FAIL' and short_report['checks']['media']['status']=='FAIL'
        assert short_report['checks']['audio_packets']['status']=='FAIL'
        args.final=good
        cases.append('32-sample valid-audio shortfall rejected despite being less than one AAC frame')
        args.project_style = None
        legacy = verify(args)
        assert legacy["status"] == "REVIEW" and legacy["checks"]["chapter_roi"]["checked_frames"] == 78
        assert legacy["checks"]["chapter_scope"]["status"] == "REVIEW"
        cases.append("Missing project-style stays REVIEW with explicit legacy core-only scope")
        args.project_style = project_style
        config = chapter_config(project_style)
        configured_transitions = check_shots(plan, config)["transitions"]
        different_guards = dict(config, guard_before_frames=3, guard_after_frames=5)
        custom = check_shots(plan, different_guards)["transitions"][0]
        assert (custom["check_start_frame"], custom["check_end_frame"]) == (87,173)
        edge_plan = dict(fps=60, frames=78, shots=[dict(id="edge", kind="transition", start_frame=0, end_frame=78, presenter_visible=False, subtitle_visible=False)])
        edge = check_shots(edge_plan, config)["transitions"][0]
        assert (edge["check_start_frame"],edge["check_end_frame"]) == (0,78)
        assert edge["guard_clipped_at_timeline_start"] and edge["guard_clipped_at_timeline_end"]
        cases.append("Per-lesson guard lengths derive from config and clamp to current timeline boundaries")
        bad_gain = root / "bad-gain.mp4"
        run([ffmpeg, "-v", "error", "-i", str(good), "-i", str(reference), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-af", "volume=0.5", "-c:a", "aac", "-b:a", "256k", "-threads", "2", "-y", str(bad_gain)])
        args.final = bad_gain
        report = verify(args)
        assert report["status"] == "FAIL" and report["checks"]["audio_waveform"]["status"] == "FAIL"
        cases.append("Wrong audio gain rejected despite high Pearson correlation")
        bad_roi = root / "bad-roi.mp4"
        encode(bad_roi, video_filter="drawbox=x=2164:y=945:w=340:h=340:color=red:t=fill:enable='eq(n,90)+eq(n,167)'")
        args.final = bad_roi
        report = verify(args)
        failed = report["checks"]["chapter_roi"]["chapters"][0]["failed_frames"]
        assert report["status"] == "FAIL" and {90, 167}.issubset({x["frame"] for x in failed})
        cases.append("Only right-side lecturer on first and last core frames rejected")
        bad_guards = root / "bad-guards.mp4"
        encode(bad_guards, video_filter="drawbox=x=2164:y=945:w=340:h=340:color=red:t=fill:enable='eq(n,72)+eq(n,89)+eq(n,168)+eq(n,185)'")
        guard_report = check_roi(bad_guards, configured_transitions, ffmpeg, config)
        failed = guard_report["chapters"][0]["failed_frames"]
        assert guard_report["status"] == "FAIL" and {72,89,168,185}.issubset({row["frame"] for row in failed})
        assert all("lecturer" in row["regions"] for row in failed)
        cases.append("Single-frame right-side lecturer leaks at each pre/post guard boundary rejected")
        bad_subtitle = root / "bad-guard-subtitle.mp4"
        encode(bad_subtitle, video_filter="drawbox=x=100:y=1320:w=600:h=60:color=black:t=fill:enable='eq(n,185)'")
        subtitle_report = check_roi(bad_subtitle, configured_transitions, ffmpeg, config)
        failed = subtitle_report["chapters"][0]["failed_frames"]
        assert subtitle_report["status"] == "FAIL" and any(row["frame"] == 185 and "subtitle_band" in row["regions"] for row in failed)
        cases.append("One-frame subtitle pixels in final post-guard frame rejected")
        atomic_json(captions, [dict(start_frame=72, end_frame=73, text="仅保护区一帧字幕")])
        assert check_captions(captions, None, frames, configured_transitions)["status"] == "FAIL"
        atomic_json(captions, [dict(start_frame=185, end_frame=186, text="仅末保护区一帧字幕")])
        assert check_captions(captions, None, frames, configured_transitions)["status"] == "FAIL"
        atomic_json(captions, clean_captions)
        cases.append("One-frame caption records in first pre-guard and last post-guard frames rejected")
        bad_time = root / "bad-time.mp4"
        run([ffmpeg, "-v", "error", "-itsscale", "1.01", "-i", str(good), "-map", "0", "-c", "copy", "-y", str(bad_time)])
        args.final = bad_time
        report = verify(args)
        assert report["status"] == "FAIL" and report["checks"]["video_packets"]["status"] == "FAIL"
        cases.append("Retimed actual MP4 PTS/duration rejected with Fraction arithmetic")
        assert check_packets([dict(pts=0, duration=255), dict(pts=256, duration=256)], "1/15360", 2)["status"] == "FAIL"
        cases.append("Single packet duration off by one tick rejected")
        args.final = good
        plan["frames"] += 1
        atomic_json(shots, plan)
        report = verify(args)
        assert report["status"] == "FAIL" and report["checks"]["reference_pcm"]["status"] == "FAIL"
        cases.append("Declared timeline/reference sample count mismatch rejected")
        plan["frames"] -= 1
        plan["shots"][1]["background"] = "#FFFFFF"
        atomic_json(shots, plan)
        report = verify(args)
        assert report["status"] == "REVIEW"
        cases.append("Unsupported chapter background yields REVIEW, never PASS")
        plan["shots"][1].pop("background")
        atomic_json(shots, plan)
        atomic_json(captions, [dict(start_frame=88, end_frame=94, text="故意重叠")])
        report = verify(args)
        assert report["status"] == "FAIL" and report["checks"]["captions"]["status"] == "FAIL"
        cases.append("Subtitle crossing a blank chapter core rejected")
    return dict(status="PASS", cases=cases, fixtures="Temporary generated flat RGB, synthetic chirps/noise, fictional text; all deleted. No people or private media.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("final", "shots", "reference-audio", "out", "captions", "srt", "chapters", "project-style", "ffmpeg", "ffprobe"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--self-test", "--selftest", action="store_true", help="Run only synthetic temporary regression tests.")
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = ["--" + k.replace("_", "-") for k in ("final", "shots", "reference_audio", "out") if getattr(args, k) is None]
        if missing:
            parser.error("Required: " + ", ".join(missing))
    try:
        if args.self_test:
            report = self_test(executable("ffmpeg", args.ffmpeg), executable("ffprobe", args.ffprobe))
        else:
            report = verify(args)
            atomic_json(args.out, report)
        print(json.dumps(report if args.self_test else dict(status=report["status"], frames=report["frames"], duration_seconds=report["duration_seconds"], checks={k: v["status"] for k, v in report["checks"].items()}, report=str(args.out.resolve()), limitations=LIMITATIONS), ensure_ascii=False, indent=2))
        return 0 if report["status"] in ("PASS", "TECHNICAL_PASS") else 2 if report["status"] == "REVIEW" else 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, AssertionError) as exc:
        print(json.dumps(dict(status="FAIL", error=str(exc)), ensure_ascii=False, indent=2), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
