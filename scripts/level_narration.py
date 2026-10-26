#!/usr/bin/env python3
"""Calibrate an edited narration WAV to an explicitly measured LUFS target.

This helper changes level, never the edit or sample clock.  It uses ffmpeg's
loudnorm only to measure input_* statistics; rendering is uniform gain, with
an oversampled peak limiter only when necessary.  The original and existing
deliverables are protected.  A failed calibration never publishes an output.
Requires Python 3 and ffmpeg with loudnorm, aresample and alimiter.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
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
import time
import wave

RATE, CHANNELS, WIDTH = 48000, 2, 2
MAX_ATTEMPTS = 3


class CalibrationError(ValueError):
    pass


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def identity(path):
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(path=str(path), size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns,
                sha256=sha256(path))


def exists(path):
    return os.path.lexists(path)


def pcm_info(path):
    with wave.open(str(path), "rb") as stream:
        fields = (stream.getframerate(), stream.getnchannels(),
                  stream.getsampwidth(), stream.getcomptype())
        if fields != (RATE, CHANNELS, WIDTH, "NONE"):
            raise CalibrationError(f"Expected 48000 Hz stereo PCM16 WAV; got {fields}.")
        count = stream.getnframes()
        if count < RATE * .4:
            raise CalibrationError("At least 0.4 seconds of non-silent audio are required for integrated LUFS measurement.")
        # Verify the declared frame count, including a partial final block.
        consumed = 0
        while True:
            block = stream.readframes(RATE)
            if not block:
                break
            if len(block) % (CHANNELS * WIDTH):
                raise CalibrationError("Incomplete stereo PCM sample in WAV data.")
            consumed += len(block) // (CHANNELS * WIDTH)
        if consumed != count:
            raise CalibrationError(f"WAV declares {count} samples but contains {consumed}.")
    return dict(sample_rate=RATE, channels=CHANNELS, sample_width_bytes=WIDTH,
                samples=count, duration_seconds=count / RATE)


def find_ffmpeg(requested):
    choices = [requested or os.environ.get("FFMPEG") or "ffmpeg"]
    for choice in choices:
        found = shutil.which(choice)
        if found:
            return found
    raise CalibrationError("ffmpeg not found; use --ffmpeg /absolute/path/to/ffmpeg.")


def run(command, commands):
    commands.append(command)
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise CalibrationError(f"ffmpeg exited {result.returncode}: {result.stderr[-4000:]}")
    return result


def measure(ffmpeg, source, commands):
    # The output_* statistics describe loudnorm's discarded processing and must
    # never be mistaken for actual levels of the supplied WAV.
    result = run([ffmpeg, "-hide_banner", "-nostdin", "-threads", "2", "-i", str(source),
                  "-map", "0:a:0", "-vn", "-filter_threads", "2", "-af",
                  "loudnorm=I=-24:TP=-2:LRA=50:print_format=json", "-f", "null", "-"], commands)
    blocks = re.findall(r'\{\s*"input_i"\s*:.*?\}', result.stderr, re.S)
    if len(blocks) != 1:
        raise CalibrationError("Could not obtain exactly one loudnorm input_* measurement.")
    raw = json.loads(blocks[0])
    selected = {name: float(raw[name]) for name in ("input_i", "input_tp", "input_lra", "input_thresh")}
    if not all(math.isfinite(value) for value in selected.values()):
        raise CalibrationError("LUFS/true peak measurement is non-finite, typically because the source is silent or too short.")
    return dict(integrated_lufs=selected["input_i"], true_peak_dbtp=selected["input_tp"],
                lra_lu=selected["input_lra"], threshold_lufs=selected["input_thresh"],
                ffmpeg_input_statistics=selected)


def render_filter(gain_db, limiter, ceiling_db):
    filters = [f"volume={gain_db:.12f}dB:precision=double"]
    if limiter:
        limit = 10 ** (ceiling_db / 20.)
        if not .0625 <= limit <= 1.:
            raise CalibrationError("Required limiter ceiling is outside alimiter's supported linear range.")
        filters.extend([
            "aresample=192000:resampler=swr:filter_size=64:phase_shift=10",
            f"alimiter=limit={limit:.12f}:attack=5:release=100:level=false:latency=true",
            "aresample=48000:resampler=swr:filter_size=64:phase_shift=10:dither_method=none",
        ])
    return ",".join(filters)


def render(ffmpeg, source, destination, filter_text, commands):
    run([ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-threads", "2",
         "-i", str(source), "-map", "0:a:0", "-vn", "-filter_threads", "2",
         "-af", filter_text, "-ar", str(RATE), "-ac", str(CHANNELS),
         "-c:a", "pcm_s16le", "-n", str(destination)], commands)


def publish_json(path, report):
    # Hard links provide no-overwrite publication even if another process
    # creates the requested report after the initial existence check.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", required=True, type=Path, help="Edited 48 kHz stereo PCM16 narration WAV.")
    cli.add_argument("--out", required=True, type=Path, help="New calibrated WAV; existing paths are never replaced.")
    cli.add_argument("--report", required=True, type=Path, help="New JSON report, also written on a calibration failure.")
    cli.add_argument("--target-lufs", required=True, type=float, help="Measured reference target; no implicit default.")
    cli.add_argument("--max-true-peak", type=float, default=-2., help="Maximum permitted measured dBTP (default -2.0).")
    cli.add_argument("--tolerance", type=float, default=.3, help="Allowed integrated LUFS difference (default 0.3 LU).")
    cli.add_argument("--ffmpeg", help="Optional ffmpeg executable.")
    return cli


def main(argv=None):
    args = parser().parse_args(argv)
    started = time.perf_counter()
    source = args.input.expanduser().resolve()
    destination = args.out.expanduser().absolute()
    report_path = args.report.expanduser().absolute()
    report = dict(status="FAIL", tool="level_narration.py", schema_version=1,
                  created_utc=datetime.now(timezone.utc).isoformat(), target_lufs=args.target_lufs,
                  tolerance_lu=args.tolerance, max_true_peak_dbtp=args.max_true_peak,
                  max_attempts=MAX_ATTEMPTS, attempts=[], commands=[], time_mapping="identity",
                  source_preserved=None, output_published=False,
                  limitations=["Narration level changes; do not claim PCM identity with the source.",
                               "Peak protection may attenuate local transients. No noise gate, denoising, other dynamic compression, cuts, fades or speed changes are applied.",
                               "Measured WAV compliance does not replace listening or checking the final encoded AAC and full mix."])
    published = False
    original = None
    try:
        for name, number in (("target-lufs", args.target_lufs), ("max-true-peak", args.max_true_peak), ("tolerance", args.tolerance)):
            if not math.isfinite(number):
                raise CalibrationError(f"--{name} must be finite.")
        if not -70 <= args.target_lufs <= -5:
            raise CalibrationError("--target-lufs must be between -70 and -5 LUFS.")
        if not -9 <= args.max_true_peak <= 0:
            raise CalibrationError("--max-true-peak must be between -9 and 0 dBTP.")
        if not 0 < args.tolerance <= 3:
            raise CalibrationError("--tolerance must be greater than 0 and no more than 3 LU.")
        if source in (destination.resolve(), report_path.resolve()) or destination.resolve() == report_path.resolve():
            raise CalibrationError("Input, output and report must be three distinct paths.")
        if exists(destination) or exists(report_path):
            raise CalibrationError("Refusing to replace an existing output or report.")
        original = identity(source)
        report["input_identity"] = original
        report["input_audio"] = info = pcm_info(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = find_ffmpeg(args.ffmpeg)
        report["ffmpeg"] = dict(path=ffmpeg, version=run([ffmpeg, "-version"], report["commands"]).stdout.splitlines()[0])
        before = measure(ffmpeg, source, report["commands"])
        report["before"] = before
        gain_db = args.target_lufs - before["integrated_lufs"]
        limiter = before["true_peak_dbtp"] + gain_db > args.max_true_peak
        ceiling_db = args.max_true_peak - .10
        report["initial_uniform_gain_db"] = gain_db
        with tempfile.TemporaryDirectory(prefix=".level-narration-", dir=destination.parent) as temporary:
            stage = Path(temporary)
            selected = None
            for number in range(1, MAX_ATTEMPTS + 1):
                candidate = stage / f"candidate-{number}.wav"
                filter_text = render_filter(gain_db, limiter, ceiling_db)
                attempt = dict(attempt=number, uniform_gain_db=gain_db,
                               uniform_linear_gain=10 ** (gain_db / 20.), limiter_used=limiter,
                               limiter_ceiling_dbtp=ceiling_db if limiter else None,
                               actual_filter=filter_text, render_from_original=True)
                report["attempts"].append(attempt)
                render(ffmpeg, source, candidate, filter_text, report["commands"])
                attempt["audio"] = output_info = pcm_info(candidate)
                if output_info["samples"] != info["samples"]:
                    attempt["sample_count_difference"] = output_info["samples"] - info["samples"]
                    raise CalibrationError("Processing changed the sample count. No trimming/padding or output publication is permitted.")
                attempt["measurement"] = measured = measure(ffmpeg, candidate, report["commands"])
                error = args.target_lufs - measured["integrated_lufs"]
                compliant_peak = measured["true_peak_dbtp"] <= args.max_true_peak
                attempt["loudness_error_lu"] = -error
                attempt["loudness_pass"] = abs(error) <= args.tolerance
                attempt["true_peak_pass"] = compliant_peak
                if attempt["loudness_pass"] and compliant_peak:
                    selected = candidate
                    report.update(after=measured, output_audio=output_info, uniform_gain_db=gain_db,
                                  uniform_linear_gain=attempt["uniform_linear_gain"], limiter_used=limiter,
                                  actual_filter=filter_text, selected_attempt=number)
                    break
                gain_db += error
                if not compliant_peak:
                    if limiter:
                        ceiling_db -= measured["true_peak_dbtp"] - args.max_true_peak + .10
                    limiter = True
                if before["true_peak_dbtp"] + gain_db > args.max_true_peak:
                    limiter = True
            if selected is None:
                raise CalibrationError("Actual output did not satisfy LUFS and true-peak limits within 3 renders. Review the source dynamics and target; no output was published.")
            if identity(source) != original:
                raise CalibrationError("Source identity changed during processing; refusing publication.")
            report["source_preserved"] = True
            report["sample_clock"] = dict(sample_rate=RATE, input_samples=info["samples"],
                                          output_samples=info["samples"], difference_samples=0,
                                          added_samples=0, removed_samples=0,
                                          limiter_latency_compensated=report["limiter_used"],
                                          tail_handling="ffmpeg filter EOF flush only; no atrim, apad, -t or manual sample repair")
            # Publish only after every measurement and sample-clock check.
            os.link(selected, destination)
            published = True
            report["output_identity"] = identity(destination)
            report["status"] = "PASS"
            report["output_published"] = True
            report["elapsed_seconds"] = time.perf_counter() - started
            publish_json(report_path, report)
        print(json.dumps(dict(status="PASS", output=str(destination), report=str(report_path),
                              before_lufs=before["integrated_lufs"], after_lufs=report["after"]["integrated_lufs"],
                              after_dbtp=report["after"]["true_peak_dbtp"], limiter_used=report["limiter_used"],
                              samples=info["samples"]), ensure_ascii=False))
        return 0
    except (ValueError, OSError, wave.Error, subprocess.SubprocessError) as error:
        report.update(status="FAIL", error=str(error), output_published=False,
                      elapsed_seconds=time.perf_counter() - started)
        if original is not None:
            try:
                report["source_preserved"] = identity(source) == original
            except OSError:
                report["source_preserved"] = False
        if published:
            # Only remove the file just published by this invocation.  Never
            # remove a replacement created by another process.
            try:
                if identity(destination) == report.get("output_identity"):
                    destination.unlink()
            except OSError:
                pass
        # Invalid numeric arguments are deliberately not serialized as NaN.
        for key in ("target_lufs", "tolerance_lu", "max_true_peak_dbtp"):
            if not math.isfinite(report[key]):
                report[key] = str(report[key])
        if report_path.resolve() not in (source, destination.resolve()) and not exists(report_path):
            try:
                report_path.parent.mkdir(parents=True, exist_ok=True)
                publish_json(report_path, report)
            except OSError as report_error:
                print(f"Could not publish failure report: {report_error}", file=sys.stderr)
        print(f"FAIL: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
