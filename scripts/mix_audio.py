#!/usr/bin/env python3
"""Mix an already-edited narration master with quiet, smoothly ducked music.

Requires Python 3, NumPy and ffmpeg. Narration must be 48 kHz stereo signed
16-bit PCM WAV. Output length is exactly the input sample count, even when it
is not an integer video frame. Narration is copied byte-for-byte and is never
cut, faded, resampled, normalized or limited by this helper.
Series defaults are 0.07 music / 0.035 during speech. --project-style reads a
lesson's resolved audio settings; explicit --gain / --speech-gain win. The
0.10 ceiling remains available for an expressly requested per-lesson override.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import wave

try:
    import numpy as np
except ImportError:
    raise SystemExit("NumPy is required. Use a Python environment with NumPy installed.")

RATE, CHANNELS, WIDTH = 48000, 2, 2
STYLE_PATH = Path(__file__).resolve().parents[1] / "assets/style.json"
SERIES_STYLE = json.loads(STYLE_PATH.read_text(encoding="utf-8-sig")) if STYLE_PATH.is_file() else {}
GAIN_CEILING = float(SERIES_STYLE.get("audio", {}).get("music_hard_gain_ceiling", .10))
if not math.isfinite(GAIN_CEILING) or not 0 < GAIN_CEILING <= .10:
    raise ValueError("Installed music_hard_gain_ceiling must be positive and no greater than 0.10.")
BLOCK = 480


def configured_gains(project_style=None, gain=None, speech_gain=None):
    """CLI > resolved project style > installed series defaults > safe fallback."""
    path = Path(project_style) if project_style else STYLE_PATH
    audio = json.loads(path.read_text(encoding="utf-8-sig")).get("audio", {}) if path.is_file() else {}
    if project_style and not path.is_file():
        raise ValueError(f"Project style file does not exist: {path}")
    maximum = float(audio.get("music_max_gain", .07)) if gain is None else gain
    speech = float(audio.get("speech_music_target_gain", .035)) if speech_gain is None else speech_gain
    return maximum, speech


class ClippingError(ValueError):
    def __init__(self, details):
        super().__init__("The mix would clip. Lower music gain and speech gain; narration was not changed.")
        self.details = details


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def pcm_info(path):
    with wave.open(str(path), "rb") as stream:
        rate, channels, width = stream.getframerate(), stream.getnchannels(), stream.getsampwidth()
        count = stream.getnframes()
        compression = stream.getcomptype()
    if (rate, channels, width, compression) != (RATE, CHANNELS, WIDTH, "NONE"):
        raise ValueError(f"Narration must be 48000 Hz, stereo, signed 16-bit PCM WAV; got {(rate, channels, width, compression)}.")
    if not count:
        raise ValueError("Narration is empty.")
    return dict(sample_rate=rate, channels=channels, sample_width=width, samples=count)


def wav_writer(path):
    stream = wave.open(str(path), "wb")
    stream.setparams((CHANNELS, WIDTH, RATE, 0, "NONE", "not compressed"))
    return stream


def find_ffmpeg(requested=None):
    candidates = [requested or os.environ.get("FFMPEG") or "ffmpeg"]
    for candidate in candidates:
        found = shutil.which(candidate)
        if found:
            return found
    raise ValueError("ffmpeg was not found. Add it to PATH or provide --ffmpeg /path/to/ffmpeg.")


def decode_music(ffmpeg, source, destination):
    # Float output preserves legitimate decoded peaks above 0 dBFS. Converting
    # the loud source music to s16 before attenuation would irreversibly clip it.
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-threads", "2",
                    "-i", str(source), "-map", "0:a:0", "-vn", "-filter_threads", "2",
                    "-ar", str(RATE), "-ac", "2", "-c:a", "pcm_f32le", "-f", "f32le",
                    "-y", str(destination)], check=True, stderr=subprocess.PIPE)
    if destination.stat().st_size % 8:
        raise ValueError("Decoded music has an incomplete stereo float sample.")
    samples = np.memmap(destination, dtype="<f4", mode="r").reshape(-1, 2)
    if len(samples) < 4 * RATE:
        raise ValueError("Music must contain at least 4 seconds for a 2-second, two-way loop crossfade.")
    if not np.isfinite(samples).all():
        raise ValueError("Music contains non-finite decoded samples.")
    return samples


def loop_music(music, start, count):
    overlap = 2 * RATE
    period = len(music) - overlap
    index = np.arange(start, start + count, dtype=np.int64)
    phase = index % period
    result = np.array(music[phase], dtype=np.float64)
    cross = (index >= period) & (phase < overlap)
    if np.any(cross):
        p = phase[cross]
        weight = p.astype(np.float64) / (overlap - 1)
        result[cross] = music[period + p] * (1. - weight[:, None]) + music[p] * weight[:, None]
    return result


class Envelope:
    def __init__(self, maximum, speech, keep_trace=False):
        self.maximum, self.speech, self.current = maximum, speech, maximum
        self.hold = 0
        self.keep_trace, self.trace = keep_trace, []
        self.applied_min, self.applied_max = float("inf"), 0.
        self.duck_min, self.duck_max = float("inf"), 0.
        self.blocks, self.voiced_blocks, self.samples = 0, 0, 0

    def render(self, narration, start, total):
        output = np.empty(len(narration), dtype=np.float64)
        for offset in range(0, len(narration), BLOCK):
            samples = narration[offset:offset + BLOCK].astype(np.float64) / 32768.
            rms = float(np.sqrt(np.mean(samples * samples, axis=0)).max())
            voiced = rms >= 10 ** (-42 / 20)
            if voiced:
                self.hold = round(.150 * RATE)
                self.voiced_blocks += 1
            target = self.speech if voiced or self.hold > 0 else self.maximum
            self.hold = max(0, self.hold - len(samples))
            time_constant = .030 if target < self.current else .900
            curve = target + (self.current - target) * np.exp(-np.arange(1, len(samples) + 1) / (RATE * time_constant))
            self.current = float(curve[-1])
            index = start + offset + np.arange(len(samples))
            fade_in = np.clip(index / (2 * RATE - 1), 0., 1.)
            fade_out = np.clip((total - 1 - index) / (3 * RATE - 1), 0., 1.)
            applied = curve * fade_in * fade_out
            self.applied_min = min(self.applied_min, float(applied.min()))
            self.applied_max = max(self.applied_max, float(applied.max()))
            self.duck_min = min(self.duck_min, float(curve.min()))
            self.duck_max = max(self.duck_max, float(curve.max()))
            self.blocks += 1
            self.samples += len(samples)
            if self.keep_trace:
                self.trace.append((start + offset, len(samples), rms, int(voiced), target,
                                   float(curve.min()), float(curve.max()), float(applied.min()), float(applied.max())))
            output[offset:offset + len(samples)] = applied
        if np.any(output < 0) or np.any(output > GAIN_CEILING + 1e-12):
            raise ValueError(f"Music envelope exceeded its {GAIN_CEILING:g} hard ceiling.")
        return output


def write_mix(narration, music, stage, total, gain, speech_gain, trace):
    envelope = Envelope(gain, speech_gain, trace)
    stats = {name: dict(peak=0., clipping_channel_samples=0, rail_channel_samples=0, sum_squares=0.)
             for name in ("narration", "music", "mixed")}
    safe_scale = 1.
    with wave.open(str(narration), "rb") as voice, wav_writer(stage / "music.wav") as music_out, wav_writer(stage / "mixed.wav") as mixed_out:
        for start in range(0, total, RATE):
            count = min(RATE, total - start)
            n = np.frombuffer(voice.readframes(count), dtype="<i2").reshape(-1, 2)
            if len(n) != count:
                raise ValueError("Narration ended before its declared sample count.")
            factors = envelope.render(n, start, total)
            floating_music = loop_music(music, start, count) * factors[:, None]
            # Keep diagnostics in float64 so even an invalid, extremely hot
            # music file cannot wrap during an integer cast.
            m = np.rint(floating_music * 32768.)
            combined = n.astype(np.float64) + m
            for name, signal in (("narration", n), ("music", m), ("mixed", combined)):
                values = signal.astype(np.float64) / 32768.
                stats[name]["peak"] = max(stats[name]["peak"], float(np.abs(values).max()))
                stats[name]["sum_squares"] += float(np.sum(values * values))
                stats[name]["clipping_channel_samples"] += int(np.count_nonzero((signal < -32768) | (signal > 32767)))
                stats[name]["rail_channel_samples"] += int(np.count_nonzero((signal == -32768) | (signal == 32767)))
            # Conservative music-only gain bound with one-LSB headroom. Lower
            # both gain settings by this scale; never silently change narration.
            positive, negative = floating_music > 0, floating_music < 0
            if np.any(positive):
                safe_scale = min(safe_scale, float(np.min((32766. - n[positive]) / (32768. * floating_music[positive]))))
            if np.any(negative):
                safe_scale = min(safe_scale, float(np.min((-32767. - n[negative]) / (32768. * floating_music[negative]))))
            overflow = np.any((m < -32768) | (m > 32767) | (combined < -32768) | (combined > 32767))
            if not overflow:
                music_out.writeframesraw(m.astype("<i2").tobytes())
                mixed_out.writeframesraw(combined.astype("<i2").tobytes())
            # A failed block is never saturated and published. Continue scanning
            # only to report a full-file peak and useful music-gain recommendation.
    for item in stats.values():
        item["peak_dbfs"] = 20 * math.log10(max(item["peak"], 1e-20))
        item["rms"] = math.sqrt(item.pop("sum_squares") / (total * CHANNELS))
    clips = stats["music"]["clipping_channel_samples"] + stats["mixed"]["clipping_channel_samples"]
    if clips:
        scale = max(0., min(1., safe_scale)) * .98
        raise ClippingError(dict(stem_statistics=stats, configured_gain=gain, configured_speech_gain=speech_gain,
                                 suggested_maximum_gain=math.floor(gain * scale * 1e6) / 1e6,
                                 suggested_speech_gain=math.floor(speech_gain * scale * 1e6) / 1e6,
                                 advice="Lower both music coefficients to these values or below and rerun. If zero is suggested, narration has insufficient headroom for positive music at some samples. No normalization or narration attenuation was applied."))
    details = dict(status="PASS", maximum_linear_gain_limit=GAIN_CEILING, configured_music_gain=gain,
                   speech_target_gain=speech_gain, actual_duck_gain_range=[envelope.duck_min, envelope.duck_max],
                   actual_gain_including_fades_range=[envelope.applied_min, envelope.applied_max],
                   attack_ms=30., release_ms=900., hold_ms=150., detector_block_ms=10.,
                   detector_rms_threshold_dbfs=-42., blocks=envelope.blocks, voiced_blocks=envelope.voiced_blocks,
                   envelope_sample_coverage=envelope.samples, fade_in_seconds=2., fade_out_seconds=3.,
                   loop_crossfade_seconds=2., loop_period_samples=len(music) - 2 * RATE,
                   loop_source_samples=len(music), loop_source_seconds=len(music) / RATE,
                   loop_weight_sum=1., loop_method="Linear complementary weights; no equal-power boost.",
                   stem_statistics=stats, narration_gain=1., final_normalization=False,
                   clipping_definition="Counts arithmetic samples outside signed-16-bit PCM range. Intersample true peak and encoded AAC are not measured by this helper.")
    if trace:
        np.save(stage / "gain-trace.npy", np.array(envelope.trace, dtype=np.float64), allow_pickle=False)
        details["gain_trace"] = dict(file="gain-trace.npy", columns=["start_sample", "samples", "narration_rms", "voiced", "target_gain", "duck_gain_min", "duck_gain_max", "applied_gain_min", "applied_gain_max"])
    return details


def verify_stems(narration, stage, total):
    differences = 0
    with wave.open(str(narration), "rb") as n, wave.open(str(stage / "music.wav"), "rb") as m, wave.open(str(stage / "mixed.wav"), "rb") as y:
        if any(w.getnframes() != total for w in (n, m, y)):
            raise ValueError("Stem sample counts differ from narration.")
        for start in range(0, total, RATE):
            count = min(RATE, total - start)
            a, b, c = [np.frombuffer(w.readframes(count), dtype="<i2").astype(np.int32) for w in (n, m, y)]
            differences += int(np.count_nonzero(a + b != c))
    if differences:
        raise ValueError(f"Mixed PCM does not equal unchanged narration + music: {differences} differences.")
    return dict(status="PASS", differing_channel_samples=0, compared_channel_samples=total * 2,
                narration_gain=1., method="Full-file integer comparison: mixed = original narration + delivered music.")


def publish(stage, destination, names):
    # Verify before publication. Back up old outputs for rollback if a later
    # rename fails. qa.json is committed last as the completion marker.
    backup = stage / "previous"
    backup.mkdir()
    saved, installed = [], []
    try:
        for name in names:
            old = destination / name
            if old.exists() or old.is_symlink():
                if old.is_dir():
                    raise ValueError(f"Output target is a directory: {old}")
                old.replace(backup / name)
                saved.append(name)
            (stage / name).replace(old)
            installed.append(name)
    except BaseException:
        for name in reversed(installed):
            (destination / name).unlink()
        for name in reversed(saved):
            (backup / name).replace(destination / name)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--narration", required=True, type=Path)
    parser.add_argument("--bgm", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--project-style", type=Path, help="Use the resolved project-style.json audio settings; explicit gains override it.")
    parser.add_argument("--gain", type=float, default=None, help="Music maximum linear coefficient, 0–0.10; series default 0.07.")
    parser.add_argument("--speech-gain", type=float, default=None, help="Ducked coefficient, 0–gain; series default 0.035.")
    parser.add_argument("--fps", type=float, default=60., help="QA metadata only; audio is never padded to a video frame.")
    parser.add_argument("--gain-trace", action="store_true", help="Save the detailed 10 ms envelope audit as gain-trace.npy.")
    parser.add_argument("--ffmpeg", help="Optional ffmpeg executable; otherwise discover it on PATH.")
    args = parser.parse_args(argv)
    try:
        args.gain, args.speech_gain = configured_gains(args.project_style, args.gain, args.speech_gain)
        if not math.isfinite(args.gain + args.speech_gain + args.fps) or not 0 <= args.speech_gain <= args.gain <= GAIN_CEILING or args.fps <= 0:
            raise ValueError("Require 0 <= speech-gain <= gain <= 0.10 and a positive finite fps.")
        ffmpeg = find_ffmpeg(args.ffmpeg)
        info = pcm_info(args.narration)
        inputs = {"narration": file_identity(args.narration), "bgm": file_identity(args.bgm)}
        args.out.mkdir(parents=True, exist_ok=True)
        names = ["narration.wav", "music.wav", "mixed.wav"] + (["gain-trace.npy"] if args.gain_trace else []) + ["qa.json"]
        for name in names:
            candidate = args.out / name
            if any(candidate.resolve() == Path(value["path"]) for value in inputs.values()):
                raise ValueError(f"Output would overwrite an input: {candidate}")
        with tempfile.TemporaryDirectory(prefix=".mix-audio-", dir=args.out) as temporary:
            stage = Path(temporary)
            music = decode_music(ffmpeg, args.bgm, stage / "bgm.f32")
            decoded_peak = float(np.abs(music).max())
            result = write_mix(args.narration, music, stage, info["samples"], args.gain, args.speech_gain, args.gain_trace)
            del music
            stem_check = verify_stems(args.narration, stage, info["samples"])
            shutil.copyfile(args.narration, stage / "narration.wav")
            source_hash = digest(args.narration)
            if digest(stage / "narration.wav") != source_hash:
                raise ValueError("Copied narration is not byte-identical to the input WAV.")
            for before in inputs.values():
                if file_identity(before["path"]) != before:
                    raise ValueError("An input changed during mixing; staged output was discarded.")
            frames = info["samples"] / RATE * args.fps
            report = dict(schema_version=1, status="PASS", inputs=inputs, format=info,
                          duration=info["samples"] / RATE, fps=args.fps, duration_in_video_frames=frames,
                          whole_video_frames=abs(frames - round(frames)) < 1e-8,
                          ffmpeg=ffmpeg, original_music_decoded_float_peak=decoded_peak,
                          over_zero_music_peaks_preserved_before_attenuation=True,
                          narration_preservation=dict(status="PASS", wav_byte_identical=True, sha256=source_hash,
                                                      samples_unchanged=info["samples"], narration_gain=1., edits_applied=0),
                          mix=result, exact_stem_sum=stem_check, outputs=names,
                          settings_source=str((args.project_style or STYLE_PATH).resolve()),
                          input_stability=dict(status="PASS"))
            (stage / "qa.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            publish(stage, args.out, names)
        print(json.dumps(dict(status="PASS", qa=str((args.out / "qa.json").resolve()),
                              samples=info["samples"], duration=info["samples"] / RATE,
                              mixed_peak_dbfs=result["stem_statistics"]["mixed"]["peak_dbfs"]), indent=2))
        return 0
    except Exception as exc:
        report = dict(status="FAIL", error=f"{type(exc).__name__}: {exc}", existing_outputs_preserved=True)
        if isinstance(exc, ClippingError):
            report["clipping_diagnostic"] = exc.details
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            report["ffmpeg_error"] = exc.stderr.decode(errors="replace")[-3000:]
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 3 if isinstance(exc, ClippingError) else 1


if __name__ == "__main__":
    sys.exit(main())
