#!/usr/bin/env python3
"""CPU-only, synthetic-media regression checks for narration calibration."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave

import numpy as np

SCRIPT = Path(__file__).with_name("level_narration.py")
SPEC = importlib.util.spec_from_file_location("level_narration", SCRIPT)
LEVEL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LEVEL)
FFMPEG = LEVEL.find_ffmpeg(None)
RATE = 48000


def write(path, pcm):
    with wave.open(str(path), "wb") as writer:
        writer.setparams((2, 2, RATE, 0, "NONE", "not compressed"))
        writer.writeframes(pcm.astype("<i2").tobytes())


def read(path):
    with wave.open(str(path), "rb") as reader:
        return np.frombuffer(reader.readframes(reader.getnframes()), "<i2").reshape(-1, 2)


class NarrationLevelTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="test-narration-level-")
        self.root = Path(self.temporary.name)
        self.source, self.out, self.report = [self.root / name for name in ("input.wav", "output.wav", "report.json")]
        self.count = 8 * RATE + 137
        t = np.arange(self.count) / RATE
        # Two channels differ, so a channel swap or mono conversion is visible.
        self.pcm = np.rint(np.stack((.03 * np.sin(2 * np.pi * 317 * t),
                                    .025 * np.sin(2 * np.pi * 521 * t)), axis=1) * 32768).astype("<i2")

    def tearDown(self):
        self.temporary.cleanup()

    def call(self, target, extra=()):
        result = subprocess.run([sys.executable, str(SCRIPT), "--input", str(self.source),
                                 "--out", str(self.out), "--report", str(self.report),
                                 "--target-lufs", str(target), "--ffmpeg", FFMPEG, *extra], capture_output=True, text=True)
        details = json.loads(self.report.read_text()) if self.report.exists() else None
        return result, details

    def baseline(self):
        return LEVEL.measure(FFMPEG, self.source, [])

    def test_gain_only_exact_sample_clock_and_single_gain(self):
        write(self.source, self.pcm)
        original = LEVEL.identity(self.source)
        before = self.baseline()
        result, report = self.call(before["integrated_lufs"] + 6)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(report["status"], "PASS")
        self.assertFalse(report["limiter_used"])
        self.assertEqual(LEVEL.identity(self.source), original)
        actual = read(self.out)
        self.assertEqual(len(actual), self.count)
        expected = np.rint(self.pcm.astype(np.float64) * report["uniform_linear_gain"])
        self.assertLessEqual(float(np.abs(actual.astype(float) - expected).max()), 1.)
        self.assertLessEqual(abs(report["after"]["integrated_lufs"] - report["target_lufs"]), .3)
        self.assertIn("input_i", report["before"]["ffmpeg_input_statistics"])

    def test_peak_protection_preserves_last_impulse_and_clock(self):
        pcm = self.pcm.copy()
        pcm[2 * RATE] = (25000, -25000)
        pcm[5 * RATE] = (-24000, 24000)
        pcm[-1] = (20000, -18000)
        write(self.source, pcm)
        target = self.baseline()["integrated_lufs"] + 12
        result, report = self.call(target)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(report["limiter_used"])
        self.assertEqual(len(read(self.out)), self.count)
        self.assertLessEqual(report["after"]["true_peak_dbtp"], -2.)
        self.assertLessEqual(abs(report["after"]["integrated_lufs"] - target), .3)
        self.assertIn("latency=true", report["actual_filter"])
        self.assertNotIn("atrim", report["actual_filter"])
        self.assertEqual(report["sample_clock"]["added_samples"], 0)
        self.assertEqual(report["sample_clock"]["removed_samples"], 0)
        actual = read(self.out)
        # An uncompensated limiter loses this boundary impulse or leaves it
        # 5 ms late. It must still be present at the original final sample.
        self.assertGreater(abs(int(actual[-1, 0])), 5000)
        self.assertGreater(abs(int(actual[-1, 1])), 5000)
        self.assertEqual(np.argmax(np.abs(actual[-100:, 0].astype(int))), 99)
        # Check alignment against the original carrier, away from transients.
        for start in (RATE // 2, 3 * RATE, 6 * RATE):
            a = pcm[start:start + 10000, 0].astype(float)
            b = actual[start - 8:start + 10008, 0].astype(float)
            offsets = [np.corrcoef(a, b[k:k + len(a)])[0, 1] for k in range(17)]
            self.assertEqual(int(np.argmax(offsets)), 8)

    def test_existing_output_never_overwritten(self):
        write(self.source, self.pcm)
        self.out.write_bytes(b"existing-deliverable")
        result, report = self.call(-18)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.out.read_bytes(), b"existing-deliverable")
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["output_published"])

    def test_existing_report_never_overwritten(self):
        write(self.source, self.pcm)
        self.report.write_text("existing-report")
        result = subprocess.run([sys.executable, str(SCRIPT), "--input", str(self.source),
                                 "--out", str(self.out), "--report", str(self.report),
                                 "--target-lufs", "-18"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.report.read_text(), "existing-report")
        self.assertFalse(self.out.exists())

    def test_silence_fails_without_output(self):
        write(self.source, np.zeros_like(self.pcm))
        original = LEVEL.identity(self.source)
        result, report = self.call(-18)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.out.exists())
        self.assertEqual(report["status"], "FAIL")
        self.assertEqual(LEVEL.identity(self.source), original)

    def test_unreachable_target_fails_after_three_renders(self):
        # Near-full-scale steady tones cannot reach -5 LUFS with a -9 dBTP
        # ceiling. The tool must fail, rather than stamp PASS after a cap.
        t = np.arange(self.count) / RATE
        tone = np.rint(.8 * np.sin(2 * np.pi * 317 * t) * 32768).astype("<i2")
        write(self.source, np.stack((tone, tone), axis=1))
        result, report = self.call(-5, ["--max-true-peak", "-9"])
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(len(report["attempts"]), 3)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(self.out.exists())

    def test_alias_cannot_overwrite_input_or_publish_output(self):
        write(self.source, self.pcm)
        original = LEVEL.identity(self.source)
        result = subprocess.run([sys.executable, str(SCRIPT), "--input", str(self.source),
                                 "--out", str(self.out), "--report", str(self.out),
                                 "--target-lufs", "-18"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.out.exists())
        self.assertEqual(LEVEL.identity(self.source), original)

    def test_invalid_number_records_failure_without_invalid_json(self):
        write(self.source, self.pcm)
        result, report = self.call("nan")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.out.exists())
        self.assertEqual(report["target_lufs"], "nan")
        self.assertEqual(report["status"], "FAIL")

    def test_wrong_rate_rejected_before_render(self):
        with wave.open(str(self.source), "wb") as writer:
            writer.setparams((2, 2, 44100, 0, "NONE", "not compressed"))
            writer.writeframes(self.pcm.tobytes())
        original = LEVEL.identity(self.source)
        result, report = self.call(-18)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(report["attempts"], [])
        self.assertFalse(self.out.exists())
        self.assertEqual(LEVEL.identity(self.source), original)


if __name__ == "__main__":
    unittest.main(verbosity=2)
