#!/usr/bin/env python3
"""Synthetic-only regression tests; no real voice, music or course files used."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave
import numpy as np

sys.dont_write_bytecode = True
SCRIPT = Path(__file__).with_name("mix_audio.py")
spec = importlib.util.spec_from_file_location("generic_audio_mixer", SCRIPT)
mixer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixer)


def save_pcm(path, data, rate=48000):
    with wave.open(str(path), "wb") as out:
        out.setparams((data.shape[1], 2, rate, 0, "NONE", "not compressed"))
        out.writeframes(data.astype("<i2").tobytes())


def load_pcm(path):
    with wave.open(str(path), "rb") as source:
        return source.getnframes(), np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").reshape(-1, source.getnchannels())


class SyntheticMixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ffmpeg = mixer.find_ffmpeg()

    def make_float_music(self, directory):
        # A constant signal is intentionally unmusical: any loop gain boost or
        # loss of over-zero float peaks is then directly observable.
        music = np.full((int(4.25 * 48000), 2), 1.125, dtype="<f4")
        path = directory / "synthetic-float-music.wav"
        subprocess.run([self.ffmpeg, "-v", "error", "-f", "f32le", "-ar", "48000", "-ac", "2",
                        "-i", "pipe:0", "-c:a", "pcm_f32le", "-y", str(path)],
                       input=music.tobytes(), check=True, stderr=subprocess.PIPE)
        return path

    def run_cli(self, narration, bgm, output, *extra):
        process = subprocess.run([sys.executable, str(SCRIPT), "--narration", str(narration),
                                  "--bgm", str(bgm), "--out", str(output), "--fps", "60", *extra],
                                 capture_output=True, text=True)
        self.assertTrue(process.stdout.strip(), process.stderr)
        return process.returncode, json.loads(process.stdout)

    def test_exact_voice_duration_gain_and_loop(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-mix-test-") as temp:
            directory = Path(temp)
            count = 12 * 48000 + 137  # Neither a video frame nor a 10 ms block.
            time = np.arange(count) / 48000
            data = np.zeros((count, 2), dtype=np.int16)
            voiced = (time < 3.) | ((time >= 7.) & (time < 10.))
            data[voiced, 0] = np.rint(np.sin(time[voiced] * 2 * np.pi * 223) * 7000).astype(np.int16)
            data[voiced, 1] = np.rint(np.sin(time[voiced] * 2 * np.pi * 271) * 6000).astype(np.int16)
            narration, output = directory / "synthetic-narration.wav", directory / "out"
            save_pcm(narration, data)
            bgm = self.make_float_music(directory)
            code, result = self.run_cli(narration, bgm, output, "--gain-trace")
            self.assertEqual(code, 0, result)
            qa = json.loads((output / "qa.json").read_text())
            self.assertEqual((output / "narration.wav").read_bytes(), narration.read_bytes())
            self.assertEqual(qa["format"]["samples"], count)
            self.assertFalse(qa["whole_video_frames"])
            self.assertEqual(qa["mix"]["configured_music_gain"], .07)
            self.assertEqual(qa["mix"]["speech_target_gain"], .035)
            self.assertEqual(qa["mix"]["loop_weight_sum"], 1.)
            self.assertGreater(qa["original_music_decoded_float_peak"], 1.)
            trace = np.load(output / "gain-trace.npy")
            self.assertEqual(int(trace[:, 1].sum()), count)
            self.assertEqual(trace[-1, 1], 137)
            self.assertTrue(np.all(trace[1:, 0] == trace[:-1, 0] + trace[:-1, 1]))
            self.assertTrue(np.all((trace[:, 7] >= 0.) & (trace[:, 8] <= .07)))
            self.assertTrue(np.all(trace[trace[:, 3] == 1, 4] == .035))
            self.assertLess(trace[(trace[:, 0] > 48000) & (trace[:, 0] < 2 * 48000), 6].max(), .035001)
            n_samples, n = load_pcm(output / "narration.wav")
            m_samples, m = load_pcm(output / "music.wav")
            y_samples, y = load_pcm(output / "mixed.wav")
            self.assertEqual((n_samples, m_samples, y_samples), (count, count, count))
            np.testing.assert_array_equal(n.astype(np.int32) + m.astype(np.int32), y.astype(np.int32))
            self.assertEqual(m[0].tolist(), [0, 0])
            self.assertEqual(m[-1].tolist(), [0, 0])
            self.assertEqual(qa["mix"]["stem_statistics"]["mixed"]["clipping_channel_samples"], 0)
            for start, length, _, _, _, _, _, low, high in trace:
                values = m[int(start):int(start + length)].astype(float) / 32768.
                # Constant 1.125 input stays constant through every overlap.
                self.assertLessEqual(values.max(), high * 1.125 + .500001 / 32768)
                self.assertGreaterEqual(values.min(), low * 1.125 - .500001 / 32768)

    def test_clipping_refuses_and_preserves_every_existing_output(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-clip-test-") as temp:
            directory = Path(temp)
            narration, output = directory / "high-headroom-test.wav", directory / "out"
            save_pcm(narration, np.full((6 * 48000, 2), 32700, dtype=np.int16))
            bgm = self.make_float_music(directory)
            output.mkdir()
            names = ["narration.wav", "music.wav", "mixed.wav", "gain-trace.npy", "qa.json"]
            for name in names:
                (output / name).write_bytes(("previous valid output: " + name).encode())
            previous = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
            code, result = self.run_cli(narration, bgm, output, "--gain-trace")
            self.assertEqual(code, 3, result)
            self.assertEqual(result["status"], "FAIL")
            self.assertTrue(result["existing_outputs_preserved"])
            diagnostic = result["clipping_diagnostic"]
            self.assertGreater(diagnostic["stem_statistics"]["mixed"]["clipping_channel_samples"], 0)
            self.assertTrue(0 < diagnostic["suggested_maximum_gain"] < .07)
            self.assertTrue(0 < diagnostic["suggested_speech_gain"] < .035)
            self.assertEqual(set(p.name for p in output.iterdir()), set(names))
            for p in output.iterdir():
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), previous[p.name])
            # The suggested music-only settings must actually resolve this case.
            lower = directory / "lower-gain"
            code, success = self.run_cli(narration, bgm, lower,
                "--gain", str(diagnostic["suggested_maximum_gain"]),
                "--speech-gain", str(diagnostic["suggested_speech_gain"]))
            self.assertEqual(code, 0, success)
            self.assertEqual((lower / "narration.wav").read_bytes(), narration.read_bytes())

    def test_input_format_and_gain_ceiling_are_enforced(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-input-test-") as temp:
            directory = Path(temp)
            source = directory / "mono.wav"
            save_pcm(source, np.zeros((48000, 1), dtype=np.int16))
            bgm = self.make_float_music(directory)
            code, result = self.run_cli(source, bgm, directory / "mono-out")
            self.assertNotEqual(code, 0)
            self.assertIn("stereo", result["error"])
            save_pcm(source, np.zeros((48000, 2), dtype=np.int16), rate=44100)
            code, result = self.run_cli(source, bgm, directory / "rate-out")
            self.assertNotEqual(code, 0)
            self.assertIn("48000", result["error"])
            save_pcm(source, np.zeros((48000, 2), dtype=np.int16))
            code, result = self.run_cli(source, bgm, directory / "gain-out", "--gain", ".11")
            self.assertNotEqual(code, 0)
            self.assertIn("0.10", result["error"])

    def test_loop_is_chunk_invariant_and_never_boosts_constant_input(self):
        constant = np.full((5 * 48000, 2), 1.125, dtype=np.float32)
        whole = mixer.loop_music(constant, 0, 12 * 48000 + 137)
        blocks = [mixer.loop_music(constant, p, min(877, len(whole) - p)) for p in range(0, len(whole), 877)]
        np.testing.assert_array_equal(whole, np.concatenate(blocks))
        self.assertLessEqual(np.abs(whole - 1.125).max(), 5e-16)

    def test_project_audio_settings_and_explicit_overrides(self):
        with tempfile.TemporaryDirectory(prefix="synthetic-style-test-") as temp:
            directory = Path(temp)
            narration = directory / "synthetic-narration.wav"
            count = 6 * 48000
            time = np.arange(count) / 48000
            data = np.zeros((count, 2), dtype=np.int16)
            voiced = (time >= 2.) & (time < 4.)
            data[voiced, :] = np.rint(np.sin(time[voiced, None] * 2 * np.pi * 223) * 7000).astype(np.int16)
            save_pcm(narration, data)
            bgm = self.make_float_music(directory)
            project_style = directory / "project-style.json"
            project_style.write_text(json.dumps({"profile": "practical", "audio": {"music_max_gain": .06, "speech_music_target_gain": .03}}))
            code, result = self.run_cli(narration, bgm, directory / "from-project", "--project-style", str(project_style), "--gain-trace")
            self.assertEqual(code, 0, result)
            qa = json.loads((directory / "from-project/qa.json").read_text())
            self.assertEqual((qa["mix"]["configured_music_gain"], qa["mix"]["speech_target_gain"]), (.06, .03))
            trace = np.load(directory / "from-project/gain-trace.npy")
            self.assertTrue(np.all(trace[:, 8] <= .06))
            self.assertTrue(np.any(trace[:, 3] == 1))
            self.assertTrue(np.all(trace[trace[:, 3] == 1, 4] == .03))
            code, result = self.run_cli(narration, bgm, directory / "explicit", "--project-style", str(project_style), "--gain", ".10", "--speech-gain", ".05")
            self.assertEqual(code, 0, result)
            qa = json.loads((directory / "explicit/qa.json").read_text())
            self.assertEqual((qa["mix"]["configured_music_gain"], qa["mix"]["speech_target_gain"]), (.10, .05))
            self.assertLessEqual(qa["mix"]["actual_gain_including_fades_range"][1], .10)
            code, result = self.run_cli(narration, bgm, directory / "missing", "--project-style", str(directory / "absent.json"))
            self.assertNotEqual(code, 0)
            self.assertIn("does not exist", result["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
