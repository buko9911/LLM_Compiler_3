"""Backend routing regression; no simulator or model needed."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "runtime"))
from decode import Device


class BackendRouting(unittest.TestCase):
    def test_default_and_explicit_backends(self):
        for backend, library in (("ramulator", None), ("ramulator", Path("/tmp/ramulator.so"))):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)
                (out / "system.json").write_text(json.dumps({"logical_to_physical": [0]}))
                (out / "io.json").write_text(json.dumps({"args": [], "outputs": [], "pim_commands": 0}))
                settings = out / "settings.toml"
                settings.write_text("num_cores = 1\n")
                device = Device(out, settings, emulator=Path("/tmp/emulator"),
                                lp6_backend=backend, lp6_library=library)

                def fake_run(command, **kwargs):
                    self.assertEqual(command[command.index("--lp6-backend") + 1], backend)
                    self.assertEqual("--lp6-library" in command, library is not None)
                    if library:
                        self.assertEqual(command[command.index("--lp6-library") + 1], str(library))
                    self.assertIn("--timeline-out", command)
                    self.assertEqual(command[command.index("--timeline-merge-gap") + 1], "0")
                    (out / device.dump_name).write_bytes(b"")
                    (out / "commands.json").write_text("{}")
                    return subprocess.CompletedProcess(command, 0, "", "")

                with patch("decode.subprocess.run", side_effect=fake_run):
                    outputs, _, _ = device.run(profile=True)
                self.assertEqual(outputs, {})

    def test_unknown_backend_rejected_before_files_are_read(self):
        with self.assertRaisesRegex(ValueError, "unsupported LP6 backend"):
            Device(Path("/nonexistent"), lp6_backend="typo")

    def test_retired_dramsim3_backend_rejected(self):
        with self.assertRaisesRegex(ValueError, "unsupported LP6 backend"):
            Device(Path("/nonexistent"), lp6_backend="dramsim3")


if __name__ == "__main__":
    unittest.main()
