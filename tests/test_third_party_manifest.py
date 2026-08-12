#!/usr/bin/env python3

import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ThirdPartyManifestTests(unittest.TestCase):
    def test_manifest_matches_gitmodules(self):
        manifest = json.loads((ROOT / "third_party" / "manifest.json").read_text())
        gitmodules = (ROOT / ".gitmodules").read_text()
        for dependency in manifest["public_submodules"].values():
            self.assertIn(dependency["path"], gitmodules)
            self.assertIn(dependency["url"], gitmodules)

    def test_local_assets_use_project_relative_paths(self):
        manifest = json.loads((ROOT / "third_party" / "manifest.json").read_text())
        for dependency in manifest["local_assets"].values():
            for value in dependency["required_files"]:
                path = Path(value)
                self.assertFalse(path.is_absolute())
                self.assertNotIn("..", path.parts)

    def test_runtime_installer_has_pinned_checksum(self):
        manifest = json.loads((ROOT / "third_party" / "manifest.json").read_text())
        runtime = manifest["local_assets"]["basalt_runtime_linux_x86_64"]
        installer = (ROOT / "scripts" / "install_basalt_runtime.sh").read_text()
        self.assertIn(runtime["sha256"], installer)
        self.assertIn(runtime["artifact"], installer)
        setup = (ROOT / "scripts" / "setup_third_party.sh").read_text()
        self.assertIn("install_basalt_runtime.sh", setup)

    def test_mediapipe_installer_has_pinned_checksum(self):
        manifest = json.loads((ROOT / "third_party" / "manifest.json").read_text())
        model = manifest["local_assets"]["mediapipe_hand_landmarker"]
        installer = (ROOT / "scripts" / "install_mediapipe_model.sh").read_text()
        self.assertIn(model["url"], installer)
        self.assertIn(model["sha256"], installer)

    def test_mano_installer_names_both_models(self):
        installer = (ROOT / "scripts" / "install_mano_models.py").read_text()
        self.assertIn("MANO_LEFT.pkl", installer)
        self.assertIn("MANO_RIGHT.pkl", installer)

    def test_manifest_has_private_asset_install_instructions(self):
        manifest = json.loads((ROOT / "third_party" / "manifest.json").read_text())
        for name in ("mano_models", "orbbec_sdk_linux_x86_64"):
            self.assertTrue(manifest["local_assets"][name]["install"])

    def test_wilor_submodule_and_assets_are_declared(self):
        manifest = json.loads((ROOT / "third_party" / "manifest.json").read_text())
        self.assertIn("WiLoR", manifest["public_submodules"])
        self.assertIn("third_party/WiLoR", (ROOT / ".gitmodules").read_text())
        self.assertIn("wilor_models", manifest["local_assets"])
        self.assertIn(
            "models/wilor/wilor_final.ckpt",
            manifest["local_assets"]["wilor_models"]["required_files"],
        )

    def test_private_asset_installer_checks_required_model_paths(self):
        installer = (ROOT / "scripts" / "install_local_assets.sh").read_text()
        self.assertIn("models/hand_landmarker.task", installer)
        self.assertIn("models/mano/MANO_LEFT.pkl", installer)
        self.assertIn("models/mano/MANO_RIGHT.pkl", installer)


if __name__ == "__main__":
    unittest.main()
