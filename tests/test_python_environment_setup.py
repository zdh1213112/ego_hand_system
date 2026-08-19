from __future__ import annotations

from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


class PythonEnvironmentSetupTests(unittest.TestCase):
    def test_conda_file_is_only_a_stable_bootstrap(self):
        value = yaml.safe_load((ROOT / "environment.yml").read_text(encoding="utf-8"))
        self.assertEqual(value["name"], "ego-hand")
        self.assertFalse(
            any(isinstance(item, dict) and "pip" in item for item in value["dependencies"])
        )
        self.assertIn("pip", value["dependencies"])

    def test_installer_separates_indexes_and_chumpy_build(self):
        installer = (ROOT / "scripts" / "setup_python_environment.sh").read_text(encoding="utf-8")
        self.assertIn("requirements/core.txt", installer)
        self.assertIn("requirements/wilor.txt", installer)
        self.assertIn("--no-build-isolation", installer)
        self.assertIn("580566eafc9ac68b2614b64d6f7aaa84eebb70da", installer)
        self.assertIn("--resume-retries", installer)
        self.assertIn("--no-deps ultralytics==8.4.56", installer)
        self.assertIn("uninstall -y thop", installer)
        self.assertIn("uninstall -y opencv-python opencv-python-headless", installer)
        self.assertIn("--force-reinstall --no-deps", installer)
        self.assertIn("env -u ALL_PROXY -u all_proxy", installer)

    def test_requirements_do_not_install_duplicate_opencv(self):
        requirements = "\n".join(
            (ROOT / name).read_text(encoding="utf-8")
            for name in ("requirements/core.txt", "requirements/wilor.txt")
        )
        self.assertIn("opencv-contrib-python==5.0.0.93", requirements)
        self.assertIn("psutil==7.2.2", requirements)
        self.assertIn("requests==2.32.5", requirements)
        self.assertIn("polars==1.43.2", requirements)
        self.assertIn("ultralytics-thop==2.0.18", requirements)
        self.assertNotIn("\nthop==", f"\n{requirements}")
        self.assertNotIn("\nopencv-python==", f"\n{requirements}")


if __name__ == "__main__":
    unittest.main()
