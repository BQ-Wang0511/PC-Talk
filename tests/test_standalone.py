from pathlib import Path
import unittest

from pctalk.config import AudioConfig, CHECKPOINT_ROOT


class StandalonePackageTest(unittest.TestCase):
    def test_bundled_runtime_assets_exist(self):
        paths = (
            AudioConfig().checkpoint,
            CHECKPOINT_ROOT / "lac.pth",
            CHECKPOINT_ROOT / "emc.pth",
            CHECKPOINT_ROOT / "style_encoder.pth",
            CHECKPOINT_ROOT
            / "liveportrait/base_models/appearance_feature_extractor.pth",
            CHECKPOINT_ROOT / "liveportrait/base_models/motion_extractor.pth",
            CHECKPOINT_ROOT / "liveportrait/base_models/warping_module.pth",
            CHECKPOINT_ROOT / "liveportrait/base_models/spade_generator.pth",
            CHECKPOINT_ROOT
            / "liveportrait/retargeting_models/stitching_retargeting_module.pth",
            CHECKPOINT_ROOT / "liveportrait/landmark.onnx",
            CHECKPOINT_ROOT / "insightface/models/buffalo_l/det_10g.onnx",
            CHECKPOINT_ROOT / "insightface/models/buffalo_l/2d106det.onnx",
        )
        missing = [str(path) for path in paths if not Path(path).is_file()]
        self.assertEqual(missing, [])

    def test_release_source_has_no_parent_repository_imports(self):
        package_root = Path(__file__).resolve().parents[1]
        forbidden = (
            "from " + "src",
            "import " + "src",
            "from " + "audio2exp",
            "import " + "audio2exp",
        )
        violations = []
        for path in package_root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if any(item in text for item in forbidden):
                violations.append(str(path.relative_to(package_root)))
        self.assertEqual(violations, [])

    def test_public_source_uses_project_model_names(self):
        package_root = Path(__file__).resolve().parents[1]
        model_root = package_root / "pctalk/models"
        forbidden = (
            "face" + "former",
            "style" + "former",
            "wav2" + "lip",
        )
        violations = []
        for path in model_root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".py":
                text = path.read_text(encoding="utf-8").lower()
                relative = str(path.relative_to(package_root)).lower()
                if any(item in text or item in relative for item in forbidden):
                    violations.append(relative)
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
