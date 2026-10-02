import argparse
import unittest

try:
    from opensource.inference import parse_articulation, parse_emotion
    from opensource.pctalk import EmotionCondition, LipArticulationEdit
except ImportError:  # tests executed from the standalone directory
    from pctalk import EmotionCondition, LipArticulationEdit
    from pctalk.cli import parse_articulation, parse_emotion


class ControlParsingTest(unittest.TestCase):
    def test_default_emotion_intensity(self):
        self.assertEqual(
            parse_emotion("happy"),
            EmotionCondition("happy", intensity=0.5, region="all"),
        )

    def test_emotion_condition(self):
        self.assertEqual(
            parse_emotion("happy:lips:0.8"),
            EmotionCondition("happy", intensity=0.8, region="lips"),
        )

    def test_articulation_edit(self):
        self.assertEqual(
            parse_articulation("pursing:1.25"),
            LipArticulationEdit("pursing", scale=1.25),
        )

    def test_invalid_region(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_emotion("happy:nose:1")

    def test_invalid_articulation(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_articulation("smile:2")


if __name__ == "__main__":
    unittest.main()
