import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from pctalk.output import random_numeric_output_name


class OutputNamingTest(unittest.TestCase):
    def test_name_is_ten_digit_number(self):
        with tempfile.TemporaryDirectory() as output_dir:
            name = random_numeric_output_name(output_dir)

        self.assertTrue(name.isdecimal())
        self.assertEqual(len(name), 10)

    def test_existing_output_name_is_skipped(self):
        with tempfile.TemporaryDirectory() as output_dir:
            Path(output_dir, "1000000001.mp4").touch()
            with patch("pctalk.output.secrets.randbelow", side_effect=(1, 2)):
                name = random_numeric_output_name(output_dir)

        self.assertEqual(name, "1000000002")


if __name__ == "__main__":
    unittest.main()
