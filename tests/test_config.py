from __future__ import annotations

import unittest

from dr_detection.config import load_config


class ConfigTests(unittest.TestCase):
    def test_loads_efficientnet_config(self) -> None:
        cfg = load_config("configs/efficientnet_b0.json")
        self.assertEqual(cfg.model.name, "efficientnet_b0")
        self.assertEqual(cfg.data.num_classes, 5)
        self.assertTrue(cfg.data.quality_filter)


if __name__ == "__main__":
    unittest.main()
