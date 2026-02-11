import unittest

from obs_simulator.config import load_config
from obs_simulator.utils.parse import ParseError, duration_to_ticks, parse_bytes, parse_duration_seconds, parse_int_list


class TestParse(unittest.TestCase):
    def test_parse_duration_seconds(self) -> None:
        self.assertAlmostEqual(parse_duration_seconds("100ms"), 0.1)
        self.assertAlmostEqual(parse_duration_seconds("5s"), 5.0)
        self.assertAlmostEqual(parse_duration_seconds("2m"), 120.0)
        self.assertAlmostEqual(parse_duration_seconds("1h"), 3600.0)
        self.assertAlmostEqual(parse_duration_seconds("0"), 0.0)
        with self.assertRaises(ParseError):
            parse_duration_seconds("5")
        with self.assertRaises(ParseError):
            parse_duration_seconds("5sec")

    def test_parse_bytes(self) -> None:
        self.assertEqual(parse_bytes("123B"), 123)
        self.assertEqual(parse_bytes("1Ki"), 1024)
        self.assertEqual(parse_bytes("2Mi"), 2 * 1024 * 1024)
        self.assertEqual(parse_bytes("3Gi"), 3 * 1024 * 1024 * 1024)
        self.assertEqual(parse_bytes("0"), 0)
        with self.assertRaises(ParseError):
            parse_bytes("1KB")
        with self.assertRaises(ParseError):
            parse_bytes("Mi")

    def test_duration_to_ticks(self) -> None:
        self.assertEqual(duration_to_ticks(60.0, 5.0), 12)
        self.assertEqual(duration_to_ticks(3.0, 1.0), 3)
        with self.assertRaises(ParseError):
            duration_to_ticks(0.0, 1.0)

    def test_parse_int_list(self) -> None:
        self.assertEqual(parse_int_list("10,30,60,90"), [10, 30, 60, 90])
        self.assertEqual(parse_int_list(" 1 , 2 , , 3 "), [1, 2, 3])

    def test_strict_idle_mode_config(self) -> None:
        cfg = load_config({"SIM_INTERVAL": "1s", "STRICT_IDLE_MODE": "true"})
        self.assertTrue(cfg.global_cfg.strict_idle_mode)


if __name__ == "__main__":
    unittest.main()
