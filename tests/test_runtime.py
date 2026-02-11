import unittest

from obs_simulator.config import load_config
from obs_simulator.runtime import should_start_cpu, should_start_mem


class TestRuntimePolicy(unittest.TestCase):
    def test_default_non_strict_starts_cpu_mem(self) -> None:
        cfg = load_config({"SIM_INTERVAL": "1s", "LOG_ENABLE": "false"})
        self.assertTrue(should_start_cpu(cfg))
        self.assertTrue(should_start_mem(cfg))

    def test_strict_idle_disables_zero_cpu_mem(self) -> None:
        cfg = load_config(
            {
                "SIM_INTERVAL": "1s",
                "STRICT_IDLE_MODE": "true",
                "CPU_TARGET_PCT": "0",
                "MEM_TARGET_PCT": "0",
                "MEM_TARGET_BYTES": "0",
                "CPU_MODE": "steady",
                "MEM_MODE": "steady",
                "LOG_ENABLE": "false",
            }
        )
        self.assertFalse(should_start_cpu(cfg))
        self.assertFalse(should_start_mem(cfg))

    def test_strict_idle_keeps_non_steady_cpu(self) -> None:
        cfg = load_config(
            {
                "SIM_INTERVAL": "1s",
                "STRICT_IDLE_MODE": "true",
                "CPU_MODE": "jitter",
                "CPU_TARGET_PCT": "0",
                "MEM_ENABLE": "false",
                "LOG_ENABLE": "false",
            }
        )
        self.assertTrue(should_start_cpu(cfg))

    def test_strict_idle_keeps_non_zero_mem(self) -> None:
        cfg = load_config(
            {
                "SIM_INTERVAL": "1s",
                "STRICT_IDLE_MODE": "true",
                "CPU_ENABLE": "false",
                "MEM_TARGET_BYTES": "64Mi",
                "MEM_MODE": "steady",
                "LOG_ENABLE": "false",
            }
        )
        self.assertTrue(should_start_mem(cfg))


if __name__ == "__main__":
    unittest.main()

