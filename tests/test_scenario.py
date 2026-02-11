import unittest

from obs_simulator.config import load_config
from obs_simulator.scenario import Scenario


class TestScenario(unittest.TestCase):
    def test_cpu_spike_window(self) -> None:
        cfg = load_config(
            {
                "SIM_INTERVAL": "1s",
                "SIM_MODE": "steady",
                "CPU_MODE": "spike",
                "CPU_TARGET_PCT": "10",
                "CPU_SPIKE_PCT": "100",
                "CPU_SPIKE_EVERY": "4s",
                "CPU_SPIKE_LAST": "2s",
                "MEM_ENABLE": "false",
                "LOG_ENABLE": "false",
            }
        )
        s = Scenario(cfg)
        vals = [s.tick(i)[0].cpu_target_pct for i in range(8)]
        # tick%4 in {0,1} => spike to 100; else base 10
        self.assertEqual(vals, [100, 100, 10, 10, 100, 100, 10, 10])

    def test_cpu_outlier(self) -> None:
        cfg = load_config(
            {
                "SIM_INTERVAL": "1s",
                "CPU_TARGET_PCT": "10",
                "CPU_OUTLIER_ENABLE": "true",
                "CPU_OUTLIER_EVERY": "3s",
                "CPU_OUTLIER_PCT": "99",
                "MEM_ENABLE": "false",
                "LOG_ENABLE": "false",
            }
        )
        s = Scenario(cfg)
        vals = [s.tick(i)[0].cpu_target_pct for i in range(7)]
        # outlier tick triggers at 3,6 (not 0)
        self.assertEqual(vals, [10, 10, 10, 99, 10, 10, 99])

    def test_cpu_step(self) -> None:
        cfg = load_config(
            {
                "SIM_INTERVAL": "1s",
                "CPU_MODE": "step",
                "CPU_STEP_SERIES": "10,20,30",
                "CPU_STEP_EVERY": "2s",
                "MEM_ENABLE": "false",
                "LOG_ENABLE": "false",
            }
        )
        s = Scenario(cfg)
        vals = [s.tick(i)[0].cpu_target_pct for i in range(7)]
        self.assertEqual(vals, [10, 10, 20, 20, 30, 30, 10])

    def test_cpu_sine_basic(self) -> None:
        cfg = load_config(
            {
                "SIM_INTERVAL": "1s",
                "CPU_MODE": "sine",
                "CPU_SINE_MIN": "0",
                "CPU_SINE_MAX": "100",
                "CPU_SINE_PERIOD": "4s",
                "MEM_ENABLE": "false",
                "LOG_ENABLE": "false",
            }
        )
        s = Scenario(cfg)
        vals = [s.tick(i)[0].cpu_target_pct for i in range(4)]
        # sin(0)=0 => 50; sin(pi/2)=1 => 100; sin(pi)=0 => 50; sin(3pi/2)=-1 => 0
        self.assertEqual(vals, [50, 100, 50, 0])


if __name__ == "__main__":
    unittest.main()

