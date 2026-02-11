import json
import os
import subprocess
import unittest


class TestOrchestratorStrictMode(unittest.TestCase):
    def test_strict_idle_zero_targets_skips_cpu_mem_metrics(self) -> None:
        cmd = ["python3", "-m", "obs_simulator"]
        env = {
            "SIM_DURATION": "2s",
            "SIM_INTERVAL": "1s",
            "STRICT_IDLE_MODE": "true",
            "CPU_TARGET_PCT": "0",
            "MEM_TARGET_PCT": "0",
            "MEM_TARGET_BYTES": "0",
            "CPU_MODE": "steady",
            "MEM_MODE": "steady",
            "LOG_ENABLE": "false",
        }
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            env={**os.environ, **env},
            cwd=".",
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)

        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        records = [json.loads(line) for line in lines]
        heartbeats = [r for r in records if r.get("kind") == "heartbeat"]
        self.assertTrue(heartbeats, "should emit heartbeat records")

        actuals = heartbeats[0].get("actuals", {})
        self.assertNotIn("mem_alloc_bytes", actuals)
        self.assertNotIn("mem_worker_rss_bytes", actuals)


if __name__ == "__main__":
    unittest.main()
