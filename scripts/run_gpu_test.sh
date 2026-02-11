#!/usr/bin/env bash

set -euo pipefail

# GPU 测试脚本说明：
# 1) 启动前置检查：验证 python3、torch 导入、CUDA 可用性与 GPU 列表。
# 2) 按 quick/full 套件执行 GPU util 与 GPU 显存场景（steady/step/spike 组合）。
# 3) 每个 case 输出 JSONL 日志（heartbeat/event/error），并在结束后打印摘要统计。
# 4) 支持 dry-run、跳过前置检查、自定义输出目录与显存回落策略。

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${PROJECT_DIR}/.venv/bin/activate" ]]; then
  source "${PROJECT_DIR}/.venv/bin/activate"
fi

SUITE="quick"
DEVICE="${GPU_DEVICE:-0}"
MEM_POLICY="${GPU_MEM_FREE_POLICY:-subprocess-restart}"
DURATION="${SIM_DURATION:-120s}"
INTERVAL="${SIM_INTERVAL:-1s}"
GPU_MEM_MAX_MB="${GPU_MEM_MAX_MB:-4000}"
OUT_DIR="${PROJECT_DIR}/artifacts/gpu-tests/$(date +%Y%m%d_%H%M%S)"
DRY_RUN="false"
SKIP_PRECHECK="false"

usage() {
  cat <<'EOF'
GPU 测试脚本（0~100 范围）

用法:
  bash scripts/run_gpu_test.sh [选项]

选项:
  --suite <quick|full>      测试套件，默认 quick
  --device <0|1|all>        GPU 设备，默认 0
  --duration <duration>     每个用例运行时长，默认 120s
  --interval <duration>     tick 间隔，默认 1s
  --gpu-mem-max-mb <int>    GPU 显存测试上限（MB），默认 4000
  --mem-policy <policy>     显存回落策略: none|empty_cache|subprocess-restart
  --out-dir <path>          日志输出目录
  --skip-precheck           跳过 torch/cuda 前置检测
  --dry-run                 只打印将执行的命令
  -h, --help                显示帮助

示例:
  bash scripts/run_gpu_test.sh --suite quick
  bash scripts/run_gpu_test.sh --suite full --device all --duration 90s
  bash scripts/run_gpu_test.sh --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --suite)
      SUITE="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --duration)
      DURATION="$2"
      shift 2
      ;;
    --interval)
      INTERVAL="$2"
      shift 2
      ;;
    --gpu-mem-max-mb)
      GPU_MEM_MAX_MB="$2"
      shift 2
      ;;
    --mem-policy)
      MEM_POLICY="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --skip-precheck)
      SKIP_PRECHECK="true"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] 未知参数: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "${SUITE}" != "quick" && "${SUITE}" != "full" ]]; then
  echo "[ERROR] --suite 仅支持 quick/full，当前: ${SUITE}" >&2
  exit 2
fi

if [[ "${MEM_POLICY}" != "none" && "${MEM_POLICY}" != "empty_cache" && "${MEM_POLICY}" != "subprocess-restart" ]]; then
  echo "[ERROR] --mem-policy 仅支持 none|empty_cache|subprocess-restart，当前: ${MEM_POLICY}" >&2
  exit 2
fi

if ! [[ "${GPU_MEM_MAX_MB}" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --gpu-mem-max-mb 必须是非负整数，当前: ${GPU_MEM_MAX_MB}" >&2
  exit 2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "[ERROR] 当前环境未找到 python3" >&2
  exit 127
fi

if [[ "${SKIP_PRECHECK}" != "true" ]]; then
  echo "[INFO] 进行 GPU 前置检查（torch + CUDA）..."
  if ! python3 - <<'PY'
import sys

try:
    import torch
except Exception as e:
    print(f"[ERROR] import torch 失败: {e}")
    sys.exit(1)

if not torch.cuda.is_available():
    print("[ERROR] CUDA 不可用，请确认驱动/CUDA/torch-cuda 环境")
    sys.exit(2)

count = torch.cuda.device_count()
names = [torch.cuda.get_device_name(i) for i in range(count)]
print(f"[INFO] CUDA 可用，GPU 数量: {count}")
for idx, name in enumerate(names):
    print(f"[INFO] GPU[{idx}]: {name}")
PY
  then
    echo "[ERROR] GPU 前置检查失败，可用 --skip-precheck 跳过检查" >&2
    exit 1
  fi
fi

mkdir -p "${OUT_DIR}"

COMMON_ENV=(
  "SIM_DURATION=${DURATION}"
  "SIM_INTERVAL=${INTERVAL}"
  "GPU_ENABLE=true"
  "GPU_DEVICE=${DEVICE}"
  "GPU_MEM_FREE_POLICY=${MEM_POLICY}"
  "CPU_ENABLE=false"
  "MEM_ENABLE=false"
  "LOG_ENABLE=false"
  "DB_ENABLE=false"
  "HEARTBEAT_ENABLE=true"
  "EVENT_LOG_ENABLE=true"
)

summarize_case() {
  local logfile="$1"

  python3 - "$logfile" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
heartbeats = 0
events = 0
errors = 0
targets_util = []
targets_mem = []

for raw in path.read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line:
        continue
    try:
        record = json.loads(line)
    except Exception:
        continue

    kind = record.get("kind")
    if kind == "heartbeat":
        heartbeats += 1
        targets = record.get("targets", {})
        if isinstance(targets, dict):
            util = targets.get("gpu_util_pct")
            mem = targets.get("gpu_mem_mb")
            if isinstance(util, (int, float)):
                targets_util.append(float(util))
            if isinstance(mem, (int, float)):
                targets_mem.append(float(mem))
    elif kind == "event":
        events += 1
    elif kind == "error":
        errors += 1

summary = {
    "heartbeats": heartbeats,
    "events": events,
    "errors": errors,
    "gpu_util_target_min": min(targets_util) if targets_util else None,
    "gpu_util_target_max": max(targets_util) if targets_util else None,
    "gpu_mem_target_mb_min": min(targets_mem) if targets_mem else None,
    "gpu_mem_target_mb_max": max(targets_mem) if targets_mem else None,
}
print("[SUMMARY]", json.dumps(summary, ensure_ascii=False))
PY
}

run_case() {
  local case_name="$1"
  shift
  local logfile="${OUT_DIR}/${case_name}.jsonl"
  local -a case_env=("$@")

  echo
  echo "=============================="
  echo "[CASE] ${case_name}"
  echo "[LOG ] ${logfile}"
  echo "=============================="

  if [[ "${DRY_RUN}" == "true" ]]; then
    echo "[DRY-RUN] cd \"${PROJECT_DIR}\" && env ${COMMON_ENV[*]} ${case_env[*]} python3 -m obs_simulator"
    return
  fi

  (
    cd "${PROJECT_DIR}"
    env "${COMMON_ENV[@]}" "${case_env[@]}" python3 -m obs_simulator
  ) | tee "${logfile}"

  summarize_case "${logfile}"
}

run_quick_suite() {
  local mem25=$((GPU_MEM_MAX_MB * 25 / 100))
  local mem50=$((GPU_MEM_MAX_MB * 50 / 100))
  local mem75=$((GPU_MEM_MAX_MB * 75 / 100))

  run_case "gpu_util_baseline_0" \
    "GPU_MODE=steady" "GPU_MEM_MODE=steady" \
    "GPU_UTIL_TARGET_PCT=0" "GPU_MEM_TARGET_MB=0"

  run_case "gpu_util_step_0_25_50_75_100" \
    "GPU_MODE=step" "GPU_UTIL_STEP_SERIES=0,25,50,75,100" "GPU_UTIL_STEP_EVERY=20s" \
    "GPU_MEM_MODE=steady" "GPU_MEM_TARGET_MB=0"

  run_case "gpu_util_spike_20_to_100" \
    "GPU_MODE=spike" "GPU_UTIL_TARGET_PCT=20" "GPU_UTIL_SPIKE_PCT=100" \
    "GPU_UTIL_SPIKE_EVERY=30s" "GPU_UTIL_SPIKE_LAST=3s" \
    "GPU_MEM_MODE=steady" "GPU_MEM_TARGET_MB=0"

  run_case "gpu_mem_step_0_25_50_75_100" \
    "GPU_MODE=steady" "GPU_UTIL_TARGET_PCT=20" \
    "GPU_MEM_MODE=step" "GPU_MEM_TARGET_MB=0" \
    "GPU_MEM_STEP_SERIES_MB=0,${mem25},${mem50},${mem75},${GPU_MEM_MAX_MB}" \
    "GPU_MEM_STEP_EVERY=20s"
}

run_full_suite() {
  local mem20=$((GPU_MEM_MAX_MB * 20 / 100))
  local mem_delta=$((GPU_MEM_MAX_MB - mem20))

  run_quick_suite

  run_case "gpu_util_steady_100" \
    "GPU_MODE=steady" "GPU_UTIL_TARGET_PCT=100" \
    "GPU_MEM_MODE=steady" "GPU_MEM_TARGET_MB=0"

  run_case "gpu_util_jitter_0_100" \
    "GPU_MODE=jitter" "GPU_UTIL_TARGET_PCT=50" "GPU_UTIL_JITTER_PCT=50" \
    "GPU_MEM_MODE=steady" "GPU_MEM_TARGET_MB=0"

  run_case "gpu_mem_spike_20_to_100" \
    "GPU_MODE=steady" "GPU_UTIL_TARGET_PCT=30" \
    "GPU_MEM_MODE=spike" "GPU_MEM_TARGET_MB=${mem20}" \
    "GPU_MEM_SPIKE_MB=${mem_delta}" "GPU_MEM_SPIKE_EVERY=30s" "GPU_MEM_SPIKE_LAST=3s"

  run_case "gpu_combined_step_util_mem" \
    "GPU_MODE=step" "GPU_UTIL_STEP_SERIES=0,25,50,75,100" "GPU_UTIL_STEP_EVERY=20s" \
    "GPU_MEM_MODE=step" "GPU_MEM_STEP_SERIES_MB=0,${mem20},$((mem20 * 2)),$((mem20 * 3)),${GPU_MEM_MAX_MB}" "GPU_MEM_STEP_EVERY=20s"
}

echo "[INFO] 项目目录: ${PROJECT_DIR}"
echo "[INFO] 输出目录: ${OUT_DIR}"
echo "[INFO] 套件类型: ${SUITE}"
echo "[INFO] GPU 设备: ${DEVICE}"
echo "[INFO] 显存上限(MB): ${GPU_MEM_MAX_MB}"
echo "[INFO] 显存回落策略: ${MEM_POLICY}"

if [[ "${SUITE}" == "quick" ]]; then
  run_quick_suite
else
  run_full_suite
fi

echo
echo "[DONE] GPU 测试完成，日志目录: ${OUT_DIR}"
