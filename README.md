# obs-simulator（Monitor-Test）

这是一个“可观测性压测/曲线模拟器”单服务：通过环境变量控制 CPU / 内存 / GPU / 日志 / DB 的负载形态，用来验证你们平台的曲线展示、尖刺/离群点处理、高频抖动下的渲染与交互稳定性。

实现以仓库根目录的契约 `monitor_env.md` 为准（变量名、默认值、优先级、映射规则都按它走）。

## 0) 统一入口与框架规范（长期迭代建议）

统一入口：

```bash
python3 -m obs_simulator
```

内部流程是固定的四段式，方便长期迭代：

1. `config`：读取并校验 env，输出生效配置（JSON）
2. `scenario`：按 tick 生成统一 targets（CPU/MEM/GPU/LOG/DB）
3. `runtime`：统一生命周期管理（模块 start/apply/snapshot/stop）
4. `orchestrator`：主循环、heartbeat/event、优雅退出

新增运行策略（推荐）：
- `STRICT_IDLE_MODE=false`（默认）：CPU/MEM 默认启用（target=0 为空载基线）
- `STRICT_IDLE_MODE=true`：当 CPU/MEM 都是 steady 且目标为 0 时，不启动对应模块，避免空转

## 1) 快速开始（本地）

进入目录后直接启动（默认会输出心跳 + 业务日志）：

```bash
cd "Monitor-Test"
python3 -m obs_simulator
```

常用例子：

```bash
# CPU 70% steady
CPU_TARGET_PCT=70 python3 -m obs_simulator

# MEM 99% steady（按 cgroup limit）
MEM_TARGET_PCT=99 python3 -m obs_simulator

# CPU 每分钟 3 秒尖刺到 100%
CPU_MODE=spike CPU_TARGET_PCT=10 CPU_SPIKE_PCT=100 CPU_SPIKE_EVERY=60s CPU_SPIKE_LAST=3s \\
  python3 -m obs_simulator

# 高频抖动（每秒刷新一次）
SIM_INTERVAL=1s CPU_MODE=jitter CPU_TARGET_PCT=50 CPU_JITTER_PCT=40 \\
  python3 -m obs_simulator
```

## 2) 当前环境直接运行（不依赖 Docker）

本项目默认可直接在当前环境运行（核心路径只依赖 Python 标准库）。

```bash
cd "Monitor-Test"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

可选依赖（仅在你启用对应功能时需要）：
- GPU 模块：安装 `torch` + 可用 CUDA 环境
- DB/Kubeblocks 模块：`pip install -r requirements-extra.txt`

建议先做一个 10 秒可用性自检：

```bash
SIM_DURATION=10s python3 -m obs_simulator
```

如需“严格空载不启动 CPU/MEM”：

```bash
SIM_DURATION=10s SIM_INTERVAL=1s STRICT_IDLE_MODE=true \
CPU_MODE=steady CPU_TARGET_PCT=0 MEM_MODE=steady MEM_TARGET_PCT=0 MEM_TARGET_BYTES=0 \
python3 -m obs_simulator
```

## 3) 输出日志（stdout）

stdout 会混合输出三类 JSON 单行日志（便于采集与检索）：
- `kind=heartbeat`：每 tick 一条，包含 targets 与部分实际值采样
- `kind=event`：spike/step/outlier/conn_storm 等边界事件
- `kind=log`：日志模块按速率输出的业务日志（可配置 burst/multiline/超长行等）

## 4) 契约与变量

变量与默认值以 `monitor_env.md` 为准（仓库根目录）。

## 5) 测试与验证（推荐日常命令）

基础单测：

```bash
cd "Monitor-Test"
python3 -m unittest discover -s tests -q
```

指定测试（strict 策略）：

```bash
python3 -m unittest -q tests.test_runtime tests.test_orchestrator_strict
```

快速健康检查（10s）：

```bash
SIM_DURATION=10s SIM_INTERVAL=1s python3 -m obs_simulator
```

你应看到：
- 启动 `kind=config`
- 过程 `kind=heartbeat`（每 tick）
- 场景边界时 `kind=event`
- 结束 `kind=shutdown`

## 6) 监控用例（范围全部 0~100）

说明：下面所有百分比都控制在 0~100，便于你们平台统一展示和阈值验证。

### 6.1 CPU 用例

1) CPU 0% 基线（观察空闲）

```bash
SIM_DURATION=60s SIM_INTERVAL=1s CPU_TARGET_PCT=0 python3 -m obs_simulator
```

2) CPU 100% 满载（上限）

```bash
SIM_DURATION=60s SIM_INTERVAL=1s CPU_TARGET_PCT=100 python3 -m obs_simulator
```

3) CPU 阶跃（0→25→50→75→100）

```bash
SIM_DURATION=120s SIM_INTERVAL=2s CPU_MODE=step CPU_STEP_SERIES=0,25,50,75,100 CPU_STEP_EVERY=20s \
python3 -m obs_simulator
```

4) CPU 抖动（围绕 50，幅度 50）

```bash
SIM_DURATION=120s SIM_INTERVAL=1s CPU_MODE=jitter CPU_TARGET_PCT=50 CPU_JITTER_PCT=50 \
python3 -m obs_simulator
```

5) CPU 尖刺（平时 10，每 30s 尖刺到 100 持续 3s）

```bash
SIM_DURATION=120s SIM_INTERVAL=1s CPU_MODE=spike CPU_TARGET_PCT=10 CPU_SPIKE_PCT=100 CPU_SPIKE_EVERY=30s CPU_SPIKE_LAST=3s \
python3 -m obs_simulator
```

### 6.2 内存用例（0~100%）

1) MEM 0% 基线

```bash
SIM_DURATION=60s SIM_INTERVAL=1s MEM_TARGET_PCT=0 python3 -m obs_simulator
```

2) MEM 100%（按 cgroup 上限）

```bash
SIM_DURATION=60s SIM_INTERVAL=1s MEM_TARGET_PCT=100 python3 -m obs_simulator
```

3) MEM 阶跃（0→25→50→75→100）

```bash
SIM_DURATION=120s SIM_INTERVAL=2s MEM_MODE=step MEM_STEP_SERIES=0B,256Mi,512Mi,768Mi,1Gi MEM_STEP_EVERY=20s \
python3 -m obs_simulator
```

> 如果你的 cgroup 限额不是 1Gi，推荐改用 `MEM_TARGET_PCT` 分段跑来保持严格 0~100%。

4) MEM 纯百分比分段（严格 0→25→50→75→100）

```bash
for pct in 0 25 50 75 100; do
  SIM_DURATION=30s SIM_INTERVAL=1s MEM_MODE=steady MEM_TARGET_PCT=${pct} python3 -m obs_simulator
done
```

### 6.3 GPU 用例（0~100%，需 CUDA + torch）

1) GPU Util 0~100 阶跃

```bash
SIM_DURATION=120s SIM_INTERVAL=2s GPU_ENABLE=true GPU_MODE=step GPU_UTIL_STEP_SERIES=0,25,50,75,100 GPU_UTIL_STEP_EVERY=20s \
python3 -m obs_simulator
```

2) GPU Util 尖刺到 100

```bash
SIM_DURATION=120s SIM_INTERVAL=1s GPU_ENABLE=true GPU_MODE=spike GPU_UTIL_TARGET_PCT=20 GPU_UTIL_SPIKE_PCT=100 GPU_UTIL_SPIKE_EVERY=30s GPU_UTIL_SPIKE_LAST=3s \
python3 -m obs_simulator
```

### 6.4 日志与 DB 用例（按 0~100 控制占比）

1) 日志错误率 0~100

```bash
SIM_DURATION=120s SIM_INTERVAL=1s LOG_ENABLE=true LOG_RATE=100 LOG_ERROR_PCT=0 python3 -m obs_simulator
SIM_DURATION=120s SIM_INTERVAL=1s LOG_ENABLE=true LOG_RATE=100 LOG_ERROR_PCT=100 python3 -m obs_simulator
```

2) DB 读比例 0~100（需 DB_URI）

```bash
SIM_DURATION=120s SIM_INTERVAL=1s DB_ENABLE=true DB_URI='postgres://user:pass@127.0.0.1:5432/test' DB_TYPE=postgres DB_QPS=200 DB_READ_PCT=0 python3 -m obs_simulator
SIM_DURATION=120s SIM_INTERVAL=1s DB_ENABLE=true DB_URI='postgres://user:pass@127.0.0.1:5432/test' DB_TYPE=postgres DB_QPS=200 DB_READ_PCT=100 python3 -m obs_simulator
```

## 7) 推荐迭代约束（团队规范）

建议把以下约束写进评审标准：

1. 所有新模块必须接入 `runtime` 统一生命周期
2. 所有新 env 必须补充到 `monitor_env.md`
3. 所有新场景至少补 1 个单测（正常/边界/异常）
4. 默认行为必须能在无外部依赖下直接启动（GPU/DB/KB 失败要降级不崩）
