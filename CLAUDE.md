# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Isaac Lab is a GPU-accelerated robotics simulation and reinforcement learning framework built on NVIDIA Isaac Sim. It provides modular environments, sensor abstractions, and integrations with popular RL frameworks (RSL-RL, Stable-Baselines3, SKRL, RL-Games, Ray RLlib).

- **Version**: 2.3.2
- **Python**: 3.11 (supports 3.10+)
- **Key dependencies**: Isaac Sim 5.1, PyTorch 2.7+, CUDA 12.8, `numpy<2`, `gymnasium==1.2.1`

## Essential Commands

All commands go through the main shell script:

```bash
./isaaclab.sh --install [all|none|<framework>]  # Install packages + RL frameworks
./isaaclab.sh --python <script.py> [args]        # Run script with Isaac Sim Python env
./isaaclab.sh --format                           # Run pre-commit hooks (ruff + linting)
./isaaclab.sh --test                             # Run all pytest tests
./isaaclab.sh --docs                             # Build Sphinx documentation
./isaaclab.sh --sim [args]                       # Launch Isaac Sim directly
```

**Running a single test file:**
```bash
python -m pytest source/isaaclab/test/test_math_utils.py -v
python -m pytest source/isaaclab_tasks/test/ -k "test_cartpole"
```

**List all registered environments:**
```bash
./isaaclab.sh --python scripts/environments/list_envs.py
```

**Training with an RL framework:**
```bash
./isaaclab.sh --python scripts/reinforcement_learning/skrl/train.py --task=Isaac-Ant-v0 --headless
./isaaclab.sh --python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Humanoid-v0 --num_envs=256
```

## Code Style & Quality

- **Formatter/Linter**: Ruff (line length 120, rules E/W/F/I/UP/C90/SIM/RET)
- **Type checking**: Pyright in `basic` mode
- **Pre-commit hooks** handle license headers, trailing whitespace, spell checking
- All Python and YAML files automatically get license headers on commit
- Run `./isaaclab.sh --format` before committing

## Repository Architecture

```
source/
  isaaclab/          # Core framework
  isaaclab_tasks/    # Pre-built environments and benchmark tasks
  isaaclab_rl/       # Thin wrappers for RL framework integrations
  isaaclab_assets/   # Robot and object asset configurations
  isaaclab_mimic/    # Imitation learning / motion capture (Apache 2.0)
  isaaclab_contrib/  # Community extensions
scripts/             # Runnable training/demo/tutorial scripts
docs/                # Sphinx documentation source
tools/               # Test runner, project templates, utilities
apps/                # Isaac Kit application configs
docker/              # Container configurations
```

Each `source/*` package is installed in editable mode (`pip install -e`). Package metadata lives in `config/extension.toml` within each package.

### Core Framework (`source/isaaclab/`)

Key modules:

| Module | Purpose |
|--------|---------|
| `isaaclab/sim/` | Physics simulation abstraction (wraps Isaac Sim APIs) |
| `isaaclab/envs/` | Environment base classes (`ManagerBasedRLEnv`, `DirectRLEnv`) |
| `isaaclab/managers/` | Modular MDP components (action, observation, reward, termination, event) |
| `isaaclab/assets/` | Articulation, rigid body, and deformable object interfaces |
| `isaaclab/sensors/` | Camera, IMU, ray caster, contact sensor abstractions |
| `isaaclab/scene/` | Scene graph and world management |
| `isaaclab/terrains/` | Procedural terrain generation |
| `isaaclab/controllers/` | IK solvers, operational space control |
| `isaaclab/devices/` | Teleoperation (gamepad, keyboard, SpaceMouse, OpenXR) |
| `isaaclab/utils/` | Buffers, math, noise, Warp GPU kernels |

### Two Environment Development Patterns

**1. Manager-Based** (`isaaclab_tasks/manager_based/`) — Recommended for reusable, modular environments:
- Define separate config classes for actions, observations, rewards, terminations, and events
- Compose via `SceneCfg` + `ManagerBasedRLEnvCfg`
- Managers handle batching and vectorization automatically

**2. Direct RL** (`isaaclab_tasks/direct/`) — For custom environments with less abstraction:
- Subclass `DirectRLEnv` and implement `_get_observations`, `_get_rewards`, `_get_dones`
- Faster to prototype, less modular

### Configuration System

Environments use Hydra `@configclass` dataclasses for all configs. This enables:
- CLI overrides: `--task=Isaac-Ant-v0 num_envs=512 seed=42`
- Nested config composition
- Type-safe configuration

### Vectorization Model

Environments run N copies in parallel (controlled by `--num_envs`). All tensors are batched along the first dimension with shape `(num_envs, ...)`. Code must be fully vectorized—no Python loops over environments.

### Task Registration

Tasks register with Gymnasium via `gym.register()` in `__init__.py` files. The task name format is `Isaac-<TaskName>-v<N>`. To add a new task, add a registration entry and ensure the module is imported.

## Testing

Tests mirror source structure: each module's tests live in `source/*/test/`. The pytest marker `isaacsim_ci` gates CI-only tests.

```bash
python -m pytest source/isaaclab/test/            # Core framework tests
python -m pytest source/isaaclab_tasks/test/      # Environment tests
python -m pytest source/isaaclab_rl/test/         # RL wrapper tests
```

## Training Hardware (本机实测 2026-05-06)

### GPU 规格

| 参数 | 数值 |
|------|------|
| 型号 | NVIDIA GeForce RTX 5070 |
| 显存 | 12GB GDDR7（可用 11.940GB） |
| GPU 核心频率 | 2872 MHz |
| 显存频率 | 13801 MHz |
| TDP | 250W |
| PCIe | GEN 5 × 16 |
| 训练时典型功耗 | ~108W（43% TDP） |
| 训练时典型温度 | ~55°C |

### Isaac Lab 训练实测性能（Go2 平地行走，SKRL PPO）

| 配置 | it/s | env_steps/s | GPU 利用率 | 显存占用 | CPU |
|------|------|-------------|-----------|---------|-----|
| 2048 envs（优化前） | 74.70 | ~153k | 58% | 3.1GB (26%) | ~200% |
| 4096 envs（优化后） | ~60 | ~246k | 70% | 4.52GB (38%) | ~180% |

- PCIe 带宽：RX ~1.28 GiB/s，TX ~1.45 GiB/s（不是瓶颈）
- **真正瓶颈**：物理步之间的串行依赖，GPU 无法跑满 100%

### 训练时长估算（4096 envs，~59 it/s）

| timesteps | 预计时长 |
|-----------|---------|
| 2,500,000 | ~11.6 小时（实测验证） |
| 8,000,000 | ~37 小时 |
| 50,000,000 | ~10 天 |
| 100,000,000 | ~20 天 |

### 推荐配置

```python
# Go2 locomotion 最优配置（本机）
num_envs = 4096          # 显存余量充足，GPU 利用率提升至 70%

# ContactSensorCfg 注意事项：
# - prim_path 必须覆盖机器人全部链接（Go2 共 19 个），不能用子集
# - filter_prim_paths_expr 是"与哪些外部物体接触"的过滤器，不是限制传感器范围的
# - 用子集 prim_path 或错误的 filter_prim_paths_expr 会报 "did not match expected entries" 错误
contact_forces = ContactSensorCfg(
    prim_path="{ENV_REGEX_NS}/Robot/.*",  # 必须全身覆盖
    history_length=3,
    track_air_time=True,
)
```

## Important Constraints

- **`numpy<2`**: Hard requirement; numpy 2.x breaks Isaac Sim internals
- **Isaac Sim version pinning**: `isaacsim==5.1.0` — version mismatches cause silent failures
- **GPU required**: Most code paths require a CUDA-capable GPU; CPU fallbacks are limited
- **Warp**: GPU tensor operations use NVIDIA Warp (`warp-lang`), not standard PyTorch ops in some modules
- **Missing import suppression**: Pyright is configured to ignore missing Isaac Sim imports since they resolve at runtime
