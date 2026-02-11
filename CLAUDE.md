# CLAUDE.md — AI 助手指南

## 项目概述

ORDINAL-SPATIAL: 序空间推理基准评估框架。
评估视觉语言模型 (VLM) 的空间推理能力。

## 当前状态

### 已完成

- [x] 渲染管线: Blender 5.0+ 多视角渲染（GPU/CPU）
- [x] 扁平数据集生成: 单进程 (`os-benchmark`) + 多 GPU (`build_parallel.py`)
- [x] 多生成+裁剪策略: 物体数量严格均分
- [x] 数据集验证 (`os-validate`)
- [x] DSL 核心模型: ObjectSpec, OSD, QRR/TRR 约束
- [x] 评估指标: T1/T2/T3 (metrics.py, consistency.py, constraint_diff.py)
- [x] VLM 约束提取智能体: 单视角/多视角 (agents/)
- [x] VLM 提示词系统: 7 种约束定义、思考角度、组合数验证、自适应 token
- [x] 约束求解器: 梯度下降重建 (reconstruction/)
- [x] 感知上界设计文档: docs/plans/2026-02-10-perception-upper-bound-brainstorm.md
- [x] 文档: README, docs/usage.md, docs/architecture.md

### 待完成 / 已知问题

- [ ] `baselines/run_baseline.py` 仍使用旧版 split 格式 — 需适配扁平模式
- [ ] 多 GPU 并行裁剪后有孤立图片文件残留（不影响正确性，浪费磁盘）
- [ ] `_trim_to_balanced()` 在 build_benchmark 和 multi_gpu_builder 中重复实现 — 可提取为共享工具
- [ ] 尚未在真实 Blender 环境中跑通端到端测试
- [ ] 评估管线 (tasks/, baselines/) 尚未与扁平数据集完全对齐
- [ ] 感知上界 (PerceptionAgent): SAM2 + UniDepth V2 管线 — 设计完成，代码待实现
- [ ] VLM 端到端评估: 实际调用 VLM API 进行约束提取 — 需配置 API key

## 项目结构

```
ordinal-spatial/
├── src/ordinal_spatial/        # 主包 (src layout)
│   ├── dsl/                    # 核心: 数据模型、谓词、比较器
│   ├── generation/             # 约束提取与生成
│   ├── evaluation/             # T1/T2/T3 评估指标
│   ├── reconstruction/         # 约束求解与场景重建
│   ├── agents/                 # VLM 约束提取智能体
│   ├── baselines/              # 基线模型 (⚠️ 部分仍用旧 split 格式)
│   ├── tasks/                  # T1/T2/T3 任务运行器
│   ├── prompts/                # VLM 提示模板
│   ├── rendering/              # Blender 5.0+ 渲染
│   ├── scripts/                # CLI 脚本实现
│   ├── cli/                    # CLI 入口
│   └── utils/                  # 工具函数
├── scripts/                    # 独立脚本
│   ├── build_parallel.py       # 多 GPU 并行生成
│   └── lib/                    # 并行构建库
├── tests/                      # 测试
└── docs/                       # 文档
```

## 数据集格式（v2.0 扁平模式）

无 train/val/test 划分。输出结构：

```
data/
├── dataset.json          # 扁平列表 [{scene_id, image_path, ...}, ...]
├── dataset_info.json     # 配置参数和统计信息
├── images/
│   ├── single_view/      # scene_NNNNNN.png
│   └── multi_view/       # scene_NNNNNN/view_{0-3}.png
└── metadata/             # scene_NNNNNN.json (含约束)
```

生成策略: 多生成 ~10% → 按 n_objects 分组 → 每组裁剪到 `n // levels` → 严格均分。

## 代码规范

- **缩进**: 2 空格
- **行宽**: 80 字符
- **命名**: snake_case（函数）, PascalCase（类）
- **注释**: 中文注释
- **输出**: print/log/LLM 字符串用英文
- **Blender**: 仅支持 5.0+（无旧版兼容代码）

## 开发命令

```bash
# 安装
uv sync --extra dev

# 测试
uv run pytest

# CLI 工具
uv run os-benchmark --help      # 单进程数据集生成
uv run os-validate --help       # 扁平数据集验证
uv run os-baseline --help       # 基线评估
uv run os-agent --help          # VLM 智能体

# 多 GPU 并行生成
python scripts/build_parallel.py --help
```

## 关键文件

| 文件 | 用途 | 状态 |
|------|------|------|
| `dsl/schema.py` | 核心数据模型 (ObjectSpec, OSD, Constraints) | ✅ |
| `dsl/predicates.py` | QRR/TRR 约束计算 | ✅ |
| `evaluation/metrics.py` | T1/T2/T3 评估指标 | ✅ |
| `evaluation/constraint_diff.py` | Constraint-Diff 评估 | ✅ |
| `agents/vlm_constraint_agent.py` | VLM 约束提取 (Task-2/3) | ✅ |
| `agents/prompts/constraint_extraction.py` | VLM 提示词 (7 约束定义+思考角度) | ✅ |
| `agents/blender_constraint_agent.py` | Blender 真值提取 (Task-1) | ✅ |
| `rendering/render_multiview.py` | Blender 多视角渲染 (--prefix) | ✅ |
| `rendering/blender_utils.py` | Blender API 封装 | ✅ |
| `scripts/build_benchmark.py` | 单进程数据集构建 (os-benchmark) | ✅ |
| `scripts/validate_benchmark.py` | 扁平数据集验证 (os-validate) | ✅ |
| `reconstruction/pipeline.py` | 端到端重建流水线 | ✅ |
| `baselines/run_baseline.py` | 基线评估运行器 | ⚠️ 待适配扁平模式 |

## 生成数据集

```bash
# 单进程 (os-benchmark)
uv run os-benchmark -o ./data -b $BLENDER_PATH -n 1000 --use-gpu

# 多 GPU 并行
python scripts/build_parallel.py -o ./data -n 10000 --n-gpus 4 -y

# 验证
uv run os-validate -d ./data
```

两条路径统一使用扁平模式，多生成 ~10% 后按物体数量裁剪到精确均分。

## VLM 约束提取 (三种输入模式)

三种模式统一输出 `ConstraintSet`，包含 7 种约束类型:
axial, topology, size, occlusion, closer, trr, qrr。

```python
# Mode 1: Blender 元数据 JSON → 真值 (Task-1)
from ordinal_spatial.agents import BlenderConstraintAgent
agent = BlenderConstraintAgent()
cs = agent.extract_from_single_view("metadata/scene_000001.json", tau=0.10)

# Mode 2: 多视角图像 → VLM 提取 (Task-2)
from ordinal_spatial.agents import VLMConstraintAgent, VLMAgentConfig
config = VLMAgentConfig(model="openai/gpt-4o")
agent = VLMConstraintAgent(config)
cs = agent.extract_from_multi_view(
    images=["view_0.png", "view_1.png", "view_2.png", "view_3.png"],
    objects=[{"id": "obj_0", "color": "red", "shape": "cube",
              "size": "large", "material": "metal"}, ...],
    tau=0.10,
)

# Mode 3: 单视角图像 → VLM 提取 (Task-3)
cs = agent.extract_from_single_view(
    image="scene.png",
    objects=[...],  # GT 物体列表
    tau=0.10,
)
```

提示词特性:
- 7 种约束类型的正式定义 (字段+规则+示例)
- 视觉推理角度提示 (深度线索、透视、地平面、遮挡序)
- 自动组合数计算和验证 (C(N,2), C(N,3), 3×C(N,4))
- 多视角: view-invariant vs view-dependent 分类
- 自适应 token: N≤5→4096, N≤8→8192, N>8→16384

## Blender 5.0+ 关键要点

- `bpy.ops.wm.append()` 需要 `filepath/directory/filename` 三参数
- GPU 初始化: 调用 `refresh_devices()` 刷新设备列表，CUDA 优先于 OptiX
- `tile_size` 在 Cycles X (Blender 3.0+) 中已移除，设置会崩溃
- 每次 `open_mainfile()` 后必须重新调用 `_setup_gpu()`
- 节点名称可能被本地化（如 '材质输出'），按 `node.type` 查找而非名称

## 环境

- **Python**: >=3.10
- **Blender**: 5.0+ only
- **包管理**: uv（推荐）或 pip
- **目标平台**: Linux CUDA + macOS
