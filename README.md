# ORDINAL-SPATIAL

**序空间推理基准评估框架 / Ordinal Spatial Reasoning Benchmark**

评估视觉语言模型 (VLM) 在序空间推理任务上的能力。

A benchmark for evaluating Vision-Language Models on ordinal spatial reasoning tasks.

---

## 功能概览 / Overview

| 功能 | 说明 |
|------|------|
| **场景渲染** | 使用 Blender 5.0+ 渲染合成 3D 场景 (单视角 + 多视角) |
| **约束提取** | 从场景中提取 QRR/TRR 等空间约束 |
| **VLM 评估** | 评估 VLM 在 T1 (分类) / T2 (提取) / T3 (重建) 任务上的表现 |
| **智能体** | VLM 约束提取智能体 (Task-2/3) + Blender 真值提取 (Task-1) |

## 快速开始 / Quick Start

### 环境安装 / Installation

```bash
# 克隆仓库
git clone https://github.com/tasd12-ty/Redraw-3d.git ordinal-spatial
cd ordinal-spatial

# 使用 uv 安装 (推荐)
uv sync

# 安装开发依赖
uv sync --extra dev

# 或使用 pip
pip install -e ".[dev]"
```

### 验证安装 / Verify Installation

```bash
uv run python -c "import ordinal_spatial; print(ordinal_spatial.__version__)"
uv run pytest
```

### 生成基准数据集 / Generate Benchmark

```bash
# 小型测试数据集
uv run os-benchmark \
    --output-dir ./data/benchmark_tiny \
    --blender-path /path/to/blender \
    --tiny

# 完整数据集
uv run os-benchmark \
    --output-dir ./data/benchmark_full \
    --blender-path /path/to/blender \
    --n-train 1000 --n-val 200 --n-test 500

# 验证数据集
uv run os-validate --dataset-dir ./data/benchmark_tiny
```

### 运行基线评估 / Run Baseline

```bash
# Oracle 基线 (100% 准确率)
uv run os-baseline --baseline oracle --task t1-q --data ./data --split test_iid

# VLM 直接预测
uv run os-baseline --baseline vlm_direct --task t2 --model openai/gpt-4o
```

### VLM 约束提取 / VLM Constraint Extraction

```bash
# 单视角提取 (Task-3)
uv run os-agent extract --image scene.png --output constraints.json --tau 0.10

# 多视角提取 (Task-2)
uv run os-agent extract --images view1.png view2.png view3.png --output constraints.json

# 使用自定义模型
uv run os-agent extract --image scene.png --model openai/gpt-4o --output constraints.json
```

## 项目结构 / Project Structure

```
ordinal-spatial/
├── pyproject.toml              # 项目配置 (uv)
├── src/ordinal_spatial/        # 主包
│   ├── dsl/                    # 核心数据模型与谓词
│   ├── generation/             # 约束生成
│   ├── evaluation/             # 评估指标
│   ├── reconstruction/         # 场景重建求解器
│   ├── agents/                 # VLM 约束提取智能体
│   ├── baselines/              # 基线实现
│   ├── tasks/                  # T1/T2/T3 任务运行器
│   ├── prompts/                # VLM 提示模板
│   ├── rendering/              # Blender 渲染 (5.0+)
│   ├── scripts/                # 脚本实现
│   ├── cli/                    # CLI 入口
│   └── utils/                  # 工具函数
├── scripts/                    # 独立脚本
├── tests/                      # 测试
└── docs/                       # 文档
```

## 任务定义 / Task Definitions

| 任务 | 输入 | 输出 | 指标 |
|------|------|------|------|
| **T1-Q** | 4 个对象 + 场景 | 距离关系 (<, >, ~=) | Accuracy, F1 |
| **T1-C** | 3 个对象 + 场景 | 时钟位置 (时 + 象限) | Hour/Quadrant Accuracy |
| **T2** | 图像 + 对象列表 | 完整 QRR/TRR 约束集 | Precision, Recall, F1 |
| **T3** | 约束集 | 3D 点配置 | NRMS (Procrustes), Constraint % |

## 约束类型 / Constraint Types

| 类型 | 说明 |
|------|------|
| **QRR** | 距离比较 (Quantitative Ratio Relations) |
| **TRR** | 时钟方向 (Ternary Reference Relations) |
| **Topology** | 相离/相切关系 |
| **Occlusion** | 遮挡关系 |
| **Axial** | 左/右/前/后关系 |
| **Size** | 大小比较 |
| **Closer** | 距离排序 |

## Blender 渲染 / Blender Rendering

本项目仅支持 **Blender 5.0+**。

### 首次设置 / First-time Setup

```bash
cd src/ordinal_spatial/rendering

# 方式一: 一键设置
./setup_blender5.sh /path/to/blender

# 方式二: 手动设置
blender --background --python create_base_scene.py
blender --background --python create_materials.py
blender --background --python create_shapes.py
```

### 渲染图像 / Render Images

```bash
# 单视角渲染
blender --background --python render_images.py -- \
    --base_scene_blendfile data/base_scene_v5.blend \
    --material_dir data/materials_v5 \
    --shape_dir data/shapes_v5 \
    --num_images 10 \
    --use_gpu 1

# 多视角渲染
blender --background --python render_multiview.py -- \
    --num_images 10 \
    --n_views 4 \
    --use_gpu 1
```

## 服务器部署 / Server Deployment

```bash
# 1. 克隆并安装
git clone https://github.com/tasd12-ty/Redraw-3d.git ordinal-spatial
cd ordinal-spatial
uv sync --extra dev

# 2. 配置 Blender 路径 (环境变量)
export BLENDER_PATH=/path/to/blender

# 3. 配置 API Key (如需 VLM 评估)
export OPENROUTER_API_KEY=your-key

# 4. 运行
uv run os-benchmark --output-dir ./data/benchmark --blender-path $BLENDER_PATH --small
```

### GPU 渲染 (Linux CUDA)

```bash
# 确保 CUDA 驱动已安装
nvidia-smi

# 使用 GPU 渲染
blender --background --python render_images.py -- --use_gpu 1

# 自动检测: CUDA -> OptiX -> HIP -> OneAPI
```

### 多 GPU 并行生成 / Multi-GPU Parallel Generation

```bash
# CPU 渲染 (单进程)
python scripts/build_parallel.py --output ./data/test --size tiny --n-gpus 0

# 单 GPU
python scripts/build_parallel.py --output ./data/small --size small --n-gpus 1

# 多 GPU 并行
python scripts/build_parallel.py --output ./data/full --size large --n-gpus 8

# 指定随机种子 (默认 42, 确保可复现)
python scripts/build_parallel.py --output ./data/full --size large --n-gpus 4 --seed 42
```

| Size preset | Scenes | Splits |
|-------------|--------|--------|
| `tiny` | 40 | 8 per split |
| `small` | 1,520 | train 800 / val 160 / test 400+80+80 |
| `medium` | 15,200 | train 8k / val 1.6k / test 4k+800+800 |
| `large` | 152,000 | train 80k / val 16k / test 40k+8k+8k |

## Python API

```python
from ordinal_spatial.dsl.comparators import compare
from ordinal_spatial.dsl.predicates import compute_qrr, compute_trr
from ordinal_spatial.agents import VLMConstraintAgent, BlenderConstraintAgent
from ordinal_spatial.evaluation.metrics import compute_t2_metrics
from ordinal_spatial.evaluation.consistency import check_qrr_consistency
```

## 开发 / Development

```bash
# 安装开发依赖
uv sync --extra dev

# 运行测试
uv run pytest

# 运行带覆盖率的测试
uv run pytest --cov=ordinal_spatial

# 代码检查
uv run ruff check src/
```

## 许可证 / License

BSD-3-Clause. See [LICENSE](LICENSE).
