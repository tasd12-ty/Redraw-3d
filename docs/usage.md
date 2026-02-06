# 使用指南 / Usage Guide

## 环境配置 / Environment Setup

### 基本安装 / Basic Installation

```bash
cd ordinal-spatial

# 推荐: 使用 uv (自动管理 Python 版本和虚拟环境)
uv sync

# 安装所有可选依赖
uv sync --extra dev
```

### 可选依赖组 / Optional Dependency Groups

| 组名 | 包含 | 用途 |
|------|------|------|
| `ml` | torch | GPU 约束求解 |
| `tokenizer` | tiktoken, transformers | Token 计数 |
| `viz` | seaborn, plotly | 高级可视化 |
| `test` | pytest, pytest-cov | 测试 |
| `dev` | 以上全部 | 开发 |

```bash
# 仅安装 ML 依赖
uv sync --extra ml

# 安装多个组
uv sync --extra ml --extra viz
```

### Blender 5.0+ 配置 / Blender Setup

```bash
# 下载 Blender 5.0+
# Linux: https://www.blender.org/download/
# macOS: brew install --cask blender

# 设置环境变量
export BLENDER_PATH=/path/to/blender

# 首次运行资源生成
cd src/ordinal_spatial/rendering
$BLENDER_PATH --background --python create_base_scene.py
$BLENDER_PATH --background --python create_materials.py
$BLENDER_PATH --background --python create_shapes.py
```

## 核心用法 / Core Usage

### 1. 生成基准数据集

```bash
# 微型数据集 (测试用，~40 场景)
uv run os-benchmark \
    --output-dir ./data/tiny \
    --blender-path $BLENDER_PATH \
    --tiny

# 小型数据集 (~1500 场景)
uv run os-benchmark \
    --output-dir ./data/small \
    --blender-path $BLENDER_PATH \
    --small

# 自定义大小
uv run os-benchmark \
    --output-dir ./data/custom \
    --blender-path $BLENDER_PATH \
    --n-train 500 --n-val 100 --n-test 200
```

**数据集划分 / Dataset Splits:**

| 划分 | 对象数 | Tau | 用途 |
|------|--------|-----|------|
| train | 4-10 | 0.10 | 训练 |
| val | 4-10 | 0.10 | 验证 |
| test_iid | 4-10 | 0.10 | IID 测试 |
| test_comp | 10-15 | 0.10 | 组合泛化 (更多对象) |
| test_hard | 4-10 | 0.05 | 困难测试 (更严格阈值) |

### 2. 验证数据集

```bash
uv run os-validate --dataset-dir ./data/tiny
```

### 3. 运行基线评估

```bash
# Oracle 基线 (真值，作为上界参考)
uv run os-baseline --baseline oracle --task t1-q --data ./data --split test_iid

# VLM 直接预测
uv run os-baseline --baseline vlm_direct --task t2 --model openai/gpt-4o --data ./data

# 混合基线 (预测-验证-修复)
uv run os-baseline --baseline hybrid --task t2 --model openai/gpt-4o --data ./data
```

**基线类型 / Baseline Types:**

| 基线 | 说明 |
|------|------|
| `oracle` | 真值基线 (100% 准确率) |
| `vlm_direct` | VLM 零样本预测 |
| `vlm_cot` | VLM + 思维链 |
| `hybrid` | 预测-验证-修复循环 |
| `embedding` | 序嵌入优化 |

### 4. VLM 约束提取

```python
from ordinal_spatial.agents import VLMConstraintAgent, VLMAgentConfig

# 配置智能体
config = VLMAgentConfig(model="google/gemma-3-27b-it")
agent = VLMConstraintAgent(config)

# 单视角提取
result = agent.extract_from_single_view("scene.png", tau=0.10)

# 多视角提取
result = agent.extract_from_multi_view(
    ["view1.png", "view2.png", "view3.png"],
    tau=0.10
)
```

### 5. 场景重建

```python
from ordinal_spatial.reconstruction.pipeline import reconstruct_and_visualize

# 从约束重建 3D 布局
result = reconstruct_and_visualize(
    constraints_path="constraints.json",
    output_path="reconstruction.json"
)
```

## 服务器部署 / Server Deployment

### Linux CUDA 服务器

```bash
# 1. 安装 uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. 克隆并安装
git clone <repo-url> ordinal-spatial
cd ordinal-spatial
uv sync --extra dev

# 3. 安装 headless Blender
wget https://download.blender.org/release/Blender5.0/blender-5.0.1-linux-x64.tar.xz
tar xf blender-5.0.1-linux-x64.tar.xz
export BLENDER_PATH=$PWD/blender-5.0.1-linux-x64/blender

# 4. 配置环境变量
export OPENROUTER_API_KEY=your-key  # 如需 VLM 评估
export CUDA_VISIBLE_DEVICES=0      # 指定 GPU

# 5. 运行
uv run os-benchmark --output-dir ./data --blender-path $BLENDER_PATH --small
```

### macOS 本地开发

```bash
# 安装 Blender
brew install --cask blender
export BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender

# 安装并运行
uv sync --extra dev
uv run pytest
```

### 多 GPU 数据集生成

```bash
# 多 GPU 并行生成 (按实际 GPU 数量设置 --n-gpus)
uv run python scripts/build_parallel.py \
    --output ./data/benchmark \
    --size small \
    --n-gpus 4 \
    --seed 42 \
    --blender $BLENDER_PATH
```

## 常见问题 / Troubleshooting

### ImportError: cannot import ordinal_spatial

确保在项目根目录运行，且虚拟环境已激活:
```bash
cd ordinal-spatial
uv sync
uv run python -c "import ordinal_spatial"
```

### Blender GPU 渲染失败

1. 检查 GPU 驱动: `nvidia-smi`
2. 使用 CPU 渲染测试: `--use_gpu 0`
3. 代码自动尝试: CUDA -> OptiX -> HIP -> OneAPI

### VLM API 调用失败

确保设置了 API Key:
```bash
export OPENROUTER_API_KEY=your-key
```
