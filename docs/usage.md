# 使用指南

## 环境配置

### 基本安装

```bash
cd ordinal-spatial

# 推荐: 使用 uv（自动管理 Python 版本和虚拟环境）
uv sync

# 安装所有可选依赖
uv sync --extra dev
```

### 可选依赖组

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

### Blender 5.0+ 配置

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

## 核心用法

### 1. 生成基准数据集

数据集为**扁平结构**，无 train/val/test 划分，物体数量严格均分。

```bash
# 生成 1000 个场景（默认物体范围 3-10）
uv run os-benchmark \
    -o ./data/benchmark \
    -b $BLENDER_PATH \
    -n 1000

# 高清 1080p + GPU 加速
uv run os-benchmark \
    -o ./data/benchmark_hd \
    -b $BLENDER_PATH \
    -n 1000 \
    --width 1920 --height 1080 \
    --render-samples 512 \
    --use-gpu

# 自定义参数
uv run os-benchmark \
    -o ./data/custom \
    -b $BLENDER_PATH \
    -n 500 \
    --min-objects 4 --max-objects 8 \
    --tau 0.05
```

**生成策略：** 系统会多生成约 10% 的场景，按物体数量分组后裁剪到目标数量，
确保每个物体数量级别的场景数严格均等。

**os-benchmark 参数：**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-o, --output-dir` | 必填 | 输出目录 |
| `-b, --blender-path` | `blender` | Blender 路径 |
| `-n, --n-scenes` | 1000 | 总场景数 |
| `--min-objects` | 3 | 最少物体数 |
| `--max-objects` | 10 | 最多物体数 |
| `--tau` | 0.10 | 约束容差阈值 |
| `--width / --height` | 480 / 320 | 图片分辨率 |
| `--render-samples` | 256 | Cycles 采样数 |
| `--use-gpu` | 否 | 启用 GPU 渲染 |
| `--seed` | 42 | 随机种子 |

### 2. 多 GPU 并行生成

适用于服务器上大规模数据集生成。

```bash
# 4 GPU 并行
python scripts/build_parallel.py \
    -o ./data/large \
    -n 10000 \
    --n-gpus 4 \
    --quality high \
    --seed 42 \
    --blender $BLENDER_PATH \
    -y

# CPU 单进程
python scripts/build_parallel.py \
    -o ./data/test \
    -n 80 \
    --n-gpus 0 \
    -y
```

**质量预设：**

| 质量 | 采样数 | 分辨率 |
|------|--------|--------|
| `draft` | 64 | 480x320 |
| `normal` | 256 | 1024x768 |
| `high` | 512 | 1024x768 |

### 3. 验证数据集

```bash
# 验证数据集完整性
uv run os-validate -d ./data/benchmark

# 输出 JSON 报告
uv run os-validate -d ./data/benchmark -o report.json
```

验证器检查：
- 目录结构（images/, metadata/, dataset.json）
- 图片文件是否存在且可读
- 元数据 JSON 格式正确性
- 物体数量分布均衡性

### 4. 运行基线评估

```bash
# Oracle 基线（真值，作为上界参考）
uv run os-baseline --baseline oracle --task t1-q --data ./data

# VLM 直接预测
uv run os-baseline --baseline vlm_direct --task t2 --model openai/gpt-4o --data ./data

# 混合基线（预测-验证-修复）
uv run os-baseline --baseline hybrid --task t2 --model openai/gpt-4o --data ./data
```

**基线类型：**

| 基线 | 说明 |
|------|------|
| `oracle` | 真值基线（100% 准确率） |
| `vlm_direct` | VLM 零样本预测 |
| `vlm_cot` | VLM + 思维链 |
| `hybrid` | 预测-验证-修复循环 |
| `embedding` | 序嵌入优化 |

### 5. VLM 约束提取

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

### 6. 场景重建

```python
from ordinal_spatial.reconstruction.pipeline import reconstruct_and_visualize

# 从约束重建 3D 布局
result = reconstruct_and_visualize(
    constraints_path="constraints.json",
    output_path="reconstruction.json"
)
```

## 服务器部署

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

# 5. 生成数据集
uv run os-benchmark -o ./data -b $BLENDER_PATH -n 1000 --use-gpu
```

### macOS 本地开发

```bash
brew install --cask blender
export BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender

uv sync --extra dev
uv run pytest
```

## 常见问题

### ImportError: cannot import ordinal_spatial

确保在项目根目录运行，且虚拟环境已激活:
```bash
cd ordinal-spatial
uv sync
uv run python -c "import ordinal_spatial"
```

### GPU 渲染不工作（占用率 0%）

常见原因：
1. **Blender 版本过低**：需要 5.0+
2. **CUDA 驱动未安装**：运行 `nvidia-smi` 检查
3. **OptiX 初始化失败**：代码会自动回退到 CUDA
4. 使用 `--use-gpu` 参数确保启用

### VLM API 调用失败

确保设置了 API Key:
```bash
export OPENROUTER_API_KEY=your-key
```
