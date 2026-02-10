# ORDINAL-SPATIAL

**序空间推理基准评估框架**

评估视觉语言模型 (VLM) 在序空间推理任务上的能力。使用 Blender 5.0+ 渲染合成 3D 场景，
自动提取空间约束，并评估 VLM 的空间理解表现。

---

## 功能概览

| 功能 | 说明 |
|------|------|
| **场景渲染** | Blender 5.0+ 渲染合成 3D 场景（单视角 + 多视角） |
| **均衡数据集** | 物体数量 3-10 严格均分，多生成后裁剪保证精确均衡 |
| **约束提取** | 从场景中提取 QRR/TRR 等空间约束 |
| **VLM 评估** | T1（分类）/ T2（提取）/ T3（重建）任务评估 |
| **智能体** | VLM 约束提取智能体（Task-2/3）+ Blender 真值提取（Task-1） |
| **多 GPU 并行** | 支持多 GPU 并行渲染大规模数据集 |

## 环境要求

- **Python** >= 3.10
- **Blender** 5.0+（仅支持此版本）
- **包管理**: uv（推荐）或 pip
- **GPU**（可选）: NVIDIA CUDA 显卡，推荐用于大规模生成

## 安装

```bash
git clone https://github.com/tasd12-ty/Redraw-3d.git ordinal-spatial
cd ordinal-spatial

# 使用 uv 安装（推荐）
uv sync

# 安装开发依赖
uv sync --extra dev

# 验证安装
uv run python -c "import ordinal_spatial; print('OK')"
```

### Blender 配置

```bash
# Linux 服务器: 下载 headless Blender
wget https://download.blender.org/release/Blender5.0/blender-5.0.1-linux-x64.tar.xz
tar xf blender-5.0.1-linux-x64.tar.xz
export BLENDER_PATH=$PWD/blender-5.0.1-linux-x64/blender

# macOS
brew install --cask blender
export BLENDER_PATH=/Applications/Blender.app/Contents/MacOS/Blender

# 首次运行：生成渲染资源文件
cd src/ordinal_spatial/rendering
$BLENDER_PATH --background --python create_base_scene.py
$BLENDER_PATH --background --python create_materials.py
$BLENDER_PATH --background --python create_shapes.py
cd -
```

---

## 数据集生成

数据集为**扁平结构**（无 train/val/test 划分），物体数量在 `min_objects` ~ `max_objects` 范围内严格均分。
系统会多生成约 10% 的场景，按物体数量分组后裁剪到目标数量，确保每个级别的场景数精确均等。

### 方式一：单进程生成（os-benchmark）

适用于单机、小规模生成。

```bash
uv run os-benchmark \
    -o <输出目录> \
    -b <Blender路径> \
    -n <场景数>
```

#### 使用示例

```bash
# 最简单的用法：生成 1000 个场景（CPU 渲染，默认分辨率）
uv run os-benchmark -o ./data/benchmark -b $BLENDER_PATH -n 1000

# GPU 加速渲染
uv run os-benchmark -o ./data/benchmark -b $BLENDER_PATH -n 1000 --use-gpu

# 1080p 高清 + GPU（适合最终数据集）
uv run os-benchmark \
    -o ./data/benchmark_hd \
    -b $BLENDER_PATH \
    -n 1000 \
    --width 1920 --height 1080 \
    --render-samples 512 \
    --use-gpu

# 1080p 高清，纯 CPU 渲染（GPU 被占用时）
uv run os-benchmark \
    -o ./data/benchmark_hd \
    -b $BLENDER_PATH \
    -n 1000 \
    --width 1920 --height 1080 \
    --render-samples 512

# 快速测试（少量场景、低分辨率）
uv run os-benchmark -o ./data/test -b $BLENDER_PATH -n 8

# 自定义物体数量范围和容差阈值
uv run os-benchmark \
    -o ./data/custom \
    -b $BLENDER_PATH \
    -n 500 \
    --min-objects 4 --max-objects 8 \
    --tau 0.05
```

#### os-benchmark 完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-o, --output-dir` | **必填** | 输出目录 |
| `-b, --blender-path` | `blender` | Blender 可执行文件路径 |
| `-n, --n-scenes` | 1000 | 最终保留的场景数（实际渲染约多 10%） |
| `--min-objects` | 3 | 每个场景的最少物体数 |
| `--max-objects` | 10 | 每个场景的最多物体数 |
| `--tau` | 0.10 | 约束容差阈值（越小越严格） |
| `--n-views` | 4 | 每个场景的多视角图片数 |
| `--width` | 480 | 渲染图片宽度（像素） |
| `--height` | 320 | 渲染图片高度（像素） |
| `--render-samples` | 256 | Cycles 渲染采样数（越高越清晰，越慢） |
| `--use-gpu` | 否 | 启用 GPU 渲染（加 `--use-gpu` 即开启） |
| `--camera-distance` | 12.0 | 相机到场景中心的距离 |
| `--elevation` | 30.0 | 相机仰角（度） |
| `--seed` | 42 | 随机种子（确保可复现） |

#### 分辨率与质量参考

| 用途 | width | height | render-samples | 预计单张耗时 |
|------|-------|--------|----------------|-------------|
| 快速测试 | 480 | 320 | 64 | ~2s (GPU) |
| 默认 | 480 | 320 | 256 | ~5s (GPU) |
| 中等质量 | 1024 | 768 | 256 | ~15s (GPU) |
| 1080p 高清 | 1920 | 1080 | 512 | ~30s (GPU) |
| 1080p CPU | 1920 | 1080 | 512 | ~3-5min (CPU) |

### 方式二：多 GPU 并行生成（build_parallel.py）

适用于服务器多卡大规模生成。

```bash
python scripts/build_parallel.py \
    -o <输出目录> \
    -n <场景数> \
    --n-gpus <GPU数量>
```

#### 使用示例

```bash
# 4 GPU 并行生成 10000 个场景（high 质量）
python scripts/build_parallel.py \
    -o ./data/large \
    -n 10000 \
    --n-gpus 4 \
    --quality high \
    --blender $BLENDER_PATH \
    -y

# 2 GPU，指定使用 GPU 4 和 5（其他 GPU 跑着大模型）
CUDA_VISIBLE_DEVICES=4,5 python scripts/build_parallel.py \
    -o ./data/benchmark \
    -n 1000 \
    --n-gpus 2 \
    --blender $BLENDER_PATH \
    -y

# CPU 模式（无 GPU 可用时）
python scripts/build_parallel.py \
    -o ./data/test \
    -n 80 \
    --n-gpus 0 \
    --blender $BLENDER_PATH \
    -y
```

#### build_parallel.py 完整参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-o, --output` | **必填** | 输出目录 |
| `-n, --n-scenes` | 1000 | 最终保留的场景数 |
| `--n-gpus` | 1 | GPU 数量（0 = CPU 模式） |
| `--quality` | normal | 渲染质量预设 |
| `--min-objects` | 3 | 最少物体数 |
| `--max-objects` | 10 | 最多物体数 |
| `--tau` | 0.10 | 约束容差阈值 |
| `--blender` | `$BLENDER_PATH` 或 `blender` | Blender 路径 |
| `--seed` | 42 | 随机种子 |
| `-y, --yes` | 否 | 跳过确认提示（服务器/脚本使用） |

#### 质量预设

| 预设 | 采样数 | 分辨率 | 适用场景 |
|------|--------|--------|---------|
| `draft` | 64 | 480×320 | 快速测试、调试 |
| `normal` | 256 | 1024×768 | 常规生成 |
| `high` | 512 | 1024×768 | 高质量最终数据集 |

### GPU 使用说明

#### GPU 渲染（推荐）

GPU 渲染速度比 CPU 快 10-50 倍。本项目 GPU 后端优先级：**CUDA > OptiX > HIP > OneAPI**。

```bash
# 检查 GPU 状态
nvidia-smi

# 单卡 GPU 渲染
uv run os-benchmark -o ./data -b $BLENDER_PATH -n 100 --use-gpu

# 指定特定 GPU（当多卡且部分被占用时）
CUDA_VISIBLE_DEVICES=2 uv run os-benchmark -o ./data -b $BLENDER_PATH -n 100 --use-gpu
```

#### GPU 被占用时的方案

| 场景 | 方案 |
|------|------|
| GPU 全被大模型占用 | 不加 `--use-gpu`，用 CPU 渲染 |
| 部分 GPU 空闲 | `CUDA_VISIBLE_DEVICES=4,5` 指定空闲卡 |
| GPU 显存充足（如 48GB L20，大模型占 30GB） | 可以共享同一张卡，Blender 通常占 2-4GB |

---

## 数据集验证

```bash
# 验证数据集完整性
uv run os-validate -d ./data/benchmark

# 输出 JSON 格式报告
uv run os-validate -d ./data/benchmark -o report.json

# 详细模式
uv run os-validate -d ./data/benchmark -v
```

验证器检查项：
- 目录结构完整性（images/、metadata/、dataset.json）
- 图片文件是否存在且可读
- 元数据 JSON 格式正确性
- 物体数量分布均衡性（输出分布百分比表）

---

## 数据集格式

生成的数据集为扁平结构：

```
data/benchmark/
├── dataset.json            # 场景索引（扁平列表，所有场景的入口）
├── dataset_info.json       # 生成配置和统计信息
├── images/
│   ├── single_view/        # 单视角图片
│   │   ├── scene_000000.png
│   │   ├── scene_000001.png
│   │   └── ...
│   └── multi_view/         # 多视角图片（每场景 4 张）
│       ├── scene_000000/
│       │   ├── view_0.png
│       │   ├── view_1.png
│       │   ├── view_2.png
│       │   └── view_3.png
│       └── ...
└── metadata/               # 场景元数据（物体信息、3D 坐标、约束）
    ├── scene_000000.json
    └── ...
```

### dataset.json 格式

```json
[
  {
    "scene_id": "scene_000042",
    "single_view_image": "images/single_view/scene_000042.png",
    "multi_view_images": [
      "images/multi_view/scene_000042/view_0.png",
      "images/multi_view/scene_000042/view_1.png",
      "images/multi_view/scene_000042/view_2.png",
      "images/multi_view/scene_000042/view_3.png"
    ],
    "metadata_path": "metadata/scene_000042.json",
    "n_objects": 5,
    "tau": 0.10
  }
]
```

### 物体数量均衡分布

以 1000 个场景、物体范围 3-10（8 个级别）为例：

| 物体数 | 场景数 | 占比 |
|--------|--------|------|
| 3 | 125 | 12.5% |
| 4 | 125 | 12.5% |
| 5 | 125 | 12.5% |
| 6 | 125 | 12.5% |
| 7 | 125 | 12.5% |
| 8 | 125 | 12.5% |
| 9 | 125 | 12.5% |
| 10 | 125 | 12.5% |

---

## 完整使用流程

### 场景一：本地快速测试

```bash
# 安装
uv sync

# 生成 8 个场景（快速验证流程是否正常）
uv run os-benchmark -o ./data/test -b $BLENDER_PATH -n 8

# 验证
uv run os-validate -d ./data/test
```

### 场景二：服务器生成 1080p 高清数据集

```bash
# 设置环境
export BLENDER_PATH=/path/to/blender

# 生成 1000 个 1080p 场景（GPU 加速）
uv run os-benchmark \
    -o ./data/benchmark_hd \
    -b $BLENDER_PATH \
    -n 1000 \
    --width 1920 --height 1080 \
    --render-samples 512 \
    --use-gpu

# 验证
uv run os-validate -d ./data/benchmark_hd
```

### 场景三：多 GPU 大规模生成

```bash
# 8 GPU 并行生成 10000 个场景
python scripts/build_parallel.py \
    -o ./data/large \
    -n 10000 \
    --n-gpus 8 \
    --quality high \
    --blender $BLENDER_PATH \
    -y

# 验证
uv run os-validate -d ./data/large
```

### 场景四：GPU 被大模型占用，只有 CPU 可用

```bash
# 降低分辨率和采样数以加快速度
uv run os-benchmark \
    -o ./data/benchmark \
    -b $BLENDER_PATH \
    -n 1000 \
    --width 1024 --height 768 \
    --render-samples 128
```

### 场景五：GPU 部分被占用

```bash
# 只使用 GPU 4 和 5
CUDA_VISIBLE_DEVICES=4,5 python scripts/build_parallel.py \
    -o ./data/benchmark \
    -n 1000 \
    --n-gpus 2 \
    --blender $BLENDER_PATH \
    -y
```

---

## VLM 评估

### 基线评估

```bash
# Oracle 基线（真值上界）
uv run os-baseline --baseline oracle --task t1-q --data ./data

# VLM 直接预测
uv run os-baseline --baseline vlm_direct --task t2 --model openai/gpt-4o

# 混合基线（预测-验证-修复）
uv run os-baseline --baseline hybrid --task t2 --model openai/gpt-4o --data ./data
```

### VLM 约束提取

```bash
# 单视角提取（Task-3）
uv run os-agent extract --image scene.png --output constraints.json --tau 0.10

# 多视角提取（Task-2）
uv run os-agent extract --images view1.png view2.png view3.png --output constraints.json

# 使用自定义模型
uv run os-agent extract --image scene.png --model openai/gpt-4o --output constraints.json
```

### Python API

```python
from ordinal_spatial.agents import VLMConstraintAgent, VLMAgentConfig

config = VLMAgentConfig(model="google/gemma-3-27b-it")
agent = VLMConstraintAgent(config)

# 单视角
result = agent.extract_from_single_view("scene.png", tau=0.10)

# 多视角
result = agent.extract_from_multi_view(
    ["view1.png", "view2.png", "view3.png"], tau=0.10
)
```

---

## 任务定义

| 任务 | 输入 | 输出 | 指标 |
|------|------|------|------|
| **T1-Q** | 4 个对象 + 场景 | 距离关系 (<, >, ~=) | Accuracy, F1 |
| **T1-C** | 3 个对象 + 场景 | 时钟位置 | Hour/Quadrant Accuracy |
| **T2** | 图像 + 对象列表 | 完整 QRR/TRR 约束集 | Precision, Recall, F1 |
| **T3** | 约束集 | 3D 点配置 | NRMS, Constraint % |

## 约束类型

| 类型 | 说明 |
|------|------|
| **QRR** | 距离比较（Quantitative Ratio Relations） |
| **TRR** | 时钟方向（Ternary Reference Relations） |
| **Topology** | 相离/相切关系 |
| **Occlusion** | 遮挡关系 |
| **Axial** | 左/右/前/后关系 |
| **Size** | 大小比较 |
| **Closer** | 距离排序 |

---

## 项目结构

```
ordinal-spatial/
├── pyproject.toml              # 项目配置
├── src/ordinal_spatial/        # 主包
│   ├── dsl/                    # 核心数据模型与谓词
│   ├── generation/             # 约束生成
│   ├── evaluation/             # 评估指标
│   ├── reconstruction/         # 场景重建求解器
│   ├── agents/                 # VLM 约束提取智能体
│   ├── baselines/              # 基线实现
│   ├── tasks/                  # T1/T2/T3 任务运行器
│   ├── prompts/                # VLM 提示模板
│   ├── rendering/              # Blender 渲染（5.0+）
│   ├── scripts/                # CLI 脚本实现
│   ├── cli/                    # CLI 入口
│   └── utils/                  # 工具函数
├── scripts/                    # 独立脚本（多 GPU 并行等）
│   ├── build_parallel.py       # 多 GPU 并行生成
│   └── lib/                    # 并行构建库
├── tests/                      # 测试
└── docs/                       # 文档
```

## CLI 命令一览

| 命令 | 用途 | 示例 |
|------|------|------|
| `os-benchmark` | 生成数据集（单进程） | `uv run os-benchmark -o ./data -b $BLENDER_PATH -n 1000` |
| `os-validate` | 验证数据集 | `uv run os-validate -d ./data` |
| `os-baseline` | 基线评估 | `uv run os-baseline --baseline oracle --task t1-q --data ./data` |
| `os-agent` | VLM 智能体 | `uv run os-agent extract --image scene.png --output out.json` |
| `build_parallel.py` | 多 GPU 并行 | `python scripts/build_parallel.py -o ./data -n 10000 --n-gpus 4 -y` |

## 常见问题

### GPU 渲染不工作（占用率 0%）

1. 确认使用了 `--use-gpu` 参数
2. 检查 CUDA 驱动：`nvidia-smi`
3. 确认 Blender 版本 >= 5.0：`$BLENDER_PATH --version`
4. 代码会自动尝试 CUDA → OptiX → HIP → OneAPI

### 渲染速度太慢

- 降低分辨率：`--width 480 --height 320`
- 降低采样：`--render-samples 64`
- 使用 GPU：`--use-gpu`
- 使用多 GPU 并行：`build_parallel.py --n-gpus 4`

### ImportError: cannot import ordinal_spatial

```bash
cd ordinal-spatial
uv sync
uv run python -c "import ordinal_spatial"
```

## 开发

```bash
uv sync --extra dev
uv run pytest
uv run pytest --cov=ordinal_spatial
```

## 许可证

BSD-3-Clause. See [LICENSE](LICENSE).
