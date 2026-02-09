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

## 快速开始

### 安装

```bash
git clone https://github.com/tasd12-ty/Redraw-3d.git ordinal-spatial
cd ordinal-spatial

# 使用 uv 安装（推荐）
uv sync

# 安装开发依赖
uv sync --extra dev
```

### 生成数据集

数据集为**扁平结构**（无 train/val/test 划分），物体数量在指定范围内严格均分。

```bash
# 生成 1000 个场景（物体数量 3-10，每级 ~125 个）
uv run os-benchmark \
    -o ./data/benchmark \
    -b /path/to/blender \
    -n 1000

# 使用 GPU 加速 + 1080p 高清
uv run os-benchmark \
    -o ./data/benchmark_hd \
    -b /path/to/blender \
    -n 1000 \
    --width 1920 --height 1080 \
    --render-samples 512 \
    --use-gpu

# 自定义物体数量范围和 tau
uv run os-benchmark \
    -o ./data/custom \
    -b /path/to/blender \
    -n 500 \
    --min-objects 4 --max-objects 8 \
    --tau 0.05
```

### 多 GPU 并行生成

```bash
# 4 GPU 并行生成 10000 个场景
python scripts/build_parallel.py \
    -o ./data/large \
    -n 10000 \
    --n-gpus 4 \
    --quality high \
    -y

# CPU 模式（单进程）
python scripts/build_parallel.py \
    -o ./data/test \
    -n 80 \
    --n-gpus 0 \
    -y
```

### 验证数据集

```bash
uv run os-validate -d ./data/benchmark
```

验证器会检查图片完整性、元数据一致性，并输出物体数量分布报告。

### 运行基线评估

```bash
# Oracle 基线（100% 准确率上界）
uv run os-baseline --baseline oracle --task t1-q --data ./data

# VLM 直接预测
uv run os-baseline --baseline vlm_direct --task t2 --model openai/gpt-4o
```

### VLM 约束提取

```bash
# 单视角提取（Task-3）
uv run os-agent extract --image scene.png --output constraints.json --tau 0.10

# 多视角提取（Task-2）
uv run os-agent extract --images view1.png view2.png view3.png --output constraints.json
```

## 数据集格式

生成的数据集为扁平结构：

```
data/benchmark/
├── dataset.json            # 场景索引（扁平列表）
├── dataset_info.json       # 配置和统计信息
├── images/
│   ├── single_view/        # scene_000000.png ...
│   └── multi_view/         # scene_000000/view_{0-3}.png ...
└── metadata/               # scene_000000.json ...
```

`dataset.json` 中每条记录：

```json
{
  "scene_id": "scene_000042",
  "single_view_image": "images/single_view/scene_000042.png",
  "multi_view_images": ["images/multi_view/scene_000042/view_0.png", ...],
  "metadata_path": "metadata/scene_000042.json",
  "n_objects": 5,
  "tau": 0.10
}
```

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

## CLI 命令

| 命令 | 用途 |
|------|------|
| `os-benchmark` | 生成基准数据集（单进程） |
| `os-validate` | 验证数据集完整性 |
| `os-baseline` | 运行基线评估 |
| `os-agent` | VLM 约束提取智能体 |

### os-benchmark 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-o, --output-dir` | 必填 | 输出目录 |
| `-b, --blender-path` | `blender` | Blender 可执行文件路径 |
| `-n, --n-scenes` | 1000 | 总场景数 |
| `--min-objects` | 3 | 最少物体数 |
| `--max-objects` | 10 | 最多物体数 |
| `--tau` | 0.10 | 约束容差阈值 |
| `--width` | 480 | 图片宽度 |
| `--height` | 320 | 图片高度 |
| `--render-samples` | 256 | 渲染采样数 |
| `--use-gpu` | 否 | 启用 GPU 渲染 |

### build_parallel.py 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `-o, --output` | 必填 | 输出目录 |
| `-n, --n-scenes` | 1000 | 总场景数 |
| `--n-gpus` | 1 | GPU 数量（0=CPU） |
| `--quality` | normal | 渲染质量（draft/normal/high） |
| `--min-objects` | 3 | 最少物体数 |
| `--max-objects` | 10 | 最多物体数 |
| `--tau` | 0.10 | 约束容差阈值 |
| `--seed` | 42 | 随机种子 |

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

## GPU 渲染

本项目仅支持 **Blender 5.0+**，GPU 后端优先级：CUDA > OptiX > HIP > OneAPI。

```bash
# 检查 GPU
nvidia-smi

# 使用 GPU 渲染
uv run os-benchmark -o ./data -b $BLENDER_PATH -n 100 --use-gpu
```

## 开发

```bash
uv sync --extra dev
uv run pytest
uv run pytest --cov=ordinal_spatial
```

## 许可证

BSD-3-Clause. See [LICENSE](LICENSE).
