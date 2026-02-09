# CLAUDE.md — AI 助手指南

## 项目概述

ORDINAL-SPATIAL: 序空间推理基准评估框架。
评估视觉语言模型 (VLM) 的空间推理能力。

## 项目结构

```
ordinal-spatial/
├── src/ordinal_spatial/        # 主包 (src layout)
│   ├── dsl/                    # 核心: 数据模型、谓词、比较器
│   ├── generation/             # 约束提取与生成
│   ├── evaluation/             # T1/T2/T3 评估指标
│   ├── reconstruction/         # 约束求解与场景重建
│   ├── agents/                 # VLM 约束提取智能体
│   ├── baselines/              # 基线模型
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
├── dataset.json          # 扁平列表
├── dataset_info.json     # 配置和统计
├── images/
│   ├── single_view/      # scene_NNNNNN.png
│   └── multi_view/       # scene_NNNNNN/view_{0-3}.png
└── metadata/             # scene_NNNNNN.json
```

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
uv run os-benchmark --help      # 数据集生成
uv run os-validate --help       # 数据集验证
uv run os-baseline --help       # 基线评估
uv run os-agent --help          # VLM 智能体

# 多 GPU 并行生成
python scripts/build_parallel.py --help
```

## 关键文件

| 文件 | 用途 |
|------|------|
| `dsl/schema.py` | 核心数据模型 |
| `dsl/predicates.py` | QRR/TRR 约束计算 |
| `evaluation/metrics.py` | T1/T2/T3 评估指标 |
| `agents/vlm_constraint_agent.py` | VLM 约束提取 |
| `rendering/render_multiview.py` | 多视角渲染（--prefix 参数） |
| `rendering/blender_utils.py` | Blender API 封装 |
| `scripts/build_benchmark.py` | 单进程数据集构建 |
| `scripts/validate_benchmark.py` | 扁平数据集验证 |

## 生成数据集

```bash
# 单进程
uv run os-benchmark -o ./data -b $BLENDER_PATH -n 1000 --use-gpu

# 多 GPU
python scripts/build_parallel.py -o ./data -n 10000 --n-gpus 4 -y
```

两条路径统一使用扁平模式，多生成 ~10% 后按物体数量裁剪到精确均分。

## Blender 5.0+ 关键要点

- `bpy.ops.wm.append()` 需要 `filepath/directory/filename` 三参数
- GPU 初始化: 调用 `refresh_devices()` 刷新设备列表，CUDA 优先
- `tile_size` 在 Cycles X 中已移除
- 每次 `open_mainfile()` 后必须重新调用 `_setup_gpu()`

## 环境

- **Python**: >=3.10
- **Blender**: 5.0+ only
- **包管理**: uv（推荐）或 pip
- **目标平台**: Linux CUDA + macOS
