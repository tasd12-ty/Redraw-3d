# 架构设计 / Architecture

## 模块关系图 / Module Dependency Graph

```
rendering/          (Blender 5.0+ 场景渲染)
    |
    v
generation/         (约束提取与生成)
    |
    v
dsl/                (核心数据模型: ObjectSpec, OSD, Constraints)
    ^          ^          ^
    |          |          |
evaluation/    tasks/     reconstruction/
(指标计算)    (T1/T2/T3) (场景重建)
    ^          ^
    |          |
baselines/    agents/
(基线模型)    (VLM 智能体)
    ^          ^
    |          |
prompts/      cli/
(提示模板)    (命令行入口)
```

## 核心模块 / Core Modules

### dsl/ - 领域特定语言

核心数据结构定义，所有其他模块的基础。

| 文件 | 职责 |
|------|------|
| `schema.py` | Pydantic 数据模型 (ObjectSpec, OSD, WorldConstraints, ViewConstraints) |
| `predicates.py` | 约束计算 (compute_qrr, compute_trr, extract_all_qrr) |
| `comparators.py` | 容差比较代数 (compare, ordinal_distance, Comparator) |

**关键类型:**
- `Comparator`: 枚举 (<, >, ~=) 表示序关系
- `ObjectSpec`: 对象规格 (形状、颜色、大小、位置、材质)
- `QRRConstraintSchema`: 四元距离比较约束
- `TRRConstraintSchema`: 三元时钟方向约束
- `OrdinalSceneDescription (OSD)`: 完整场景描述

### generation/ - 约束生成

从 3D 场景数据中提取约束。

| 文件 | 职责 |
|------|------|
| `constraint_extractor.py` | 从 Blender 场景提取 QRR/TRR 约束 |
| `degeneracy_checker.py` | 检测退化空间配置 |
| `difficulty_control.py` | 控制约束难度分布 |

### evaluation/ - 评估框架

| 文件 | 职责 |
|------|------|
| `metrics.py` | T1/T2/T3 评估指标 (Accuracy, F1, NRMS) |
| `consistency.py` | 约束一致性检查 (基于图的环检测) |
| `constraint_diff.py` | Constraint-Diff 评估指标 |

### reconstruction/ - 场景重建

从约束恢复 3D 空间配置。

| 文件 | 职责 |
|------|------|
| `constraint_solver.py` | 基于梯度下降的约束满足求解器 (PyTorch/NumPy) |
| `dsl_parser.py` | 解析 JSON 约束到优化问题 |
| `pipeline.py` | 端到端重建流水线 |
| `scene_builder.py` | 从求解结果构建场景描述 |
| `visualizer.py` | 3D 可视化 (Matplotlib/Plotly) |

### agents/ - VLM 约束提取智能体

| 文件 | 职责 |
|------|------|
| `base.py` | 基类 (ConstraintAgent, ObjectInfo, ConstraintSet) |
| `vlm_constraint_agent.py` | VLM 约束提取 (Task-2/3) |
| `blender_constraint_agent.py` | Blender 真值提取 (Task-1) |
| `cli.py` | 命令行接口 |

### rendering/ - Blender 渲染

仅支持 Blender 5.0+，已移除所有旧版兼容代码。

| 文件 | 职责 |
|------|------|
| `render_images.py` | 单视角场景渲染 |
| `render_multiview.py` | 多视角场景渲染 |
| `blender_utils.py` | Blender API 封装 |
| `create_*.py` | 资源文件生成脚本 |

## 数据流 / Data Flow

### 基准数据集生成

```
Blender 5.0+ → render_images/multiview → CLEVR_scenes.json
                                              |
                                              v
                                    constraint_extractor
                                              |
                                              v
                                    OSD (约束集 + 场景数据)
                                              |
                                              v
                                    数据集划分 (train/val/test)
```

### VLM 评估流程

```
图像 + 对象列表
      |
      v
VLMConstraintAgent → 预测约束集
      |
      v
evaluation/metrics → T1/T2/T3 指标
      |
      v
evaluation/consistency → 一致性分数
```

## 设计原则 / Design Principles

1. **Pydantic 数据验证**: 所有核心数据结构使用 Pydantic v2 模型
2. **容差比较**: 使用 tau 参数控制 ~= (近似相等) 的容差范围
3. **模块解耦**: 各模块通过 DSL 层交互，避免直接依赖
4. **Blender 5.0+ 专用**: 不维护旧版兼容性，代码更简洁
