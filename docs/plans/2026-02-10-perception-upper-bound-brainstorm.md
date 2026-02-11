# 感知上界算法头脑风暴

> 日期: 2026-02-10 ~ 2026-02-11
> 参与者: 用户、Claude、Codex
> 状态: v4 — VLM 提示词系统已实现

### 实现进度

| 组件 | 状态 | 说明 |
|------|------|------|
| 设计文档 | ✅ | 本文档 (v4) |
| VLM 提示词系统 | ✅ | `agents/prompts/constraint_extraction.py` 重写完成 |
| VLM Agent 改进 | ✅ | view labels, token 自适应, metric 默认值, model metadata |
| 测试 | ✅ | 159 tests passed |
| PerceptionAgent (SAM2+UniDepth) | ⏳ | 设计完成，代码待实现 |
| Blender 渲染扩展 (depth/instance pass) | ⏳ | 控制实验所需 |
| 端到端 VLM 评估 | ⏳ | 需配置 API key 后实测 |

---

## 1. 问题定义

### 1.1 论文背景

NeurIPS benchmark 论文，评估当前 VLM 的 3D 空间理解能力。
感知上界 (Perception Upper Bound, PU) 的作用：

```
Oracle (GT 3D)        100%   ← 任务定义正确的证明
Perception UB          ??%   ← 代表人类视觉水平
  ├── PU-MV            ??%   ← 多视角：人类可看 4 张图
  └── PU-SV            ??%   ← 单视角：人类可看 1 张图
VLM (GPT-4o 等)        ??%   ← 被评估模型
Random baseline       ~33%   ← 三分类随机猜
```

**核心论点**：PU 代表"理想视觉系统从图像中能提取多少约束"。
PU 与 VLM 的差距 = VLM 在空间推理上还需改进多少。

### 1.2 核心约束条件

| 约束 | 说明 |
|------|------|
| **数据正交** | 使用的预训练模型不能在 CLEVR 或类似合成场景上训练过 |
| **Zero-shot** | 不在当前 benchmark 数据上做任何微调 |
| **类人视觉** | 算法应模拟人类观察空间关系的过程 |
| **无人类实验** | 仅用 CV 管线作为人类感知的计算代理 |

### 1.3 需要提取的约束类型

| 约束类型 | 说明 | 所需信息 | 预估难度 |
|----------|------|---------|----------|
| **TRR** | A 在 B→C 轴的时钟方向 | 2D 坐标 | 易 |
| **Axial** | 左/右、上/下 | 2D 坐标 | 易 |
| **Size** | 更大/更小 | 掩码面积 + 深度修正 | 易 |
| **Occlusion** | 谁遮挡谁 | 深度序 + 2D 重叠 | 中 |
| **Axial** | 前/后 | 深度序 | 中 |
| **Topology** | 分离/接触/重叠 | 3D 距离 + 尺寸 | 中 |
| **Closer** | 三元距离序 | 3D 距离 | 难 |
| **QRR** | d(A,B) vs d(C,D) | 准确 3D 距离 | 难 |

### 1.4 场景特点

- 合成 CLEVR 风格：3-15 简单几何体在平坦地面上
- 3 形状 × 8 色 × 2 材质 × 2 大小
- 已知相机：球面坐标 (d=12, el=30°, az=45°/135°/225°/315°)
- 所有物体底部在 z=0 地平面
- 单视角 (1 图) / 多视角 (4 图, 间隔 90°)

---

## 2. 模型选型（数据正交性论证）

### 2.1 候选模型对比

| 模型 | 发表 | 训练数据 | 与 CLEVR 正交？ | 能力 | 适合环节 |
|------|------|---------|---------------|------|---------|
| **SAM2** | Meta, 2024 | SA-1B (11M 真实图像) | ✅ 纯真实图像 | Zero-shot 实例分割 | 物体分割 |
| **Depth Anything V2** | NeurIPS 2024 | 合成(Hypersim/VKITTI) + 62M 真实 | ✅ 室内/驾驶场景 | 仿射不变深度 | 相对深度 |
| **UniDepth V2** | CVPR 2024 / ICLR 2025 | 3M 真实图像 (NYU/KITTI/...) | ✅ 真实场景 | 度量深度 + 自适应相机 | 度量深度 |
| **Metric3D V2** | TPAMI 2024 | 16M 多源图像 | ✅ 多域真实 | 度量深度 + 法线 | 度量深度 |
| **DINOv2** | Meta, 2024 | LVD-142M (真实图像) | ✅ 自监督真实 | 通用视觉特征 | 跨视角匹配 |

> **Depth Anything V2 的训练数据包含 Hypersim/Virtual KITTI 合成数据**，
> 但这些是室内/驾驶场景，与 CLEVR 的简单几何体场景在域上正交。
> UniDepth 和 Metric3D 更"纯净"——主要在真实场景上训练。

### 2.2 推荐组合

```
SAM2             → 物体实例分割 (zero-shot)
UniDepth V2      → 度量深度估计 (可利用已知相机内参)
DINOv2           → 跨视角特征匹配 (多视角场景)
predicates.py    → 约束计算 (复用现有代码)
```

**为什么选 UniDepth 而非 Depth Anything V2？**
- UniDepth 输出**度量深度**（米），不是仿射不变深度
- 可以直接传入已知相机内参提升精度
- QRR 约束需要准确的距离比较，度量深度更可靠
- Codex 也推荐 "prefer UniDepth / Metric3D alongside DA V2"

**备选方案**：三者都跑，作为 ablation study 报告在论文中。

---

## 3. 物体协议与算法管线

### 3.1 物体协议：GT 物体 + Grounding（与 VLM 相同）

**设计决策**：PU 管线接收与 VLM 相同的 GT 物体列表，确保公平对比。

#### 为什么？

当前 VLM 评估协议中，T2 任务**向 VLM 提供 GT 物体列表**：

```
VLM 收到的 prompt:
  "Known objects in scene:
   - obj_0: front-left red large metal cube
   - obj_1: back-right blue small rubber sphere
   - obj_2: center green medium metal cylinder
  请提取这些物体之间的空间约束。"
```

VLM 不需要自己检测物体，只需在已知物体间提取关系。
**PU 管线采用完全相同的输入协议**——接收同样的 GT 物体列表。

#### 对比表：三条路径的输入一致性

| 步骤 | BlenderAgent (Oracle) | PU (ZVB) | VLM |
|------|----------------------|----------|-----|
| **物体列表** | GT 3D 数据直接读取 | **接收 GT 列表** | **接收 GT 列表** |
| **物体定位** | 不需要（已有 3D 坐标） | SAM2 grounding | VLM 视觉理解 |
| **3D 恢复** | 不需要（已有） | 深度估计 + 反投影 | VLM 隐式推理 |
| **约束计算** | predicates.py | predicates.py | VLM 直接输出 |

**评估聚焦点**：纯空间关系理解能力，不混入物体检测误差。

#### 管线中 "Grounding" 的含义

"Grounding" = 将文字描述的物体定位到图像中的具体像素区域。
类比人类：看到 "红色大立方体" → 在图中找到它 → 判断位置。

```
输入: GT 物体列表 + 图像
       ↓
Step 1: SAM2 分割图像 → 候选掩码
Step 2: 匈牙利匹配 GT 物体 ↔ SAM2 掩码 (按颜色+大小代价)
Step 3: 每个 GT 物体获得一个精确掩码 → 2D 中心 + 轮廓
       ↓
输出: 每个 GT 物体的图像定位 (掩码, 中心, 边界框)
```

### 3.2 总体架构

```
  GT 物体列表                     图像
  (id, 颜色, 形状,               (RGB)
   大小, 材质)                      │
       │                           │
       │         ┌─────────────────┼──────────────────┐
       │         │  感知前端        │                  │
       │         │                 ▼                  │
       │         │        SAM2 → 候选掩码              │
       │         │        UniDepth → 度量深度图         │
       │         └────────┬───────────────────────────┘
       │                  │
       ▼                  ▼
  ┌──────────────────────────────┐
  │  Grounding (匈牙利匹配)       │
  │                              │
  │  GT物体 × SAM2掩码 → 最优匹配  │
  │  代价 = 颜色距离 + 大小距离    │
  │                              │
  │  输出: 每个 GT 物体的掩码定位  │
  └──────────────┬───────────────┘
                 │
    ┌────────────┼────────────────┐
    │ SV         │                │ MV
    │            │                │
    ┌────────────▼─┐   ┌─────────▼──────────────┐
    │ 单视角 3D 恢复│   │ 多视角融合               │
    │              │   │                         │
    │ 方式 A: 地平面│   │ 跨视角匹配 (同一 GT ID)  │
    │   反投影     │   │ + 三角测量               │
    │ 方式 B: 度量 │   │ + 可选深度融合            │
    │   深度反投影 │   │                         │
    └──────┬───────┘   └──────────┬──────────────┘
           │                      │
           └──────────┬───────────┘
                      │
           ┌──────────▼──────────┐
           │  约束提取后端 (共享)  │
           │  predicates.py      │
           │  物体 ID = GT ID    │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │  一致性修复 (可选)    │
           └─────────────────────┘
```

### 3.3 单视角管线 (ZVB-SV) 详细设计

#### Step 1: SAM2 分割 + Grounding

```python
from sam2 import SAM2AutoMaskGenerator
from scipy.optimize import linear_sum_assignment

def segment_and_ground(image, gt_objects, color_rgb_dict):
    """
    SAM2 分割 → 匈牙利匹配 → 每个 GT 物体获得掩码。

    参数:
      image: RGB 图像 (H, W, 3)
      gt_objects: GT 物体列表
        [{"id": "obj_0", "color": "red", "shape": "cube",
          "size": "large", "material": "metal"}, ...]
      color_rgb_dict: 颜色名 → RGB 映射
        {"red": [255, 0, 0], "blue": [0, 0, 255], ...}

    返回:
      grounded: 每个 GT 物体的掩码定位
        [{"obj": gt_objects[i], "mask": mask_array,
          "centroid": (cx, cy), "bottom": (bx, by)}, ...]
    """
    # 1. SAM2 auto-mask 生成候选掩码
    generator = SAM2AutoMaskGenerator(model="sam2-large")
    all_masks = generator.generate(image)

    # 2. 过滤: 去掉背景/地面/天空
    candidates = [
        m for m in all_masks
        if MIN_AREA < m["area"] < MAX_AREA
    ]

    # 3. 构建代价矩阵: GT 物体 × SAM2 掩码
    n_gt, n_cand = len(gt_objects), len(candidates)
    cost = np.full((n_gt, n_cand), 1e6)

    for i, obj in enumerate(gt_objects):
        gt_rgb = np.array(color_rgb_dict[obj["color"]])
        gt_size = obj["size"]  # "large" or "small"
        for j, mask in enumerate(candidates):
            # 颜色代价: 掩码区域中位 RGB vs GT 颜色
            pixels = image[mask["segmentation"]]
            median_rgb = np.median(pixels, axis=0)
            color_cost = np.linalg.norm(gt_rgb - median_rgb) / 255.0

            # 大小代价: 大物体 → 掩码面积应较大
            area = mask["area"]
            if gt_size == "large":
                size_cost = max(0, 1.0 - area / LARGE_AREA_REF)
            else:
                size_cost = max(0, area / LARGE_AREA_REF - 0.5)

            cost[i, j] = color_cost + 0.5 * size_cost

    # 4. 匈牙利最优匹配
    row_ind, col_ind = linear_sum_assignment(cost)

    # 5. 构建结果
    grounded = []
    for r, c in zip(row_ind, col_ind):
        seg = candidates[c]["segmentation"]
        ys, xs = np.where(seg)
        grounded.append({
            "obj": gt_objects[r],
            "mask": seg,
            "centroid": (xs.mean(), ys.mean()),
            "bottom": (xs[ys == ys.max()].mean(), ys.max()),
            "bbox": candidates[c]["bbox"],
        })
    return grounded
```

**为什么用匈牙利匹配而非贪心？**
- 贪心可能导致全局次优（物体 A 抢了物体 B 的最佳掩码）
- 匈牙利算法保证**全局最优对齐** (O(n³))
- 对 3-15 个物体完全可行

#### Step 2: 度量深度估计

```python
def estimate_depth(image, camera_intrinsics=None):
    """
    UniDepth V2 度量深度估计。
    可选传入相机内参以提升精度。
    """
    from unidepth import UniDepthV2

    model = UniDepthV2.from_pretrained("unidepthv2-vitl14")
    result = model.infer(image, intrinsics=camera_intrinsics)
    return result["depth"]  # (H, W) 度量深度图，单位: 米
```

#### Step 3: 3D 坐标恢复

对每个 grounded 物体，估计其 3D 世界坐标。

**方式 A — 地平面反投影（推荐）：**

```python
def ground_plane_backproject(u, v, K, R, t):
    """
    从掩码底部像素 (u, v) 反投影到地平面 z=0。
    人类类比: 看到物体"在地面上的哪个位置"。
    """
    ray_cam = np.linalg.inv(K) @ np.array([u, v, 1.0])
    ray_world = R.T @ ray_cam
    cam_pos = -R.T @ t
    t_intersect = -cam_pos[2] / ray_world[2]
    point = cam_pos + t_intersect * ray_world
    return point[:2]  # (x, y) on ground plane

def recover_3d_ground_plane(grounded, camera, size_heights):
    """
    从 grounding 结果恢复 3D 坐标。
    size_heights = {"large": 0.35, "small": 0.175}  # 中心高度
    """
    K, R, t = camera["K"], camera["R"], camera["t"]
    for g in grounded:
        bx, by = g["bottom"]  # 掩码底部中点
        xy = ground_plane_backproject(bx, by, K, R, t)
        z = size_heights.get(g["obj"]["size"], 0.25)
        g["position_3d"] = [xy[0], xy[1], z]
```

**方式 B — 度量深度反投影：**

```python
def recover_3d_metric_depth(grounded, depth_map, camera):
    """
    从掩码中心 + 度量深度反投影到 3D。
    不依赖地平面假设。
    """
    K, R, t = camera["K"], camera["R"], camera["t"]
    for g in grounded:
        cx, cy = g["centroid"]
        # 掩码区域的中位深度（比单点更鲁棒）
        mask_depth = np.median(depth_map[g["mask"]])
        point_cam = mask_depth * np.linalg.inv(K) @ np.array([cx, cy, 1.0])
        g["position_3d"] = list(R.T @ (point_cam - t))
```

两种方式都实现，论文中对比报告。

#### Step 4: 约束提取

```python
from ordinal_spatial.dsl.predicates import (
    extract_all_qrr, extract_all_trr, MetricType
)

def extract_constraints(grounded, tau=0.10):
    """
    从 grounded 物体构建约束。
    物体 ID 直接使用 GT ID → 与评估系统完全对齐。
    """
    # 构造物体字典 — ID 来自 GT
    obj_dict = {
        g["obj"]["id"]: {
            "id": g["obj"]["id"],
            "position_3d": g["position_3d"],
            "position_2d": list(g["centroid"]),
            "size": g["obj"]["size"],
        }
        for g in grounded
    }

    # 3D 约束: QRR, Axial, Topology, Size, Closer
    qrr = extract_all_qrr(obj_dict, MetricType.DIST_3D, tau)

    # 2D 约束: TRR (仅需 2D 中心，不经过深度)
    trr = extract_all_trr(obj_dict, use_3d=False)

    return {"qrr": qrr, "trr": trr, ...}
```

**关键点**：
- 物体 ID 直接使用 GT 的 `"obj_0"`, `"obj_1"` 等
- 无需 ID 对齐——因为 grounding 已经建立了 GT 物体 ↔ 掩码的映射
- TRR 直接从 2D 掩码中心计算（不需要深度）

### 3.4 多视角管线 (ZVB-MV) 详细设计

多视角管线同样接收 GT 物体列表，每个视角独立 grounding。

#### Step 1: 每个视角独立 grounding

```python
def perceive_multi_view(images, cameras, gt_objects):
    """
    每个视角: SAM2 分割 → GT 物体 grounding → 2D 定位。
    物体 ID 始终来自 GT，跨视角天然对齐。
    """
    per_view = []
    for img, cam in zip(images, cameras):
        grounded = segment_and_ground(img, gt_objects, COLOR_RGB)
        per_view.append(grounded)
    return per_view
```

**关键优势**：由于每个视角都用同一套 GT 物体做 grounding，
**跨视角物体匹配是免费的**——同一个 `obj_0` 在 4 个视角中
天然对应。不需要 DINOv2 特征匹配。

#### Step 2: 三角测量

```python
def triangulate_all(per_view, cameras):
    """
    从多视角 2D 中心三角测量 3D 坐标。
    每个物体在 4 个视角中有 4 个 2D 观测。
    """
    positions_3d = {}
    obj_ids = [g["obj"]["id"] for g in per_view[0]]

    for obj_id in obj_ids:
        observations = []
        proj_matrices = []
        for view_idx, (grounded, cam) in enumerate(
            zip(per_view, cameras)
        ):
            g = next(g for g in grounded if g["obj"]["id"] == obj_id)
            observations.append(g["centroid"])
            P = cam["K"] @ np.hstack([cam["R"], cam["t"].reshape(3,1)])
            proj_matrices.append(P)

        positions_3d[obj_id] = triangulate_dlt(
            observations, proj_matrices
        )
    return positions_3d

def triangulate_dlt(observations, proj_matrices):
    """标准 DLT 三角测量"""
    A = []
    for (u, v), P in zip(observations, proj_matrices):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3]).tolist()
```

#### Step 3: 约束提取

同单视角 Step 4，使用三角测量后的 3D 坐标。

### 3.5 各约束类型的提取策略

| 约束 | SV 提取方式 | MV 提取方式 | 关键依赖 |
|------|-----------|-----------|---------|
| TRR | 2D 掩码中心 → 时钟角度 | 同 SV (选 view_0) | 仅 grounding 精度 |
| Axial L/R | 2D x 坐标比较 | 三角测量后 3D x 比较 | 仅 grounding 精度 |
| Axial U/D | 2D y 坐标比较 | 三角测量后 3D z 比较 | 仅 grounding 精度 |
| Axial F/B | 深度序比较 | 三角测量后 3D y 比较 | 深度精度 |
| Size | GT 直接获取 ✓ | GT 直接获取 ✓ | 无（GT 物体已含大小） |
| Occlusion | 掩码重叠 + 深度序 | 多视角遮挡投票 | grounding + 深度 |
| Topology | 3D 距离 vs 半径和 | 三角测量后 3D 距离 | 3D 精度 |
| Closer | 3D 距离序 | 三角测量后 3D 距离 | 3D 精度 |
| QRR | d(A,B) vs d(C,D) 3D | 三角测量后 3D 距离 | 3D 精度 (最苛刻) |

> **注意**: Size 约束在方案 A 中直接从 GT 物体列表获取
> （因为 GT 已提供 `"size": "large"/"small"`），无需视觉估计。
> 这与 VLM 也能从 prompt 中读取 size 信息一致——公平。

---

## 4. 三方意见综合

### Claude

1. **地平面先验是杀手锏**：z=0 + 已知相机 → 单视角 3D 恢复变成射线-平面求交
2. **两种 3D 恢复方式都做**：地平面法 vs 度量深度法，论文中对比
3. **约束类型有天然难度梯度**：TRR/Axial 仅需 2D，QRR 需准确 3D → 预计不同约束精度有区分度
4. **这个难度梯度本身就是论文贡献**：说明哪些空间关系"容易看出来"，哪些"需要深度 3D 理解"

### Codex

1. **管线架构**：感知 → 3D → 约束 → 一致性修复，前端可换后端复用
2. **模型选择**：SAM2 + UniDepth/Metric3D 优先，DA V2 作为对比
3. **多视角用经典三角测量**，不用 learned MVS
4. **物体表示最小化**：3D 中心 + 大小类别 + 遮挡标记，不需要完整重建
5. **一致性优化器**：图/ILP 修复约束矛盾，提升最终精度
6. **论文表述**：称为"功能类比 (functional analogy)"而非认知等价

### 用户

1. 感知上界 = 对标人类视觉水平的计算代理
2. 数据正交 + zero-shot → 证明通用视觉能力的可达性
3. 与 CV 管线对比 → 暴露 VLM 具体视觉缺陷

---

## 5. 推荐实现路径

### Phase 0: Blender 渲染扩展（前置）

- 添加 depth pass + instance ID pass 渲染
- 生成 GT 掩码和 GT 深度，用于控制实验

### Phase 1: 基础管线（ZVB-SV）

1. **SAM2 分割 + GT 物体 grounding**（匈牙利匹配）
2. **UniDepth V2 度量深度**
3. **单视角 3D 恢复**（地平面法 + 度量深度法 两种）
4. **约束提取**（predicates.py，物体 ID = GT ID）
5. **评估** → ZVB-SV 数值

### Phase 2: 多视角扩展（ZVB-MV）

6. **每视角独立 grounding**（GT 物体 → 跨视角天然对齐）
7. **三角测量 3D**
8. **约束提取** → ZVB-MV 数值

### Phase 3: 控制实验

9. **PU-PerfectVis**: GT 掩码 + GT 深度
10. **PU-GTMask**: GT 掩码 + 预测深度
11. **PU-GTDepth**: 预测掩码 + GT 深度
12. **经典 CV 基线**: 颜色阈值 + 轮廓 + 几何

### Phase 4: 论文实验

13. 按约束类型分析 (TRR 易 → QRR 难)
14. 按物体数量分析 (3 vs 15 个)
15. 先验消融 (全先验 / 仅相机 / 无先验)
16. VLM 对比: GPT-4o / Gemini / Gemma vs ZVB

---

## 6. 代码集成方案

作为新的 Agent 类型加入 `agents/` 模块，**接口与 VLM Agent 对齐**：

```python
# agents/perception_agent.py
class PerceptionAgent(ConstraintAgent):
    """
    基于 CV 管线的约束提取 (Zero-shot Vision Baseline)。
    接收 GT 物体列表（与 VLM 相同协议），通过 SAM2 grounding
    定位物体，再估计 3D 坐标并提取约束。
    """

    def extract_from_single_view(
        self,
        image: str,
        objects: List[Dict],  # GT 物体列表 (与 VLM 收到的相同)
        camera: Dict,         # 已知相机参数
        tau: float = 0.10,
    ) -> ConstraintSet:
        """ZVB-SV: 单视角 zero-shot 视觉基线"""
        # 1. SAM2 分割 + GT 物体 grounding
        grounded = self.segment_and_ground(image, objects)
        # 2. 度量深度估计
        depth_map = self.depth_model.predict(image)
        # 3. 3D 坐标恢复
        self.recover_3d(grounded, depth_map, camera)
        # 4. 约束提取 (物体 ID = GT ID)
        return self.build_constraint_set(grounded, tau)

    def extract_from_multi_view(
        self,
        images: List[str],
        objects: List[Dict],  # 同一套 GT 物体
        cameras: List[Dict],
        tau: float = 0.10,
    ) -> ConstraintSet:
        """ZVB-MV: 多视角 zero-shot 视觉基线"""
        # 1. 每视角独立 grounding (同一 GT 物体 → 跨视角天然对齐)
        per_view = [
            self.segment_and_ground(img, objects)
            for img in images
        ]
        # 2. 三角测量 (无需跨视角匹配，ID 已对齐)
        positions_3d = self.triangulate(per_view, cameras)
        # 3. 约束提取
        return self.build_constraint_set_3d(
            objects, positions_3d, tau
        )
```

三条路径统一接口：

| Agent | 输入 | 物体来源 | 3D 来源 |
|-------|------|---------|--------|
| `BlenderConstraintAgent` | GT 场景数据 | GT 3D 数据 | GT 坐标 |
| `PerceptionAgent` | 图像 + **GT 物体列表** | **GT 列表 + SAM2 grounding** | 深度/三角测量 |
| `VLMConstraintAgent` | 图像 + **GT 物体列表** | **GT 列表 + VLM 理解** | VLM 推理 |

---

## 7. 技术依赖

| 组件 | 模型 | 训练数据 | 数据正交 | 用途 |
|------|------|---------|---------|------|
| 分割 | SAM2-Large | SA-1B (11M 真实) | ✅ | GT 物体 grounding |
| 深度 | UniDepth V2 | 3M 真实 | ✅ | 度量深度估计 |
| 深度(备选) | Metric3D V2 | 16M 多域 | ✅ | ablation |
| 深度(对比) | Depth Anything V2 | 合成+62M 真实 | ✅ | ablation |
| 约束计算 | predicates.py | — | — | 复用现有代码 |
| 物体匹配 | SciPy (Hungarian) | — | — | grounding 最优对齐 |
| 几何计算 | NumPy | — | — | 反投影 + 三角测量 |

> **注意**: 方案 A (GT 物体协议) 下**不需要 DINOv2**。
> 跨视角匹配通过同一 GT ID 天然对齐，无需特征匹配。

---

## 8. Codex 审查：设计缺陷与修补

> 以下为 Codex 审查后识别的问题及三方协商的修补方案。

### 8.1 定义严谨性 — "上界"命名风险

**问题**：NeurIPS 审稿人可能质疑 "Upper Bound" 措辞——
使用有噪声的 CV 模型怎么算"上界"？这只是一个 baseline。

**修补**：
- 论文中称为 **"Zero-shot Vision Baseline (ZVB)"** 或 **"Perceptual Reference"**
- 同时报告分层体系，明确定义边界：

```
Oracle (GT 3D)         = 任务天花板（验证任务定义合理性）
PU-PerfectVis          = GT 掩码 + GT 深度 + 约束提取器（感知完美时的理论上界）
ZVB-MV / ZVB-SV        = Zero-shot CV 管线（从 RGB 可达到的参考水平）
VLM baselines          = 被评估模型
Random                 = 下界
```

- **PU-PerfectVis 必须包含**：它才是真正的"感知上界"
- ZVB 是"实际可达的视觉参考"，不是严格上界

### 8.2 物体 ID 对齐 — 匈牙利匹配

**问题**：当前设计用 (颜色, 形状, 大小) 精确匹配物体 ID → 脆弱。
属性识别出错时直接 ID 错位，所有约束全错。

**修补**：使用**排列不变匹配 (Hungarian algorithm)**：

```python
from scipy.optimize import linear_sum_assignment

def align_objects(pred_objects, gt_objects):
    """
    匈牙利匹配: 最小化检测物体与 GT 物体的对齐代价。
    代价 = 属性不匹配惩罚 + 位置距离。
    """
    n_pred, n_gt = len(pred_objects), len(gt_objects)
    cost = np.zeros((n_pred, n_gt))
    for i, p in enumerate(pred_objects):
        for j, g in enumerate(gt_objects):
            # 属性匹配代价
            attr_cost = 0
            if p.color != g.color: attr_cost += 1.0
            if p.shape != g.shape: attr_cost += 1.0
            if p.size != g.size: attr_cost += 0.5
            # 位置距离代价 (归一化)
            pos_cost = np.linalg.norm(
                np.array(p.pos_2d) - np.array(g.pos_2d)
            ) / image_diagonal
            cost[i, j] = attr_cost + pos_cost
    row_ind, col_ind = linear_sum_assignment(cost)
    return list(zip(row_ind, col_ind))
```

**额外处理**：
- 检测多了 → 丢弃未匹配的检测（false positive）
- 检测少了 → 未匹配的 GT 物体标记为 miss（false negative）
- 评估指标中报告 **object detection recall** 作为辅助指标

### 8.3 误差归因 — 控制变量消融

**问题**：PU 错误来自哪里？分割错误？深度错误？约束计算？
论文需要分解误差来源。

**修补**：设计 4 个控制实验：

| 实验 | 分割 | 深度 | 说明 |
|------|------|------|------|
| PU-PerfectVis | GT 掩码 | GT 深度 | 纯约束提取误差（≈0） |
| PU-GTMask | GT 掩码 | 预测深度 | 仅深度模型误差 |
| PU-GTDepth | 预测掩码 | GT 深度 | 仅分割误差 |
| ZVB (full) | 预测掩码 | 预测深度 | 完整管线误差 |

**实现方式**：Blender 渲染时输出 depth pass + instance ID pass 作为 GT。

### 8.4 遮挡检测 — 具体规则

**问题**：遮挡提取设计模糊。

**修补**：定义明确规则：

```python
def detect_occlusion(mask_i, mask_j, depth_i, depth_j):
    """
    判断物体 i 是否遮挡物体 j。
    条件: (1) 掩码有重叠区域, (2) i 的深度更浅。
    """
    overlap = mask_i & mask_j
    overlap_ratio = overlap.sum() / mask_j.sum()

    if overlap_ratio < 0.01:  # 无显著重叠
        return None

    # 重叠区域内 i 比 j 近
    i_closer = (depth_i[overlap] < depth_j[overlap]).mean()

    if i_closer > 0.5:
        return OcclusionConstraint(
            occluder=obj_i.id, occluded=obj_j.id,
            partial=(overlap_ratio < 0.5)
        )
    return None
```

**多视角投票**：
- 每个视角独立判断遮挡
- 同一对 (i, j) 在多数视角被判为遮挡 → 确认

### 8.5 已知相机 + 地平面先验 — 消融必要性

**问题**：审稿人可能认为先验太强，让结果"不公平"。

**修补**：报告 3 级先验消融：

| 配置 | 相机参数 | 地平面 | 说明 |
|------|---------|--------|------|
| Full priors | ✅ 已知 | ✅ z=0 | 最强，利用所有场景知识 |
| Camera only | ✅ 已知 | ❌ | 仅用度量深度反投影 |
| No priors | ❌ | ❌ | 纯仿射深度 + 2D 几何 |

**论文论点**：人类观察者也具备"理解场景结构"的能力，
使用先验是合理的类人建模，但消融证明即使不用先验结果也有意义。

### 8.6 TRR 不需要深度

**确认（Claude + Codex 一致）**：TRR 约束仅依赖 2D 投影坐标。

```python
# TRR: 直接从 SAM2 掩码中心计算，不经过深度/3D 恢复
trr_objects = {
    obj_id: {"position_2d": mask_centroid}
    for obj_id, mask_centroid in detected_centroids.items()
}
trr_constraints = extract_all_trr(trr_objects, use_3d=False)
```

这意味着 **PU-SV 的 TRR 精度应该非常高**（仅取决于分割精度），
可作为论文中 "2D 约束 vs 3D 约束难度差异" 的有力证据。

### 8.7 经典基线锚点

**Codex 建议**：增加一个"令人惊讶的强"经典基线作为参考锚：

```
颜色阈值分割 + 轮廓检测 + 标定几何 (无深度学习)
```

**论文价值**：
- 如果经典方法接近 ZVB → 说明 "这个任务主要靠几何，不靠学习"
- 如果经典方法远低于 ZVB → 说明 "预训练模型的泛化能力有价值"
- 两种结果都是有意义的发现

---

## 9. 修订后的评估协议

### 9.1 主要评估维度

| 任务 | 说明 | PU 是否参与 |
|------|------|-----------|
| **T2** | 约束提取 (P/R/F1 per relation) | ✅ 核心指标 |
| **T1** | 约束分类 (给定 query 判断 <, ~=, >) | ✅ 衍生指标 |
| **T3** | 场景重建 (从约束恢复 3D) | ⚠️ 可选 |

### 9.2 主表设计

```
Table 1: Per-Relation T2 F1 Scores

| Method          | QRR  | TRR  | Axial | Size | Closer | Topo | Occl | Avg  |
|-----------------|------|------|-------|------|--------|------|------|------|
| Oracle (GT 3D)  | 100  | 100  | 100   | 100  | 100    | 100  | 100  | 100  |
| PU-PerfectVis   |  ??  |  ??  |  ??   |  ??  |  ??    |  ??  |  ??  |  ??  |
| ZVB-MV          |  ??  |  ??  |  ??   |  ??  |  ??    |  ??  |  ??  |  ??  |
| ZVB-SV          |  ??  |  ??  |  ??   |  ??  |  ??    |  ??  |  ??  |  ??  |
| GPT-4o (SV)     |  ??  |  ??  |  ??   |  ??  |  ??    |  ??  |  ??  |  ??  |
| Gemini (SV)     |  ??  |  ??  |  ??   |  ??  |  ??    |  ??  |  ??  |  ??  |
| Classical CV     |  ??  |  ??  |  ??   |  ??  |  ??    |  ??  |  ??  |  ??  |
| Random          | ~33  | ~8   | ~50   | ~50  | ~33    | ~33  | ~50  | ~37  |
```

### 9.3 消融表设计

```
Table 2: Error Attribution (Single-View)

| Config          | Mask | Depth | QRR  | TRR  | Axial | Avg  |
|-----------------|------|-------|------|------|-------|------|
| PU-PerfectVis   | GT   | GT    |  ??  |  ??  |  ??   |  ??  |
| PU-GTMask       | GT   | Pred  |  ??  |  ??  |  ??   |  ??  |
| PU-GTDepth      | Pred | GT    |  ??  |  ??  |  ??   |  ??  |
| ZVB-SV (full)   | Pred | Pred  |  ??  |  ??  |  ??   |  ??  |

Table 3: Prior Ablation (Single-View)

| Config          | Camera | Plane | QRR  | Closer | Avg  |
|-----------------|--------|-------|------|--------|------|
| Full priors     | ✅     | ✅    |  ??  |  ??    |  ??  |
| Camera only     | ✅     | ❌    |  ??  |  ??    |  ??  |
| No priors       | ❌     | ❌    |  ??  |  ??    |  ??  |
```

---

## 10. 修订后的实现路径

### Phase 0: Blender 渲染扩展（前置）
- 添加 depth pass + instance ID pass 渲染
- 生成 GT 掩码和 GT 深度作为控制实验输入

### Phase 1: 核心管线
1. SAM2 分割 + 属性识别
2. UniDepth V2 度量深度
3. 匈牙利物体对齐
4. 单视角 3D 恢复（地平面 + 度量深度 两种）
5. 约束提取 (predicates.py)
6. 评估 → ZVB-SV 数值

### Phase 2: 多视角 + 控制实验
7. 跨视角匹配 + 三角测量 → ZVB-MV
8. PU-PerfectVis (GT mask + GT depth)
9. 控制实验: PU-GTMask, PU-GTDepth
10. 经典 CV 基线

### Phase 3: 论文实验
11. 按约束类型分析
12. 按物体数量分析
13. 先验消融
14. VLM 对比

---

## 11. 预期论文贡献（修订）

1. **分层参考体系**：Oracle / PU-PerfectVis / ZVB-MV / ZVB-SV，量化信息上界
2. **约束类型难度梯度**：TRR(易,仅2D) → QRR(难,需3D)，揭示层次性
3. **误差归因**：控制变量消融分离分割/深度/几何的各自贡献
4. **Zero-shot 视觉参考**：数据正交模型即可达高水平，VLM 差距明确可量化
5. **经典 vs 学习对比**：简单几何方法的意外有效性（或学习模型的必要性）

---

## 附录 A: 相关工作参考

- SAM2: [Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714)
- Depth Anything V2: [NeurIPS 2024](https://arxiv.org/abs/2406.09414)
- UniDepth: [CVPR 2024](https://arxiv.org/abs/2403.18913)
- Metric3D V2: [TPAMI 2024](https://arxiv.org/abs/2404.15506)
- DINOv2: [Meta AI, 2024](https://arxiv.org/abs/2304.07193)

## 附录 B: 待确认问题

1. **PU-PerfectVis 是否列入主表？** → Codex 强烈建议是
2. **匈牙利匹配是否作为默认评估协议？** → 建议是，避免 ID 错位惩罚过重
3. **经典 CV 基线是否列入主表？** → 建议是，作为锚点
4. **深度模型 ablation 需要跑几个？** → 建议 UniDepth + Metric3D + DA V2 三个
