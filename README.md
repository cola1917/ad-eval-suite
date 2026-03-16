# ad-eval-suite

面向自动驾驶评测的工程化工具集，目标是“快速迭代、可复现、可展示”。

项目当前重点能力：

- 感知评测（检测 + 跟踪）
- 失败样本挖掘（最差场景 / 最差帧）
- 回放可视化（PNG / GIF）
- OpenSCENARIO 导出与校验（XOSC + esmini smoke）

## 功能展示

### 项目闭环流程图（评测 -> 挖掘 -> 可视化 -> 仿真）

![项目闭环流程图](images/pipeline-overview.svg)

### 失败场景 GIF（已挖掘 bad case）

![失败场景GIF](images/scene-0103.gif)

### 指标结果截图（评测输出摘要）

![指标结果截图](images/metrics-summary.png)

### BEV 回放静态帧（含地图覆盖）

![BEV回放静态图](images/scene-0061-frame.png)

### 地图覆盖对比图（同帧对比）

![地图覆盖对比图](images/map-overlay-comparison.png)

## 项目功能模块

### 1. 评测模块（eval/）

- 检测指标：Precision / Recall / F1 / AP / mAP
- 跟踪指标：MOTA / MOTP / IDF1 / IDSW / MT / ML
- 通过策略 YAML 统一管理参数（raw / detection_10cls / l2_planning）

### 2. 挖掘模块（mining/）

- 基于加权错误分数排序最差场景与最差帧
- 导出 snapshot JSON，便于复现与定位问题
- 挖掘结果可直接联动可视化与仿真导出

### 3. 仿真模块（simulation/）

- snapshot 导出 OpenSCENARIO（.xosc）
- XML/XSD 结构校验
- esmini headless smoke 快速验证可运行性

### 4. 可视化模块（visualization/）

- 俯视图逐帧渲染
- GIF 导出
- 可选 nuScenes 地图覆盖（车道线、可行驶区域）

## 运行方式（当前代码基线）

当前主入口为：`scripts/run_perception_eval.py`

```bash
# 查看参数
python scripts/run_perception_eval.py --help

# 1) 默认配置运行（读取 configs/dataset.yaml + configs/eval.yaml）
python scripts/run_perception_eval.py

# 2) 指定评测策略
python scripts/run_perception_eval.py --strategy detection_10cls
python scripts/run_perception_eval.py --strategy l2_planning

# 3) 场景与帧范围控制
python scripts/run_perception_eval.py --scenes first --max-frames full
python scripts/run_perception_eval.py --scenes full --max-frames full

# 4) generator profile 切换（good / baseline / bad）
python scripts/run_perception_eval.py --generator-profile good
python scripts/run_perception_eval.py --generator-profile bad

# 5) profile 基础上再用 CLI 精调（CLI 优先）
python scripts/run_perception_eval.py \
    --generator-profile good \
    --drop-rate 0.02 \
    --fp-rate 0.04

# 6) 导出失败场景 GIF（可选）
python scripts/run_perception_eval.py \
    --generator-profile good \
    --scenes first \
    --max-frames full \
    --export-replay-gif \
    --run-name good_fullframes_gif

# 7) 回放默认行为
# - 默认开启地图覆盖（车道线、可行驶区域）
# - 默认固定视角（ego_fixed）以保持帧尺度一致
# 如需关闭地图层或切回自动视角：
python scripts/run_perception_eval.py \
    --export-replay-gif \
    --no-overlay-map \
    --replay-view-mode auto
```

## 配置架构

![配置分层图](images/config-layering.svg)

配置按“数据 / 评测策略”解耦：

- `configs/dataset.yaml`
    - 数据集注册
    - active dataset
    - 类别映射 schema
- `configs/eval.yaml`
    - 策略默认参数（matcher、IoU、metrics level）
    - perception/prediction/planning 设置
    - synthetic generator base 参数
    - `generator_profiles`（good / baseline / bad）

## 快速运行

### 1. 环境准备

```bash
git clone https://github.com/cola1917/ad-eval-suite.git
cd ad-eval-suite

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 数据准备

将 nuScenes mini 放在：

```text
data/nuscenes-mini/
    maps/
    samples/
    sweeps/
    v1.0-mini/
```

### 3. 首次运行

```bash
python scripts/run_perception_eval.py --run-name first_run
```

输出目录：

```text
outputs/perception/first_run/
    report.json
    report.md
    topn/
    failure_mining/
```

## 当前范围说明

- 已可稳定使用：perception eval、mining、visualization、simulation export/validation/smoke
- 仍是占位脚手架：prediction/planning/e2e 的独立 CLI

## 目录速览

```text
configs/         dataset/eval/sim 配置
datasets/        数据加载
eval/            指标评测
mining/          失败挖掘与排序
simulation/      xosc 导出与校验
visualization/   回放与地图覆盖
scripts/         统一 CLI 与兼容脚本
tests/           单测与回归测试
images/          README 展示素材
```