# Robot Fundamentals

从零实现串联机械臂的运动学、动力学与轨迹规划算法。全部基于 NumPy 构建，不依赖任何外部机器人库，体现对底层数学原理的完整理解。

<p align="center">
  <img src="assets/01_fk.png" width="100%"/>
</p>

## 项目亮点

- **DH 参数运动学** — 正/逆运动学求解，支持解析法与数值法（阻尼最小二乘）
- **几何雅可比矩阵** — 速度映射、可操作性椭圆、奇异性分析
- **拉格朗日动力学** — 质量矩阵、科里奥利/离心力（Christoffel 符号）、重力项；正/逆动力学
- **仿真引擎** — 欧拉积分，支持自定义力矩策略（自由落体、PD+重力补偿）
- **轨迹规划** — 梯形与 S 曲线速度规划，关节空间与笛卡尔直线规划
- **可视化** — 每个模块均生成高质量 matplotlib 图表，无需安装 MuJoCo / Isaac Sim 即可运行

## 项目结构

```
core/
  kinematics.py      DH 变换、正/逆运动学、雅可比矩阵、可操作性
  dynamics.py        质量矩阵、科里奥利矩阵、重力项、正/逆动力学、仿真
  trajectory.py      梯形与 S 曲线规划器、笛卡尔直线规划
  visualization.py   机械臂绘制、椭圆、轨迹与动力学图表

examples/
  01_kinematics.py   多构型正运动学、解析逆运动学（肘上/肘下）、工作空间
  02_jacobian.py     可操作性热力图、速度椭圆、速度映射
  03_dynamics.py     正/逆动力学、自由落体仿真、PD 轨迹跟踪
  04_trajectory.py   梯形 vs S 曲线对比、笛卡尔路径、关节轨迹

assets/              自动生成的图表（见下方）
```

## 快速开始

```bash
# 克隆仓库
git clone https://github.com/<your-username>/robot-fundamentals.git
cd robot-fundamentals

# 安装依赖（仅需 NumPy + matplotlib）
pip install numpy matplotlib

# 运行示例并生成图表
python examples/01_kinematics.py
python examples/02_jacobian.py
python examples/03_dynamics.py
python examples/04_trajectory.py
```

## 运行效果

### 1. 正运动学与逆运动学

| 正运动学 | 逆运动学（肘上 / 肘下） |
|---|---|
| ![FK](assets/01_fk.png) | ![IK](assets/01_ik.png) |

**工作空间覆盖：**

<p align="center"><img src="assets/01_workspace.png" width="45%"/></p>

### 2. 雅可比矩阵与可操作性分析

| 可操作性热力图 | 速度椭圆 |
|---|---|
| ![map](assets/02_manipulability_map.png) | ![ellipses](assets/02_ellipses.png) |

### 3. 动力学与控制

| 自由落体（快照） | PD + 重力补偿跟踪 |
|---|---|
| ![fall](assets/03_freefall_snapshots.png) | ![pd](assets/03_pd_tracking.png) |

### 4. 轨迹规划

| 梯形 vs S 曲线 | 笛卡尔直线路径 |
|---|---|
| ![cmp](assets/04_profile_comparison.png) | ![cart](assets/04_cartesian_path.png) |

## 理论参考

实现遵循标准机器人学教材：

- **运动学**：Denavit-Hartenberg 参数法、几何雅可比矩阵、Yoshikawa 可操作性指标
- **动力学**：基于复合刚体算法的拉格朗日公式；由第一类 Christoffel 符号构造科里奥利矩阵
- **轨迹规划**：时间最优梯形（bang-coast-bang）速度曲线与七段 S 曲线

## License

MIT
