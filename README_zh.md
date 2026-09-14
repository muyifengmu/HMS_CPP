# HMS-CPP: Hybrid Macro-Action SAC Coverage Path Planning

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-blue.svg)](https://docs.ros.org/en/humble/index.html)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**[English README]**](README.md)

**基于几何先验与宏动作强化学习融合的农田全覆盖路径规划方法**

HMS-CPP (Hybrid Macro-Action SAC Coverage Path Planning) 是一种专为移动机器人（尤其是面对高度不规则边界与复杂未知障碍物的覆盖区域）设计的两阶段全覆盖路径规划框架。

### 🌟 核心特性 (Key Features)
1. **两阶段混合架构 (Two-Stage Architecture)**: 
   - **Phase 1 (几何先验)**: 利用图像形态学提取“绝对安全核心区”，采用传统弓字形（Boustrophedon）算法快速消耗大面积规整区域。
   - **Phase 2 (智能探索)**: 针对非凸几何边缘与狭窄死角，引入改进的 SAC 强化学习算法进行目标航点动态生成与精细化补漏。
2. **宏动作控制 (Macro-Action Control)**: 区别于传统端到端RL输出底层电机转速，本框架的 SAC 智能体输出连续空间向量 `[dx, dy]` 作为目标航点，底层局部避障与运动学约束交由 ROS 2 Nav2 栈处理，彻底解决长视野死锁问题。
3. **全变分平滑约束 (Total Variation Penalty)**: 创新性引入轨迹平滑惩罚，约束运动机器人转弯曲率，减轻机械磨损。
4. **在线行为克隆 (Online BC)**: 在模型训练受挫陷入死锁时，基于前沿探测（Frontier-based）算法实时注入专家动作，有效抑制 Critic 网络过估计偏差。

### ⚙️ 环境依赖 (Dependencies)
- OS: Ubuntu 22.04
- Framework: ROS 2 Humble
- Python 依赖: 请通过 `requirements.txt` 安装

### 🚀 快速开始 (Quick Start)

**1. 安装 Python 依赖**
```bash
pip install -r requirements.txt
```

**2. 编译工作空间**
```bash
cd ~/MyProject/HMS_CPP
colcon build --symlink-install
```

**3. 启动环境**
```bash
source install/setup.bash
```

**4. 运行测试/推理模式 (Inference)**
```bash
python3 src/sac_python_bridge/sac_python_bridge/inference_master.py
```

**5. 运行训练模式 (Training)**
```bash
python3 src/sac_python_bridge/sac_python_bridge/curriculum_master.py
```

### ⚠️ 复现注意事项
本仓库代码的启动脚本中可能包含硬编码绝对路径（如 `/home/cmh/MyProject/HMS_CPP/`）。在复现前，请全局搜索并替换为您本机的克隆路径。由于文件体积过大，训练好的经验池文件 (`.pkl`) 未包含在仓库中，但可以直接使用提供的 `sac_model.zip` 权重进行推理。
