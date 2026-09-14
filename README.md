# HMS-CPP: Hybrid Macro-Action SAC Coverage Path Planning

[![ROS 2](https://img.shields.io/badge/ROS_2-Humble-blue.svg)](https://docs.ros.org/en/humble/index.html)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[**[中文版 README (Chinese)]**](README_zh.md)

**A Full Coverage Path Planning Method Based on the Fusion of Macro-Action Reinforcement Learning and Geometric Priors**

HMS-CPP (Hybrid Macro-Action SAC Coverage Path Planning) is a two-stage complete coverage path planning framework specifically designed for mobile robots operating in coverage areas with highly irregular boundaries and complex, unknown obstacles.

### 🌟 Key Features
1. **Two-Stage Hybrid Architecture**: 
   - **Phase 1 (Geometric Prior)**: Utilizes image morphology to extract the "absolute safe core area" and applies the traditional Boustrophedon algorithm to quickly cover large, regular areas.
   - **Phase 2 (Intelligent Exploration)**: For non-convex edges and narrow dead ends, an improved SAC (Soft Actor-Critic) algorithm is introduced to dynamically generate target waypoints for precise leak-filling.
2. **Macro-Action Control**: Unlike traditional end-to-end RL that outputs low-level motor speeds, the SAC agent in this framework outputs a continuous spatial vector `[dx, dy]` as a target waypoint. Local obstacle avoidance and kinematic constraints are handled by the ROS 2 Nav2 stack, effectively solving the deadlock problem in long-horizon navigation.
3. **Total Variation (TV) Penalty**: A novel trajectory smoothness penalty is introduced to constrain the turning curvature of the mobile robot and reduce mechanical wear.
4. **Online Behavioral Cloning (BC)**: Injects expert actions based on Frontier-based exploration in real-time when the model training stalls or deadlocks, effectively suppressing the overestimation bias of the Critic network.

### ⚙️ Dependencies
- OS: Ubuntu 22.04
- Framework: ROS 2 Humble
- Python Dependencies: Install via `requirements.txt`

### 🚀 Quick Start

**1. Install Python Dependencies**
```bash
pip install -r requirements.txt
```

**2. Build the Workspace**
```bash
cd ~/MyProject/HMS_CPP
colcon build --symlink-install
```

**3. Source the Environment**
```bash
source install/setup.bash
```

**4. Run Inference/Evaluation**
```bash
python3 src/sac_python_bridge/sac_python_bridge/inference_master.py
```

**5. Run Curriculum Training**
```bash
python3 src/sac_python_bridge/sac_python_bridge/curriculum_master.py
```

### ⚠️ Notes for Reproduction
The startup scripts in this repository may contain hard-coded absolute paths (e.g., `/home/cmh/MyProject/HMS_CPP/`). Before running, please perform a global search and replace them with your local clone path. Due to file size limits, the trained replay buffer files (`.pkl`) are not included in the repository, but you can directly use the provided `sac_model.zip` weights for inference.