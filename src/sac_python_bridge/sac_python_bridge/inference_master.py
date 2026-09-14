#!/usr/bin/env python3
import subprocess
import time
import os
import signal
import sys
import json

# 导入我们的生成器
sys.path.append("/home/cmh/MyProject/Random_boundaries")
try:
    from generate_sac_arena import generate_level_env
except ImportError:
    print("❌ 无法导入生成器，请确保路径正确: /home/cmh/MyProject/Random_boundaries")
    sys.exit(1)

# --- 测试关卡配置 ---
STAGES = [
    {"level": 1}, 
    {"level": 2}, 
    {"level": 3}, 
    {"level": 4},
    {"level": 5}
]

PROJECT_ROOT = "/home/cmh/MyProject/HMS_CPP"

class InferenceMaster:
    def __init__(self, start_stage_idx=0, start_ep=0):
        self.current_stage_idx = start_stage_idx
        self.total_episodes = start_ep
        self.active_processes = []

    def kill_all_ros(self):
        print("\n🛑 正在清理战场，关闭所有节点...")
        for p in self.active_processes:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except Exception:
                pass
        time.sleep(5)
        progs_to_kill = [
            "gzserver", "gzclient", "gazebo",
            "nav2_controller", "nav2_planner", "nav2_bt_navigator",
            "nav2_lifecycle_manager", "nav2_map_server", "nav2_amcl",
            "nav2_costmap", "nav2_waypoint", "nav2_velocity",
            "sac_server_grid", "automated_training",
            "lifecycle_manager", "bt_navigator", "rviz2",
            "component_container",
        ]
        for prog in progs_to_kill:
            subprocess.run(["pkill", "-9", prog],
                           stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "-f", "ros2 launch"],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "-f", "gazebo_sim.launch"],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)        # 清理后台残留进程
        subprocess.run(["pkill", "-9", "-f", "sac_training.launch"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)
        print("✅ 清理完成。\n")
        self.active_processes = []

    def start_launch_stack(self, level, map_name, json_path, target_coverage, method='hybrid'):
        world_path = os.path.join(PROJECT_ROOT, "src/agricultural_world/world", f"{map_name}.world")
        map_path = os.path.join(PROJECT_ROOT, "src/navigation2_config/maps", f"{map_name}.yaml")
        print(f"🚀 [Level {level}] 启动仿真 | World: {map_name}...")
        p1 = subprocess.Popen(["ros2", "launch", "robot_description", "gazebo_sim.launch.py", f"world:={world_path}"], preexec_fn=os.setsid, stdout=subprocess.DEVNULL)
        self.active_processes.append(p1)
        time.sleep(8)
        print(f"🚀 [Level {level}] 启动导航+推理 | Map: {map_name}...")
        
        
        # 配置各基准算法的参数
        skip_phase1 = "false"
        use_heuristic = "false"
        stop_after_phase1 = "false"
        if method == 'pure_sac':
            skip_phase1 = "true"
        elif method == 'heuristic':
            skip_phase1 = "true"
            use_heuristic = "true"
        elif method == 'pure_boustrophedon':
            stop_after_phase1 = "true"
            
        # 强制使用 inference 模式
        p2 = subprocess.Popen([
            "ros2", "launch", "navigation2_config", "sac_training.launch.py",
            f"map:={map_path}", "mode:=inference",
            f"json_path:={json_path}", f"skip_phase1:={skip_phase1}",
            f"use_heuristic:={use_heuristic}",
            f"stop_after_phase1:={stop_after_phase1}",
            f"target_exit_coverage:={target_coverage}",
            f"curriculum_name:={map_name}",
        ], preexec_fn=os.setsid)
        self.active_processes.append(p2)
        time.sleep(15)

    def run_stage(self, stage, method='hybrid', reuse_env=None):
        level = stage["level"]
        target = 100.0 
        
        if reuse_env:
            unique_name = reuse_env
            json_path = os.path.join(PROJECT_ROOT, "src/agricultural_world/world", f"{unique_name}.json")
            print(f"\n🔄 复用既有环境地图: {unique_name}")
        else:
            unique_name = f"eval_lvl{level}_ep{self.total_episodes:03d}"
            json_path = generate_level_env(level, unique_name)[1]
            print(f"\n🆕 随机生成新地图: {unique_name}")
        
        print(f"\n" + "="*50)
        print(f"🔍 性能验收测试 | 关卡 {level} | 极限目标: {target}%")
        print(f"📂 隔离存档: {unique_name}")
        print("="*50)
        
        # 每次启动前彻底清理残留进程
        self.kill_all_ros()
        self.start_launch_stack(level, unique_name, json_path, target, method)
        
        try:
            launch_proc = self.active_processes[-1]
            launch_proc.wait()
        except KeyboardInterrupt:
            print("\n🛑 收到退出指令，正在关闭验收测试...")
            self.kill_all_ros()
            sys.exit(0)
            
        self.total_episodes += 1
        self.kill_all_ros()
    
    def start(self, method='hybrid', reuse_env=None):
        stage = STAGES[self.current_stage_idx]
        self.run_stage(stage, method, reuse_env)
        print(f"\n🌟 Level {stage['level']} ({method}) 测试已跑完，等待您手动中止 (Ctrl+C)...")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Inference Evaluation Master')
    parser.add_argument('--level', type=int, default=1, help='Which level to evaluate (e.g. 1, 2, 3, 4, 5)')
    parser.add_argument('--ep', type=int, default=0, help='Episode ID for isolation (e.g. 0)')
    parser.add_argument('--method', type=str, default='hybrid', choices=['hybrid', 'pure_sac', 'heuristic', 'pure_boustrophedon'], help='Algorithm baseline to test')
    parser.add_argument('--reuse_env', type=str, default='', help='Reuse an existing map (e.g., eval_lvl5_ep000)')
    args = parser.parse_args()
    
    stage_idx = 0
    for i, s in enumerate(STAGES):
        if s['level'] == args.level:
            stage_idx = i
            break
            
    master = InferenceMaster(start_stage_idx=stage_idx, start_ep=args.ep)
    try:
        reuse_env_val = args.reuse_env if args.reuse_env else None
        master.start(method=args.method, reuse_env=reuse_env_val)
    except KeyboardInterrupt:
        master.kill_all_ros()
