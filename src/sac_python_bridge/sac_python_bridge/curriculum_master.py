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

# --- 课程学习配置 ---
STAGES = [
    {"level": 1, "target": 70.0}, 
    {"level": 2, "target": 85.0}, 
    {"level": 3, "target": 95.0}, 
    {"level": 4, "target": 98.0},
    {"level": 5, "target": 100.0}
]
EPISODES_PER_STAGE = 5  # 每个等级至少练 5 张不同的图再升级

MAP_NAME = "sac_curriculum"
PROJECT_ROOT = "/home/cmh/MyProject/HMS_CPP"

class CurriculumMaster:
    def __init__(self, start_stage_idx=0, start_ep=0):
        self.current_stage_idx = start_stage_idx
        self.total_episodes = start_ep
        self.stage_episodes_counter = start_ep % EPISODES_PER_STAGE
        self.active_processes = []

    def kill_all_ros(self):
        print("\n🛑 正在清理战场，关闭所有节点...")
        # 第一波：优雅地向所有子进程组发送 SIGINT
        for p in self.active_processes:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGINT)
            except Exception:
                pass
        time.sleep(5)
        # 第二波：强制 SIGKILL 所有已知的 ROS2/Gazebo 相关进程
        # 覆盖完整的进程列表，防止僵尸进程占用端口干扰下一轮启动
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
        # 额外清理：杀掉所有残留的 ros2 launch 子进程
        subprocess.run(["pkill", "-9", "-f", "ros2 launch"],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "-f", "gazebo_sim.launch"],
                       stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
        subprocess.run(["pkill", "-9", "-f", "sac_training.launch"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(5)  # 等待端口和共享内存完全释放
        print("✅ 清理完成。\n")
        self.active_processes = []

    def start_launch_stack(self, level, map_name, json_path, target_coverage):
        world_path = os.path.join(PROJECT_ROOT, "src/agricultural_world/world", f"{map_name}.world")
        map_path = os.path.join(PROJECT_ROOT, "src/navigation2_config/maps", f"{map_name}.yaml")
        print(f"🚀 [Level {level}] 启动仿真 | World: {map_name}...")
        p1 = subprocess.Popen(["ros2", "launch", "robot_description", "gazebo_sim.launch.py", f"world:={world_path}"], preexec_fn=os.setsid, stdout=subprocess.DEVNULL)
        self.active_processes.append(p1)
        time.sleep(8)
        print(f"🚀 [Level {level}] 启动导航+训练 | Map: {map_name}...")
        # 【根因修复】所有参数通过 launch 文件传递给内部的 automated_training 节点
        # 不再单独 subprocess.run automated_training，否则会出现两个实例抢 Nav2 导致跳变
        p2 = subprocess.Popen([
            "ros2", "launch", "navigation2_config", "sac_training.launch.py",
            f"map:={map_path}", "mode:=train",
            f"json_path:={json_path}", "skip_phase1:=true",
            f"curriculum_name:={map_name}",
            f"target_exit_coverage:={target_coverage}",
        ], preexec_fn=os.setsid)
        self.active_processes.append(p2)
        time.sleep(15)

    def run_stage(self, stage):
        level = stage["level"]
        target = stage["target"]
        unique_name = f"lvl{level}_ep{self.total_episodes:03d}"
        
        print(f"\n" + "="*50)
        print(f"🏆 关卡 {level} | 第 {self.stage_episodes_counter+1}/{EPISODES_PER_STAGE} 场 | 目标: {target}%")
        print(f"📂 存档: {unique_name} | 总进度: Ep.{self.total_episodes}")
        print("="*50)
        
        json_path = generate_level_env(level, unique_name)[1]
        
        MAX_RETRIES = 3
        for attempt in range(1, MAX_RETRIES + 1):
            # 每次启动前先彻底清理残留进程
            self.kill_all_ros()
            
            self.start_launch_stack(level, unique_name, json_path, target)
            
            try:
                launch_proc = self.active_processes[-1]
                launch_proc.wait()
            except KeyboardInterrupt:
                self.kill_all_ros()
                sys.exit(0)
            
            exit_code = launch_proc.returncode
            
            # 检查是否有实际训练产出（progress.csv 是否存在且有内容）
            progress_file = os.path.join(
                PROJECT_ROOT, "src/sac_python_bridge/models_and_logs", unique_name, "progress.csv")
            has_data = os.path.exists(progress_file) and os.path.getsize(progress_file) > 50
            
            if has_data:
                print(f"✅ Episode {unique_name} 训练完成 (exit={exit_code})")
                break
            else:
                print(f"⚠️ Episode {unique_name} 异常退出 (exit={exit_code}, attempt={attempt}/{MAX_RETRIES})")
                self.kill_all_ros()
                if attempt < MAX_RETRIES:
                    print(f"🔄 等待 15 秒后重试...")
                    time.sleep(15)
                else:
                    print(f"❌ {unique_name} 连续 {MAX_RETRIES} 次失败，跳过此 Episode。")
        
        self.total_episodes += 1
        self.stage_episodes_counter += 1
        self.kill_all_ros()
    
    def start(self):
        while True:
            stage = STAGES[self.current_stage_idx]
            self.run_stage(stage)
            
            # 升级判断：只有当本级练够了场次才升下一级
            if self.stage_episodes_counter >= EPISODES_PER_STAGE:
                print(f"\n🌟 表现出色！Level {stage['level']} 毕业，晋升下一级！")
                self.current_stage_idx += 1
                self.stage_episodes_counter = 0 # 重置计数
                
                if self.current_stage_idx >= len(STAGES):
                    print("🎊 终极满级！从头开始高强度循环...")
                    self.current_stage_idx = 0

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Curriculum Learning Master')
    parser.add_argument('--start-level', type=int, default=1, help='Starting level (e.g. 1, 2, 3, 4, 5)')
    parser.add_argument('--start-ep', type=int, default=0, help='Starting total episode count (e.g. 1 for ep001)')
    args = parser.parse_args()
    
    # 将用户输入的 level 转换为 STAGES 数组的索引
    stage_idx = 0
    for i, s in enumerate(STAGES):
        if s['level'] == args.start_level:
            stage_idx = i
            break
            
    master = CurriculumMaster(start_stage_idx=stage_idx, start_ep=args.start_ep)
    try:
        master.start()
    except KeyboardInterrupt:
        master.kill_all_ros()