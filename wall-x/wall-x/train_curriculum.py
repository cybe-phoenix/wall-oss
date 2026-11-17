import os
import numpy as np
import torch
import torch.nn as nn
import gym
from collections import deque
from tqdm import tqdm  #进度条，需安装：pip install tqdm

# ====================== 1. 数据解析模块（核心调整：适配JSONL格式） ======================
class VideoJSONLParser:
    def __init__(self, data_root):
        """
        初始化解析器（适配JSONL+MP4）
        :param data_root: 数据集根目录（包含MP4和JSONL文件）
        """
        self.data_root = data_root
        # 获取所有视频-JSONL对（确保文件名一致，仅后缀不同：mp4 ↔ jsonl）
        self.video_paths = [f for f in os.listdir(data_root) if f.endswith(".mp4")]
        self.jsonl_paths = [f.replace(".mp4", ".jsonl") for f in self.video_paths]
        assert len(self.video_paths) == len(self.jsonl_paths), "视频和JSONL文件数量不匹配！"
        # 校验每个JSONL文件是否存在
        for vid_path, jsonl_path in zip(self.video_paths, self.jsonl_paths):
            jsonl_full_path = os.path.join(self.data_root, jsonl_path)
            assert os.path.exists(jsonl_full_path), f"JSONL文件缺失：{jsonl_full_path}"

    def parse_single_pair(self, video_path, jsonl_path):
        """
        解析单个视频+JSONL对，提取状态-动作对
        :return: states (N,7), actions (N,7)  N为有效帧数
        """
        # 1. 加载JSONL标注（核心修改：逐行读取JSONL）
        annotations = []
        jsonl_full_path = os.path.join(self.data_root, jsonl_path)
        with open(jsonl_full_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                try:
                    frame_anno = json.loads(line)  # 逐行解析JSON对象
                    annotations.append(frame_anno)
                except json.JSONDecodeError as e:
                    print(f"⚠️ 解析JSONL行失败（已跳过）：{line} | 错误：{e}")
                    continue
        
        # 2. 读取视频（校验帧数，逻辑不变）
        cap = cv2.VideoCapture(os.path.join(self.data_root, video_path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        # 允许微小帧数差异（比如JSONL少1-2帧，可根据实际情况调整阈值）
        frame_diff = abs(len(annotations) - frame_count)
        if frame_diff > 2:  # 超过2帧差异才报错
            raise AssertionError(f"{video_path}帧数({frame_count})与JSONL标注数({len(annotations)})差异过大（差{frame_diff}帧）！")
        elif frame_diff > 0:
            print(f"⚠️ {video_path}帧数({frame_count})与JSONL标注数({len(annotations)})差{frame_diff}帧，已按标注数处理")
        
        # 3. 提取状态和动作（替换为你JSONL中的实际字段名）
        states = []
        actions = []
        for frame_anno in annotations:
            #  关键：替换为你JSONL每行的实际字段名 
            # 示例：如果你的JSONL每行是 {"eef": [0.1,0.2,0.3], "bowl": [0.5,0.0,0.8], "grasp": true, "act": [0.1,...0.7]}
            eef_pos = frame_anno["【机械臂末端位置字段名】"]  # 需是3维列表 [x,y,z]
            bowl_pos = frame_anno["【碗位置字段名】"]        # 需是3维列表 [x,y,z]
            is_grasped = frame_anno["【抓取状态字段名】"]    # 需是布尔值 True/False
            action = frame_anno["【动作字段名】"]            # 需是7维列表 [a1,a2,a3,a4,a5,a6,a7]
            
            # 拼接7维状态：eef_pos(3) + bowl_pos(3) + is_grasped(1)（无需修改）
            state = np.concatenate([
                np.array(eef_pos, dtype=np.float32),
                np.array(bowl_pos, dtype=np.float32),
                np.array([float(is_grasped)], dtype=np.float32)
            ])
            states.append(state)
            actions.append(np.array(action, dtype=np.float32))
        
        return np.array(states), np.array(actions)

    def parse_all_data(self):
        """解析所有视频+JSONL，合并为完整数据集"""
        all_states = []
        all_actions = []
        for vid_path, jsonl_path in tqdm(zip(self.video_paths, self.jsonl_paths), desc="解析数据集"):
            try:
                states, actions = self.parse_single_pair(vid_path, jsonl_path)
                all_states.append(states)
                all_actions.append(actions)
            except Exception as e:
                print(f"⚠️ 解析{vid_path}失败（已跳过）：{e}")
                continue
        
        # 合并所有数据并过滤异常值
        if not all_states:
            raise ValueError("❌ 无有效数据解析，请检查数据集路径和JSONL格式！")
        all_states = np.concatenate(all_states, axis=0)
        all_actions = np.concatenate(all_actions, axis=0)
        
        # 过滤异常值（可选：根据你的数据范围调整阈值）
        valid_mask = (
            (np.abs(all_states) < 10.0).all(axis=1) &  # 坐标范围限制
            (np.abs(all_actions) < 1.0).all(axis=1)    # 动作范围（模型输出Tanh，建议动作在[-1,1]）
        )
        all_states = all_states[valid_mask]
        all_actions = all_actions[valid_mask]
        
        print(f"✅ 数据集解析完成！总有效样本数：{len(all_states)}")
        return all_states, all_actions

# ====================== 1. 适配器 (保持元组返回，确保数据类型一致) ======================
class InfoAdapter:
    @staticmethod
    def process(info, prev_processed_info=None):
        processed = {}
        processed["task_success"] = info.get("task_success", False)
        processed["is_grasped"] = info.get("is_grasped", False) or info.get("robot0_contact_object", False)
        
        obj_pos_dict = info.get("object_pos", {})
        bowl_key = next((k for k in obj_pos_dict.keys() if "bowl" in k.lower()), None)
        
        # 强制使用 float32 array，确保奖励函数中的减法和 norm 计算不出错
        processed["bowl_pos"] = np.array(obj_pos_dict.get(bowl_key, [0.5, 0.0, 0.8]), dtype=np.float32)
        processed["eef_pos"] = np.array(info.get("robot0_eef_pos", [0.0, 0.0, 1.0]), dtype=np.float32)
        processed["target_pos"] = np.array(info.get("target_pos", [0.5, 0.0, 1.2]), dtype=np.float32)
        processed["in_target_zone"] = info.get("in_container", False)
        
        if prev_processed_info is None:
            return processed, processed
        return processed

# ====================== 2. 奖励函数 (按照你的逻辑重写) ======================
def compute_reward(curr_p, next_p, current_stage):
    # 解包 next_p 元组
    next_p_dict = next_p[0] if isinstance(next_p, tuple) else next_p
    
    # 1. 计算距离
    dist_to_bowl = np.linalg.norm(curr_p["eef_pos"] - curr_p["bowl_pos"])
    next_dist_to_bowl = np.linalg.norm(next_p_dict["eef_pos"] - next_p_dict["bowl_pos"])
    
    reward = 0.0
    
    # --- 稠密奖励（Potential Reward）：低倍数，减轻远离惩罚 ---
    dist_diff = dist_to_bowl - next_dist_to_bowl
    reward += dist_diff * 10.0  # 降低权重，模型更敢于探索
    # --- 限制单步最大负数奖励，避免惩罚爆炸
    if dist_diff < 0:
        reward = max(reward, -0.5) #单步负数奖励上限为-0.5，防止训练震荡
    # --- 状态保持奖励：扩大激励范围 ---
    max_dist = 1.0  # 假设环境中最大距离
    norm_dist = np.clip(next_dist_to_bowl / max_dist, 0, 1)
    reward += (1.0 - norm_dist) * 2.0  # 靠近即给分，远距离也有基础保底
    
    # --- 抓取奖励（强化信号）：大幅提高权重 ---
    if next_p_dict["is_grasped"]:
        reward += 30.0  
        
        # 如果抓住了且在 MOVE 阶段，计算到目标的距离奖励
        if current_stage >= 2:
            dist_to_target = np.linalg.norm(next_p_dict["eef_pos"] - next_p_dict["target_pos"])
            norm_target_dist = np.clip(dist_to_target / max_dist, 0, 1)
            reward += (1.0 - norm_target_dist) * 5.0
    
    # --- 任务成功奖励（终极大奖） ---
    if next_p_dict["task_success"]:
        reward += 200.0  
    
    return reward

# ====================== 3. 模型与策略 ======================
class WallXPolicy(nn.Module):
    def __init__(self, state_dim=7, action_dim=7):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        ).to(self.device)

    def get_action(self, state_vec):
        self.eval()
        t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device)
        return self.net(t.unsqueeze(0)).cpu().detach().numpy().squeeze(0)

# ====================== 4. 主运行循环 ======================
def main():
    print("🚀 启动训练：奖励逻辑已重置（减轻惩罚，强化大奖）")
    model = WallXPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    success_history = deque(maxlen=20)
    current_stage = 1 # 初始为靠近/抓取阶段
    proximity_count = 0 

    for episode in range(100000):
        # 模拟环境初始 info
        info = {"object_pos": {"bowl": np.random.rand(3)}, "robot0_eef_pos": np.random.rand(3)}
        curr_p = InfoAdapter.process(info)[0]
        
        total_reward = 0
        min_dist = 999.0

        for step in range(50):
            # 状态拼接 (7维: EEF_POS(3) + BOWL_POS(3) + GRASP(1))
            state_vec = np.concatenate([curr_p["eef_pos"], curr_p["bowl_pos"], [float(curr_p["is_grasped"])]])
            
            # 获取动作 + 探索噪声
            action = model.get_action(state_vec)
            if episode < 50: action += np.random.normal(0, 0.1, size=7)
            
            # 模拟执行 (步长 0.1)
            info_next = info.copy()
            new_eef_pos = curr_p["eef_pos"] + action[:3] * 0.1
            new_eef_pos = np.clip(new_eef_pos, 0.0, 1.0)
            info_next["robot0_eef_pos"] = new_eef_pos
            
            # 环境反馈
            next_p = InfoAdapter.process(info_next)
            
            # 计算奖励
            step_reward = compute_reward(curr_p, next_p, current_stage)
            total_reward += step_reward
            
            # 更新模型
            model.train()
            loss = nn.MSELoss()(model.net(torch.as_tensor(state_vec, dtype=torch.float32, device=model.device)), 
                               torch.as_tensor(action, dtype=torch.float32, device=model.device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            # 滚动状态
            curr_p = next_p[0] if isinstance(next_p, tuple) else next_p
            
            # 统计最近距离
            min_dist = min(min_dist, np.linalg.norm(curr_p["eef_pos"] - curr_p["bowl_pos"]))

        success_history.append(curr_p["task_success"])
        
        # 阶段切换逻辑
        if min_dist < 0.1:
            proximity_count += 1  #达标时计数+1
        else:
            proximity_count = max(0, proximity_count - 1)  #不达标时计算衰减，不清零

        #连续3轮达标后，再升级到Stage 2
        if proximity_count >= 3 and current_stage == 1:
            current_stage = 2
            print("🌟 成功靠近碗，进入 MOVE 阶段训练")

        if episode % 10 == 0:
            print(f"Epi {episode:3d} | Reward: {total_reward:7.2f} | MinDist: {min_dist:.3f} | SR: {np.mean(success_history):.1%}")

if __name__ == "__main__":
    main()