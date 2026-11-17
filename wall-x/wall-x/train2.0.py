import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from collections import deque
from tqdm import tqdm  # 进度条，需安装：pip install tqdm

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
            # 🔴 关键：替换为你JSONL每行的实际字段名 🔴
            # 示例：如果你的JSONL每行是 {"eef": [0.1,0.2,0.3], "bowl": [0.5,0.0,0.8], "grasp": true, "act": [0.1,...0.7]}
            eef_pos = data['eef_pos']  # 或 'end_effector_position', 'robot_eef_pos'
            target_pos = data.get('bowl_pos', data.get('target_object_pos', [0.5, 0, 0.8]))  # 碗或其他目标物
            is_grasped = data.get('is_grasped', data.get('gripper_state', 0) > 0.5)  # bool 或 float 0/1
            action = data['action']  # 7维 list，通常 [dx,dy,dz, drx,dry,drz, gripper]
            
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

# ====================== 2. 原有模块（无需修改） ======================
class InfoAdapter:
    @staticmethod
    def process(info, prev_processed_info=None):
        processed = {}
        processed["task_success"] = info.get("task_success", False)
        processed["is_grasped"] = info.get("is_grasped", False) or info.get("robot0_contact_object", False)
        
        obj_pos_dict = info.get("object_pos", {})
        bowl_key = next((k for k in obj_pos_dict.keys() if "bowl" in k.lower()), None)
        processed["bowl_pos"] = np.array(obj_pos_dict.get(bowl_key, [0.5, 0.0, 0.8]), dtype=np.float32)
        processed["eef_pos"] = np.array(info.get("robot0_eef_pos", [0.0, 0.0, 1.0]), dtype=np.float32)
        processed["target_pos"] = np.array(info.get("target_pos", [0.5, 0.0, 1.2]), dtype=np.float32)
        
        if prev_processed_info is None:
            return processed, processed
        return processed

def compute_reward(curr_p, next_p, current_stage):
    next_p_dict = next_p[0] if isinstance(next_p, tuple) else next_p
    dist_to_bowl = np.linalg.norm(curr_p["eef_pos"] - curr_p["bowl_pos"])
    next_dist_to_bowl = np.linalg.norm(next_p_dict["eef_pos"] - next_p_dict["bowl_pos"])
    
    reward = 0.0
    if next_dist_to_bowl < dist_to_bowl:
        reward += (dist_to_bowl - next_dist_to_bowl) * 80.0
    else:
        reward -= 0.005
    reward += np.exp(-next_dist_to_bowl * 4.0) * 0.2
    if next_dist_to_bowl < 0.1:
        reward += 1.0
        if next_p_dict["is_grasped"]:
            reward += 20.0
    if current_stage >= 2 and next_p_dict["is_grasped"]:
        dist_to_target = np.linalg.norm(next_p_dict["bowl_pos"] - next_p_dict["target_pos"])
        reward += (1.0 - np.clip(dist_to_target, 0, 1)) * 5.0
    if next_p_dict["task_success"]:
        reward += 50.0
    return reward

class WallXPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(7, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 7), nn.Tanh()
        ).to(self.device)
    
    def get_action(self, state_vec):
        self.eval()
        t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device)
        return self.net(t.unsqueeze(0)).cpu().detach().numpy().squeeze(0)

# ====================== 3. 数据集封装 + 训练主循环（仅修改路径） ======================
def load_dataset_from_video_jsonl(data_root):
    """从视频+JSONL解析并封装为DataLoader"""
    # 1. 解析数据
    parser = VideoJSONLParser(data_root)
    states, actions = parser.parse_all_data()
    
    # 2. 数据校验
    assert states.shape[1] == 7, f"状态维度错误！需7维，实际{states.shape[1]}维"
    assert actions.shape[1] == 7, f"动作维度错误！需7维，实际{actions.shape[1]}维"
    
    # 3. 划分训练/验证集（8:2）
    split_idx = int(len(states) * 0.8)
    train_states, val_states = states[:split_idx], states[split_idx:]
    train_actions, val_actions = actions[:split_idx], actions[split_idx:]
    
    # 4. 封装为PyTorch DataLoader
    train_dataset = Data.TensorDataset(
        torch.from_numpy(train_states),
        torch.from_numpy(train_actions)
    )
    val_dataset = Data.TensorDataset(
        torch.from_numpy(val_states),
        torch.from_numpy(val_actions)
    )
    
    train_loader = Data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = Data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    return train_loader, val_loader

def main():
    # ---------------------- 第一步：配置数据集路径 ----------------------
    # 🔴 替换为你的数据集根目录（包含所有MP4和JSONL文件）🔴
    DATA_ROOT = "/data/wall_x/wall-x/datasets/libero_all"  # 例如：./robot_data/ 下有 demo1.mp4、demo1.jsonl、demo2.mp4、demo2.jsonl...
    
    # ---------------------- 第二步：初始化模型和优化器 ----------------------
    print("✅ 基于视频+JSONL数据集启动训练！")
    model = WallXPolicy()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    # 随机种子（可复现）
    np.random.seed(42)
    torch.manual_seed(42)
    
    # ---------------------- 第三步：加载数据集 ----------------------
    train_loader, val_loader = load_dataset_from_video_jsonl(DATA_ROOT)
    
    # ---------------------- 第四步：训练模型 ----------------------
    epochs = 100
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        total_train_loss = 0.0
        for batch_states, batch_actions in tqdm(train_loader, desc=f"Epoch {epoch} 训练"):
            batch_states = batch_states.to(model.device)
            batch_actions = batch_actions.to(model.device)
            
            pred_actions = model.net(batch_states)
            loss = loss_fn(pred_actions, batch_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * batch_states.size(0)
        
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch_states, batch_actions in val_loader:
                batch_states = batch_states.to(model.device)
                batch_actions = batch_actions.to(model.device)
                
                pred_actions = model.net(batch_states)
                loss = loss_fn(pred_actions, batch_actions)
                total_val_loss += loss.item() * batch_states.size(0)
        
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "./best_policy_model.pth")
            print(f"\n📌 第{epoch}轮验证Loss最优，已保存模型！")
        
        # 打印日志
        if epoch % 5 == 0:
            print(f"\nEpoch {epoch:3d} | 训练Loss: {avg_train_loss:.4f} | 验证Loss: {avg_val_loss:.4f} | 最优Val Loss: {best_val_loss:.4f}")
    
    # ---------------------- 第五步：训练完成 ----------------------
    print("\n🎉 训练完成！")
    print(f"最终训练Loss: {train_losses[-1]:.4f}")
    print(f"最终验证Loss: {val_losses[-1]:.4f}")
    print(f"最优验证Loss: {best_val_loss:.4f}")
    print("✅ 最优模型已保存至：./best_policy_model.pth")

if __name__ == "__main__":
    main()
