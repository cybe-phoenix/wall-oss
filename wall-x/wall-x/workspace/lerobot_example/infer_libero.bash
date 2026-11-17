#!/bin/bash

export MUJOCO_GL=osmesa
export PYTHONPATH="/data/wall_x/wall-x/LIBERO:${PYTHONPATH}"

# ================= 核心模型配置 =================
# 指向你刚刚跑完的 Epoch 5 或 Epoch 4
checkpoint_path="/data/save/save_dx/1"
mode="flow"
norm_key="libero_all"
initial_states_path="DEFAULT"

# 每个任务测 5 次
num_trials_per_task=5 
seed=3407

# ================= 目录配置 =================
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
rollout_dir="./rollouts/eval_${mode}_${seed}_${TIMESTAMP}"
log_dir="./logs/eval_${mode}_${seed}_${TIMESTAMP}"
mkdir -p "$log_dir"

cam_names=("face_view" "right_wrist_view")
suite_name="libero_goal"

echo "=================================================="
echo "🚀 启动 Wall-X 单卡(GPU 4)仿真评测"
echo "模型路径: $checkpoint_path"
echo "日志目录: $log_dir"
echo "视频目录: $rollout_dir"
echo "=================================================="

# ================= 单卡(4)运行核心逻辑 =================
# 定义要测试的 6 个任务的 ID (Libero Spatial 通常有 10 个任务，我们先测前 6 个)
task_ids=(0 1 2 3 4 5)

# 固定使用 GPU 4
export CUDA_VISIBLE_DEVICES=4

# 遍历 6 个任务，在单卡上顺序执行
for current_task_id in "${task_ids[@]}"; do
    log_file="$log_dir/${suite_name}_task${current_task_id}.log"
    
    echo "🎯 正在测试 Task ID $current_task_id (GPU 4)..."
    
    # 顺序执行，不使用 & 后台运行
    python /home/node/WALL-OSS/wall-x/scripts/infer_libero.py \
        --seed "$seed" \
        --mode "$mode" \
        --checkpoint_path "$checkpoint_path" \
        --norm_key "$norm_key" \
        --cam_names "${cam_names[@]}" \
        --task_suite_name "$suite_name" \
        --id "$current_task_id" \
        --initial_states_path "$initial_states_path" \
        --num_trials_per_task "$num_trials_per_task" \
        --rollout_dir "$rollout_dir" \
        > "$log_file" 2>&1
    
    echo "✅ Task ID $current_task_id 完成"
    echo "----------------------------------------"
done

echo "=================================================="
echo "🔥 所有 6 个任务已全部在 GPU 4 上完成！"
echo "查看日志：tail -f $log_dir/*.log"
echo "评测程序已运行完毕"
echo "=================================================="

echo "✅ 所有 6 个任务评测结束！赶紧去 $rollout_dir 看视频吧！"