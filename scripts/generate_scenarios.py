import json
import numpy as np
import math

def generate_cutin_trajectory(num_frames=24, output_path="output/cutin_scenario.json"):
    """
    生成左后方车辆切入的轨迹数据 (Ego-centric coordinates)
    """
    frames_data = {}
    
    # === 场景参数设置 ===
    # 目标车初始位置：左后方 (x<0, y<0)
    start_x = -10.0  # 后方 10 米
    start_y = 3.5   # 左侧车道 (假设车道宽 3.5m)
    
    # 目标车结束位置：正前方 (x>0, y=0)
    end_x = 20.0     # 前方 20 米
    end_y = 0.0      # 回到主车道中心
    
    # 车辆尺寸 (以普通轿车为例)
    car_size = {"dx": 4.5, "dy": 2.0, "dz": 1.6} # length, width, height
    class_name = "vehicle.car"

    # 生成轨迹插值
    for i in range(num_frames):
        # 进度 t (0.0 到 1.0)
        t = i / (num_frames - 1)
        
        # === 1. 计算位置 (X, Y) ===
        # X轴：线性加速超越 (简单的线性插值)
        # 实际情况可能是加速，这里简化为匀速相对运动
        curr_x = start_x + (end_x - start_x) * t
        
        # Y轴：使用 Sigmoid 函数模拟平滑切入 (Lane Change)
        # t 从 0->1，我们希望变道发生在中间段
        # 将 t 映射到 sigmoid 的有效区间，例如 -6 到 6
        k = 12 * (t - 0.5) 
        sigmoid_value = 1 / (1 + np.exp(-k))
        curr_y = start_y + (end_y - start_y) * sigmoid_value
        
        # Z轴：假设平地
        curr_z = -1.0 # nuScenes 中 ego z=0 通常是传感器高度，地面在负值，具体视标定而定
        
        # === 2. 计算朝向角 (Yaw) ===
        # 车辆的朝向应该是轨迹的切线方向
        # 简单的差分计算：下一帧位置 - 当前位置
        if i < num_frames - 1:
            next_t = (i + 1) / (num_frames - 1)
            next_x = start_x + (end_x - start_x) * next_t
            
            next_k = 12 * (next_t - 0.5)
            next_sigmoid = 1 / (1 + np.exp(-next_k))
            next_y = start_y + (end_y - start_y) * next_sigmoid
            
            dx = next_x - curr_x
            dy = next_y - curr_y
            yaw = math.atan2(dy, dx) # 弧度
        else:
            # 最后一帧保持上一帧的角度
            yaw = frames_data[str(i-1)]["objects"][0]["yaw"]

        # === 3. 构建对象数据 ===
        frame_obj = {
            "class_name": class_name,
            "x": round(float(curr_x), 3),
            "y": round(float(curr_y), 3),
            "z": round(float(curr_z), 3),
            "dx": car_size["dx"], # Length (x方向尺寸)
            "dy": car_size["dy"], # Width (y方向尺寸)
            "dz": car_size["dz"], # Height
            "yaw": round(float(yaw), 3)
        }
        
        frames_data[str(i)] = {
            "frame_idx": i,
            "objects": [frame_obj] # 当前帧只有一个切入车辆
        }

    # === 保存 ===
    with open(output_path, 'w') as f:
        json.dump(frames_data, f, indent=4)
    
    print(f"Scenario generated: {output_path}")
    print(f"Start Pos: ({start_x}, {start_y}), End Pos: ({end_x}, {end_y})")

if __name__ == "__main__":
    generate_cutin_trajectory()