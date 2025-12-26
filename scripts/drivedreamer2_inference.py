import os
import sys
import warnings
import torch
import numpy as np
import cv2
import json
import random
import pickle
from PIL import Image

# 忽略警告
warnings.filterwarnings("ignore")

# 路径配置
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "dreamer-datasets"))

import ENV
ENV.init_paths(project_name="DriveDreamer2")

from dreamer_models.pipelines.drivedreamer2.pipeline_drivedreamer2 import DriveDreamer2Pipeline
from dreamer_datasets import load_dataset, boxes3d_utils
from transformers import CLIPTextModel, CLIPTokenizer

CAM_NAMES = ['CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']

class DriveDreamerInference:
    def __init__(self, svd_path, dd2_weight_path, embed_map_path, device="cuda"): 
        self.svd_path = svd_path
        self.dd2_weight_path = dd2_weight_path
        self.embed_map_path = embed_map_path
        self.device = device
        self.pipe = None
        self.prompt_embed_map = None
        
    def setup(self):
        print(f"Loading SVD from {self.svd_path}...")
        self.pipe = DriveDreamer2Pipeline.from_pretrained(
            self.svd_path,
            variant="fp16",
            torch_dtype=torch.float16
        )
        self.pipe.load_weights(self.dd2_weight_path)
        self.pipe.to(self.device)
        
        print(f"Loading Prompt Embeddings from {self.embed_map_path}...")
        with open(self.embed_map_path, 'rb') as f:
            content = pickle.load(f)
            if isinstance(content, (list, tuple)) and len(content) > 1 and isinstance(content[1], dict):
                self.prompt_embed_map = content[1]
            elif isinstance(content, dict):
                self.prompt_embed_map = content
            else:
                raise ValueError(f"Unknown pickle structure: {type(content)}")

    def encode_prompt(self, prompt):
        if self.prompt_embed_map is None: raise ValueError("Prompt map not loaded!")
        if prompt not in self.prompt_embed_map:
            print(f"Warning: Prompt '{prompt}' not found. Using default.")
            prompt = "realistic autonomous driving scene."
        embed = self.prompt_embed_map[prompt]
        if not isinstance(embed, torch.Tensor): embed = torch.tensor(embed)
        if embed.dim() == 1: embed = embed.unsqueeze(0).unsqueeze(0) 
        elif embed.dim() == 2: embed = embed.unsqueeze(1) 
        return embed.to(self.device, dtype=torch.float16)
    
    def build_index_map(self, dataset):
        print("Building dataset index map...")
        index_map = {} 
        for i in range(len(dataset)):
            try:
                sample = dataset[i]
                scene_token = sample['scene_token']
                frame_idx = sample['frame_idx']
                cam_type = sample['cam_type'].upper()
                if scene_token not in index_map: index_map[scene_token] = {}
                if frame_idx not in index_map[scene_token]: index_map[scene_token][frame_idx] = {}
                index_map[scene_token][frame_idx][cam_type] = i
            except: continue
        self.index_map = index_map
        print(f"Index map built. Found {len(index_map)} scenes.")

    def find_valid_sequence(self, dataset, seq_len=24):
        if not hasattr(self, 'index_map'): self.build_index_map(dataset)
        scenes = sorted(list(self.index_map.keys()), reverse=True)
        for scene in scenes:
            frames = sorted(self.index_map[scene].keys())
            if len(frames) < seq_len: continue
            for i in range(len(frames) - seq_len + 1):
                start_frame = frames[i]
                valid = True
                for k in range(seq_len):
                    curr_frame = frames[i+k]
                    if len(self.index_map[scene][curr_frame]) < 6:
                        valid = False; break
                if valid: return scene, start_frame
        return scenes[0], sorted(self.index_map[scenes[0]].keys())[0]

    def prepare_inputs_chunk(self, dataset, scene_token, start_frame_idx, cutin_boxes_ego, frame_offset, chunk_len, use_real_image=True):
        all_hdmaps, all_layouts, all_images = [], [], []
        scene_frames = sorted(self.index_map[scene_token].keys())
        try: start_list_idx = scene_frames.index(start_frame_idx)
        except: start_list_idx = 0
            
        for t in range(frame_offset, frame_offset + chunk_len):
            frame_hdmaps, frame_layouts, frame_images = [], [], []
            current_list_idx = min(start_list_idx + t, len(scene_frames) - 1)
            current_frame_idx = scene_frames[current_list_idx]
            frame_cam_indices = self.index_map[scene_token][current_frame_idx]
            
            for cam_name in CAM_NAMES:
                idx = frame_cam_indices.get(cam_name, list(frame_cam_indices.values())[0])
                sample = dataset[idx]
                
                # HDMap
                hdmap = sample['image_hdmap'].resize((448, 256))
                hdmap_t = (torch.tensor(np.array(hdmap)).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5
                frame_hdmaps.append(hdmap_t)
                
                # Image
                if t == 0: 
                    if use_real_image:
                        img = sample['image'].resize((448, 256))
                        img_t = (torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.0 - 0.5) / 0.5
                    else:
                        img_t = torch.zeros((3, 256, 448), dtype=torch.float32)
                    frame_images.append(img_t)
                
                # Layout
                layout_t = self._process_layout(sample, cutin_boxes_ego, t)
                frame_layouts.append(layout_t)
            
            all_hdmaps.append(torch.cat(frame_hdmaps, dim=2))
            all_layouts.append(torch.cat(frame_layouts, dim=2))
            if t == 0: all_images.append(torch.cat(frame_images, dim=2))

        hdmap_final = torch.stack(all_hdmaps, dim=0).to(self.device, dtype=torch.float16)
        layout_final = torch.stack(all_layouts, dim=0).to(self.device, dtype=torch.float16)
        img_cond = all_images[0].unsqueeze(0).to(self.device, dtype=torch.float16) if len(all_images) > 0 else None
            
        return hdmap_final, layout_final, img_cond

    def _process_layout(self, sample, cutin_boxes_ego, t):
        boxes_cam_obj = sample['boxes3d']
        if hasattr(boxes_cam_obj, 'tensor'): boxes_cam = boxes_cam_obj.tensor.numpy()
        elif isinstance(boxes_cam_obj, torch.Tensor): boxes_cam = boxes_cam_obj.cpu().numpy()
        else: boxes_cam = boxes_cam_obj
        
        # [FIX 2: NPC 车辆修复]
        # 如果 NPC 还是横向的，我们强制交换 长度(idx 3) 和 宽度(idx 4)
        if len(boxes_cam) > 0:
            boxes_cam = boxes_cam.copy()
            # 交换 Length 和 Width。NuScenes 原始通常是 [dx, dy, dz]
            # 如果绘图看起来是横的，说明 dx 和 dy 反了
            boxes_cam[:, [3, 4]] = boxes_cam[:, [4, 3]]
            
            # [Step 2] 统一转为 [Length, Height, Width] 格式供 draw_single_layout 使用
            # 经过上面的交换后，现在假设 idx 3 是真正视觉上的 Length
            # 我们需要输出 [L, H, W] -> 取 [3, 5, 4]
            # (注意：上面的交换已经改变了 boxes_cam 的内容，所以现在 idx 3 是旧的 W，idx 4 是旧的 L)
            # 为了逻辑清晰，我们重新组织：
            
            l = boxes_cam[:, 3:4] # 目前的 idx 3
            w = boxes_cam[:, 4:5] # 目前的 idx 4
            h = boxes_cam[:, 5:6] # idx 5
            
            # 重新拼装为 [x, y, z, L, H, W, yaw, ...]
            # 注意：这里我们构造一个新的数组，用于传给 draw
            # 这里的 L, H, W 顺序必须匹配 draw_single_layout 内部逻辑
            # 通常 graphics utils 喜欢 L, H, W
            boxes_cam_processed = np.concatenate([
                boxes_cam[:, :3], l, h, w, boxes_cam[:, 6:]
            ], axis=1)
        else:
            boxes_cam_processed = np.zeros((0, 9))

        # Cut-in 处理
        cam2ego = sample['calib']['cam2ego']
        ego2cam = np.linalg.inv(cam2ego)
        
        cutin_idx = min(t, len(cutin_boxes_ego) - 1)
        cutin_box_ego = cutin_boxes_ego[cutin_idx]
        
        # 调试打印 (检查 Cut-in 坐标)
        if t == 0 or t == 10:
             print(f"[Debug Frame {t}] Cut-in Ego Pos: X(Forward)={cutin_box_ego[0,0]:.2f}, Y(Left)={cutin_box_ego[0,1]:.2f}")
        
        cutin_box_cam = self.transform_box_ego2cam(cutin_box_ego, ego2cam)
        
        # 过滤 Z > 0.5 (在相机前方)
        if cutin_box_cam[0, 2] > 0.5:
            if len(boxes_cam_processed) > 0:
                min_dims = min(boxes_cam_processed.shape[1], cutin_box_cam.shape[1])
                combined_boxes = np.concatenate([boxes_cam_processed[:, :min_dims], cutin_box_cam[:, :min_dims]], axis=0)
            else:
                combined_boxes = cutin_box_cam
        else:
            combined_boxes = boxes_cam_processed
            
        cam_intrinsic = sample['calib']['cam_intrinsic']
        return self.draw_single_layout(combined_boxes, cam_intrinsic, resolution=(256, 448))

    @staticmethod
    def transform_box_ego2cam(box_ego, ego2cam):
        center_ego = box_ego[:, :3].astype(np.float32)
        if center_ego.ndim == 1: center_ego = center_ego.reshape(1, 3)
        
        # 1. Transform Center
        center_ego_hom = np.concatenate([center_ego, np.ones((1, 1), dtype=np.float32)], axis=1)
        center_cam_hom = (ego2cam @ center_ego_hom.T).T
        center_cam = center_cam_hom[:, :3]
        
        # 2. Dimensions: 必须输出 [Length, Height, Width]
        # box_ego 是 [L, W, H] (idx 3, 4, 5)
        # 我们要变为 [L, H, W] -> 取 [0, 2, 1]
        dims_ego = box_ego[:, 3:6]
        dims_cam = dims_ego[:, [0, 2, 1]] 
        
        # 3. Orientation
        yaw_ego = box_ego[:, 6]
        vec_ego = np.stack([np.cos(yaw_ego), np.sin(yaw_ego), np.zeros_like(yaw_ego)], axis=1)
        R_ego2cam = ego2cam[:3, :3]
        vec_cam = (R_ego2cam @ vec_ego.T).T
        yaw_cam = np.arctan2(vec_cam[:, 0], vec_cam[:, 2]) # Cam 坐标系下 Yaw 是 X-Z 平面
        
        vel = np.zeros((1, 2))
        box_cam = np.concatenate([center_cam, dims_cam, yaw_cam[:, None], vel], axis=1)
        return box_cam

    @staticmethod
    def draw_single_layout(boxes, cam_intrinsic, resolution=(256, 448)):
        tgt_H, tgt_W = resolution
        orig_H, orig_W = 900, 1600
        canvas = np.zeros((19, orig_H, orig_W), dtype=np.float32)
        
        if len(boxes) == 0:
             layout_tensor = torch.from_numpy(np.zeros((19, tgt_H, tgt_W), dtype=np.float32)).float()
             return (layout_tensor - 0.5) / 0.5

        # boxes 预期是 [x, y, z, L, H, W, yaw]
        boxes_7 = boxes[:, :7]
        
        # 注意：boxes3d_to_corners3d 这里的 rot_axis=1 (Y轴) 是符合 Camera 坐标系的
        corners3d = boxes3d_utils.boxes3d_to_corners3d(boxes_7, rot_axis=1)
        
        valid_mask = (corners3d[..., 2] > 0.1).any(axis=1)
        corners3d = corners3d[valid_mask]
        
        if len(corners3d) == 0:
             layout_tensor = torch.from_numpy(np.zeros((19, tgt_H, tgt_W), dtype=np.float32)).float()
             return (layout_tensor - 0.5) / 0.5
             
        corners2d = boxes3d_utils.corners3d_to_corners2d(corners3d, cam_intrinsic)
        
        box_skeleton = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
        car_idx = 8 

        def pt_out_img(pt):
            return pt[0]<0 or pt[0]>=orig_W or pt[1]<0 or pt[1]>=orig_H

        for corner in corners2d:
            if (corner[:, 0].min() >= orig_W or corner[:, 0].max() < 0 or 
                corner[:, 1].min() >= orig_H or corner[:, 1].max() < 0): continue
            
            corner = corner.astype(np.int32)
            
            # 简单的 Box 绘制
            for i_st, i_end in box_skeleton:
                if pt_out_img(corner[i_st]) and pt_out_img(corner[i_end]): continue
                cv2.line(canvas[car_idx], tuple(corner[i_st]), tuple(corner[i_end]), color=1.0, thickness=5)
            
            # 简单的面填充 (取前面的面)
            cv2.fillPoly(canvas[car_idx], [np.array([corner[4], corner[5], corner[6], corner[7]])], color=0.8)

        canvas_resized_list = []
        for i in range(19):
            canvas_resized_list.append(cv2.resize(canvas[i], (tgt_W, tgt_H), interpolation=cv2.INTER_NEAREST))
        
        return (torch.from_numpy(np.stack(canvas_resized_list, axis=0)).float() - 0.5) / 0.5
        
    def load_cutin_scenario(self, json_path):
        print(f"Loading cut-in scenario from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        boxes_list = []
        sorted_keys = sorted(data.keys(), key=lambda k: int(k))
        
        for k in sorted_keys:
            frame_data = data[k]
            obj = frame_data['objects'][0]
            
            # [FIX 3: 坐标系完全修正]
            # NuScenes Ego: X = Forward (纵向), Y = Left (横向)
            # 假设 JSON 中: x 是纵向距离, y 是横向距离
            
            # 1. 映射逻辑
            nu_x = obj['x']  # 纵向距离
            nu_y = obj['y']  # 横向距离
            
            # 2. 角度修正
            # 如果 JSON 中 yaw=0 是沿着 X 轴正方向，则不需要修正
            # 如果车横着走，可能需要 nu_yaw = obj['yaw'] + np.pi/2
            # 暂时保持原始
            nu_yaw = obj['yaw'] + np.pi / 2
            
            # 3. 尺寸
            dx = obj['dx'] # Length
            dy = obj['dy'] # Width
            dz = obj['dz'] # Height
            
            # 构造 Ego 坐标系下的 Box [x, y, z, L, W, H, yaw]
            # z=0.8 大致是地面以上车中心的高度
            box = np.array([nu_x, nu_y, 0.8, dx, dy, dz, nu_yaw, 0.0, 0.0])
            boxes_list.append(box.reshape(1, -1))
            
        return boxes_list

    def run(self, prompt, ref_data_path, output_path, use_wo_img=True):
        cutin_scenario_path = os.path.join(ROOT_DIR, "cutin_scenario.json")
        cutin_boxes_ego = self.load_cutin_scenario(cutin_scenario_path)
        
        print(f"Loading dataset from {ref_data_path}...")
        dataset = load_dataset(ref_data_path)
        
        scene_token, start_frame_idx = self.find_valid_sequence(dataset, seq_len=24)
        print(f"Using scene {scene_token} starting at frame: {start_frame_idx}")
        
        prompt_embeds = self.encode_prompt(prompt)
        chunk_size = 24 
        
        # 调试: 打印第一帧 Cut-in 的位置
        if len(cutin_boxes_ego) > 0:
            print(f"DEBUG: Start Cut-in Pos: {cutin_boxes_ego[0][0, :3]}")

        hdmap_chunk, layout_chunk, first_img_cond = self.prepare_inputs_chunk(
            dataset, scene_token, start_frame_idx, cutin_boxes_ego, 
            frame_offset=0, chunk_len=chunk_size, 
            use_real_image=(not use_wo_img)
        )
        
        input_dict = {
            "grounding_downsampler_input": hdmap_chunk,
            "box_downsampler_input": layout_chunk,
        }
        if not use_wo_img: input_dict["img_cond"] = first_img_cond

        self.pipe.frame_num = chunk_size
        generator = torch.Generator(device=self.device).manual_seed(42)
        with torch.no_grad():
            output = self.pipe(
                prompt_embeddings=prompt_embeds,
                input_dict=input_dict,
                height=256,
                width=448 * 6, 
                num_frames=chunk_size,
                num_inference_steps=50,
                min_guidance_scale=2.5,
                max_guidance_scale=5.0,
                motion_bucket_id=60,
                noise_aug_strength=0.01,
                first_frame=True,
                generator=generator
            )
        
        print(f"Saving video to {output_path}...")
        video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 4, (448 * 6, 256))
        for img in output.frames[0]:
            video.write(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        video.release()
        print("Done.")

if __name__ == "__main__":
    # 配置
    SVD_PATH = "/DriveDreamer2/models/huggingface/stable-video-diffusion-img2vid-xt-1-1"
    DD2_WEIGHTS = "/DriveDreamer2/pretrained_models/drivedreamer2_wo_img/pytorch_gligen_weights.bin"
    DATA_PATH = "/DriveDreamer2/dreamer-data/v1.0-mini/cam_all_val/v0.0.2"
    
    # embedding pkl 文件
    EMBED_PATH = "/DriveDreamer2/dreamer-data/clip_text_transform_after_pool_panoramic.pkl"
    
    inference = DriveDreamerInference(SVD_PATH, DD2_WEIGHTS, EMBED_PATH) # 传入 Embedding 路径
    inference.setup()
    
    # 使用固定的 Key
    prompt = "rainy, realistic autonomous driving scene."
    
    # 确保生成足够长的时间以看到 Cut-in
    inference.run(prompt, DATA_PATH, "output/output.mp4", use_wo_img=True)