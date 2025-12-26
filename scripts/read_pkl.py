import pickle
import numpy as np
import os
import torch # 可能包含 Tensor

def inspect_pkl(file_path):
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            
        print(f"=== Content of {file_path} ===")
        
        if isinstance(data, list):
            print(f"Type: List with {len(data)} elements")
            for i, item in enumerate(data):
                print(f"\n--- Item [{i}] ---")
                if isinstance(item, dict):
                    print(f"Type: Dictionary with {len(item)} keys")
                    for key, value in item.items():
                        print(f"\n  Key: '{key}'")
                        # === 修改：打印详细的值信息 ===
                        print_detailed_value(value, indent="    ")
                else:
                    print_detailed_value(item, indent="  ")
        else:
            # 处理非 List 的情况 (虽然这里已知是 List)
            print_detailed_value(data)
            
    except Exception as e:
        print(f"Error reading {file_path}: {e}")

def print_detailed_value(value, indent=""):
    """打印值的详细信息，包括形状和部分内容"""
    if isinstance(value, (np.ndarray, torch.Tensor)):
        shape = value.shape
        dtype = value.dtype
        print(f"{indent}Type: {type(value).__name__}, Shape: {shape}, Dtype: {dtype}")
        
        # 打印前几个数值作为示例
        flat_val = value.flatten()
        if len(flat_val) > 0:
            preview = flat_val[:8] # 只看前8个数字
            print(f"{indent}Preview: {preview}")
            
    elif isinstance(value, list):
        print(f"{indent}Type: List, Length: {len(value)}")
        if len(value) > 0:
            print(f"{indent}First element: {value[0]}")
            
    elif isinstance(value, dict):
        print(f"{indent}Type: Dict, Keys: {list(value.keys())}")
        
    else:
        print(f"{indent}Value: {value}")

if __name__ == "__main__":
    file_path = "clip_text_transform_after_pool_panoramic.pkl"
    
    if os.path.exists(file_path):
        inspect_pkl(file_path)
    else:
        print(f"File not found: {file_path}")