import random
import numpy as np
import math
try:
    import torch
except ImportError:
    torch = None

def set_random_seed(enable=False, seed=42):
    """
    设置随机种子以确保实验的可重复性
    
    Args:
        enable (bool): 是否启用固定随机种子
        seed (int): 当 enable 为 True 时使用的随机种子
    """
    if enable:
        # 设置 Python 随机种子
        random.seed(seed)
        # 设置 NumPy 随机种子
        np.random.seed(seed)
        
        # 设置 PyTorch 随机种子（如果可用）
        if torch is not None:
            torch.manual_seed(seed)  # CPU 随机种子
            torch.cuda.manual_seed(seed)  # 当前 GPU 随机种子
            torch.cuda.manual_seed_all(seed)  # 所有 GPU 随机种子
            # 确保 CUDA 操作的确定性
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        print(f"随机种子已设置为: {seed}")
    else:
        # 重置为随机性，使用系统时间作为种子
        random.seed()
        np.random.seed(None)
        
        print("随机种子已禁用，使用完全随机模式")
        
def vec2(p):
    '''
    将点转换为二维向量。
    
    输入:
    p (list or tuple): 一个包含两个元素的列表或元组，分别表示点的 x 和 y 坐标。
    
    返回:
    np.ndarray: 一个包含两个元素的 numpy 数组，分别表示点的 x 和 y 坐标。
    '''
    return np.array([p[0], p[1]], dtype=float)

def norm(v):
    '''
    归一化向量。
    
    输入:
    v (np.ndarray): 一个包含两个元素的 numpy 数组，分别表示向量的 x 和 y 坐标。
    
    返回:
    np.ndarray: 一个包含两个元素的 numpy 数组，分别表示归一化后的向量的 x 和 y 坐标。
    '''
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n

def angle_deg(v):
    '''
    计算向量的角度（单位：度）。
    
    输入:
    v (np.ndarray): 一个包含两个元素的 numpy 数组，分别表示向量的 x 和 y 坐标。
    
    返回:
    float: 向量的角度（范围：0到360度）。
    '''
    return (math.degrees(math.atan2(v[1], v[0])) + 360.0) % 360.0

def seg_dist(a, b, p):
    '''
    计算点到线段的距离。
    
    输入:
    a (np.ndarray): 线段的起点，一个包含两个元素的 numpy 数组，分别表示点的 x 和 y 坐标。
    b (np.ndarray): 线段的终点，一个包含两个元素的 numpy 数组，分别表示点的 x 和 y 坐标。
    p (np.ndarray): 要计算距离的点，一个包含两个元素的 numpy 数组，分别表示点的 x 和 y 坐标。
    
    返回:
    float: 点到线段的距离。
    '''
    ab = b - a
    t = 0.0 if (ab[0]==0 and ab[1]==0) else max(0.0, min(1.0, np.dot(p - a, ab) / (np.dot(ab, ab)+1e-9)))
    proj = a + t * ab
    return np.linalg.norm(p - proj)
