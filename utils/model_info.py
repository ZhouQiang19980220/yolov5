import torch
import sys
import os
from pathlib import Path

# 确保可以导入YOLOv5模块
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加ROOT到PATH

from models.common import DetectMultiBackend
from utils.torch_utils import select_device, time_sync

def main():
    # 设置模型路径
    model_path = ROOT / 'yolov5s.pt'
    
    # 加载模型
    print(f"加载模型: {model_path}")
    device = select_device('')  # 自动选择设备
    model = DetectMultiBackend(model_path, device=device)

    # 打印模型信息
    print(f"模型参数量: {sum(x.numel() for x in model.parameters()) / 1e6:.1f}M 参数")

    # 测试推理速度
    img = torch.rand(1, 3, 640, 640).to(device)  # 创建随机输入

    # 预热
    for _ in range(10):
        _ = model(img)

    # 测速
    iterations = 100
    start = time_sync()
    for _ in range(iterations):
        _ = model(img)
    end = time_sync()

    # 计算平均推理时间
    avg_time = (end - start) * 1000 / iterations  # 毫秒
    print(f"推理速度: {avg_time:.2f} ms per image (在 {device})")
    print(f"FPS: {1000 / avg_time:.1f}")

if __name__ == '__main__':
    main() 