import torch
import sys
import os
from pathlib import Path
import time

# 确保可以导入YOLOv5模块
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 添加ROOT到PATH

def main():
    try:
        # 尝试导入ultralytics库
        from ultralytics import YOLO
        
        # 设置模型路径
        model_path = ROOT / 'weights' / 'yolov8s.pt'
        
        if not os.path.exists(model_path):
            print(f"错误: 模型文件 {model_path} 不存在")
            return
            
        print(f"加载模型: {model_path}")
        
        # 确保使用GPU
        if torch.cuda.is_available():
            device = 'cuda:0'  # 使用第一个GPU
            print(f"使用GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = 'cpu'
            print("警告: 未检测到GPU, 将在CPU上运行 (这会很慢)")
        
        # 加载YOLOv8模型，明确指定设备
        model = YOLO(model_path)
        model.to(device)  # 明确移动到指定设备
        
        # 打印模型信息
        print(f"\nYOLOv8s 模型信息:")
        print(f"模型文件: {model_path}")
        print(f"运行设备: {device}")
        
        # 准备测试图像
        img_size = 640
        img = torch.rand(1, 3, img_size, img_size).to(device)  # 确保图像在GPU上
        
        # 预热
        print("预热中...")
        for _ in range(10):
            _ = model.predict(source=img, verbose=False, device=device)
        
        # 测速
        print("测速中...")
        iterations = 100
        start = time.time()
        for _ in range(iterations):
            _ = model.predict(source=img, verbose=False, device=device)
        end = time.time()
        
        # 计算平均推理时间
        avg_time = (end - start) * 1000 / iterations  # 毫秒
        
        # 尝试获取参数量
        try:
            param_count = sum(p.numel() for p in model.model.parameters())
            print(f"模型参数量: {param_count/1e6:.1f}M 参数")
        except:
            print("无法计算精确参数量")
            
        print(f"推理速度: {avg_time:.2f} ms per image (在 {device})")
        print(f"FPS: {1000 / avg_time:.1f}")
        
    except ImportError:
        print("错误: 未找到ultralytics库，请先安装: pip install ultralytics")
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 

"""
YOLOv8s 模型信息:
模型文件: /root/yolov5/weights/yolov8s.pt
运行设备: cuda:0
预热中...
测速中...
模型参数量: 11.2M 参数
推理速度: 5.23 ms per image (在 cuda:0)
FPS: 191.0
"""