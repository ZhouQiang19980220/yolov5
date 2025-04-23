import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from models.common import DetectMultiBackend

# 定义特征图提取钩子
class FeatureExtractor:
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.outputs = {layer: None for layer in target_layers}
        self.hooks = []
        
        # 注册钩子
        for layer_name in target_layers:
            # 使用eval()动态获取模型中的层
            layer = eval(f"model.{layer_name}")
            hook = layer.register_forward_hook(self._get_hook(layer_name))
            self.hooks.append(hook)
    
    def _get_hook(self, layer_name):
        def hook_fn(module, input, output):
            self.outputs[layer_name] = output
        return hook_fn
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_features(self):
        return self.outputs

# 可视化特征图
def visualize_feature_maps(features, output_dir, input_img):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 可视化输入图像
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    plt.title("Input Image")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "input_image.png"))
    plt.close()
    
    # 遍历每层的特征图
    for layer_name, feature_map in features.items():
        # 确保张量在CPU上并转换为numpy数组
        if isinstance(feature_map, torch.Tensor):
            feature_map = feature_map.detach().cpu().numpy()
        
        # 对于3D特征图(C,H,W)，计算通道间平均值
        if len(feature_map.shape) == 4:  # (B,C,H,W) -> (C,H,W)
            feature_map = feature_map[0]  # 取第一个样本
        
        # 计算通道间平均值
        avg_feature = np.mean(feature_map, axis=0)
        
        # 归一化到[0,1]
        avg_feature = (avg_feature - avg_feature.min()) / (avg_feature.max() - avg_feature.min() + 1e-8)
        
        # 调整大小至原始输入图像尺寸，便于比较
        h, w = input_img.shape[:2]
        resized_feature = cv2.resize(avg_feature, (w, h))
        
        # 使用热力图可视化
        plt.figure(figsize=(10, 10))
        plt.imshow(resized_feature, cmap='viridis')
        plt.colorbar(label='Feature Activation')
        plt.title(f"Feature Map: {layer_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"feature_map_{layer_name.replace('.', '_')}.png"))
        plt.close()
        
        # 与原图叠加的热力图
        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
        plt.imshow(resized_feature, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Feature Activation')
        plt.title(f"Feature Map Overlay: {layer_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"overlay_{layer_name.replace('.', '_')}.png"))
        plt.close()

# 主函数
def main():
    # 参数设置
    weights = 'weights/yolov5s.pt'
    img_path = 'data/images/bus.jpg'
    output_dir = 'feature_maps'
    
    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights, device=device)
    model.eval()
    
    # 输入图像预处理
    img = cv2.imread(img_path)
    img_copy = img.copy()  # 保存原始图像用于可视化
    
    # 图像预处理，调整大小到模型期望输入
    img = cv2.resize(img, (640, 640))
    img = img / 255.0  # 归一化
    img = img.transpose(2, 0, 1)  # HWC -> CHW
    img = np.ascontiguousarray(img)  # 确保内存连续
    img = torch.from_numpy(img).float().to(device)
    img = img.unsqueeze(0)  # 添加批次维度 (1, 3, 640, 640)
    
    # 定义要提取特征的目标层
    target_layers = [
        # 骨干网络 (Backbone) 早期层
        "model.model[1]",
        # 骨干网络中部层
        "model.model[4]",
        # 骨干网络较深层
        "model.model[8]",
        # 颈部网络 (Neck/PANet) 输出
        "model.model[17]",
        "model.model[20]",
        "model.model[23]"
    ]
    
    # 初始化特征提取器
    extractor = FeatureExtractor(model, target_layers)
    
    # 执行前向传播
    with torch.no_grad():
        model(img)
    
    # 获取特征图
    features = extractor.get_features()
    
    # 可视化特征图
    visualize_feature_maps(features, output_dir, img_copy)
    
    # 移除钩子
    extractor.remove_hooks()
    
    print(f"特征图可视化完成，结果保存在 {output_dir} 目录下")

if __name__ == "__main__":
    main() 