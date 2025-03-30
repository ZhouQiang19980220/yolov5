import torch
import os

script_dir = os.path.dirname(__file__)
save_dir = os.path.join(script_dir, 'output/')
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

img = 'https://ultralytics.com/images/zidane.jpg'

results = model(img)

results.print()
results.show()
results.save(save_dir = 'runs/detect/zq_learn', exist_ok=True)