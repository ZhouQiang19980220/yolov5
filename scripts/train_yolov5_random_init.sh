# python train.py --img 640 --epochs 200 --data SARAIRcraft1.0.yaml --weights '' --cache ram --cfg yolov5s.yaml

python train.py --img 640 --epochs 200 --data SARAIRcraft1.0.yaml --weights '' --cache ram --cfg yolov5s.yaml --project runs/train/yolov5_baseline_随机初始化

# python train.py --img 640 --epochs 200 --data SARAIRcraft1.0.yaml --weights '' --cache ram --cfg yolov5s_sparse_conv_transformer_neck.yaml

python train.py --img 640 --epochs 200 --data SARAIRcraft1.0.yaml --weights '' --cache ram --cfg yolov5s_sparse_conv.yaml --hyp data/hyps/hyp.scratch-low_label_smooth.yaml

python train.py --img 640 --epochs 200 --data SARAIRcraft1.0.yaml --weights '' --cache ram --cfg yolov5s_sparse_conv.yaml --hyp data/hyps/hyp.scratch-low_label_smooth=0.1.yaml

python train.py --img 640 --epochs 200 --data SARAIRcraft1.0.yaml --weights '' --cache ram --cfg yolov5s_sparse_conv.yaml --hyp data/hyps/hyp.scratch-low_label_smooth=0.3.yaml

python train.py --img 640 --epochs 200 --data SARAIRcraft1.0.yaml --weights '' --cache ram --cfg yolov5s_sparse_conv.yaml --hyp data/hyps/hyp.scratch-low_label_smooth=0.7.yaml

python train.py --img 640 --epochs 200 --data SARAIRcraft1.0.yaml --weights '' --cache ram --cfg yolov5s_sparse_conv.yaml --hyp data/hyps/hyp.scratch-low_label_smooth=0.9.yaml

python train.py --img 640 --epochs 200 --data SARAIRcraft1.0.yaml --weights '' --cache ram --cfg yolov5s_sparse_conv_average_trainsformer_neck.yaml

yolo train data=SARAIRcraft1.0.yaml model=yolov8s.pt epochs=200 lr0=0.01