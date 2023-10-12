from os import path
import shutil
import random
import os
import copy

retention_rate = 0.9
annotation_source_dir = 'datasets/SAR-AIRcraft-1.0-yolo/labels/train'
annotation_des_dir = f'datasets/SAR-AIRcraft-1.0-yolo/labels/train_{retention_rate}'
os.makedirs(annotation_des_dir, exist_ok=True)      


annotation_files = os.listdir(annotation_source_dir)

number_annotation = 0           # 原始数据集中标注框的数量
number_new_annotation = 0       # 新数据集中标注框的数量
all_delete_line = []
for file in annotation_files:
    with open(path.join(annotation_source_dir, file), 'r') as fp:
        lines = fp.readlines()

    new_lines = []
    for l in lines:
        if random.uniform(0, 1) < retention_rate:
            new_lines.append(l)
            number_new_annotation += 1
        else:
            all_delete_line.append((f'{file}:{l.strip()}\n'))
        number_annotation += 1
    
    with open(path.join(annotation_des_dir, file), 'w') as fp:
        fp.writelines(new_lines)

with open('datasets/SAR-AIRcraft-1.0-yolo/labels/all_delete_line.txt', 'w') as fp:
    fp.writelines(all_delete_line)
print(f'number_annotation = {number_annotation}')
print(f'number_new_annotation = {number_new_annotation}')
assert len(all_delete_line) + number_new_annotation == number_annotation