from os import path, makedirs
import shutil

sourcr_images_dir = '../datasets/SAR-AIRcraft-1.0/JPEGImages'
txt_files_dir = '../datasets/SAR-AIRcraft-1.0/ImageSets/Main'
txt_files = ['test.txt', 'train.txt', 'val.txt']

target_dir = '../datasets/SAR-AIRcraft-1.0-yolo'
img_extension = '.jpg'
label_extension = '.txt'


for txt_file in txt_files:
    target_images_dir = path.join(target_dir, 'images',txt_file.split('.')[0])
    makedirs(target_images_dir, exist_ok=True)

    basenames = []
    with open(path.join(txt_files_dir, txt_file), 'r') as fp:
        for line in fp.readlines():
            basenames.append(line.strip())

    for bn in basenames:
        img_file = bn+img_extension
        shutil.copyfile(src=path.join(sourcr_images_dir, img_file), dst=path.join(target_images_dir, img_file))

print()