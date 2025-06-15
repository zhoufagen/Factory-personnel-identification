import os
import random
import shutil
from sklearn.model_selection import train_test_split

# 设置随机种子以保证可重复性
random.seed(42)

# 原始数据集路径
dataset_path = "dataset"
images_path = os.path.join(dataset_path, "images", "images")
labels_path = os.path.join(dataset_path, "labels", "labels")

# 新数据集路径
new_dataset_path = "split_dataset"
os.makedirs(new_dataset_path, exist_ok=True)

# 创建新目录结构
new_images_train = os.path.join(new_dataset_path, "images", "train")
new_images_val = os.path.join(new_dataset_path, "images", "val")
new_labels_train = os.path.join(new_dataset_path, "labels", "train")
new_labels_val = os.path.join(new_dataset_path, "labels", "val")

os.makedirs(new_images_train, exist_ok=True)
os.makedirs(new_images_val, exist_ok=True)
os.makedirs(new_labels_train, exist_ok=True)
os.makedirs(new_labels_val, exist_ok=True)

# 获取所有图像文件名（不带扩展名）
image_files = [f.split('.')[0] for f in os.listdir(images_path) if f.endswith('.jpg')]

# 按4:1比例划分训练集和验证集
train_files, val_files = train_test_split(image_files, test_size=0.2, random_state=42)

print(f"总样本数: {len(image_files)}")
print(f"训练集样本数: {len(train_files)}")
print(f"验证集样本数: {len(val_files)}")

# 复制训练集文件
for file in train_files:
    # 复制图像
    shutil.copy(
        os.path.join(images_path, f"{file}.jpg"),
        os.path.join(new_images_train, f"{file}.jpg")
    )
    # 复制标签
    label_file = f"{file}.txt"
    if os.path.exists(os.path.join(labels_path, label_file)):
        shutil.copy(
            os.path.join(labels_path, label_file),
            os.path.join(new_labels_train, label_file)
        )

# 复制验证集文件
for file in val_files:
    # 复制图像
    shutil.copy(
        os.path.join(images_path, f"{file}.jpg"),
        os.path.join(new_images_val, f"{file}.jpg")
    )
    # 复制标签
    label_file = f"{file}.txt"
    if os.path.exists(os.path.join(labels_path, label_file)):
        shutil.copy(
            os.path.join(labels_path, label_file),
            os.path.join(new_labels_val, label_file)
        )

print("数据集划分完成！")