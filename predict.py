from ultralytics import YOLO
import os

# 加载训练好的模型
model = YOLO('person_detection/exp15/weights/best.pt')  # 替换为你的模型路径

# 设置输入输出路径
input_folder = 'img'  # 包含要预测的图片的文件夹
output_folder = 'img_out'  # 预测结果保存路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 对文件夹中的所有图片进行预测
results = model.predict(
    source=input_folder,
    save=True,  # 保存预测结果图片
    save_txt=True,  # 保存检测结果为txt文件
    save_conf=True,  # 保存置信度分数
    save_crop=False,  # 是否保存裁剪的检测目标
    show_labels=True,  # 在图片上显示标签
    show_conf=True,  # 在图片上显示置信度
    conf=0.25,  # 置信度阈值
    imgsz=640,  # 推理尺寸
    device='0',  # 使用GPU 0，如果是CPU则设为'cpu'
    project=output_folder  # 结果保存路径
)

print(f"预测完成，结果保存在: {output_folder}")