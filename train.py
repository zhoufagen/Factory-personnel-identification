from ultralytics import YOLO


if __name__ == '__main__':
        
    # 加载预训练模型
    model = YOLO('yolov8s.pt')  # 使用YOLOv8s预训练模型

    # 训练模型
    results = model.train(
        data='person_detection.yaml',
        epochs=100,
        imgsz=640,
        batch=4,
        patience=50,
        device='0',  # 使用GPU 0，如果是CPU则设为'cpu'
        project='person_detection',
        name='exp1',
        save=True,
        save_period=10,
        visualize=True,
        val=True,
        plots=True
    )