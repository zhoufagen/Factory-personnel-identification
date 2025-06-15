from ultralytics import YOLO
if __name__ == '__main__':

    # 加载训练好的模型
    model = YOLO('person_detection/exp15/weights/best.pt')

    # 在验证集上评估
    metrics = model.val()  # 可以添加data参数指定验证数据集

    print(f"""
    性能指标:
    - 精确度(P): {metrics.p.mean():.4f}  # 使用metrics.p代替metrics.box.precision
    - 召回率(R): {metrics.r.mean():.4f}  # 使用metrics.r代替metrics.box.recall
    - mAP50: {metrics.map50:.4f}
    - mAP50-95: {metrics.map:.4f}
    - 速度指标:
      - 预处理: {metrics.speed['preprocess']:.2f} ms/image
      - 推理: {metrics.speed['inference']:.2f} ms/image
      - 后处理: {metrics.speed['postprocess']:.2f} ms/image
      - FPS: {1000 / (metrics.speed['preprocess'] + metrics.speed['inference'] + metrics.speed['postprocess']):.2f}
    """)

    # 获取模型大小和计算量信息
    model_info = model.info()
    print(f"""
    模型复杂度:
    - 参数量: {model_info['params'] / 1e6:.2f} M
    - GFLOPs: {model_info['gflops']:.2f}
    """)

    #
    # metrics = model.val(data='person_detection.yaml', imgsz=640, batch=1, device='0')  # device='0'表示使用GPU
    # print(f"FPS: {1000 / metrics.speed['inference']:.2f}")  # 转换为FPS