import pandas as pd
import matplotlib.pyplot as plt

# 读取训练结果
results = pd.read_csv('runs/detect/exp1/results.csv')

# 绘制训练和验证损失曲线
plt.figure(figsize=(12, 6))
plt.plot(results['                  train/box_loss'], label='Train Box Loss')
plt.plot(results['                  train/cls_loss'], label='Train Class Loss')
plt.plot(results['                  train/dfl_loss'], label='Train DFL Loss')
plt.plot(results['                  val/box_loss'], label='Validation Box Loss')
plt.plot(results['                  val/cls_loss'], label='Validation Class Loss')
plt.plot(results['                  val/dfl_loss'], label='Validation DFL Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()
plt.savefig('custom_loss_curve.png')
plt.show()