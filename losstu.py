import matplotlib.pyplot as plt
import pandas as pd

# 读取loss值文件
with open('epoch_losses.txt', 'r') as f:
    losses = [float(line.strip()) for line in f]

# 将损失列表转换为Pandas的Series对象
loss_series = pd.Series(losses)

# 计算移动平均，窗口大小可以调整（例如5）
smoothed_losses = loss_series.rolling(window=5).mean()

# 绘制原始损失曲线
plt.plot(losses, label='Original Loss')

# 绘制平滑后的损失曲线
plt.plot(smoothed_losses, label='Smoothed Loss', linewidth=2)

# 添加图例和标签
plt.title('Training Loss over Epochs with Smoothing')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
