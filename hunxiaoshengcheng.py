import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 加载 .npz 文件
data = np.load('predictions_and_labels_test6.npz')

# 获取预测结果和真实标签
preds = data['preds']
golds = data['golds']

# 定义标签名称
labels = ['Happiness', 'Sadness', 'Neutral', 'Anger', 'Excited', 'Frustrated']

# 生成混淆矩阵
cm = confusion_matrix(golds, preds)

# 设置图形的尺寸
plt.figure(figsize=(8, 6))

# 绘制热力图
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, cbar=True)

# 设置轴标签和标题
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)

# 调整字体大小和旋转角度
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)

# 显示图像
plt.tight_layout()
plt.show()


