import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


data = np.load('test_features.npz')
features = data['features']
labels = data['labels']


tsne = TSNE(n_components=2, random_state=0)
features_2d = tsne.fit_transform(features)


plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    indices = labels == label
    plt.scatter(features_2d[indices, 0], features_2d[indices, 1], label=label)

plt.legend()
plt.show()
