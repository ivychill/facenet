import numpy as np
from matplotlib.pyplot import cm
from matplotlib import pyplot as plt
from tsne import bh_sne
# load up data

# convert image data to float64 matrix. float64 is need for bh_sne
data = np.load( "./data/signatures.npy")
data =data.astype('float64')
x_data = data.reshape((data.shape[0], -1))
y_train = np.load("./data/labels_name.npy")
num_classes = len(np.unique(y_train))
colors = cm.Spectral(np.linspace(0, 1, num_classes))
# For speed of computation, only run on a subset
n = 20000
x_data = x_data[:n]
# y_data = y_data[:n]
# perform t-SNE embedding
vis_data = bh_sne(x_data)
# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]
plt.scatter(vis_x, vis_y, c=colors, cmap=plt.cm.get_cmap("jet", 10))
# plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()