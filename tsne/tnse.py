import pickle as pkl
import numpy as np
from matplotlib import pyplot as plt

from tsne import bh_sne
import sys
from matplotlib.pyplot import cm
# with open("data", 'rb') as f:
#             if sys.version_info > (3, 0):
#                 data = pkl.load(f, encoding='latin1')
#             else:
#                 data = pkl.load(f)
data = np.load( "/home/kc/mmz/doublefacent/signatures/0512_ID+camera/signatures.npy")
data =data.astype('float64')


# with open("label", 'rb') as f:
#             if sys.version_info > (3, 0):
#                 y_data = pkl.load(f, encoding='latin1')
#             else:
#                 y_data = pkl.load(f)
#
# classNum = 20
# y_data = np.where(y_data==1)[1]*(9.0/classNum)

vis_data = bh_sne(data)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

y_train = np.load("/home/kc/mmz/doublefacent/signatures/0512_ID+camera/labels_name.npy")
num_classes = len(np.unique(y_train))
print("num_class:", num_classes)
colors = cm.Spectral(np.linspace(0, 1, num_classes))
fig = plt.figure()
plt.scatter(vis_x, vis_y, c=colors, s=10, cmap=plt.cm.get_cmap("jet", 10))
# plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()
fig.savefig('mytest.png')