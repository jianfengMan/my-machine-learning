'''
@Desc  : 线性可分
@Date  : 2018/12/28
@Author: zhangjianfeng 
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC


def plot_svc_decision_function(model, ax=None, plot_support=True):
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)

    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # 绘制超平面
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # 标识出支持向量
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, edgecolors='blue', facecolors='none')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


# 用make_blobs生成样本数据
from sklearn.datasets.samples_generator import make_blobs

X, y = make_blobs(n_samples=50, centers=2,
                  random_state=0, cluster_std=0.60)

# 将样本数据绘制在直角坐标中
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
plt.show()

# 用线性核函数的SVM来对样本进行分类
model = SVC(kernel='linear')
model.fit(X, y)

# 在直角坐标中绘制出分割超平面、辅助超平面和支持向量
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model);
plt.show()

X, y = make_blobs(n_samples=100, centers=2,
                  random_state=0, cluster_std=0.9)
plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn');
plt.show()

model = SVC(kernel='linear')
model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.show()

# 所用的 SVC 类，有一个 C 参数，对应的是错误项（Error Term）的惩罚系数。
# 这个系数设置得越高，容错性也就越小，分隔空间的硬度也就越强

# 加大惩罚系数
model = SVC(kernel='linear', C=10.0)
model.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.show()
