'''
@Desc  : 
@Date  : 2019/1/2
@Author: zhangjianfeng 
'''

from matplotlib.pylab import array, diag
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import linalg
from sklearn import mixture


def sample_gaussian_mixture(centroids, ccov, mc=None, samples=1):
    """
    Draw samples from a Mixture of Gaussians (MoG)
    Parameters
    ----------
    centroids : list
        List of cluster centers - [ [x1,y1,..],..,[xN, yN,..] ]
    ccov : list
        List of cluster co-variances DxD matrices
    mc : list
        Mixing cofficients for each cluster (must sum to one)
                  by default equal for each cluster.
    Returns
    -------
    X : 2d np array
         A matrix with samples rows, and input dimension columns.
    Examples
    --------
    ::
        from pypr.clustering import *
        from numpy import *
        centroids=[array([10,10])]
        ccov=[array([[1,0],[0,1]])]
        samples = 10
        gmm.sample_gaussian_mixture(centroids, ccov, samples=samples)
    """
    cc = centroids
    D = len(cc[0])  # Determin dimensionality

    # Check if inputs are ok:
    K = len(cc)
    if mc is None:  # Default equally likely clusters
        mc = np.ones(K) / K
    if len(ccov) != K:
        raise Exception(ValueError, "centroids and ccov must contain the same number" + \
                        "of elements.")
    if len(mc) != K:
        raise Exception(ValueError, "centroids and mc must contain the same number" + \
                        "of elements.")

    # Check if the mixing coefficients sum to one:
    EPS = 1E-15
    if np.abs(1 - np.sum(mc)) > EPS:
        raise Exception(ValueError, "The sum of mc must be 1.0")

    # Cluster selection
    cs_mc = np.cumsum(mc)
    cs_mc = np.concatenate(([0], cs_mc))
    sel_idx = np.random.rand(samples)

    # Draw samples
    res = np.zeros((samples, D))
    for k in range(K):
        idx = (sel_idx >= cs_mc[k]) * (sel_idx < cs_mc[k + 1])
        ksamples = np.sum(idx)
        drawn_samples = np.random.multivariate_normal( \
            cc[k], ccov[k], ksamples)
        res[idx, :] = drawn_samples
    return res
# 将样本点显示在二维坐标中
def plot_results(X, Y_, means, covariances, colors, eclipsed, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, colors)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        if eclipsed:
            # Plot an ellipse to show the Gaussian component
            angle = np.arctan(u[1] / u[0])
            angle = 180. * angle / np.pi  # convert to degrees
            ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
            ell.set_clip_box(splot.bbox)
            ell.set_alpha(0.5)
            splot.add_artist(ell)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)


# 创建样本数据，一共3个簇
mc = [0.4, 0.4, 0.2]
centroids = [array([0, 0]), array([3, 3]), array([0, 4])]
ccov = [array([[1, 0.4], [0.4, 1]]), diag((1, 2)), diag((0.4, 0.1))]

X = sample_gaussian_mixture(centroids, ccov, mc, samples=1000)

# 用plot_results函数显示未经聚类的样本数据
gmm = mixture.GaussianMixture(n_components=1, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, ['grey'], False, 0, "Sample data")

# 用EM算法的GMM将样本分为3个簇，并按不同颜色显示在二维坐标系中
gmm = mixture.GaussianMixture(n_components=3, covariance_type='full').fit(X)
plot_results(X, gmm.predict(X), gmm.means_, gmm.covariances_, ['red', 'blue', 'green'], True, 1, "GMM clustered data")

plt.show()
