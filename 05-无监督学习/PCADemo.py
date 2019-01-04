'''
@Desc  : 
@Date  : 2019/1/3
@Author: zhangjianfeng 
'''
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.decomposition import PCA


def pca(om, cn):
    ipca = PCA(cn).fit(om)
    img_c = ipca.transform(om)

    print(img_c.shape)
    print
    np.sum(ipca.explained_variance_ratio_)

    temp = ipca.inverse_transform(img_c)
    print(
        temp.shape)

    return temp


def compressImage(image, k):
    redChannel = image[..., 0]
    greenChannel = image[..., 1]
    blueChannel = image[..., 2]

    cmpRed = pca(redChannel, k)
    cmpGreen = pca(greenChannel, k)
    cmpBlue = pca(blueChannel, k)

    newImage = np.zeros((image.shape[0], image.shape[1], 3), 'uint8')

    newImage[..., 0] = cmpRed
    newImage[..., 1] = cmpGreen
    newImage[..., 2] = cmpBlue

    return newImage


path = 'zjf.jpg'
img = mpimg.imread(path)

title = "Original Image"
plt.title(title)
plt.imshow(img)
plt.show()

weights = [100, 50, 20, 5]

for k in weights:
    newImg = compressImage(img, k)

    title = " Image after =  %s" % k
    plt.title(title)
    plt.imshow(newImg)
    plt.show()

    newname = os.path.splitext(path)[0] + '_pca_' + str(k) + '.jpg'
    mpimg.imsave(newname, newImg)
