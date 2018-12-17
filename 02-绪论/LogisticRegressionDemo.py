'''
@Desc  :
@Date  : 2018/12/18
@Author: zhangjianfeng
'''
print(__doc__)


from numpy import *
from sklearn.datasets import load_iris

#加载鸢尾花数据集
iris = load_iris()
samples = iris.data
print(samples.shape)

target = iris.target
print(set(target))
# {0, 1, 2} ,有三种分类结果

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

# 训练模型
classifier.fit(samples,target)

x = classifier.predict(array([5,3,5,2.5]).reshape(1,-1))
# reshape(x,y),-1代表未知维度，比如，reshape(-1,2),代表2列，n * 2 = x * y
print(x)