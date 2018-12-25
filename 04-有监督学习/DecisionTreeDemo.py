'''
@Desc  : 
@Date  : 2018/12/25
@Author: zhangjianfeng 
'''

from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np

# 9个女孩和8只猫的数据，对应7个feature，yes取值为1，no为0
features = np.array([
    [1, 1, 0, 0, 1, 0, 1],
    [1, 1, 1, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 0, 1],
    [1, 1, 0, 0, 1, 0, 1],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [1, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 1, 0, 1, 1, 1, 1],
    [1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 0],
    [0, 0, 0, 1, 1, 1, 0],
    [1, 0, 0, 1, 1, 1, 0],
    [0, 0, 1, 0, 1, 1, 0],
    [1, 1, 1, 1, 1, 1, 0],
    [1, 0, 1, 1, 1, 1, 0]
])

# 1 表示是女孩，0表示是猫
labels = np.array([
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
    [0],
])

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels,
                                                    test_size=0.2,
                                                    random_state=0)

# 训练分类树模型
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

print(clf.predict(X_test))

# 比对结果
print(clf.score(X_test, y_test))

HelloKitty = np.array([[1, 1, 1, 1, 1, 1, 1]])
print(clf.predict(HelloKitty))
