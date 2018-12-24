'''
@Desc  : 朴素贝叶斯分类器
    GaussianNB_高斯朴素贝叶斯
    BernoulliNB_伯努利朴素贝叶斯
    MultinomialNB_多项朴素贝叶斯
@Date  : 2018/12/23
@Author: zhangjianfeng 
'''

import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

# 导入数据集
data = pd.read_csv("career_data.csv")
# 将条件转换为数字
data['985_cleaned'] = np.where(data["985"] == "Yes", 1, 0)
data["education_cleaned"] = np.where(data["education"] == "bachlor", 1,
                                     np.where(data["education"] == "master", 2,
                                              np.where(data["education"] == "phd", 3, 4)))
data["skill_cleaned"] = np.where(data["skill"] == "c++", 1,
                                 np.where(data["skill"] == "java", 2, 3))
data["enrolled_cleaned"] = np.where(data["enrolled"] == "Yes", 1, 0)
# 切分数据集 9：1
X_train, X_test = train_test_split(data, test_size=0.1, random_state=int(time.time()))
print(len(X_train) / len(X_test))
# print(X_train)
# 初始化分类器
gnb = GaussianNB()
used_features = ["985_cleaned",
                 "education_cleaned",
                 "skill_cleaned"
                 ]
# 训练分类器
gnb.fit(X_train[used_features].values, X_train["enrolled_cleaned"])
y_pred = gnb.predict(X_test[used_features])

#输出预测结果
print("Number of mislabeled points out of a total {} points : {}, performance {:05.2f}%"
    .format(
    X_test.shape[0],
    (X_test["enrolled_cleaned"] != y_pred).sum(),
    100 * (1 - (X_test["enrolled_cleaned"] != y_pred).sum() / X_test.shape[0])
))
