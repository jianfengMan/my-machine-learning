'''
@Desc  : 简历过滤器
@Date  : 2019/1/2
@Author: zhangjianfeng 
'''
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report


# 加载数据集
def loadDataSet(file_name):
    data = pd.read_csv(file_name)
    # print(data.tail())
    # 学历（degree）,
    # 毕业院校（education）,
    # 技能（skills）,
    # 曾经工作过的公司（working_experience）,
    # 当前职位（position） target
    features = ["degree", "education", "skills", "working_experience"]
    label_name = "position"
    # dev（开发工程师），qa（测试工程师）和 manager（经理）
    return data[features], data[label_name]


# 切分特征，一个变多个
def convert(feature, split_str=";"):
    return feature.split(split_str)


labelEncoder = LabelEncoder()
enc = OneHotEncoder()


# LabelEncoder编码，将特征值转化为数字
def convertLabelEncoder(trans_values):
    le = labelEncoder.fit(trans_values)
    return le.transform(trans_values)


# LabelEncoder编码，将特征值转化为数字,每行带分隔符
def convertSplitLabelEncoder(trans_values):
    arr_values = list(map(convert, trans_values))
    all_values = [x for j in arr_values for x in j]
    le = labelEncoder.fit(all_values)
    labels = list(map(lambda x: le.transform(x), arr_values))
    return labels, len(set(all_values))


# 进行OneHotEcoder编码
def convertOneHotEncoder(labels):
    reshape_labels = labels.reshape(-1, 1)
    tran_enc = enc.fit(reshape_labels)
    return tran_enc.transform(reshape_labels).toarray()


def convertSplitEncoder(labels, label_encoder_len):
    all_arr_data = np.zeros((len(labels), label_encoder_len))
    for index, single_arr_index in enumerate(labels):
        single_arr_data = np.zeros(label_encoder_len)
        single_arr_data[single_arr_index] = 1
        all_arr_data[index] = single_arr_data
    return all_arr_data


# 加载数据集
file_name = "employees_dataset.csv"
features_mat, label_mat = loadDataSet(file_name)
# print(label_mat)

features_len = len(label_mat)

# 将label转换为数字
for index, label in enumerate(set(label_mat)):
    label_mat = np.where(label_mat == label, index, label_mat)

# LabelEncoder编码，将特征值转化为数字
degree_labels = convertLabelEncoder(features_mat['degree'])
education_labels = convertLabelEncoder(features_mat['education'])

# 带分隔符的LabelEncoder转换
working_labels, working_label_len = convertSplitLabelEncoder(features_mat['working_experience'])
skills_labels, skills_label_len = convertSplitLabelEncoder(features_mat['skills'])

# OneHot转换
degree_enc = convertOneHotEncoder(degree_labels)
education_enc = convertOneHotEncoder(education_labels)

skills_enc = convertSplitEncoder(skills_labels, skills_label_len)
working_enc = convertSplitEncoder(working_labels, working_label_len)

# 进行特征拼接
features_mat = np.hstack((degree_enc, education_enc, skills_enc, working_enc))

# print(len(degree_enc))
# print(len(education_enc))
# print(len(skills_enc))
# print(len(working_enc))
# print(len(label_mat))
print(features_mat.shape)

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(features_mat,
                                                    label_mat.astype(int),
                                                    test_size=0.3,
                                                    random_state=int(time.time()))

lr = LogisticRegression(C=1e5)
lr.fit(X_train, y_train)
y_test_predict = lr.predict(X_test)
print("error_count:{}".format(sum(abs(y_test_predict - y_test))))

# F1-score precision recall
print(classification_report(y_test, y_test_predict))
'''
error_count:1
             precision    recall  f1-score   support

          0       1.00      0.67      0.80         3
          1       0.93      1.00      0.97        14
          2       1.00      1.00      1.00         5

avg / total       0.96      0.95      0.95        22


'''
