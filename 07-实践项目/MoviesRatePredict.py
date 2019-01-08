'''
@Desc  : 电影评分预测
@Date  : 2019/1/3
@Author: zhangjianfeng 
'''
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 加载数据集
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


def loadDataSet(file_name):
    '''
       not 中文电影名（chinese_name），
       not 英文电影名（english_name），
       导演（director），
       主演（starring），
       类型（type），
       发行日期（release_date），
       评分（rate），
       投票数（votes），
       发行地区（region），
       播放次数（runtime），
       分级（certification），
       语言（language），
       发行公司（company）
        '''
    data = pd.read_csv(file_name)
    return data


labelEncoder = LabelEncoder()
enc = OneHotEncoder()
stas = StandardScaler()


# 切分特征，一个变多个
def convert(feature, split_str=" "):
    return feature.split(split_str)


# LabelEncoder编码，将特征值转化为数字,每行带分隔符
def convertSplitLabelEncoder(trans_values):
    arr_values = list(map(convert, trans_values))
    all_values = [x for j in arr_values for x in j]
    print(set(all_values))
    print(len(set(all_values)))
    le = labelEncoder.fit(all_values)
    labels = list(map(lambda x: le.transform(x), arr_values))
    return labels, len(set(all_values))


def convertSplitEncoder(labels, label_encoder_len):
    all_arr_data = np.zeros((len(labels), label_encoder_len))
    for index, single_arr_index in enumerate(labels):
        single_arr_data = np.zeros(label_encoder_len)
        single_arr_data[single_arr_index] = 1
        all_arr_data[index] = single_arr_data
    return all_arr_data


# 判断是否包含汉字
def isChinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False


def validResult(realSet, predictSet, threshold=0.10):
    '''
    验证标准：
    预测准确率=（预测正确样本数）/（总测试样本数）* 100%
    可以人工指定一个 ErrorTolerance（一般是10%或者5%），
    当 |预测值-真实值| / 真实值 <= ErrorTolerance 时，
    我们认为预测正确，否则为预测错误。
    '''
    # 正确返回1 ，错误返回0
    valid_result = [0 if (abs(predictSet[index] - lable) / lable > threshold) else 1
                    for index, lable in enumerate(realSet)]
    return sum(valid_result) / len(realSet) * 100


file_name = "movies_dataset.csv"
data = loadDataSet(file_name)
used_features = ['type', 'votes', 'runtime', 'rate']
data = data[used_features]
print(data.info())
print(data.describe())

# 过滤数据
data = data[
    (data.apply(lambda x: len(x['votes']) > 4, axis=1)) &
    (data.apply(lambda x: len(x['runtime']) > 4, axis=1)) &
    (data.apply(lambda x: isChinese(x['type']), axis=1)) &
    (data.apply(lambda x: len(x['rate']) == 6, axis=1)) &
    (data.apply(lambda x: not isChinese(x['rate']), axis=1))
    ]

# 概况展示
print(data.info())
print(data.describe())
print('----------------1---------------')
data['rate'] = list(map(lambda x: x.replace(' ', '').replace('\'', ''), data['rate']))
data['votes'] = list(map(lambda x: x.replace(' ', '').replace('\'', ''), data['votes']))
data['runtime'] = list(map(lambda x: x.replace(' ', '').replace('\'', ''), data['runtime']))
data['type'] = list(map(lambda x: x.replace('\'', ''), data['type']))

# 数据标准化
data['votes'] = stas.fit_transform(data['votes'].values.reshape(-1, 1))
data['runtime'] = stas.fit_transform(data['runtime'].values.reshape(-1, 1))

print('----------------2---------------')
print(data.info())
print(data.describe())

type_labels, type_label_len = convertSplitLabelEncoder(data['type'])

type_enc = convertSplitEncoder(type_labels, type_label_len)
print(type_enc[0])

# 删除转化列
data.drop(['type'], 1, inplace=True)
print(data['votes'].values.shape)
print(data['runtime'].values.shape)
print(type_enc.shape)

# 用hstack会报错
features = np.column_stack((data['votes'].values, data['runtime'].values, type_enc))
labels = data['rate']
print(features.shape)
print(data.info())

# 切分数据集
X_train, X_test, y_train, y_test = train_test_split(features,
                                                    labels.astype(float),
                                                    test_size=0.3,
                                                    random_state=int(time.time()))

# 调用模型
regr = DecisionTreeRegressor(max_depth=10)
regr = DecisionTreeRegressor(min_samples_leaf=8)
regr.fit(X_train, y_train)

# 预测
y_predict = regr.predict(X_test)

# 绘制结果
index = np.arange(len(X_test))
plt.figure()
plt.scatter(index, y_test, c="darkorange", label="data")
plt.plot(index, y_predict, color="yellowgreen", label="max_depth=10", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()

print("准确率:%.2f%%" % (validResult(y_test, y_predict)))
