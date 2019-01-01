'''
@Desc  : 股票价格预测,预测收盘价
@Date  : 2019/1/1
@Author: zhangjianfeng 
'''

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# 加载数据集
def loadDataSet(file_name):
    data = pd.read_csv(file_name)
    print(data.tail())
    # 日最高价（High Price）
    # 最低价（Low Price）
    # 开盘价（Open Price）
    # 收盘价（Close Price)  label
    features = ["High Price", "Low Price", "Open Price", "Volume"]
    lable_name = "Close Price"
    return data[features], data[lable_name]


def validResult(realSet, predictSet, threshold=0.05):
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


file_name = "msft_stockprices_dataset.csv"
features_mat, lable_mat = loadDataSet(file_name)

# 训练集：(测试集+验证集）= 8：2
X_train, X_test_val, y_train, y_test_val = train_test_split(features_mat,
                                                            lable_mat,
                                                            test_size=0.2,
                                                            random_state=0)
# 测试集：验证集 = 1：1
X_test, X_valid, y_test, y_valid = train_test_split(X_test_val,
                                                    y_test_val,
                                                    test_size=0.5,
                                                    random_state=0)
# 输出各自的长度
print("X_train:%d,\nX_valid:%d,\nX_test:%d\n" %
      (len(X_train), len(X_valid), len(X_test)))

# 构建线性回归模型
lr_model = LinearRegression()
# 训练模型
lr_model.fit(X_train, y_train)

# 预测验证和测试集
y_valid_predict = lr_model.predict(X_valid)
y_test_predict = lr_model.predict(X_test)

# 获取校验结果
valid_predict_result = validResult(y_valid, y_valid_predict)
test_predict_result = validResult(y_test, y_test_predict)

# 输出准确率
print("valid_predict_result:%.0f%%,\ntest_predict_result:%.0f%%" % (valid_predict_result, test_predict_result))

# 绘图显示结果
index = np.arange(len(y_valid))
plt.plot(index, y_valid, "r-o")
plt.plot(index, y_valid_predict, "b-o")
plt.xlabel("index")
plt.ylabel("Close Price")
plt.title("valid_predict")

plt.show()
