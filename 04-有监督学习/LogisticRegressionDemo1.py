'''
@Desc  : 二分类
@Date  : 2018/12/24
@Author: zhangjianfeng 
'''
from sklearn.linear_model import LinearRegression, LogisticRegression
import pandas as pd

data = pd.read_csv('quiz.csv', delimiter=',')
used_features = ["Last Score", "Hours Spent"]
X = data[used_features].values
scores = data["Score"].values

X_train = X[:11]
X_test = X[11:]

y_train = scores[:11]
y_test = scores[11:]

regr = LinearRegression()
regr.fit(X_train, y_train)
y_predict = regr.predict(X_test)

print(y_predict)
# [55.33375602 54.29040467 90.76185124]


passed = []

for i in range(len(scores)):
    if (scores[i] >= 60):
        passed.append(1)
    else:
        passed.append(0)

y_train = passed[:11]
y_test = passed[11:]

classifier = LogisticRegression(C=1e5)
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)
print(y_predict)
# [1 0 1]