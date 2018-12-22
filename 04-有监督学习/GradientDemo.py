'''
@Desc  : theta迭代过程
@Date  : 2018/12/22
@Author: zhangjianfeng 
'''
import numpy as np

# 输入变量 1000个
X = 2 * np.random.rand(1000, 1)

# 加上噪声的输出结果y 此时W为[4,3]
y = 4 + 3 * X + np.random.randn(1000, 1)

# 因为y = b + ax 相当于有个y=w0*x0+w1*x1,
# x1这时候等于1 公式就变成 y = w0 + w1*x1,迭代求解 [w0,w1]
X_b = np.c_[np.ones((1000, 1)), X]

# 学习率就是步长，超参数
learning_rate = 0.1

# 迭代次数
n_iterations = 100000

# 样本的数量
m = X_b.shape[0]

# 初始化theta，w0...wn
theta = np.random.randn(2, 1)
count = 0

# θ=θ−αXT(Xθ−Y),XT(Xθ−Y)是J(θ)对θ的偏导
# 之间设置超参数：迭代次数，迭代次数到了，我们就认为收敛了
for iteration in range(n_iterations):
    count += 1
    # 接着求梯度gradient
    gradients = 1 / m * X_b.T.dot(X_b.dot(theta) - y)
    # 应用公式调整theta值，theta_t + 1 = theta_t - grad * learning_rate
    theta = theta - learning_rate * gradients

print(count)
print(theta)
# 最终求解的theta,非常接近于 [4,3]，如果增大样本，结果将会更准确
# [[4.02659459]
 # [2.96139651]]
