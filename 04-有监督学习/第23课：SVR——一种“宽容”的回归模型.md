# 第23课：SVR——一种“宽容”的回归模型

**SVR**:支持向量回归 SUpport Vector Regression

**SVR和线性回归的区别**

* 目标函数不同
* 最优化算法不同

**SVR原理**

* svr在线性函数两侧制造了一个“间隔带”,对于所有落入到间隔带内的样本，都不计算损失
* 间隔带之外的，才计入损失函数，之后再通过最小化间隔带的宽度与总损失来最优化模型

**SVR 引入两个松弛变量：ξξ 和 ξ∗**

![image-20181228155653551](/Users/zhangjianfeng/Library/Application Support/typora-user-images/image-20181228155653551.png)

* 目标函数 f(x) = wx+b
* 落在隔离带边缘超平面上的样本，才是 SVR 的支持向量！