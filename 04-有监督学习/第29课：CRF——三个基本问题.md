# 第29课：CRF——三个基本问题

**概率计算问题：**

* 已知信息：给定 CRF：
  * P(Y|X)
  * 观测序列 x
  * 状态序列 y
* 求解目标：求条件概率P(Yi=yi|x), P(Y(i−1)=y(i−1),Yi=yi|x) 以及相应的数学期望。

**预测问题：**

* 已知信息：给定 CRF:
  * P(Y|X)
  * 观测序列 
* 求解目标：求条件概率最大的状态序列 y∗，也就是对观测序列进行标注

**学习问题：**

* 已知信息：训练数据集。
* 求解目标：求 CRF 模型的参数

**CRF和HMM的区别：**

*  HMM 而言，概率计算只需要观测序列即可，无须确定的状态序列，而最终计算出的结果，则是当前观测序列出现的可能性
* CRF 则需要既有已知观测序列，又有已知状态序列，这才能够去计算概率

