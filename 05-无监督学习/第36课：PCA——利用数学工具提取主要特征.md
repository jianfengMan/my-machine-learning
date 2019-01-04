# 第36课：PCA——利用数学工具提取主要特征

**休斯现象（Hughes Phenomenon）**：

* 在训练样本固定的情况下，特征维数增加到某一个临界点后，继续增加反而会导致模型的预测能力减小——这叫做休斯现象

**常用降维算法：**

* PCA (Principal Component Analysis 、主成分分析)
* LDA（Linear Discriminant Analysis 、线性判别分析）
* LLE（Locally linear embedding ,局部线性嵌入）
* Laplacian Eigenmaps 拉普拉斯特征映射

**超平面具备的两个性质：**

* 最大可分性：样本点到这个超平面上的投影尽量能够分开
  * 要让所有样本点投影后尽量分开，那就应该**让新空间中投影点的方差尽量大**
  * 目标函数：argmaxWtr(WT * X * XT * W)
* 最近重构性：样本点到这个超平面的距离尽量近
  * 我们要的是所有 nn 个样本分别与其基于投影重构的样本点间的**距离整体最小**
  * 目标函数min∑ni=1||^x(i)−x(i)||2

