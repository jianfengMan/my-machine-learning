# 第37课：PCA——用 SVD 实现 PCA

**PCA目标函数:**

* argminW−tr(WT*X*XT*W)
* s.t.WTW=I

**SVD(奇异值分解 Singular Value Decomposition):**

* Mm×n=Um×mΣm×nVTn×n
* Σ 是一个 m×n 的非负实数对角矩阵，Σ 对角线上的元素是矩阵 M 的奇异值
* 何为酉矩阵:若一个 n×nn×n 的实数方阵 UU 满足 UTU=UUT=InUTU=UUT=In，则 UU 称为酉矩阵。

**SVD求解过程**：

* 计算 MMT 和 MTM；
* 分别计算 MMT 和 MTM 的特征向量及其特征值；
* 用 MMT 的特征向量组成 U，MTM 的特征向量组成 V；
* 对MMT 和 MTM 的非零特征值求平方根，对应上述特征向量的位置，填入 Σ 的对角元。

**PCA优缺点**

优点:

* 通过 PCA 进行降维处理，我们就可以同时获得 SVM 和决策树的优点:(得到了和决策树一样简单的分类器，同时分类间隔和SVM一样好)
* 降低数据的复杂性，识别最重要的多个特征。

缺点: 

* 不一定需要，且可能损失有用信息

 适用数据类型:数值型数据。

**SVD**

* 在推荐系统中用到了SVD



