我在学习一种新的**模型/网络**时，喜欢按照**下面步骤**进行探索。

1. 在知乎/CSDN上找相关的（解析）博客，进行初步印象的建立
2. 细读原文paper
3. 开始跑代码，看代码，深入理解

> 卷积层后面加BN层，卷积层中的bias没有作用。
>
> Focal loss CSPNet（CSPDarknet-53） spatial pyramid pooling(spp)
>
> Path Aggregation Netword(PAN)

# 机器学习

## Bagging和boosting

​        本章介绍的两种集成方法是bagging和boosting。在bagging中，是通过随机抽样的替换方式， 得到了与原始数据集规模一样的数据集。而boosting在bagging的思路上更进了一步，它在数据集上顺序应用了多个不同的分类器。另一个成功的集成方法就是随机森林，但是由于随机森林不如AdaBoost流行，所以本书并没有对它进行介绍。

​         本章介绍了boosting方法中流行的一个称为AdaBoost的算法。AdaBoost以弱学习器作为基分类器，并且输入数据，使其通过权重向量进行加权。在第一次迭代当中，所有数据都等权重。但是在后续的迭代当中，前次迭代中分错的数据的权重会增大。这种针对错误的调节能力正是 AdaBoost的长处。

​         本章以单层决策树作为弱学习器构建了AdaBoost分类器。实际上，AdaBoost函数可以应用于 任意分类器，只要该分类器能够处理加权数据即可。AdaBoost算法十分强大，它能够快速处理其 他分类器很难处理的数据集。

​         非均衡分类问题是指在分类器训练时正例数目和反例数目不相等（相差很大）。该问题在错 分正例和反例的代价不同时也存在。本章不仅考察了一种不同分类器的评价方法——ROC曲线， 还介绍了正确率和召回率这两种在类别重要性不同时，度量分类器性能的指标。

​         本章介绍了通过抽样和欠抽样方法来调节数据集中的正例和反例数目。另外一种可能更好 的非均衡问题的处理方法，就是在训练分类器时将错误的代价考虑在内。

## SoftMax和Sigmoid区别

<img src="img\001-softmax和sigmoid.png" alt="image-20250211134936070" style="zoom:80%;" />

<img src="img\001-softmax_and_sigmoid_2.png" alt="image-20250211135144926" style="zoom: 80%;" />

## 防止过拟合

<img src="img\002_overfitting.png" alt="image-20250211135352365" style="zoom:80%;" />