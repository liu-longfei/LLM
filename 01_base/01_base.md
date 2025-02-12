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

> Cv指南：https://www.pyimagesearch.com/start-here/
>
> 注意力机制：CBAM

# 分割

<img src="img\003_分割.png" alt="image-20250212080337640" style="zoom:80%;" />

## MobileNetv1

<img src="img\004_depthwise_separable_convolution.png" alt="image-20250212080505187" style="zoom:80%;" />

### 与普通卷积的计算量和参数量对比

<img src="img\004_depthwise_separable_convolution_1.png" alt="image-20250212080719680" style="zoom:80%;" />

<img src="img\004_depthwise_separable_convolution_2.png" alt="image-20250212080836475" style="zoom:80%;" />

<img src="img\004_depthwise_separable_convolution_3.png" alt="image-20250212080929574" style="zoom:80%;" />

<img src="img\004_depthwise_separable_convolution_4.png" alt="image-20250212081013446" style="zoom:80%;" />

## MobileNetv2

![image-20250212081131661](img\005_mobilenet_v2.png)

![image-20250212081223925](img\005_mobilenet_v2_1.png)

**下图左mobilenetv1、右图mobilenetv2**

![image-20250212081346234](img\005_mobilenet_v2_2.png)

## shuffleNetv1

<img src="img\006_shufflenet_v1.png" alt="image-20250212081446508" style="zoom:80%;" />

<img src="img\006_shufflenet_v1_1.png" alt="image-20250212081536501" style="zoom:80%;" />

![image-20250212081730103](img\006_shufflenet_v1_2.png)

## ShuffleNetv2

首先提出了设计轻量化网络的4的通用原则：

**内存访问成本**(memory access cost)和**平台特点**(platform characterics)。

![image-20250212081901609](img\006_shufflenet_v2_1.png)

**证明准则一：**

![image-20250212082005263](img\006_shufflenet_v2_2.png)

**证明准则二：**

![image-20250212082055141](img\006_shufflenet_v2_3.png)

![image-20250212082139332](img\006_shufflenet_v2_4.png)

**ShuffleNetv1和ShuffleNetv2对比：**

![image-20250212082223022](img\006_shufflenet_v2_5.png)

# 转置卷积

![image-20250212082447823](img\007_transpose_convolution.png)

# FCN

**创新点：**

- 不含全连接层(fc)的全卷积(fully conv)网络。可适应任意尺寸输入。 
- 增大数据尺寸的反卷积(deconv)层。能够输出精细的结果。
-  结合不同深度层结果的跳级(skip)结构。同时确保鲁棒性和精确性。

**网络结构图：**

![image-20250212082638356](img\008_FCN.png)

**FCN 的优势在于：** 

- 可以接受任意大小的输入图像（没有全连接层）
- 更加高效，避免了使用邻域带来的重复计算和空间浪费的问题。

**其不足也很突出：** 

- 得到的结果还不够精细 。进行8倍上采样虽然比32倍的效果好了很多，但是上采样的结果还是比较模糊和平滑，对图像中的细节不敏感。
- 是对各个像素进行分类，没有充分考虑像素与像素之间的关系。忽略了在通常的基于像素分类的分割方法中使用的空间规整（spatial regularization）步骤，缺乏空间一致性。

# U-Net

**使用的方法？**

继承FCN的思想，继续进行改进。但是相对于FCN，有几个改变的地方，U-Net是完全对称的，且对解码器（应该自Hinton提出编码器、解码器的概念来，即将图像->高语义feature map的过程看成编码器，高语义->像素级别的分类score map的过程看作解码器）进行了加卷积加深处理，FCN只是单纯的进行了上采样。

Skip connection：两者都用了这样的结构，虽然在现在看来这样的做法比较常见，但是对于当时，这样的结构所带来的明显好处是有目共睹的，因为可以联合高层语义和低层的细粒度表层信息，就很好的符合了分割对这两方面信息的需求。

联合：在FCN中，Skip connection的联合是通过对应像素的求和，而U-Net则是对其的channel的concat过程。

**U-Net结构图**

![image-20250212082846100](img\008_U-Net.png)

**U-Net结构的解析**

整体结构就是先编码（下采样），再解码（上采样），回归到跟原始图像一样大小的像素点的分类。

首先是输入图像的大小，这个是根据再高层的大小来进行反推的，最后取的一个比较合适的方便计算的输入大小。

下采样是通过max pool 2x2来进行1/2下采样的，下采样之间是两个conv卷积层，这里的卷积是使用valid卷积。所以在卷积过程中图像的大小是会减小的。这会造成一个问题，就是造成了在skip connection部分concat时候大小不一致，因为在上面有一个copy & crop操作，crop就是为了将大小进行裁剪的操作。

虽然上面有说到crop操作，但若是在卷积的时候使用的是same，就无需这个操作，至于这其中的影响，我觉得应该是不会造成太大的影响的，而且还会方便计算操作。这是same的代码。

上采样，相对于FCN的转置卷积进行上采样，这里是一个up-conv 2x2，具体对应的操作是：对行列进行2倍翻倍。

U-Net和FCN非常的相似，U-Net比FCN稍晚提出来，但都发表在2015年，和FCN相比，U-Net的第一个特点是完全对称，也就是左边和右边是很类似的，而FCN的decoder相对简单，只用了一个deconvolution的操作，之后并没有跟上卷积结构。第二个区别就是skip connection，FCN用的是加操作（summation），U-Net用的是叠操作（concatenation）。这些都是细节，重点是它们的结构用了一个比较经典的思路，也就是编码和解码（encoder-decoder），早在2006年就被Hinton大神提出来发表在了nature上.

当时这个结构提出的主要作用并不是分割，而是压缩图像和去噪声。输入是一幅图，经过下采样的编码，得到一串比原先图像更小的特征，相当于压缩，然后再经过一个解码，理想状况就是能还原到原来的图像。这样的话我们存一幅图的时候就只需要存一个特征和一个解码器即可。这个想法我个人认为是很漂亮了。同理，这个思路也可以用在原图像去噪，做法就是在训练的阶段在原图人为的加上噪声，然后放到这个编码解码器中，目标是可以还原得到原图。

# U-Net++