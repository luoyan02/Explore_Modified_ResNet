# Report

1. LeNet中哪些结构或思想在ResNet中仍然存在？哪些已经不用？

    **仍然存在**

    ① **使用局部感受野**

    使用局部感受野，相比于全连接，神经网络可以使用更少的参数来实现对边缘、角点等视觉特征的提取，这些特征在下一层中进行结合形成更高层的特征，从而对形变或位移导致的图像显著特征位置的变化这一问题进行改善；

    ② **权值共享**

    每一层中所有神经元形成一个平面，平面中所有神经元共享权值，神经元的所有输出构成feature map，feature map中所有单元在图像的不同位置执行相同的操作，这样他们可以在输入图像的不同位置检测到相同的特征，并且能极大的减少需要训练的参数个数；

    ③ **下采样层**

    下采样层通过求局部平均值来降低特征图的分辨率，降低输出对平移和形变的敏感度。

    => 以上几个改进可以大大提高网络对几何变换的不变形。

    **舍弃不用**

    采用卷积层、池化层、激活层、全连接层搭建一个复杂神经网络的思想是不变的，而改变的只是这些层中进行运算的方式和层间连接的方式。

    ① 激活层使用ReLu代替sigmoid；

    ② 改变了LeNet中卷积->池化->激活的层间连接顺序而采用一种残差结构，通过残差结构的堆叠形成更深的卷积神经网络；

    ③ 产生目标向量的方式由LeNet中的RBF更改为softmax。

2. AlexNet对于LeNet做了哪些改进？

    **改进之处**

    ① **ReLu激活函数的引入**

    在AlexNet中，使用的激活函数是ReLu，而在LeNet中使用的激活函数是sigmoid

    ReLu VS Sigmoid:

    * 相比于sigmoid或是tanh，ReLu的计算更为简洁，提高了运算速度；
    * 对于深层的网络而言，sigmoid和tanh函数反向传播的过程中，饱和区域非常平缓，接近于0，容易出现梯度消失的问题，减缓收敛速度。ReLu的梯度大多数情况下是常数，有助于解决深层网络的收敛问题；
    * Relu会使一部分神经元的输出为0，形成了网络的稀疏性，并且减少了参数的相互依存关系，缓解了过拟合问题。

    ② **卷积、池化、激活的顺序**

    在LeNet中顺序为：卷积->平均池化->sigmoid激活

    <img src="./ReportPic/LeNet_seq.png" width = "380" height = "200" alt="LeNet示例图片" align=center />

    在AlexNet中顺序为：卷积->ReLu激活->最大池化

    <img src="./ReportPic/AlexNet_seq.png" width = "150" height = "200" alt="LeNet示例图片" align=center />

    顺序的变化是否会对最终的结果造成较大影响，这一点不是很清楚

    ③ **池化层的设计**

    |diff|LeNet|AlexNet|comments|
    |---|---|---|---|
    |采用的池化类型|平均池化|最大池化|平均池化更倾向于保留背景信息<br />而最大池化则更倾向于保留特征纹理|
    |是否采用层叠池化|不采用|采用|AlexNet采用了层叠池化操作，即PoolingSize > stride，类似于卷积操作，可以使相邻像素间产生信息交互和保留必要的联系。论文中也证明，此操作可以有效防止过拟合的发生。


    ④ **网络层数的增加**

    与原始的LeNet相比，AlexNet网络结构更深，LeNet为5层，AlexNet为8层。我认为这一点也是AlexNet在某些网络结构上进行了一些改善的原因，由于希望能通过更深的网络结构实现更好的功能，所以AlexNet在一些网络结构上的设计也跟随着这一点做出了改善。

    ⑤ **droupout操作**

    AlexNet中相比于LeNet添加了droupout操作，dropout会以一定的概率失活一些神经元，可以起到防止过拟合的效果，减少了复杂的神经元之间的相互影响。

    <img src="./ReportPic/dropout.png" width = "450" height = "200" alt="LeNet示例图片" align=center />

    ⑥ **输出处进行的转换**

    在LeNet中输出采用的是Gaussian Connection 即径向欧氏距离函数，计算输入向量和参数向量之间的欧式距离，目前已经被softmax取代。

    ⑦ **使用GPU**
    
    AlexNet中将数据分成两块分别在两个GPU中进行训练，在网络的最后进行合并，提高了数据处理的速度。

    <img src="./ReportPic/ResNet_2gpu.jpg" width = "450" height = "200" alt="LeNet示例图片" align=center />

    ⑧ **全连接层的数量**

    AlexNet中在最终输出前放了三层连续的全连接层，而在LeNet中只在最后使用了一层全连接层，全连接层中需要训练大量的参数，这也导致AlexNet中需要训练的参数量比LeNet中大得多。

3. 这些改进中，有哪些在ResNet中仍然存在？哪些又舍弃了?

    **仍然存在：**

    ① 激活层采用的函数仍为：ReLu；

    ② 池化层的设计：
    仍然采用了层叠池化，但在ResNet中既采用了最大池化也采用了平均池化。

    **舍弃的改进：**

    ① 不再采用单一的卷积->激活->池化的连接顺序，而是采用了残差结构，通过增加short cut的路径将输入与输出直接连接(也可能是虚线结构，采用1x1的卷积使输入与输出的大小匹配)，这能有效改善因为训练层数加深而导致的梯度性消失与梯度爆炸现象。

    <img src="./ReportPic/Residual.png" width = "350" height = "350" alt="残差结构" align=center />

    ② ResNet中减少了全连接层的数量

    AlexNet中全连接层共有3层，如下(其中名为linear的网络结构)：

    <img src="./ReportPic/AlexNet_fc.png" width = "280" height = "400" alt="fc层" align=center />

    ResNet中全连接层仅有1层，如下:

    <img src="./ReportPic/ResNet_fc.png" width = "300" height = "500" alt="fc层" align=center />

    ③ ResNet中没有使用droupout层

    将通过第4题的实验来具体说明这一点
