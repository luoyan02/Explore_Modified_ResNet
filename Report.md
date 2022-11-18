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

   <img src="./ReportPic/AlexNet_seq.png" width = "150" height = "200" alt="AlexNet示例图片" align=center />

   顺序的变化是否会对最终的结果造成较大影响，这一点不是很清楚

   ③ **池化层的设计**

   | diff             | LeNet    | AlexNet  | comments                                                                                                                                                        |
   | ---------------- | -------- | -------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
   | 采用的池化类型   | 平均池化 | 最大池化 | 平均池化更倾向于保留背景信息<br />而最大池化则更倾向于保留特征纹理                                                                                              |
   | 是否采用层叠池化 | 不采用   | 采用     | AlexNet采用了层叠池化操作，即PoolingSize > stride，类似于卷积操作，可以使相邻像素间产生信息交互和保留必要的联系。论文中也证明，此操作可以有效防止过拟合的发生。 |

   ④ **网络层数的增加**

   与原始的LeNet相比，AlexNet网络结构更深，LeNet为5层，AlexNet为8层。我认为这一点也是AlexNet在某些网络结构上进行了一些改善的原因，由于希望能通过更深的网络结构实现更好的功能，所以AlexNet在一些网络结构上的设计也跟随着这一点做出了改善。

   ⑤ **droupout操作**

   AlexNet中相比于LeNet添加了droupout操作，dropout会以一定的概率失活一些神经元，可以起到防止过拟合的效果，减少了复杂的神经元之间的相互影响。


   <img src="./ReportPic/dropout.png" width = "450" height = "200" alt="dropout" align=center />

   ⑥ **输出处进行的转换**

   在LeNet中输出采用的是Gaussian Connection 即径向欧氏距离函数，计算输入向量和参数向量之间的欧式距离，目前已经被softmax取代。

   ⑦ **使用GPU**

   AlexNet中将数据分成两块分别在两个GPU中进行训练，在网络的最后进行合并，提高了数据处理的速度。

   <img src="./ReportPic/ResNet_2gpu.jpg" width = "450" height = "200" alt="Resnet_2gpu" align=center />

   ⑧ **全连接层的数量**

   AlexNet中在最终输出前放了三层连续的全连接层，而在LeNet中只在最后使用了一层全连接层，全连接层中需要训练大量的参数，这也导致AlexNet中需要训练的参数量比LeNet中大得多。
3. 这些改进中，有哪些在ResNet中仍然存在？哪些又舍弃了?

   **仍然存在：**

   ① 激活层采用的函数仍为：ReLu；

   ② 池化层的设计：
   仍然采用了层叠池化，但在ResNet中既采用了最大池化也采用了平均池化。

   **舍弃的改进：**

   ① 不再采用单一的卷积->激活->池化的连接顺序，而是采用了残差结构，通过增加short cut的路径将输入与输出直接连接(也可能是虚线结构，采用1x1的卷积使输入与输出的大小匹配)，这能有效改善因为训练层数加深而导致的梯度性消失与梯度爆炸现象。

   <img src="./ReportPic/residual.png" width = "350" height = "350" alt="残差结构" align=center />

   ② ResNet中减少了全连接层的数量

   AlexNet中全连接层共有3层，如下(其中名为linear的网络结构)：

   <img src="./ReportPic/AlexNet_fc.png" width = "280" height = "400" alt="fc层" align=center />

   ResNet中全连接层仅有1层，如下:

   <img src="./ReportPic/ResNet_fc.png" width = "300" height = "500" alt="fc层" align=center />

   ③ ResNet中没有使用droupout层

   将通过第4题的实验来具体说明这一点
4. 如果再把舍弃的改进加回ResNet，会有什么样的实验表现？请挑选一处，在Tiny-ImageNet数据集（见QQ群共享）上做实验，给出量化分析（可以从性能和运算效率角度出发；设置对照实验；如resnet现有结构不足以支撑你的实验，可改动其网络结构）。

① **实验设计**

在AlexNet的模型中作者引入了Dropout层以达到防止模型过拟合的效果，而在ResNet模型中作者又引入了BN层，BN层通过一定的规范化手段，把每层神经网络输入值的分布强行拉回到均值为0方差为1的标准正态分布，使得分布回到非线性函数对输入比较敏感的区域，让损失函数能发生较大的变化，避免梯度消失问题。但是这两个层一起使用的时候模型的性能反而会下降，下面基于这一点做一个验证性的实验。

| 实验组别   | 在原模型上的修改                                             |
| ---------- | ------------------------------------------------------------ |
| resnet18   | 原模型，作为一个对照组                                       |
| Dropout    | 将原模型中所有BN层均替换为Dropout层(p=0.2)，作为另一个对照组 |
| BN_Dropout | 原模型的残差结构中在所有BN层的后面添加Dropout层(p=0.2)       |
| Dropout_BN | 原模型的残差结构中在所有BN层的前面添加Dropout层(p=0.2)       |

同时控制其他无关因素保持相同：

* 学习率：所有组别均设置lr = 0.1, step_size = 5, gamma = 0.5
* epoch：所有组别均跑20个epoch
* gpu配置：所有组别的gpu均采用推荐配置

<img src="./net_pic/gpu.png" width = "600" height = "100" alt="resnet18网络结构" align=center />

② **实验结果**

* **组别1：resnet18**

  **网络结构：**

  <img src="./net_pic/resnet18/resnet1.png" width = "250" height = "500" alt="resnet18网络结构" align=center />
    <img src="./net_pic/resnet18/resnet2.png" width = "300" height = "300" alt="basic_block结构" align=center />

  **实验结果：**

  <img src="./net_pic/resnet18/result.png" width = "600" height = "300" alt="result" align=center /><br /><br />

  <img src="./net_pic/resnet18/train_loss.png" width = "300" height = "150" alt="train_loss" align=center />
    <img src="./net_pic/resnet18/train_acc1.png" width = "300" height = "150" alt="train_acc1" align=center /><br /><br />
    <img src="./net_pic/resnet18/train_acc5.png" width = "300" height = "150" alt="train_acc5" align=center />
    <img src="./net_pic/resnet18/test_loss.png" width = "300" height = "150" alt="test_loss" align=center /><br /><br />
    <img src="./net_pic/resnet18/test_acc1.png" width = "300" height = "150" alt="test_acc1" align=center />
    <img src="./net_pic/resnet18/test_acc5.png" width = "300" height = "150" alt="test_acc5" align=center />
* **组别2：Dropout**

  **网络结构：**

  <img src="./net_pic/dropout/dropout1.png" width = "250" height = "500" alt="dropout网络结构" align=center />
    <img src="./net_pic/dropout/dropout2.png" width = "300" height = "300" alt="basic_block结构" align=center />

  **实验结果：**

  <img src="./net_pic/dropout/result.png" width = "600" height = "300" alt="result" align=center /><br /><br />

  <img src="./net_pic/dropout/train_loss.png" width = "300" height = "150" alt="train_loss" align=center />
    <img src="./net_pic/dropout/train_acc1.png" width = "300" height = "150" alt="train_acc1" align=center /><br /><br />
    <img src="./net_pic/dropout/train_acc5.png" width = "300" height = "150" alt="train_acc5" align=center />
    <img src="./net_pic/dropout/test_loss.png" width = "300" height = "150" alt="test_loss" align=center /><br /><br />
    <img src="./net_pic/dropout/test_acc1.png" width = "300" height = "150" alt="test_acc1" align=center />
    <img src="./net_pic/dropout/test_acc5.png" width = "300" height = "150" alt="test_acc5" align=center />
* **组别3：BN_Dropout**

  **网络结构：**

  <img src="./net_pic/BN_Dropout/BN_Dropout1.png" width = "250" height = "500" alt="BN_Dropout网络结构" align=center />
    <img src="./net_pic/BN_Dropout/BN_Dropout2.png" width = "300" height = "300" alt="basic_block结构" align=center />

  **实验结果：**

  <img src="./net_pic/BN_Dropout/result.png" width = "600" height = "300" alt="result" align=center /><br /><br />

  <img src="./net_pic/BN_Dropout/train_loss.png" width = "300" height = "150" alt="train_loss" align=center />
    <img src="./net_pic/BN_Dropout/train_acc1.png" width = "300" height = "150" alt="train_acc1" align=center /><br /><br />
    <img src="./net_pic/BN_Dropout/train_acc5.png" width = "300" height = "150" alt="train_acc5" align=center />
    <img src="./net_pic/BN_Dropout/test_loss.png" width = "300" height = "150" alt="test_loss" align=center /><br /><br />
    <img src="./net_pic/BN_Dropout/test_acc1.png" width = "300" height = "150" alt="test_acc1" align=center />
    <img src="./net_pic/BN_Dropout/test_acc5.png" width = "300" height = "150" alt="test_acc5" align=center />
* **组别4：Dropout_BN**

  **网络结构：**

  <img src="./net_pic/Dropout_BN/Dropout_BN1.png" width = "250" height = "500" alt="Dropout_BN网络结构" align=center />
    <img src="./net_pic/Dropout_BN/Dropout_BN2.png" width = "300" height = "300" alt="basic_block结构" align=center />

  **实验结果：**

  <img src="./net_pic/Dropout_BN/result.png" width = "600" height = "300" alt="result" align=center /><br /><br />

  <img src="./net_pic/Dropout_BN/train_loss.png" width = "300" height = "150" alt="train_loss" align=center />
    <img src="./net_pic/Dropout_BN/train_acc1.png" width = "300" height = "150" alt="train_acc1" align=center /><br /><br />
    <img src="./net_pic/Dropout_BN/train_acc5.png" width = "300" height = "150" alt="train_acc5" align=center />
    <img src="./net_pic/Dropout_BN/test_loss.png" width = "300" height = "150" alt="test_loss" align=center /><br /><br />
    <img src="./net_pic/Dropout_BN/test_acc1.png" width = "300" height = "150" alt="test_acc1" align=center />
    <img src="./net_pic/Dropout_BN/test_acc5.png" width = "300" height = "150" alt="test_acc5" align=center />

③ **数据分析**

通过对以上数据进行分析，能够得到如下结论：

* **Dropout层能显著改善过拟合的问题**

    在组别1中，使用resnet18的网络进行训练，train的精度可以达到非常高，接近100%，但是test的精度很低，acc1仅在30%-50%左右，且观察它的test_loss曲线呈先下降后上升的趋势，过拟合的问题非常明显。而组别3和组别5中，在原resnet模型的基础上加入了dropout层，虽然模型本身训练的精度不高，但是过拟合问题得到了改善，test_loss曲线呈一直下降的趋势。

* **BN层与dropout层连用会使得模型整体性能下降**
  
    组别1中使用resnet18的原模型，模型中只有BN层，没有dropout层，训练精度很高，在组别3、4中加入了dropout层之后训练的精度下降，train_acc1最终在50%-60%左右，train_acc5最终在80%-90%左右，相比于之前接近100%的准确率有显著下降。

* **dropout层与BN层排列顺序对原模型影响程度的对比**
  
    《Understanding the Disharmony between Dropout and Batch Normalization by Variance Shift》一文的作者在文章中提出将dropout层放在BN层的后面会改善BN+Dropout联用时产生的方差偏移现象，但是从我自己训练的数据上来看，这二者好像并没有太大的差别，可能是由于p值取0.2并不是很大或者是训练集较小等因素导致的。这一组对比实验并没有得出较明显的结果。

④ **实验总结及心得体会**

**首先说一下这次实验的收获：**

  因为郑老师的实验课后期都是在线上上的，没有了助教的指导我可以说是啥都不会，所以后面几次的随堂作业我是和几个同学在一起完成的，每次都是东拼西凑，这边抄一点，那边看一眼才勉强凑合完成，导致这一阶段学的有点稀里糊涂的。本次作业的内容是让我们回顾3个经典的神经网络的内容，才倒逼着让我好好抽时间盘了一下这一阶段的知识，下面是我学习这一部分知识的过程，写在这里也算是做一个总结：

step1:了解搭建神经网络的基本组件

  卷积层、激活层、池化层、全连接层等，学习这些组件分别进行了什么运算、有哪些典型的参数以及实现了什么功能，它们之间有什么区别和联系。

step2:3种经典神经网络的结构图

  使用tensorboard做出3种网络的graph，去看组件是按照什么样的顺序进行连接的，张量x的四个维度[batch,channel,width,height]通过该组件之后又是如何变化的。

step3:读懂实现model的代码，读懂main.py各部分在干什么

  通过在网上查找资料，明白常用的几个函数都是在干什么，model中每个层次是如何通过代码给串起来的

step4:运行代码，消耗算力(bushi

  对比几个模型，找到其中舍弃与改进的部分(当然也不全是自己找到的，没少参照网上的资料)，然后修改已有的resnet18的模型并运用新模型进行训练，将训练的结果与原模型出来的结果进行对比。

**最后是吐槽**

  数据集比较小，用resnet18跑出来的结果过拟合很严重，根据助教的建议将模型层数削一点，最后将每个layer中都只保留了一个basicblock，并且删去了一层layer，只保留了前三层，过拟合的现象得到了好转，但是此时模型已经不能收敛了，train_acc大概在百分之五六十左右，效果如下：

<img src="./net_pic/SS3.png" width = "500" height = "250" alt="SS3" align=center /><br /><br />

<img src="./net_pic/SS3_loss.png" width = "300" height = "200" alt="SS3_loss" align=center />

  在折腾的过程中学到了不少，但是也确实花了不少时间，而且心好累，就差给电脑磕头了。。。