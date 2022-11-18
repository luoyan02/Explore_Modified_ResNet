# dlp_hw2
以下对此仓库中存放的文件进行几点说明：

### main.py

  **调用、训练、保存模型的代码**
  
  **更改文件第47行default后的值可对训练的模型进行更改:**
  
  若要训练Dropout.py中的模型->改为"Dropout_"
  
  若要训练BN_Dropout.py中的模型->改为"BN_Dropout_"
  
  若要训练Dropout_BN.py中的模型->改为"Dropout_BN_"
  
  若要训练resnet_SS3.py中的模型->改为"resnet_SS3_"
  
  **运行代码前需更改文件第45行default后的值为当前存放数据集的路径**

### Mymodel

  Mymodel文件夹中存放了训练过的所有模型
  
  * resnet.py
  
    resnet18的原始模型
  
  * Dropout.py, BN_Dropout.py, Dropout_BN.py
  
    修改后的模型，在作业的第四问中使用到
  
  * resnet_SS3.py
  
    尝试解决原模型过拟合的问题中得出的一个比较成功的模型。在此模型中，为了适应较小的数据集，削减了模型的层数，去掉了一层layer，且在剩下的3个layer中每层只保留一个basicblock
   
  * alexnet.py, LeNet.py
   
    其余两个经典模型的原始代码，仅用于学习模型的网络结构

### tiny_imagenet_200_reorg

  使用的数据集
  
### ReportPic, net_pic

  用于存放在Report.md中使用到的图片
  
### Report.md, 钟珞妍_PB20061329.pdf

  两种格式的实验报告
