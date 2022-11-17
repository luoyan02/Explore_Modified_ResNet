import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.tensorboard import SummaryWriter

# 首先定义一个类，这个类要继承nn.Module这个父类
class LeNet(nn.Module):

    '''
        在这个类中实现两个方法:
        一个是初始化函数：定义一些在搭建网络的时候会使用到的层的结构
        一个是前向传播函数：定义前向传播的过程
    '''
    # 定义需要用到的层的结构
    def __init__(self):
        super(LeNet, self).__init__()  # super函数用于解决在多层继承中调用父类方法可能出现的问题
        self.conv1 = nn.Conv2d(1, 6, 5)  # ctrl+鼠标左键可进入Conv2d函数中
        '''
            卷积层
            在def __init__()函数中可以看到需要输入的参数的定义：
            这里是def __init__(in_channels, out_channels, kernel_size) 
            输入通道数为3, 共有16组卷积核, 卷积核的尺寸为5x5
        '''
        self.pool1 = nn.AvgPool2d(2, 2)
        '''
            最大下采样层
            跳转到MaxPool2d函数中发现没有初始化函数, 是因为直接继承了父类
            class MaxPool2d(_MaxPoolNd) 括号内即为它的父类
            这里是def __init__(kernel_size, stride, padding = 0)
            池化核大小为2x2, 步距也为2的操作
        '''
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        '''
            全连接层
            这里是def __init__(in_features, out_features)
        '''
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        '''
            最后一个全连接层的输出是要根据训练集的分类任务个数进行修改的
            此处为10个分类任务
        '''
    
    # 定义正向传播过程, x代表输入的数据[batch, channel, height, width]
    def forward(self, x):
        # 卷积层并将输出送入ReLu激活函数  调用方式: F.relu
        x = self.conv1(x)            # input(1, 32, 32) output(6, 28, 28)
        x = torch.sigmoid(self.pool1(x)) # output(6, 14, 14)
        x = self.conv2(x)            # output(16, 10, 10)
        x = torch.sigmoid(self.pool2(x)) # output(16, 5, 5)
        x = x.view(-1, 16*5*5)       # output(16*5*5)
        '''
            将特征矩阵展平，展成特征向量的形式
            第一个参数为-1 表示自动填充batch的值
        '''
        x = torch.sigmoid(self.fc1(x))      # output(120)
        x = torch.sigmoid(self.fc2(x))      # output(84)
        x = self.fc3(x)              # output(10)
  
        return x

writer = SummaryWriter('runs/LeNet')
net = LeNet()
fake_img = torch.randn(1, 1, 32, 32)
writer.add_graph(net,fake_img)
writer.flush()