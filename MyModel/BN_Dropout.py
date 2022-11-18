import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter


class BasicBlock(nn.Module):
    expansion = 1 # 表示一个layer里面的2个conv卷积核的大小有没有变化 适用于resnet18 resnet54
    '''
        初始化函数: 用来定义我使用到的网络结构
        输入：
        in_channel: 输入特征矩阵的通道数
        out_channel: 输出特征矩阵的通道数
        stride: 卷积核移动的步距 默认是1
        downsample: 是否有下采样, 对应的是short cut那条线上的是否进行了一次1x1的卷积运算
                    每个layer的第一层的short cut都是虚线的连接
                    默认是实线结构
        tips: stride为1对应实线的残差结构
              stride为2对应虚线的残差结构
    '''
    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        # 第一个卷积层(步距可能为1或2)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False) # 不使用偏置这个参数 因为使用了bn
        # batch normalization
        self.bn1 = nn.BatchNorm2d(out_channel)
        # dropout层
        self.dropout1 = nn.Dropout(p = 0.2)
        # 激活函数
        self.relu = nn.ReLU()
        # 第二个卷积层 步距为1
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        # batch normalization
        self.bn2 = nn.BatchNorm2d(out_channel)
        # dropout层
        self.dropout2 = nn.Dropout(p = 0.2)
        # 下采样：实线结构或者是虚线结构
        self.downsample = downsample

    '''
        前向传播的过程：
        输入: 特征矩阵x
    '''
    def forward(self, x):
        identity = x # 将x赋值给identity 即shortcut分支上的输出值
        if self.downsample is not None: # 如果是虚线结构就要进行下采样的操作
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.dropout1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)

        out += identity
        out = self.relu(out)

        return out

# 定义整个网络框架部分
class BN_Dropout(nn.Module):
    '''
        初始化函数
        输入:
        block: 定义的残差结构
        block_num: 残差结构的数目 是一个list列表 
                   对于resnet18而言就是[2,2,2,2]
        num_classes: 训练集的分类个数

    '''
    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=200,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(BN_Dropout, self).__init__()
        self.include_top = include_top
        self.in_channel = 64 # 对应的是3x3maxpool之后输入特征矩阵的深度

        self.groups = groups
        self.width_per_group = width_per_group
        # 第一层卷积层
        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.dropout1 = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    '''
        make_layer函数 即用来生成convk_x(k = 2,3,4,5)的函数
        输入:
        block: 残差结构
        channel: 残差结构中第一个卷积层所使用的卷积核的个数 即该conv的output_channel
                 layer1为64, layer2为128, layer32为256, layer4为512
        block_num: 一个layer中有多少个残差结构
        stride: 步距 默认为1
    '''
    def _make_layer(self, block, channel, block_num, stride=1):
        # 首先给下采样函数赋初值，如果不符合下面的条件，则默认为None
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion: # 对于resnet18: conv3_x,conv4_x,conv5_x都要进入这里的下采样函数
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion),
                nn.Dropout(p=0.2))

        layers = [] # 定义一个空的列表
        # 先插入第一个残差结构
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion
        # 再通过循环插入后面的残差结构
        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers) # 将list列表转化为非关键字参数传入进去

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x

def BN_Dropout_(num_classes=200, include_top=True):
    return BN_Dropout(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

writer = SummaryWriter('runs/resnet18')
net = BN_Dropout_()
fake_img = torch.randn(1,3,64,64)
writer.add_graph(net,fake_img)
writer.flush()
