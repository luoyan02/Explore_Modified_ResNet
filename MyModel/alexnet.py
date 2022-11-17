import torch.nn as nn
import torch
from torch.utils.tensorboard import SummaryWriter

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000, init_weights=False):
        super(AlexNet, self).__init__()
        '''
            nn.Sequential函数可以将一系列的层结构进行打包
        '''
        self.features = nn.Sequential(
            # 原网络中这里kernel_num是96
            nn.Conv2d(3, 48, kernel_size=11, stride=4, padding=2),  # input[3, 224, 224]  output[48, 55, 55]
            nn.ReLU(inplace=True),  # 能通过inplace这个方法在内存中载入更大的一个模型
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[48, 27, 27]
            nn.Conv2d(48, 128, kernel_size=5, padding=2),           # output[128, 27, 27]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 13, 13]
            nn.Conv2d(128, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),          # output[192, 13, 13]
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 128, kernel_size=3, padding=1),          # output[128, 13, 13]
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # output[128, 6, 6]
        )
        '''
            分类器
        '''
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), # dropout一般是放在全连接层与全连接层之间 0.5表示以50%的概率失活一些神经元
            nn.Linear(128 * 6 * 6, 2048), #全连接层
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(2048, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_classes),
        )
        '''
            初始化权重
        '''
        if init_weights: 
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1) # pytorch中[batch,channel,height,width] 其中第0维batch是不动的，从1维channel开始进行展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules(): # 通过调用self.modules会遍历我们定义的每一个模块
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

writer = SummaryWriter('runs/alexnet')
net = AlexNet()
fake_img = torch.randn(1,3,227,227)
writer.add_graph(net,fake_img)
writer.flush()
