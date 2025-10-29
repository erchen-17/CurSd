import torch
from torch import nn
import torch.nn as nn  # 导入PyTorch中的神经网络模块
import math  # 提供数学函数
import torch.utils.model_zoo as model_zoo  # 用于从网络加载预训练模型
import torch  # PyTorch库

class ClassificationNetwork(nn.Module):
    """
    Module for classifying feature maps extracted from the diffusion model.
    The network uses a simple convolutional layer followed by adaptive pooling
    and a fully connected layer for classification.
    """
    def __init__(self, input_channels, num_classes, device):
        super(ClassificationNetwork, self).__init__()
        self.device = device
        self.num_classes = num_classes

        # Convolutional layer to reduce dimensionality
        self.conv = nn.Conv2d(in_channels=input_channels, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # Adaptive pooling to reduce the spatial dimensions to 1x1
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer for classification
        self.fc = nn.Linear(128, num_classes)

        # Move all layers to the specified device
        self.to(device)

    def forward(self, x):
        """
        Forward pass through the classification network.
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
        Returns:
            torch.Tensor: Classification logits of shape (B, num_classes)
        """
        x = self.conv(x)  # Convolutional layer
        x = self.relu(x)  # Activation function
        x = self.pool(x)  # Adaptive pooling to reduce to 1x1
        x = x.view(x.size(0), -1)  # Flatten the feature map
        x = self.fc(x)  # Fully connected layer
        return x


# 指定在此模块中可用的类和函数
#__all__ = ['Res2Net', 'res2net50_v1b', 'res2net101_v1b']

# 模型的URL字典，用于下载预训练模型的权重
model_urls = {
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}

# Bottle2neck模块定义，Res2Net的基本构建块
class Bottle2neck(nn.Module):
    expansion = 4  # 扩展系数

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale=4, stype='normal'):
        super(Bottle2neck, self).__init__()

        # 计算卷积层的宽度
        width = int(math.floor(planes * (baseWidth / 64.0)))
        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(width * scale)  # 批量归一化层
        
        # 设定分支数量
        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)  # 平均池化层

        # 定义卷积层和批量归一化层
        convs = []
        bns = []
        for i in range(self.nums):
            convs.append(nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, bias=False))
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)  # 使用ModuleList来存储卷积层
        self.bns = nn.ModuleList(bns)  # 使用ModuleList来存储批量归一化层

        self.conv3 = nn.Conv2d(width * scale, planes * self.expansion, kernel_size=1, bias=False)  # 第三个卷积层
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)  # 批量归一化层

        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数
        self.downsample = downsample  # 下采样层
        self.stype = stype  # 模块类型
        self.scale = scale  # 缩放因子
        self.width = width  # 宽度

    def forward(self, x):
        residual = x  # 保留输入数据以便后续与输出相加

        out = self.conv1(x)  # 第一个卷积层
        out = self.bn1(out)  # 批量归一化
        out = self.relu(out)  # 激活函数

        spx = torch.split(out, self.width, 1)  # 按宽度拆分
        for i in range(self.nums):
            if i == 0 or self.stype == 'stage':
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)  # 卷积操作
            sp = self.relu(self.bns[i](sp))  # 批量归一化和激活函数
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)  # 级联操作

        if self.scale != 1 and self.stype == 'normal':
            out = torch.cat((out, spx[self.nums]), 1)
        elif self.scale != 1 and self.stype == 'stage':
            out = torch.cat((out, self.pool(spx[self.nums])), 1)

        out = self.conv3(out)  # 第三个卷积层
        out = self.bn3(out)  # 批量归一化

        if self.downsample is not None:
            residual = self.downsample(x)  # 如果存在下采样，应用它

        out += residual  # 残差连接
        out = self.relu(out)  # 激活函数

        return out
    
# Res2Net模型定义
class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth=26, scale=4, num_classes=6):
        self.inplanes = 64  # 初始卷积通道数
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth  # 基础宽度
        self.scale = scale  # 缩放因子
        self.conv1 = nn.Sequential(
            nn.Conv2d(384, 256, 1, 1, 1, bias=False),  # 初始卷积层
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 1, 1, 1, bias=False),  # 初始卷积层
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1, bias=False)
        )            
        self.bn1 = nn.BatchNorm2d(64)  # 批量归一化
        self.relu = nn.ReLU()  # ReLU激活函数
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化层
        self.layer1 = self._make_layer(block, 64, layers[0])  # 创建网络层
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.fc = nn.Linear(512 * block.expansion, num_classes)  # 全连接层

        # 初始化网络参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')  # Kaiming初始化
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)  # 批量归一化参数初始化
                nn.init.constant_(m.bias, 0)
        self.to(torch.device('cuda:0'))

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, stype='stage', baseWidth=self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth=self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 初始卷积
        x = self.bn1(x)  # 批量归一化
        x = self.relu(x)  # 激活函数
        x = self.maxpool(x)  # 最大池化

        x = self.layer1(x)  # 第一层
        x = self.layer2(x)  # 第二层
        x = self.layer3(x)  # 第三层
        x = self.layer4(x)  # 第四层

        x = self.avgpool(x)  # 自适应平均池化
        x = x.view(x.size(0), -1)  # 展平
        x = self.fc(x)  # 全连接层

        return x

# 创建Res2Net50模型函数
def res2net50_v1b(pretrained=False, num_classes=6, **kwargs):
    model = Res2Net(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4, **kwargs)  # 实例化模型
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['res2net50_v1b_26w_4s'])  # 下载预训练模型权重
        # 删除全连接层的权重
        state_dict.pop('fc.weight', None)
        state_dict.pop('fc.bias', None)
        model.load_state_dict(state_dict, strict=False)  # 加载权重
    # 修改全连接层以匹配新的类别数
    model.fc = nn.Linear(512 * Bottle2neck.expansion, num_classes).to(torch.device('cuda:0'))
    return model