import torch.nn as nn
import torch
from torchvision import models, transforms
from archs.prototype import PrototypeNet

class FeatureExtractor(nn.Module):
    def __init__(self, num_classes, feature_dim, input_channel):
        super(FeatureExtractor, self).__init__()
        backbone = models.resnet18(pretrained=False)  # 设置为 False，使用随机初始化
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])  # 去掉最后的全连接层
        self.prototype = PrototypeNet(num_classes, feature_dim=feature_dim)
        self.simple_CNN = nn.Conv2d(in_channels=input_channel, out_channels=3, kernel_size=1)

    def forward(self, support, y_support, query):
        with torch.amp.autocast("cuda"):
            distances = []
            if query.dim() == 5:
                for i in range(support.size(1)):
                    support_scale = self.simple_CNN(support[:, i, :])
                    support_scale = self.feature_extractor(support_scale)
                    support_scale = support_scale.view(support_scale.size(0), -1)

                    query_scale = self.simple_CNN(query[:, i, :])
                    query_scale = self.feature_extractor(query_scale)
                    query_scale = query_scale.view(query_scale.size(0), -1)
                    distance =  self.prototype(support_scale, y_support, query_scale)
                    distances.append(distance)
                # 拼接为 [scale, batch_size, c, w, h]
                distances = torch.stack(distances, dim=0)
                # 如果你想要 [batch_size, scale, c, w, h]
                distances = distances.permute(1, 0, 2)
                return distances  # 展平成 (batch_size, feature_dim)
            elif query.dim() == 4:

                support_scale = self.simple_CNN(support)
                support_scale = self.feature_extractor(support_scale)
                support_scale = support_scale.view(support_scale.size(0), -1)

                query_scale = self.simple_CNN(query)
                query_scale = self.feature_extractor(query_scale)
                query_scale = query_scale.view(query_scale.size(0), -1)
                distances =  self.prototype(support_scale, y_support, query_scale)
                return distances  # 展平成 (batch_size, feature_dim)
            else:
                raise ValueError("False dim")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_only_few_shot_whole_model(num_classes, input_channel, feature_dim = 512):
    backbone = FeatureExtractor(num_classes, feature_dim, input_channel).to("cuda:0")
    return backbone