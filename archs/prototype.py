import torch
import torch.nn as nn

class PrototypeNet(nn.Module):
    def __init__(self, num_classes=6, feature_dim=512):
        super(PrototypeNet, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes

    def calculate_prototypes(self, z_support, y_support):
        prototypes = []
        for class_idx in range(self.num_classes):
            class_mask = (y_support == class_idx).squeeze(0)# Create a mask to filter out the features of the current class
            class_features = z_support[class_mask]# Select features belonging to the current class
            class_prototype = class_features.mean(dim=0)
            prototypes.append(class_prototype)
        prototypes = torch.stack(prototypes, dim=0)
        return prototypes

    def forward(self, x_support, y_support, x_query):
        prototypes = self.calculate_prototypes(x_support, y_support) # Calculate prototypes using the support set
        dists = self.euclidean_dist(x_query, prototypes)# Compute distances between query features and prototypes #x_query.shape()
        return dists

    @staticmethod
    def euclidean_dist_with_space(x, y):

        x = x.unsqueeze(2)  # [1, n, 1, channels, h, w]
        y = y.unsqueeze(1)  # [m, 1, channels, h, w]

        return torch.pow(x - y, 2).sum(dim=[3, 4, 5])
    
    @staticmethod
    def euclidean_dist(x, y):
        # Flatten x and y to [N, D], where D = channels * height * width
        x = x.view(x.size(0), -1)  # [5, 128 * 5 * 5]
        y = y.view(y.size(0), -1)  # [5, 128 * 5 * 5]

        # Add extra dimensions for broadcasting
        x = x.unsqueeze(1)  # [5, 1, D]
        y = y.unsqueeze(0)  # [1, 5, D]

        # Compute squared Euclidean distance
        return torch.pow(x - y, 2).sum(dim=2)  # [5, 5]