from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn as nn
import torch
import numpy as np
from copy import deepcopy

class ProtoCamWrapper(nn.Module):
    def __init__(self, model, pred_support, support_labels):
        super().__init__()
        self.model = deepcopy(model).train()
        self.pred_support = pred_support
        self.support_labels = support_labels
    def forward(self, x):
        dists = self.model(self.pred_support, self.support_labels, x)  # [B, C], 假设是距离
        scores = -dists  # 或 softmax(-dists)
        return scores
    
class simple_CNN_grad(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.conv1 = nn.Conv2d(scale, 3, 3, padding=1)
    
    def forward(self, x):
        out = self.conv1(x)
        return out
    
def cal_grad(model, model_for_scale_fusion, pred_support, support_labels, images, query_labels, conf_list, conf_thr):
    with torch.enable_grad():
        cam_maps = []
        target = []
        for label in query_labels.squeeze(0):
            target.append(ClassifierOutputTarget(label))
        for i in range(images.size(1)):
            model_grad = ProtoCamWrapper(model, pred_support[:, i, :], support_labels)
            target_layer = model_grad.model.feature_extractor[-2]
            cam = GradCAM(model = model_grad, target_layers=[target_layer])
            input = images[:, i, :]
            cam_map = cam(input_tensor=input, targets=target)
            if isinstance(cam_map, np.ndarray):
                cam_map = torch.from_numpy(cam_map)
            cam_maps.append(cam_map)
        multi_scale_cam = torch.stack(cam_maps, dim=1).to(pred_support.device)
        conf_tensor = conf_list.detach().permute(1, 0)
        mask = (conf_tensor < conf_thr).unsqueeze(-1).unsqueeze(-1)  # [20, 3, 1, 1]
        multi_scale_cam = torch.where(mask, torch.zeros_like(multi_scale_cam), multi_scale_cam)
        return multi_scale_cam

