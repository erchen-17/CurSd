import torch

def analyze_unet_parameters(unet):
    """
    Analyze trainable and non-trainable parameters in U-Net model
    Returns total, trainable, and non-trainable parameter counts
    Also prints detailed analysis of parameters by layer type
    """
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0
    
    # For detailed analysis by layer type
    layer_stats = {
        'crossattn': {'trainable': 0, 'non_trainable': 0},
        'other': {'trainable': 0, 'non_trainable': 0}
    }

    # Analyze each parameter
    for name, param in unet.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        # Check if parameter is trainable
        if param.requires_grad:
            trainable_params += num_params
            if 'attn' in name:
                layer_stats['crossattn']['trainable'] += num_params
            else:
                layer_stats['other']['trainable'] += num_params
        else:
            non_trainable_params += num_params
            if 'attn' in name:
                layer_stats['crossattn']['non_trainable'] += num_params
            else:
                layer_stats['other']['non_trainable'] += num_params
        
    # Print detailed analysis
    print("\n=== U-Net Parameter Analysis ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.2f}%)")
    print(f"Non-trainable parameters: {non_trainable_params:,} ({non_trainable_params/total_params*100:.2f}%)")
    
    print("\n=== Layer-wise Analysis ===")
    print("Cross-Attention Layers:")
    print(f"- Trainable: {layer_stats['crossattn']['trainable']:,}")
    print(f"- Non-trainable: {layer_stats['crossattn']['non_trainable']:,}")
    print(f"trainable")
    print("\nOther Layers:")
    print(f"- Trainable: {layer_stats['other']['trainable']:,}")
    print(f"- Non-trainable: {layer_stats['other']['non_trainable']:,}")
    
    # Check for LoRA parameters specifically
    lora_params = sum(p.numel() for name, p in unet.named_parameters() if 'lora' in name.lower())
    if lora_params > 0:
        print(f"\nLoRA parameters: {lora_params:,}")
    
    return total_params, trainable_params, non_trainable_params

# Example usage:
def check_model_params(unet):
    total, trainable, non_trainable = analyze_unet_parameters(unet)
    
    # Print memory usage estimation
    param_memory = total * 2 / (1024 * 1024)  # Approximate memory in MB for float16
    print(f"\nApproximate parameter memory usage: {param_memory:.2f} MB")
    
    return total, trainable, non_trainable

def diagnose_loss_and_gradients(loss, models_dict):
    """诊断loss和梯度的规模"""
    print(f"Loss value: {loss.item():.4e}")
    
    for name, model in models_dict.items():
        grad_norms = []
        param_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())
            param_norms.append(param.norm().item())
        
        if grad_norms:
            print(f"{name} - Grad norms: min={min(grad_norms):.4e}, "
                  f"max={max(grad_norms):.4e}, "
                  f"mean={sum(grad_norms)/len(grad_norms):.4e}")
        print(f"{name} - Param norms: min={min(param_norms):.4e}, "
                  f"max={max(param_norms):.4e}, "
                  f"mean={sum(param_norms)/len(param_norms):.4e}")

def check_feature_magnitudes(feats, name=""):
    """检查特征图的数值范围"""
    print(f"{name} features - min: {feats.min().item():.4e}, "
          f"max: {feats.max().item():.4e}, "
          f"mean: {feats.mean().item():.4e}, "
          f"std: {feats.std().item():.4e}")
    

def monitor_gradients_from_loss(loss, models_dict,log_reason):
    """
    直接从loss计算梯度
    """
    # 收集所有需要梯度的参数
    params_dict = {
        name: [p for p in model.parameters() if p.requires_grad]
        for name, model in models_dict.items()
    }
    
    # 使用autograd.grad直接计算梯度
    grads = torch.autograd.grad(
        loss,
        [p for params in params_dict.values() for p in params],
        create_graph=False,
        retain_graph=True,
        allow_unused=True
    )
    
    # 整理每个模型的梯度统计
    grad_stats = {}
    start_idx = 0
    for model_name, params in params_dict.items():
        end_idx = start_idx + len(params)
        model_grads = grads[start_idx:end_idx]
        
        if model_grads:
            all_grads = torch.cat([g.flatten() for g in model_grads])
            grad_stats.update({
                f"{model_name}_{log_reason}/grad_norm": all_grads.norm().item(),
                f"{model_name}_{log_reason}/grad_mean": all_grads.mean().item(),
                f"{model_name}_{log_reason}/grad_std": all_grads.std().item()
            })
        
        start_idx = end_idx
    
    return grad_stats

def print_message_if_all_requires_grad_false(model):
    """
    如果模型中所有参数的 requires_grad 都为 False，则打印一句话；
    如果有任何参数的 requires_grad 为 True，则什么都不做。
    
    :param model: 传入的 PyTorch 模型
    """
    if all(not param.requires_grad for param in model.parameters()):
        print("All parameters have requires_grad set to False.")

def check_parameters(self,model, name=""):
    """检查模型参数是否包含NaN或Inf值"""
    has_problem = False
    for param_name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN parameter detected in {name}.{param_name}")
            print(f"Parameter stats - mean: {param.mean()}, std: {param.std()}, "
                f"min: {param.min()}, max: {param.max()}")
            has_problem = True
        if torch.isinf(param).any():
            print(f"Inf parameter detected in {name}.{param_name}")
            print(f"Parameter stats - mean: {param.mean()}, std: {param.std()}, "
                f"min: {param.min()}, max: {param.max()}")
            has_problem = True
    return has_problem

import torch
import numpy as np
import cv2
import os

def visualize_grad_cam(
    original_imgs, grad_cam_maps, alpha=0.4, colormap=cv2.COLORMAP_JET,
    save_path=None
):
    """
    可视化多通道Grad-CAM热力图并叠加到原图上

    Args:
        original_imgs: torch.Tensor, (N, C, H, W) or (N, H, W, C)
        grad_cam_maps: torch.Tensor, (N, 3, H, W)
        alpha: 叠加权重
        colormap: OpenCV伪彩色映射方式
        save_path: 保存文件夹路径（可为None）

    Returns:
        List[List[np.ndarray]]: 每个样本对应3个通道可视化，外层list长度N，内层长度3
    """
    # 转为numpy
    if isinstance(original_imgs, torch.Tensor):
        original_imgs = original_imgs.detach().cpu().numpy()
    if isinstance(grad_cam_maps, torch.Tensor):
        grad_cam_maps = grad_cam_maps.detach().cpu().numpy()

    # 调整原图shape: (N, C, H, W) -> (N, H, W, C)
    if original_imgs.ndim == 4 and original_imgs.shape[1] == 3:
        original_imgs = np.transpose(original_imgs, (0, 2, 3, 1))

    N, cam_c, H, W = grad_cam_maps.shape
    visualizations = []

    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)

    for i in range(N):
        img = original_imgs[i]
        if img.dtype != np.uint8:
            img = np.clip(img * 255, 0, 255).astype(np.uint8)
        cam_list = []
        for c in range(cam_c):
            cam = grad_cam_maps[i, c]
            cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam_uint8 = np.uint8(255 * cam_norm)
            heatmap = cv2.applyColorMap(cam_uint8, colormap)
            heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            visualization = cv2.addWeighted(heatmap, alpha, img, 1 - alpha, 0)
            cam_list.append(visualization)
            # 保存
            if save_path is not None:
                save_file = os.path.join(save_path, f"image_{i}_cam{c}.jpg")
                cv2.imwrite(save_file, visualization)
        visualizations.append(cam_list)
    return visualizations