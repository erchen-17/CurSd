import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndimage

def create_control_images(
    input_tensor, 
    low_threshold=0.005, 
    high_threshold=0.995,
    area_threshold_ratio=0.05,
    output_format='squared_deviation'):

    assert input_tensor.dim() == 4, f"Expected 4D tensor, got {input_tensor.dim()}D"
    assert input_tensor.shape[1] == 3, f"Expected 3 channels, got {input_tensor.shape[1]}"
    
    batch_size, channels, height, width = input_tensor.shape
    
    # 创建异常值掩码（检测过小或过大的像素）
    # 对每个像素，如果任意通道是异常值，则整个像素标记为异常
    outlier_masks_low = input_tensor < low_threshold  # [batch_size, 3, height, width]
    outlier_masks_high = input_tensor > high_threshold  # [batch_size, 3, height, width]
    outlier_masks = (outlier_masks_low | outlier_masks_high).any(dim=1, keepdim=True)  # [batch_size, 1, height, width]
    
    area_threshold = area_threshold_ratio * height * width
    #连通区域面积过滤
    filtered_outlier_masks = torch.zeros_like(outlier_masks)
    for b in range(batch_size):
        mask_np = outlier_masks[b, 0].cpu().numpy().astype(np.uint8)  # [H, W]
        labeled, num_features = ndimage.label(mask_np)  # 连通区域标记
        filtered_mask = np.zeros_like(mask_np)
        for region_idx in range(1, num_features + 1):
            area = (labeled == region_idx).sum()
            if area >= area_threshold:
                filtered_mask[labeled == region_idx] = 1
        filtered_outlier_masks[b, 0] = torch.from_numpy(filtered_mask)

    outlier_masks = filtered_outlier_masks.to(input_tensor.device).bool()  # [B, 1, H, W]

    # 创建有效像素掩码（非异常值）
    valid_masks = ~outlier_masks  # [batch_size, 1, height, width]
    
    # 计算每个通道的均值，只考虑有效像素
    mean_values = []
    for b in range(batch_size):
        channel_means = []
        for c in range(channels):
            # 获取当前通道的有效像素
            valid_mask_bc = valid_masks[b, 0]  # [height, width]
            channel_data = input_tensor[b, c]  # [height, width]
            
            # 只使用有效像素计算均值
            if valid_mask_bc.sum() > 0:
                valid_pixels = channel_data[valid_mask_bc]
                mean_val = valid_pixels.mean()
            else:
                raise ValueError("Inappropriate_threshold_set")
            
            channel_means.append(mean_val)
        
        mean_values.append(torch.stack(channel_means))
    
    # 将均值整理成正确的形状 [batch_size, 3, 1, 1]
    mean_values = torch.stack(mean_values).view(batch_size, channels, 1, 1).to(input_tensor.device)
    
    # 计算中心化图像
    centered_tensor = input_tensor - mean_values  # [batch_size, 3, height, width]
    
    # 在异常值位置将输出置为0
    # 扩展outlier_masks以匹配通道数
    outlier_masks_3ch = outlier_masks.expand(-1, 3, -1, -1)  # [batch_size, 3, height, width]
    centered_tensor[outlier_masks_3ch] = 0
    
    if output_format == 'centered':
        return centered_tensor
    elif output_format == 'squared_deviation':
        squared_deviation = centered_tensor ** 2
        # 异常值位置已经是0，平方后仍然是0
        normalized_squared_deviation = (squared_deviation - squared_deviation.min())/(squared_deviation.max()-squared_deviation.min())
        images = torch.nn.functional.interpolate(normalized_squared_deviation, size=256, mode="bilinear")
        return images
    else:
        raise ValueError(f"Unknown output_format: {output_format}")

def compute_deviation_map(control_tensor):
    """
    计算偏差图，用于显示每个像素的偏差程度
    
    Args:
        control_tensor: 形状为 [batch_size, 3, height, width] 的控制图像
    
    Returns:
        deviation_map: 形状为 [batch_size, height, width] 的偏差图
    """
    # 计算每个像素在所有通道上的L2范数
    deviation_map = torch.sqrt((control_tensor ** 2).sum(dim=1))
    
    return deviation_map

def detect_defects(control_tensor, threshold=None, percentile=95):
    """
    检测潜在缺陷区域
    
    Args:
        control_tensor: 形状为 [batch_size, 3, height, width] 的控制图像
        threshold: 固定阈值，如果为None则使用自适应阈值
        percentile: 用于计算自适应阈值的百分位数
    
    Returns:
        defect_masks: 形状为 [batch_size, height, width] 的缺陷掩码
        deviation_maps: 形状为 [batch_size, height, width] 的偏差图
        thresholds: 每张图像使用的阈值
    """
    deviation_maps = compute_deviation_map(control_tensor)
    batch_size = deviation_maps.shape[0]
    
    defect_masks = torch.zeros_like(deviation_maps, dtype=torch.bool)
    thresholds = []
    
    for i in range(batch_size):
        deviation = deviation_maps[i]
        
        # 计算阈值
        if threshold is None:
            # 自适应阈值：使用百分位数
            thresh = torch.quantile(deviation.flatten(), percentile / 100.0)
        else:
            thresh = threshold
        
        thresholds.append(thresh.item() if torch.is_tensor(thresh) else thresh)
        
        # 创建缺陷掩码
        defect_masks[i] = deviation > thresh
    
    return defect_masks, deviation_maps, thresholds

def visualize_batch_results(input_tensor, control_tensor, max_display=4):
    """
    可视化批次中的结果
    
    Args:
        input_tensor: 原始图像tensor [batch_size, 3, height, width]
        control_tensor: 控制图像tensor [batch_size, 3, height, width]
        max_display: 最多显示的图像数量
    """
    batch_size = min(input_tensor.shape[0], max_display)
    
    fig, axes = plt.subplots(batch_size, 4, figsize=(16, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(batch_size):
        # 原始图像
        img = input_tensor[i].permute(1, 2, 0).cpu()
        # 归一化到[0, 1]范围以便显示
        img = (img - img.min()) / (img.max() - img.min())
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f'Original Image {i+1}')
        axes[i, 0].axis('off')
        
        # 控制图像（归一化显示）
        control = control_tensor[i].permute(1, 2, 0).cpu()
        control_normalized = (abs(control) - control.min()) / (control.max() - control.min())
        axes[i, 1].imshow(control_normalized)
        axes[i, 1].set_title(f'Control Image {i+1}')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def get_defect_statistics(input_tensor, control_tensor, threshold=None):
    """
    获取缺陷检测的统计信息
    
    Args:
        input_tensor: 原始图像tensor [batch_size, 3, height, width]
        control_tensor: 控制图像tensor [batch_size, 3, height, width]
        threshold: 检测阈值
    
    Returns:
        stats: 包含统计信息的字典列表
    """
    defect_masks, deviation_maps, thresholds = detect_defects(control_tensor, threshold)
    batch_size = input_tensor.shape[0]
    
    stats = []
    for i in range(batch_size):
        deviation = deviation_maps[i]
        defect_mask = defect_masks[i]
        
        stat = {
            'image_index': i,
            'threshold': thresholds[i],
            'max_deviation': deviation.max().item(),
            'mean_deviation': deviation.mean().item(),
            'std_deviation': deviation.std().item(),
            'defect_pixels': defect_mask.sum().item(),
            'total_pixels': defect_mask.numel(),
            'defect_ratio': defect_mask.sum().item() / defect_mask.numel()
        }
        stats.append(stat)
    
    return stats