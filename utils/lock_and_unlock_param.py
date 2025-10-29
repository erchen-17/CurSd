def lock_non_upblock_gradients_NEU(unet, freeze_ratio=0.9, step = 1):

    #print_message_if_all_requires_grad_false(unet)
    modified_params = {}
    total_trainable_params = 0  # 总可训练参数数量
    up_blocks_params = 0  # up_blocks 中可训练的参数数量
    frozen_up_blocks_params = 0  # 实际冻结的 up_blocks 参数数量

    # 冻结非 up_blocks 的参数
    for name, param in unet.named_parameters():
        if param.requires_grad:
            total_trainable_params += 1
            if 'up_blocks' not in name and "conv_out" not in name and "conv_norm_out" not in name:
                param.requires_grad = False  # 冻结非 up_blocks 的参数
                modified_params[name] = True
            else:
                up_blocks_params += 1
    #print_message_if_all_requires_grad_false(unet)
    # 额外冻结 up_blocks 的参数
    up_blocks_frozen = 0
    up_blocks_target_to_freeze = int(up_blocks_params * freeze_ratio)

    for name, param in unet.named_parameters():
        if 'up_blocks' in name and param.requires_grad:
            param.requires_grad = False  # 冻结所有 up_blocks 参数
            modified_params[name] = True
            frozen_up_blocks_params += 1
            up_blocks_frozen += 1

            # 如果达到目标冻结数量，停止
            if up_blocks_frozen >= up_blocks_target_to_freeze:
                break

    return modified_params

def lock_all_unet_gradients(unet):

    #print_message_if_all_requires_grad_false(unet)
    modified_params = {}
    total_trainable_params = 0  # 总可训练参数数量

    # 冻结非 up_blocks 的参数
    for name, param in unet.named_parameters():
        if param.requires_grad:
            total_trainable_params += 1
            param.requires_grad = False  # 冻结非 up_blocks 的参数
            modified_params[name] = True

    return modified_params

def unlock_all_gradients_NEU(unet):

    modified_params = {}
    total_params = 0
    unlocked_params = 0

    # 解锁所有参数
    for name, param in unet.named_parameters():
        total_params += 1
        if not param.requires_grad:
            param.requires_grad = True
            modified_params[name] = True
            unlocked_params += 1
    
    return modified_params

def relock_gradients_NEU(unet, modified_params):

    cnt = 0
    for name, param in unet.named_parameters():
        # 只重新锁定之前被修改的参数
        if name in modified_params:
            cnt += 1
            param.requires_grad = False

    if cnt == 0:
        raise ValueError("No parameters were relocked. Check if modified_params is empty or if parameters were correctly tracked.")

def restore_lora_gradients_NEU(unet, modified_params):

    cnt = 0
    for name, param in unet.named_parameters():
        if name in modified_params:
            cnt += 1
            param.requires_grad = True

    if cnt == 0:
        raise ValueError("fail to restore")
    
def print_message_if_all_requires_grad_false(model):
    """
    如果模型中所有参数的 requires_grad 都为 False，则打印一句话；
    如果有任何参数的 requires_grad 为 True，则什么都不做。
    
    :param model: 传入的 PyTorch 模型
    """
    if all(not param.requires_grad for param in model.parameters()):
        print("All parameters have requires_grad set to False.")
        raise ValueError("error in unet")