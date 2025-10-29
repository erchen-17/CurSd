import torch
import math

def get_scheduler_NEU(config, optimizer, len_traindataloader):
    """
    创建学习率调度器
    """
    # 计算总训练步数
    num_warmup_steps = config.get("lr_warmup_steps", 500)
    if config["transfer_learning"]:
        gradient_accumulation_steps = 30
    else:
        gradient_accumulation_steps = config["accumulation_steps"]
    
    # 计算每个epoch的更新步数
    num_update_steps_per_epoch = math.ceil(len_traindataloader / gradient_accumulation_steps)
    
    if config.get("max_train_steps") is None:
        # 如果没有指定最大训练步数，根据epoch计算
        num_training_steps = config["max_epochs"] * num_update_steps_per_epoch
    else:
        num_training_steps = config["max_train_steps"]
    
    # 创建调度器
    scheduler_type = config.get("lr_scheduler", "constant")
    
    if scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    elif scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=config.get("lr_num_cycles", 0.5),
            min_lr_ratio=config.get("min_lr_ratio", 0.0)
        )
    elif scheduler_type == "constant_with_warmup":
        return get_constant_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps
        )
    else:  # default to constant
        return get_constant_schedule(optimizer)

def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    """
    创建带预热的线性学习率调度器
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5, min_lr_ratio=0.0):
    """
    创建带预热的余弦学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_constant_schedule_with_warmup(optimizer, num_warmup_steps):
    """
    创建带预热的常数学习率调度器
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def get_constant_schedule(optimizer):
    """
    创建常数学习率调度器
    """
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)

def hook_optimizer(optimizer, model):
    """
    优化器监控，仅在检测到inf/nan时输出详细信息
    """
    param_to_module = {}
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters():
            param_to_module[param] = (name, param_name)
    
    old_step = optimizer.step
    
    def new_step(closure=None):
        prev_params = {}
        # 记录状态的代码保持不变...
        for group in optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    module_name, param_name = param_to_module.get(p, ("Unknown", "Unknown"))
                    prev_params[p] = {
                        'data': p.data.clone(),
                        'grad': p.grad.clone(),
                        'grad_norm': p.grad.norm().item(),
                        'module_name': module_name,
                        'param_name': param_name
                    }
        loss = old_step(closure)
        
        for group in optimizer.param_groups:
            for p in group['params']:
                if p in prev_params:
                    if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                        info = prev_params[p]
                        print(f"\nNaN/Inf detected in module: {info['module_name']}")
                        print(f"Parameter: {info['param_name']}")
                        
                        # 之前的统计信息保持不变...
                        
                        # 添加详细的更新计算追踪
                        if p in optimizer.state:
                            state = optimizer.state[p]
                            if 'exp_avg' in state and 'exp_avg_sq' in state:
                                beta1, beta2 = group['betas']
                                step = state['step']
                                lr = group['lr']
                                eps = group['eps']
                                
                                # 计算Adam更新各个组件
                                bias_correction1 = 1 - beta1 ** step
                                bias_correction2 = 1 - beta2 ** step
                                step_size = lr * math.sqrt(bias_correction2) / bias_correction1
                                
                                exp_avg = state['exp_avg']
                                exp_avg_sq = state['exp_avg_sq']
                                
                                # 计算实际更新步长
                                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                                update = (exp_avg / bias_correction1) / denom
                                
                                print("\nDetailed Adam update analysis:")
                                print(f"Learning rate: {lr}")
                                print(f"Step size: {step_size}")
                                print(f"Bias correction 1: {bias_correction1}")
                                print(f"Bias correction 2: {bias_correction2}")
                                print(f"Epsilon: {eps}")
                                
                                print("\nUpdate components:")
                                print(f"Denominator stats:")
                                print(f" - Mean: {denom.mean().item():.6f}")
                                print(f" - Std: {denom.std().item():.6f}")
                                print(f" - Min: {denom.min().item():.6f}")
                                print(f" - Max: {denom.max().item():.6f}")
                                
                                print(f"Final update stats:")
                                print(f" - Mean: {update.mean().item():.6f}")
                                print(f" - Std: {update.std().item():.6f}")
                                print(f" - Min: {update.min().item():.6f}")
                                print(f" - Max: {update.max().item():.6f}")
                                
                                # 检查是否有极小值导致除法问题
                                small_denom = (denom.abs() < eps).sum().item()
                                if small_denom > 0:
                                    print(f"\nWarning: {small_denom} elements in denominator are smaller than epsilon")
                                        
                                    
        return loss
    
    optimizer.step = new_step
    return optimizer