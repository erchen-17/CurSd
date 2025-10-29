import numpy as np
from PIL import Image
import PIL
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionPipeline
)
from archs.stable_diffusion.resnet import set_timestep, collect_feats,init_resnet_func
import torch.nn.functional as F
from peft import LoraConfig
from aid_methods.debug_analyse import *
from diffusers.training_utils import compute_snr
"""
Functions for running the generalized diffusion process 
(either inversion or generation) and other helpers 
related to latent diffusion models. Adapted from 
Shape-Guided Diffusion (Park et. al., 2022).
https://github.com/shape-guided-diffusion/shape-guided-diffusion/blob/main/utils.py
"""

def print_message_if_all_requires_grad_false(model):
    """
    如果模型中所有参数的 requires_grad 都为 False，则打印一句话；
    如果有任何参数的 requires_grad 为 True，则什么都不做。
    
    :param model: 传入的 PyTorch 模型
    """
    if all(not param.requires_grad for param in model.parameters()):
        print("All parameters have requires_grad set to False.")

def get_tokens_embedding(clip_tokenizer, clip, device, prompt):
  tokens = clip_tokenizer(
    prompt,
    padding="max_length",
    max_length=clip_tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
    return_overflowing_tokens=True,
  )
  input_ids = tokens.input_ids.to(device)
  embedding = clip(input_ids).last_hidden_state
  return tokens, embedding

def latent_to_image(vae, latent):
  latent = latent / 0.18215
  image = vae.decode(latent.to(vae.dtype)).sample
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
  image = (image[0] * 255).round().astype("uint8")
  image = Image.fromarray(image)
  return image

def image_to_latent(vae, image, generator=None, mult=64, w=512, h=512):
  image = image.resize((w, h), resample=PIL.Image.LANCZOS)
  image = np.array(image).astype(np.float32)
  # remove alpha channel
  if len(image.shape) == 2:
    image = image[:, :, None]
  else:
    image = image[:, :, (0, 1, 2)]
  # (b, c, w, h)
  image = image[None].transpose(0, 3, 1, 2)
  image = torch.from_numpy(image)
  image = image / 255.0
  image = 2. * image - 1.
  image = image.to(vae.device)
  image = image.to(vae.dtype)
  return vae.encode(image).latent_dist.sample(generator=generator) * 0.18215

def get_xt_next(xt, et, at, at_next, eta):
  """
  Uses the DDIM formulation for sampling xt_next
  Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
  """
  
  x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
  if eta == 0:
    c1 = 0
  else:
    c1 = (
      eta * ((1 - at / at_next) * (1 - at_next) / (1 - at)).sqrt()
    )
  c2 = ((1 - at_next) - c1 ** 2).sqrt()
  xt_next = at_next.sqrt() * x0_t + c1 * torch.randn_like(et) + c2 * et
  return x0_t, xt_next

def get_x0(xt, et, t, scheduler):
  """
  Uses the DDIM formulation for sampling xt_next
  Denoising Diffusion Implicit Models (Song et. al., ICLR 2021).
  """
  b = scheduler.betas
  b = b.to(xt.device)
  t = t.to(xt.device)
  if t.sum() == -t.shape[0]:
    at = torch.ones_like(t)
  else:
    at = (1 - b).cumprod(dim=0).index_select(0, t.long())
  at = at[:, None, None, None].to('cuda:0')
  x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
  return x0_t

def generalized_steps(x, model, scheduler, **kwargs):
  """
  Performs either the generation or inversion diffusion process.
  """
  seq = scheduler.timesteps
  seq = torch.flip(seq, dims=(0,))
  b = scheduler.betas
  b = b.to(x.device)
  n = x.size(0)
  seq_next = [-1] + list(seq[:-1])
  if kwargs.get("run_inversion", False):
    seq_iter = seq_next
    seq_next_iter = seq
  else:
    seq_iter = reversed(seq)
    seq_next_iter = reversed(seq_next)

  x0_preds = [x]
  xs = [x] #xs[0]为原图
  with torch.amp.autocast('cuda'):
        for i, (t, next_t) in enumerate(zip(seq_iter, seq_next_iter)):
          max_i = kwargs.get("max_i", None)
          min_i = kwargs.get("min_i", None)
          if max_i is not None and i >= max_i: #可以手动设定抛弃过大过小的时间步
            break
          if min_i is not None and i < min_i:
            continue
          
          t = (torch.ones(n) * t).to(x.device)
          next_t = (torch.ones(n) * next_t).to(x.device)
          if t.sum() == -t.shape[0]:
            at = torch.ones_like(t)
          else:
            at = (1 - b).cumprod(dim=0).index_select(0, t.long())
          if next_t.sum() == -next_t.shape[0]:
            at_next = torch.ones_like(t)
          else:
            at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())
          
          # Expand to the correct dim
          at, at_next = at[:, None, None, None], at_next[:, None, None, None]

          if kwargs.get("run_inversion", False):
            cur_time_step = len(seq_iter) - i - 1
            set_timestep(model, cur_time_step)
          else:
            cur_time_step = i
            set_timestep(model, cur_time_step)

          xt = xs[-1].to(x.device)
          cond = kwargs["conditional"]
          guidance_scale = kwargs.get("guidance_scale", -1)
          if guidance_scale == -1:
            if cur_time_step in kwargs.get("save_timestep", []):
              et = model(xt, t, encoder_hidden_states=cond).sample #xt.shape(5,4,64,64) t.shape(5) encoder_hidden_states.shape(5,77,768)
            else:
              with torch.no_grad():
                et = model(xt, t, encoder_hidden_states=cond).sample #xt.shape(5,4,64,64) t.shape(5) encoder_hidden_states.shape(5,77,768)
          else:
            # If using Classifier-Free Guidance, the saved feature maps
            # will be from the last call to the model, the conditional prediction
            if cur_time_step in kwargs.get("save_timestep", []):
              uncond = kwargs["unconditional"]
              et_uncond = model(xt, t, encoder_hidden_states=uncond).sample
              et_cond = model(xt, t, encoder_hidden_states=cond).sample
              et = et_uncond + guidance_scale * (et_cond - et_uncond)
            else:
              with torch.no_grad():
                uncond = kwargs["unconditional"]
                et_uncond = model(xt, t, encoder_hidden_states=uncond).sample
                et_cond = model(xt, t, encoder_hidden_states=cond).sample
                et = et_uncond + guidance_scale * (et_cond - et_uncond)
            
          eta = kwargs.get("eta", 0.0)
          x0_t, xt_next = get_xt_next(xt, et, at, at_next, eta)

          x0_preds.append(x0_t)
          xs.append(xt_next.to('cpu'))
          #del xt, et, at, at_next, x0_t, xt_next
          #torch.cuda.empty_cache()
  return x0_preds

def generalized_steps_with_xt(x, model, scheduler, **kwargs):
  """
  Performs either the generation or inversion diffusion process.
  """
  seq = scheduler.timesteps
  seq = torch.flip(seq, dims=(0,))
  b = scheduler.betas
  b = b.to(x.device)
  n = x.size(0)
  seq_next = [-1] + list(seq[:-1])
  if kwargs.get("run_inversion", False):
    seq_iter = seq_next
    seq_next_iter = seq
  else:
    seq_iter = reversed(seq)
    seq_next_iter = reversed(seq_next)
  time_step_use = {}
  x0_preds = [x]
  xs = [x] #xs[0]为原图
  xt_use = {}
  with torch.amp.autocast('cuda'):
        for i, (t, next_t) in enumerate(zip(seq_iter, seq_next_iter)):
          max_i = kwargs.get("max_i", None)
          min_i = kwargs.get("min_i", None)
          if max_i is not None and i >= max_i: #可以手动设定抛弃过大过小的时间步
            break
          if min_i is not None and i < min_i:
            continue
          
          t = (torch.ones(n) * t).to(x.device)
          next_t = (torch.ones(n) * next_t).to(x.device)
          if t.sum() == -t.shape[0]:
            at = torch.ones_like(t)
          else:
            at = (1 - b).cumprod(dim=0).index_select(0, t.long())
          if next_t.sum() == -next_t.shape[0]:
            at_next = torch.ones_like(t)
          else:
            at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())
          
          # Expand to the correct dim
          at, at_next = at[:, None, None, None], at_next[:, None, None, None]

          if kwargs.get("run_inversion", False):
            cur_time_step = len(seq_iter) - i - 1
            set_timestep(model, cur_time_step)
          else:
            cur_time_step = i
            set_timestep(model, cur_time_step)
          if cur_time_step <= min(kwargs.get("save_timestep")) - 2:
            break

          xt = xs[-1].to(x.device)
          cond = kwargs["conditional"]
          guidance_scale = kwargs.get("guidance_scale", -1)
          if guidance_scale == -1:
            if cur_time_step in kwargs.get("save_timestep", []):
              et = model(xt, t, encoder_hidden_states=cond).sample #xt.shape(5,4,64,64) t.shape(5) encoder_hidden_states.shape(5,77,768)
              time_step_use[cur_time_step] = t
            else:
              with torch.no_grad():
                et = model(xt, t, encoder_hidden_states=cond).sample #xt.shape(5,4,64,64) t.shape(5) encoder_hidden_states.shape(5,77,768)
          else:
            # If using Classifier-Free Guidance, the saved feature maps
            # will be from the last call to the model, the conditional prediction
            if cur_time_step in kwargs.get("save_timestep", []):
              uncond = kwargs["unconditional"]
              et_uncond = model(xt, t, encoder_hidden_states=uncond).sample
              et_cond = model(xt, t, encoder_hidden_states=cond).sample
              et = et_uncond + guidance_scale * (et_cond - et_uncond)
            else:
              with torch.no_grad():
                uncond = kwargs["unconditional"]
                et_uncond = model(xt, t, encoder_hidden_states=uncond).sample
                et_cond = model(xt, t, encoder_hidden_states=cond).sample
                et = et_uncond + guidance_scale * (et_cond - et_uncond)
            
          eta = kwargs.get("eta", 0.0)
          if cur_time_step in kwargs.get("save_timestep", []):
              xt_use[cur_time_step] = xt
          x0_t, xt_next = get_xt_next(xt, et, at, at_next, eta) #xt:[5,4,32,32] et:[5,4,32,32], at:[5,1,1,1]

          x0_preds.append(x0_t)
          xs.append(xt_next.to('cpu'))
          #del xt, et, at, at_next, x0_t, xt_next
          #torch.cuda.empty_cache()
  return x0_preds, xt_use, time_step_use

def process_diffusion_step(x, t, next_t, model, b, seq_len, cur_time_step, conditional, guidance_scale, eta, save_timesteps, **kwargs):

    n = x.size(0)  # Batch size

    # Compute alpha_t and alpha_t_next
    if t.sum() == -t.shape[0]:
      at = torch.ones_like(t)
    else:
      at = (1 - b).cumprod(dim=0).index_select(0, t.long())
    if next_t.sum() == -next_t.shape[0]:
      at_next = torch.ones_like(t)
    else:
      at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())

    # Expand to match dimensions
    at, at_next = at[:, None, None, None], at_next[:, None, None, None]

    # Set the current timestep for the model
    set_timestep(model, cur_time_step)
       
    # Predict noise
    if guidance_scale == -1:
        with torch.no_grad():
            et = model(x, t, encoder_hidden_states=conditional).sample
    else:
        uncond = kwargs["unconditional"]
        et_uncond = model(x, t, encoder_hidden_states=uncond).sample
        et_cond = model(x, t, encoder_hidden_states=conditional).sample
        et = et_uncond + guidance_scale * (et_cond - et_uncond)

    # Compute x0_t and xt_next using the DDIM formula
    x0_t, xt_next = get_xt_next(x, et, at, at_next, eta)

    return x0_t, xt_next, (cur_time_step in save_timesteps)

def patch_shuffling(x, patch_size=16):
    """
    对输入的图像批次进行块随机化（Patch Shuffling）
    
    参数:
        x: 形状为[batch_size, channel, height, width]的tensor
        patch_size: 每个块的大小
    
    返回:
        patch随机化后的tensor
    """
    # 获取输入维度
    batch_size, channels, height, width = x.shape
    
    # 确保图像尺寸能被patch_size整除
    assert height % patch_size == 0 and width % patch_size == 0, "图像尺寸必须能被patch_size整除"
    
    # 计算每个维度的patch数量
    num_patches_h = height // patch_size
    num_patches_w = width // patch_size
    total_patches = num_patches_h * num_patches_w
    
    # 重塑tensor为包含所有patch的形式
    # 形状变为: [batch_size, channels, num_patches_h, patch_size, num_patches_w, patch_size]
    patches = x.reshape(batch_size, channels, num_patches_h, patch_size, num_patches_w, patch_size)
    
    # 调整维度顺序以便按patch处理
    # 形状变为: [batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size]
    patches = patches.permute(0, 2, 4, 1, 3, 5)
    
    # 将所有patch展平为一个列表
    # 形状变为: [batch_size, total_patches, channels, patch_size, patch_size]
    patches = patches.reshape(batch_size, total_patches, channels, patch_size, patch_size)
    
    # 为每个批次独立地创建随机索引
    shuffled_patches = patches.clone()
    for i in range(batch_size):
        # 生成随机索引
        idx = torch.randperm(total_patches)
        # 使用随机索引进行shuffle
        shuffled_patches[i] = patches[i, idx]
    
    # 重新排列为原始形状
    # 先恢复为: [batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size]
    shuffled_patches = shuffled_patches.reshape(batch_size, num_patches_h, num_patches_w, channels, patch_size, patch_size)
    
    # 再调整回原始维度顺序
    # 形状变为: [batch_size, channels, num_patches_h, patch_size, num_patches_w, patch_size]
    shuffled_patches = shuffled_patches.permute(0, 3, 1, 4, 2, 5)
    
    # 最后合并patch维度，恢复为原始图像形状: [batch_size, channels, height, width]
    shuffled_images = shuffled_patches.reshape(batch_size, channels, height, width)
    
    return shuffled_images

def inversion_and_generation(x, model, scheduler, inversion_steps, generation_steps, validate, **kwargs):
    # Get scheduler parameters
    seq = scheduler.timesteps
    seq = torch.flip(seq, dims=(0,))  # Reverse time steps for diffusion
    b = scheduler.betas.to(x.device)  # Beta schedule for noise
    n = x.size(0)  # Batch size
    control_image = kwargs.get("control_image")
    control_net = kwargs.get("control_net")
    control_net_scale = kwargs.get("control_net_scale")
    
    # Split inversion and generation sequences
    seq_inv = [-1] + list(seq[:inversion_steps])  # Time steps for inversion
    seq_inv_next = seq[:inversion_steps + 1]
    seq_gen = seq[:generation_steps]  # Time steps for generation
    seq_gen = torch.flip(seq_gen, dims=(0,))  # Reverse time steps for generation
    seq_gen_next = list(seq_gen[1:]) + [-1]

    time_step_use_inv = {}
    time_step_use_gen = {}
    xt_use_inv = {}
    xt_use_gen = {}
    # Initialize lists to store results
    x0_preds_inv = [x]  # Store predictions of x0 during inversion
    x0_preds_gen = []   # Store predictions of x0 during generation
    xs_inv = [x]        # Store intermediate results during inversion
    transfer_hyperfeature_mode = kwargs.get("transfer_hyperfeature_mode")
    cond = kwargs["conditional"]
    # Perform inversion (add noise)
    with torch.amp.autocast("cuda"):
      with torch.no_grad():
        if transfer_hyperfeature_mode == 'inversion' or transfer_hyperfeature_mode == 'generation':
          for i, (t, next_t) in enumerate(zip(seq_inv, seq_inv_next)):
              control_net_down_feature, control_net_mid_feature =control_net(
                x,        # latent 输入 (B, 4, 64, 64)
                next_t,              # 当前 timestep
                encoder_hidden_states=cond,  # 文本条件
                controlnet_cond=control_image,
                conditioning_scale = control_net_scale,
                return_dict = False                 # 条件图像（如canny/seg/gradcam）
                )
              max_i = kwargs.get("max_i", None)
              min_i = kwargs.get("min_i", None)
              if max_i is not None and i >= max_i: #可以手动设定抛弃过大过小的时间步
                break
              if min_i is not None and i < min_i:
                continue
              
              t = (torch.ones(n) * t).to(x.device)
              next_t = (torch.ones(n) * next_t).to(x.device)
              if t.sum() == -t.shape[0]:
                at = torch.ones_like(t)
              else:
                at = (1 - b).cumprod(dim=0).index_select(0, t.long())
              if next_t.sum() == -next_t.shape[0]:
                at_next = torch.ones_like(t)
              else:
                at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())
              
              # Expand to the correct dim
              at, at_next = at[:, None, None, None], at_next[:, None, None, None]

              cur_time_step = len(seq) - i - 1
              set_timestep(model, cur_time_step)

              xt = xs_inv[-1].to(x.device)
              guidance_scale = kwargs.get("guidance_scale", -1)
              if guidance_scale == -1:
                if cur_time_step in kwargs.get("save_timestep", []) and transfer_hyperfeature_mode == "inversion":
                  with torch.no_grad():
                    et = model(xt, t, encoder_hidden_states=cond, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample #xt.shape(25,4,32,32) t.shape(5) encoder_hidden_states.shape(5,77,768)
                    time_step_use_inv[cur_time_step] = t
                else:
                  with torch.no_grad():
                    et = model(xt, t, encoder_hidden_states=cond, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample #xt.shape(5,4,64,64) t.shape(5) encoder_hidden_states.shape(5,77,768)
              else:
                # If using Classifier-Free Guidance, the saved feature maps
                # will be from the last call to the model, the conditional prediction
                if cur_time_step in kwargs.get("save_timestep", []):
                  uncond = kwargs["unconditional"]
                  et_uncond = model(xt, t, encoder_hidden_states=uncond, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample
                  et_cond = model(xt, t, encoder_hidden_states=cond, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample
                  et = et_uncond + guidance_scale * (et_cond - et_uncond)
                else:
                  with torch.no_grad():
                    uncond = kwargs["unconditional"]
                    et_uncond = model(xt, t, encoder_hidden_states=uncond, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample
                    et_cond = model(xt, t, encoder_hidden_states=cond, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample
                    et = et_uncond + guidance_scale * (et_cond - et_uncond)
                
              eta = kwargs.get("eta", 0.0)
              if cur_time_step in kwargs.get("save_timestep", []):
                  xt_use_inv[cur_time_step] = xt
              x0_t, xt_next = get_xt_next(xt, et, at, at_next, eta) #xt:[5,4,32,32] et:[5,4,32,32], at:[5,1,1,1]

              x0_preds_inv.append(x0_t)
              xs_inv.append(xt_next.to('cpu'))
        else:
              # 假设t_target是目标噪声时间步
            t_target = seq[inversion_steps]  # 默认使用最大的时间步
            t_target = (torch.ones(n) * t_target).to(x.device)
            
            # 计算目标时间步的累积alpha
            if t_target.sum() == -t_target.shape[0]:
                at_target = torch.ones_like(t_target)
            else:
                at_target = (1 - b).cumprod(dim=0).index_select(0, t_target.long())
            
            # 扩展维度
            at_target = at_target[:, None, None, None]
            
            # 获取初始干净图像
            x0 = x.to(x.device)
            
            # 获取条件
            cond = kwargs["conditional"]
            
            # 采样随机噪声
            noise = torch.randn_like(x0)
            
            # 基于DDIM公式，一步将图像加噪到目标时间步
            # DDIM公式: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
            xt = torch.sqrt(at_target) * x0 + torch.sqrt(1 - at_target) * noise
            
      if transfer_hyperfeature_mode == "inversion":
        return x0_preds_inv, xt_use_inv, time_step_use_inv
      init_resnet_func(model, save_hidden=True, reset=True, idxs=kwargs['idxs'], save_timestep=kwargs['save_timestep'])
      xs_inversion = xs_inv[-1].to("cuda:0")
      if transfer_hyperfeature_mode == 'generation' and not validate:
        with torch.no_grad():
            for i in range(generation_steps - inversion_steps):
              # 假设t_target是目标噪声时间步
              t_target = len(xs_inv) - 1 + i  # 默认使用最大的时间步
              t_target = (torch.ones(n) * seq[t_target]).to(x.device)
              
              # 计算目标时间步的累积alpha
              if t_target.sum() == -t_target.shape[0]:
                  at_target = torch.ones_like(t_target)
              else:
                  at_target = (1 - b).cumprod(dim=0).index_select(0, t_target.long())
              
              # 扩展维度
              at_target = at_target[:, None, None, None]
              
              # 获取初始干净图像
              x0 = xs_inversion.to(xs_inversion.device)
              
              # 获取条件
              cond = kwargs["conditional"]
              
              # 采样随机噪声
              noise = torch.randn_like(x0).to(xs_inversion.device)
              
              # 基于DDIM公式，一步将图像加噪到目标时间步
              # DDIM公式: x_t = sqrt(alpha_t) * x_0 + sqrt(1-alpha_t) * noise
              xs_inversion = torch.sqrt(at_target) * x0 + torch.sqrt(1 - at_target) * noise
            #xs_inversion = patch_shuffling(xs_inversion)

      xs_gen = [xs_inversion]
      for i, (t, next_t) in enumerate(zip(seq_gen, seq_gen_next)):
          with torch.no_grad():
            max_i = kwargs.get("max_i", None)
            min_i = kwargs.get("min_i", None)
            if max_i is not None and i >= max_i: #可以手动设定抛弃过大过小的时间步
              break
            if min_i is not None and i < min_i:
              continue
            control_net_down_feature, control_net_mid_feature =control_net(
                x,        # latent 输入 (B, 4, 64, 64)
                t,              # 当前 timestep
                encoder_hidden_states=cond,  # 文本条件
                controlnet_cond=control_image,
                conditioning_scale = control_net_scale,
                return_dict = False                 # 条件图像（如canny/seg/gradcam）
                )
            
            t = (torch.ones(n) * t).to(x.device)
            next_t = (torch.ones(n) * next_t).to(x.device)
            if t.sum() == -t.shape[0]:
              at = torch.ones_like(t)
            else:
              at = (1 - b).cumprod(dim=0).index_select(0, t.long())
            if next_t.sum() == -next_t.shape[0]:
              at_next = torch.ones_like(t)
            else:
              at_next = (1 - b).cumprod(dim=0).index_select(0, next_t.long())
            
            # Expand to the correct dim
            at, at_next = at[:, None, None, None], at_next[:, None, None, None]

            cur_time_step = len(seq) - generation_steps + i
            set_timestep(model, cur_time_step)

            xt = xs_gen[-1].to(x.device)
            cond = kwargs["conditional"]
            guidance_scale = kwargs.get("guidance_scale", -1)
          if guidance_scale == -1:
            if cur_time_step in kwargs.get("save_timestep", []):
              torch.cuda.empty_cache()
              et = model(xt, t, encoder_hidden_states=cond, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample #xt.shape(5,4,64,64) t.shape(5) encoder_hidden_states.shape(5,77,768)
              time_step_use_gen[cur_time_step] = t
            else:
              with torch.no_grad():
                et = model(xt, t, encoder_hidden_states=cond, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample #xt.shape(5,4,64,64) t.shape(5) encoder_hidden_states.shape(5,77,768)
          else:
            # If using Classifier-Free Guidance, the saved feature maps
            # will be from the last call to the model, the conditional prediction
            if cur_time_step in kwargs.get("save_timestep", []):
              uncond = kwargs["unconditional"]
              et_uncond = model(xt, t, encoder_hidden_states=uncond, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample
              et_cond = model(xt, t, encoder_hidden_states=cond, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample
              et = et_uncond + guidance_scale * (et_cond - et_uncond)
            else:
              with torch.no_grad():
                uncond = kwargs["unconditional"]
                et_uncond = model(xt, t, encoder_hidden_states=uncond, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample
                et_cond = model(xt, t, encoder_hidden_states=cond, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample
                et = et_uncond + guidance_scale * (et_cond - et_uncond)
          with torch.no_grad():
            eta = kwargs.get("eta", 0.0)
            if cur_time_step in kwargs.get("save_timestep", []):
                xt_use_gen[cur_time_step] = xt
            x0_t, xt_next = get_xt_next(xt, et, at, at_next, eta) #xt:[5,4,32,32] et:[5,4,32,32], at:[5,1,1,1]

            x0_preds_gen.append(x0_t)
            xs_gen.append(xt_next.to('cpu'))
    return x0_preds_gen, xt_use_gen, time_step_use_gen

def train_DDIM(batch_size, latents, unet, control_net, noise_scheduler, config, conditional=None, unconditional=None, guidance_scale=-1, control_net_pic=None, control_net_scale = 0.0, min_i=None, max_i=None):
    def compute_latent_edge_map(latents, device=None):
      """
      针对VAE潜在空间(4通道)优化的边缘检测
      Args:
          latents: [B,4,H,W] 经过VAE编码的潜在变量
      Returns:
          edge_map: [B,1,H,W] 边缘强度图
      """
      if device is None:
          device = latents.device
      
      # 可学习的通道权重（实践表明第一通道通常包含主要结构信息）
      channel_weights = torch.tensor([0.6, 0.2, 0.1, 0.1], device=device)  # 可配置
      
      # 加权合并通道
      weighted_latent = (latents * channel_weights.view(1,4,1,1)).sum(dim=1, keepdim=True)
      
      # 潜在空间特化的边缘检测核
      # 因为潜在空间分辨率低，使用更大的核尺寸(5x5)
      kernel_x = torch.tensor([
          [ [ [2, 1, 0, -1, -2], 
            [2, 1, 0, -1, -2],
            [4, 2, 0, -2, -4],
            [2, 1, 0, -1, -2],
            [2, 1, 0, -1, -2] ]]
      ], device=device).float() / 16.0
      
      kernel_y = kernel_x.transpose(2,3)
      
      # 边缘检测（分组卷积处理batch）
      edge_x = F.conv2d(weighted_latent, kernel_x, padding=2, groups=1)
      edge_y = F.conv2d(weighted_latent, kernel_y, padding=2, groups=1)
      
      # 计算边缘强度
      edge_magnitude = torch.sqrt(edge_x**2 + edge_y**2)
      
      # 动态归一化（保持batch内相对强度）
      edge_map = edge_magnitude / (edge_magnitude.amax(dim=(1,2,3), keepdim=True) + 1e-6)
      
      # 潜在空间边缘增强（经验系数）
      edge_map = torch.pow(edge_map, 0.75)  # 非线性增强
      
      return edge_map
    """
    基于train_text_to_image_lora的训练方法
    """
    # Sample noise
    noise = torch.randn_like(latents)
    if config.get("noise_offset", 0):
        # https://www.crosslabs.org//blog/diffusion-with-offset-noise
        noise += config["noise_offset"] * torch.randn(
            (latents.shape[0], latents.shape[1], 1, 1), device=latents.device
        )

    # Sample a random timestep for each image
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
    timesteps = timesteps.long()

    # Add noise to the latents according to the noise magnitude at each timestep
    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

    edge_map = None
    if config.get("use_edge_guidance", False):
        edge_map = compute_latent_edge_map(latents)
        # 时间步衰减：越后期越不依赖边缘
        edge_map *= (1 - timesteps.float()/noise_scheduler.num_train_timesteps).view(-1,1,1,1)
    if config["use_control_net"]:
      control_net_down_feature, control_net_mid_feature =control_net(
      latents,        # latent 输入 (B, 4, 64, 64)
      timesteps,              # 当前 timestep
      encoder_hidden_states=conditional,  # 文本条件
      conditioning_scale = control_net_scale,
      controlnet_cond=control_net_pic,
      return_dict = False                 # 条件图像（如canny/seg/gradcam）
      )
      # Predict the noise residual
      if guidance_scale > 0:
          # 使用classifier-free guidance
          model_pred_uncond = unet(noisy_latents, timesteps, encoder_hidden_states = unconditional, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample
          torch.cuda.empty_cache()
          model_pred_cond = unet(noisy_latents, timesteps, encoder_hidden_states = conditional, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample
          model_pred = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)
      else:
          model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=conditional, down_block_additional_residuals=control_net_down_feature,  mid_block_additional_residual=control_net_mid_feature).sample
    else:
      if guidance_scale > 0:
          # 使用classifier-free guidance
          model_pred_uncond = unet(noisy_latents, timesteps, encoder_hidden_states = unconditional).sample
          torch.cuda.empty_cache()
          model_pred_cond = unet(noisy_latents, timesteps, encoder_hidden_states = conditional).sample
          model_pred = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)
      else:
          model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=conditional).sample

    # Get the target for loss
    if config.get("prediction_type") is not None:
        # 设置prediction_type
        noise_scheduler.register_to_config(prediction_type=config["prediction_type"])

    if config["prediction_type"] is not None:
        # set prediction_type of scheduler if defined
        noise_scheduler.register_to_config(prediction_type=config["prediction_type"])

    if noise_scheduler.config.prediction_type == "epsilon":
        target = noise
    elif noise_scheduler.config.prediction_type == "v_prediction":
        target = noise_scheduler.get_velocity(latents, noise, timesteps)
    else:
        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

    if config.get("snr_gamma") is None:
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    else:
        # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
        # Since we predict the noise instead of x_0, the original formulation is slightly changed.
        # This is discussed in Section 4.2 of the same paper.
        snr = compute_snr(noise_scheduler, timesteps)
        mse_loss_weights = torch.stack([snr, config.get("snr_gamma", 5) * torch.ones_like(timesteps)], dim=1).min(
            dim=1
        )[0]
        if noise_scheduler.config.prediction_type == "epsilon":
            mse_loss_weights = mse_loss_weights / snr
        elif noise_scheduler.config.prediction_type == "v_prediction":
            mse_loss_weights = mse_loss_weights / (snr + 1)

        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss = loss * mse_loss_weights.view(-1, 1, 1, 1)
        '''
        loss = loss.mean(dim=list(range(1, len(loss.shape))))
        loss = loss.mean()
        '''
        # 应用边缘引导（如果启用）
    if edge_map is not None:
        edge_weight = config.get("edge_weight", 0.3)
        # 边缘区域损失增强
        loss = (loss * (1 + edge_weight * edge_map.expand(-1, loss.size(1), -1, -1))).mean()
    else:
        loss = loss.mean()
    return loss

def freeze_weights(weights):
  for param in weights.parameters():
    param.requires_grad = False

def init_models(
    device="cuda",
    model_id="runwayml/stable-diffusion-v1-5",
    freeze=False,
    lora_config=None
  ):
  # Set model weights to mirror since
  # runwayml took down the weights for SDv1-5
  if model_id == "runwayml/stable-diffusion-v1-5":
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
  )
  unet = pipe.unet
  vae = pipe.vae
  clip = pipe.text_encoder
  clip_tokenizer = pipe.tokenizer
  unet.to(device)
  vae.to(device)
  clip.to(device)

  use_lora = lora_config.get('use_lora', False)
  rank = lora_config.get('rank', 0)
  lora_alpha = lora_config.get('lora_alpha', 0.0)

  if use_lora:
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    # 添加LoRA adapter
    unet.add_adapter(unet_lora_config)

    # 使用filter只选择LoRA参数进行训练
    trainable_params = filter(
        lambda p: p.requires_grad, 
        unet.parameters()
    )

  if freeze:
    freeze_weights(unet)
    freeze_weights(vae)
    freeze_weights(clip)
      # 打印检查
  # 打印检查
  found_lora = False
  grad_enabled_count = 0
  grad_disabled_count = 0
  for name, param in unet.named_parameters():
      #print(name)
      if "lora" in name.lower():
          found_lora = True
          grad_enabled_count += 1
      else:
          grad_disabled_count += 1
      if "conv_in" in name or "conv_out" in name or "conv_norm_out" in name or "down_blocks.2" in name or "down_blocks.1" in name or "down_blocks.0" in name:
         param.requires_grad = True

  if not found_lora:
      print("Warning: No LoRA parameters found!")

  if use_lora:
    return unet, vae, clip, clip_tokenizer, trainable_params
  else:
    return unet, vae, clip, clip_tokenizer, None

def init_models_with_controlnet(
    device="cuda",
    model_id="runwayml/stable-diffusion-v1-5",
    control_net_id = None,
    freeze=False,
    lora_config=None
  ):
  # Set model weights to mirror since
  # runwayml took down the weights for SDv1-5
  if model_id == "runwayml/stable-diffusion-v1-5":
    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
  controlnet = ControlNetModel.from_pretrained(
    control_net_id, torch_dtype=torch.float16
  )
  pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    controlnet=controlnet,
    torch_dtype=torch.float16,
  )
  unet = pipe.unet
  vae = pipe.vae
  clip = pipe.text_encoder
  clip_tokenizer = pipe.tokenizer
  scheduler_config = pipe.scheduler.config
  unet.to(device)
  vae.to(device)
  clip.to(device)
  controlnet.to(device)

  use_lora = lora_config.get('use_lora', False)
  rank = lora_config.get('rank', 0)
  lora_alpha = lora_config.get('lora_alpha', 0.0)

  if use_lora:
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    unet_lora_config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
    )
    # 添加LoRA adapter
    unet.add_adapter(unet_lora_config)

    # 使用filter只选择LoRA参数进行训练
    trainable_params = filter(
        lambda p: p.requires_grad, 
        unet.parameters()
    )

  if freeze:
    freeze_weights(unet)
    freeze_weights(vae)
    freeze_weights(clip)
      # 打印检查
  # 打印检查
  found_lora = False
  grad_enabled_count = 0
  grad_disabled_count = 0
  for name, param in unet.named_parameters():
      if "lora" in name.lower():
          found_lora = True
          grad_enabled_count += 1
      else:
          grad_disabled_count += 1
      if "conv_out" in name or "conv_norm_out" in name or "up_blocks.3" in name or "up_blocks.2" in name:
         param.requires_grad = True

  if not found_lora:
      print("Warning: No LoRA parameters found!")

  if use_lora:
    return unet, vae, clip, clip_tokenizer, trainable_params, controlnet, scheduler_config
  else:
    return unet, vae, clip, clip_tokenizer, None, controlnet, scheduler_config

def collect_and_resize_feats(unet, idxs, timestep, resolution=-1, xt=None, scheduler=None, cond = None, time_step_use = None):
  latent_feats = collect_feats(unet, idxs=idxs)
  latent_feats = [feat[timestep] for feat in latent_feats]
  ans = []
  with torch.amp.autocast("cuda"):
    for i in latent_feats:
      #torch.manual_seed(0)  # PyTorch的随机种子
      #torch.cuda.manual_seed_all(0)  # GPU随机种子
      temp = unet.up_blocks[-1].attentions[-1](i, encoder_hidden_states = cond).sample  # 访问最后一个up_block中的最后一个transformer
      temp = unet.conv_norm_out(temp)
      temp = F.silu(temp)
      temp = unet.conv_out(temp)
      ans.append(get_x0(xt.to("cuda:0"), temp, time_step_use, scheduler))
  
  ans = torch.stack(ans, dim=1)
  return ans