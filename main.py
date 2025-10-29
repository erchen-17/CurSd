import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import argparse
import numpy as np
from PIL import Image
import torch
from tqdm import tqdm
import wandb
import torch.nn as nn

from archs.diffusion_extractor import DiffusionExtractor
from omegaconf import OmegaConf
from dataset_build.NEU import few_shot_data_loader_NEU_transfer, few_shot_data_loader_NEU_whole, get_loader_NEU
from utils.get_scheduler_optimizer import *
from dataset_build.severstal import few_shot_data_loader_severstal_whole, few_shot_data_loader_severstal_transfer
from dataset_build.NEU64 import few_shot_data_loader_NEU64_whole, few_shot_data_loader_NEU64_transfer
from dataset_build.X_SDD import few_shot_data_loader_X_SDD_whole, few_shot_data_loader_X_SDD_transfer
from archs.classification import res2net50_v1b
from utils.lock_and_unlock_param import *
import torch.nn.functional as F
from aid_methods.debug_analyse import *
from archs.only_few_shot import get_only_few_shot_whole_model
from archs.control_net.utils.grad_cam import cal_grad, simple_CNN_grad
from utils.image_minus_ava import create_control_images

def save_model_NEU(config, diffusion_extractor, scale_fusion_network, classification_model, optimizer, step,save_reason):
    # Create a dictionary to save model states
    dict_to_save = {
        "step": step,
        "config": config,
        "scale_fusion_network": scale_fusion_network.state_dict() if config["use_control_net"] else None,
        "classification_model": classification_model.state_dict() if config.get("classification", True) else None,
        "diffusion_unet": diffusion_extractor.unet.state_dict(),  # 保存扩散模型权重
        "optimizer_state_dict": optimizer.state_dict()
    }

    # Create results folder if it doesn't exist
    results_folder = f"{config['results_folder']}" # /{wandb.run.name}
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    # Save model to a checkpoint file
    if save_reason == "step":
        torch.save(dict_to_save, f"{results_folder}/checkpoint_step_{step}.pt")
    #    torch.save(dict_to_save_2, f"{results_folder}/checkpoint_step_only_unet_{step}.pt")
    elif save_reason == "epoch":
        torch.save(dict_to_save, f"{results_folder}/checkpoint_epoch_{step}.pt")
    #    torch.save(dict_to_save_2, f"{results_folder}/checkpoint_epoch_only_unet_{step}.pt")
    else:
        raise ValueError("Invalid save_reason")


def loss_fn(pred, label):
    logits=-pred.float()
    logits=logits.view(-1,logits.size(-1))
    label=label.view(-1)
    return nn.CrossEntropyLoss()(logits, label)  #pre:float16,label:int64

def triplet_loss(distances, labels, margin=1.0):
    B = distances.shape[0]
    
    row_indices = torch.arange(labels.size(1)).to("cuda:0")
    # 获取每个query样本与其真实类别原型的距离(positive距离)
    positive_dist = distances[row_indices, labels]
    
    # 获取negative distance（每个样本与其他类别prototype的距离的最小值）
    distance_matrix_masked = distances.clone()
    distance_matrix_masked[row_indices, labels] = float('inf')
    negative_dist, _ = torch.min(distance_matrix_masked, dim=1)
    
    margin_tensor = torch.tensor([margin] * labels.size(1)).to("cuda:0") # 广播成长度为5的tensor

    # 计算triplet loss
    triplet_loss = torch.max(positive_dist - negative_dist + margin_tensor, torch.zeros_like(margin_tensor).to('cuda:0'))
    loss = triplet_loss.mean()
    
    return loss.mean()

def loss_with_contrastive(distances, labels, alpha, margin):
    """
    Args:
        distances: [B,5,5] 
        labels: [B,5] 每个batch中包含[0,1,2,3,4]
    """
    with torch.amp.autocast("cuda"):
        all_loss = torch.tensor(0.0).to("cuda:0")
        all_loss_list = []
        for i in range(distances.size(1)):
        # CrossEntropy损失
            scale_dis = distances[:, i, :]
            logits = -scale_dis  # [B,5,5]
            ce_loss = nn.CrossEntropyLoss()(logits, labels.reshape(-1))
            
            # Triplet loss
            trip_loss = triplet_loss(scale_dis, labels, margin)
            loss = ce_loss + alpha * trip_loss
            all_loss += loss
            all_loss_list.append(loss.detach())
        all_loss = all_loss/distances.size(1)
        return all_loss, all_loss_list


#////////////////////////////////////////////////////////////////////////////////////////////////////////
def load_checkpoint(config, diffusion_extractor, aggregation_network, classification_model=None, optimizer=None):

    if not os.path.exists(config["checkpoint_path"]):
        print('file_folder_not_exist')
    checkpoint = torch.load(config["checkpoint_path"], map_location="cuda" if torch.cuda.is_available() else "cpu")
    if "diffusion_unet" in checkpoint:
        diffusion_extractor.unet.load_state_dict(checkpoint["diffusion_unet"])
    if "scale_fusion_network" in checkpoint and checkpoint["scale_fusion_network"]!=None:
        aggregation_network.load_state_dict(checkpoint["scale_fusion_network"])
    if "classification_model" in checkpoint and checkpoint["classification_model"]!=None:
        classification_model.load_state_dict(checkpoint["classification_model"])

    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print(f"Checkpoint loaded from '{config['checkpoint_path']}'")

def get_DDIM_loss(config,diffusion_extractor, imgs, labels = None, control_net_pic=None, control_net_scale = 0.0):
    #with torch.autocast("cuda"): 
    guidance_scale = config.get('ddim_guidance_scale', -1)
    loss = diffusion_extractor.forward_for_DDIM_loss(config, imgs, labels, guidance_scale, control_net_pic = control_net_pic, control_net_scale = control_net_scale) #此处_即为每一个时间步中对于x0的预测，理论上可以在这里计算loss
    return loss

def freeze_classification(classification_model):
        # 冻结前几层的参数 (例如 conv1, bn1, layer1 和 layer2)
    whole_num = len(list(classification_model.named_parameters()))
    freeze_ratio = 0.5
    freeze_num = whole_num * freeze_ratio
    cnt = 0
    for name, param in classification_model.named_parameters():
        cnt += 1
        if cnt >= freeze_num:
            param.requires_grad = False
            
def generate_image(config,diffusion_extractor, imgs, labels = None):
    #with torch.autocast("cuda"): 
    guidance_scale = config.get('ddim_guidance_scale', -1)
    pic = diffusion_extractor.generate_images(config, imgs, labels, guidance_scale) #此处_即为每一个时间步中对于x0的预测，理论上可以在这里计算loss
    return pic

def get_feat1(config, diffusion_extractor, imgs, transfer_mode = False, step = 0, manually_visualize = False, validate_mode=False, control_image = None, control_net = None, control_net_scale = 0.0):
    with torch.autocast("cuda"):
        feats, x0, vae_changed_latets = diffusion_extractor.forward(imgs, transfer_mode = transfer_mode, validate_mode = validate_mode, control_image=control_image, control_net_scale = control_net_scale) #此处_即为每一个时间步中对于x0的预测，理论上可以在这里计算loss，维度与时间步的数目相等
        
        with torch.no_grad():
            ret_feats = []
            for image_feats in feats:
                temp = diffusion_extractor.latents_to_images_for_feats(image_feats)
                #temp = torch.tensor(temp).permute(0,3,1,2).to(feats.device)
                ret_feats.append(temp)
                torch.cuda.empty_cache()
            ret_feats = torch.stack(ret_feats)
        
        with torch.no_grad():
            if config["visualize_hyperfeatures"] and manually_visualize: #self
                torch.cuda.empty_cache()
                i = 0
                for k in range(len(imgs)):
                    tensor = imgs[k] # Shape: [3, 200, 200]

                    # 将张量从 [C, H, W] 转换为 [H, W, C]
                    tensor = tensor.permute(1, 2, 0)  # Shape: [200, 200, 3]

                    # 转换为 numpy 数组，并将数据范围从 [0, 1] 变为 [0, 255]
                    array = (tensor.to("cpu").numpy() * 255).astype(np.uint8)

                    # 创建 PIL 图像
                    image = Image.fromarray(array)

                    # Resize 到 256*256
                    resized_image = image.resize((256, 256), Image.BICUBIC)

                    # 保存图像
                    resized_image.save(f"test_visualize/original_image_{k}.png")
                
                images = diffusion_extractor.latents_to_images_all(vae_changed_latets)
                torch.cuda.empty_cache()
                for j in range(len(images)):
                    images[j].save(f"test_visualize/only_vae_pic_{j}.png")
                torch.cuda.empty_cache()
                for imagex0 in x0:
                    images = diffusion_extractor.latents_to_images_all(imagex0)
                    for j in range(len(images)):
                        images[j].save(f"test_visualize/pic_{j}_step_{i}.png")
                    torch.cuda.empty_cache()
                    i+=1
                
                del image, images, imagex0
        
    return ret_feats

def get_pic(diffusion_extractor, num_generate_steps, prompt = None, guidance_scale = -1):
    x=diffusion_extractor.generate_images(prompt = prompt, num_inference_steps = num_generate_steps, guidance_scale = -1)
    save_path='generate_image'
    i=0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    images = diffusion_extractor.latents_to_images_for_generate_pic(x)
    for i,image in enumerate(images):
        image.save(f"{save_path}/{i}.png")
        i+=1

def load_models_NEU(config_path, inner_config, few_shot_mode=True, only_few_shot=False, debug_sep=False, only_baseline=False, vae_decode_baseline=False):
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    config = inner_config|config
    device = config.get("device", "cuda")
    if not only_few_shot:
        diffusion_extractor = DiffusionExtractor(config, device)
    else:
        diffusion_extractor = None
    if config.get("flip_timesteps", False):
        config["save_timestep"] = config["save_timestep"][::-1]
    num_classes = config.get("num_classes")
    if few_shot_mode:
        if only_few_shot:
            input_channel = 3
        else:
            if debug_sep:
                if only_baseline or vae_decode_baseline:
                    input_channel = 3
                else:
                    input_channel = 4
            else:
                input_channel = 3
        classification_model = get_only_few_shot_whole_model(num_classes, input_channel)
    else:
        classification_model = res2net50_v1b(False,config["num_classes"])
    scale_num = len(config["save_timestep"])*len(config["idxs"])
    if not config['only_few_shot']:
        scale_fusion_network = simple_CNN_grad(scale_num).cuda().half()
    else:
        scale_fusion_network = None
    return config, diffusion_extractor, scale_fusion_network, classification_model

def validate_only_few_shot(config, diffusion_extractor, classification_model, val_dataloader, alpha, margin, step = 0, writer = None):
    device = config.get("device", "cuda")
    classification_model.eval()  # Set classification model to evaluation mode
    
    scale = len(config["save_timestep"])*len(config["idxs"])
    val_loss = 0.0
    correct_predictions = [0] *  scale
    total_samples = [0] * scale
    lambda_weight = config.get("lambda_weight", 0.01)
    accumulation_steps = config.get("accumulation_steps", 4)
    val_ddim_loss=0.0
    with torch.no_grad():
            accumulated_cls_loss = 0.0
            accumulated_ddim_loss = 0.0
            batch_count = 0
            voting_total = 0
            voting_correct = 0
            #for batch_idx, (support_images, support_labels, query_images, query_labels, _, __) in enumerate(tqdm(val_dataloader)):
            for batch_idx, (support_images, support_labels, query_images, query_labels, support_class_names, query_class_names) in enumerate(tqdm(val_dataloader)):
                support_images = support_images.squeeze(0).to(device)
                support_labels = support_labels.to(device)
                query_images = query_images.squeeze(0).to(device)
                query_labels = query_labels.to(device)

                if config["classification"]:
                    with torch.autocast(device_type='cuda'):
                        #pred_support = support_images.unsqueeze(1)
                        #pred_query = query_images.unsqueeze(1)
                        vis = voting_total == 0

                        support_images = support_images.unsqueeze(1)
                        query_images = query_images.unsqueeze(1)
                        #support_images = diffusion_extractor.forward_only_use_vae(support_images).unsqueeze(1)
                        #query_images = diffusion_extractor.forward_only_use_vae(query_images).unsqueeze(1)
                        classification_result = classification_model(support_images, support_labels, query_images)
                        #classification_result = classification_result.unsqueeze(1)
                        classification_loss, loss_list = loss_with_contrastive(classification_result, query_labels,alpha,margin)
                        accumulated_cls_loss += classification_loss.item()

                    with torch.no_grad():
                        predicted_labels_list = []
                        confidence_list = []  # 存储每个预测的置信度分数

                        for i in range(classification_result.size(1)):
                            # 获取距离并转换为概率
                            distances = classification_result[:, i, :]
                            probabilities = F.softmax(-distances, dim=1)  # 负号是因为更小的距离意味着更高的概率
                            confidence, predicted_labels = torch.max(probabilities, dim=1)
                            
                            predicted_labels_list.append(predicted_labels)
                            confidence_list.append(confidence)
                            
                            correct_predictions[i] += (predicted_labels == query_labels).sum().item()
                            total_samples[i] += query_labels.size(1)

                        # 堆叠预测结果和置信度
                        stacked_predictions = torch.stack(predicted_labels_list, dim=0)  # [num_timesteps, batch_size]
                        stacked_confidences = torch.stack(confidence_list, dim=0)  # [num_timesteps, batch_size]
                        
                        # 为每个样本计算加权预测
                        vote_results = []
                        for sample_idx in range(stacked_predictions.shape[1]):
                            sample_predictions = stacked_predictions[:, sample_idx]  # [num_timesteps]
                            sample_confidences = stacked_confidences[:, sample_idx]  # [num_timesteps]
                            
                            # 创建张量存储每个类别的加权投票
                            num_classes = 5  # 假设有5个类别，根据实际情况调整
                            weighted_votes = torch.zeros(num_classes, device=query_labels.device)
                            
                            # 累积每个类别的置信度加权投票
                            for pred, conf in zip(sample_predictions, sample_confidences):
                                weighted_votes[pred] += conf
                            
                            # 获取加权投票最高的类别
                            final_prediction = torch.argmax(weighted_votes)
                            vote_results.append(final_prediction)
                        
                        # 转换投票结果为tensor并计算准确率
                        voted_predictions = torch.tensor(vote_results, device=query_labels.device)
                        voting_correct += (voted_predictions == query_labels).sum().item()
                        voting_total += query_labels.size(1)

                    del classification_result
                    #torch.cuda.empty_cache()


                batch_count += 1
                
                # 每accumulation_steps批次记录一次平均损失
                if (batch_idx + 1) % accumulation_steps == 0:
                    val_loss += accumulated_cls_loss / accumulation_steps
                    val_ddim_loss += accumulated_ddim_loss / accumulation_steps
                    
                    if config["use_wandb"]:
                        wandb.log({
                            "val/classification_loss": accumulated_cls_loss / accumulation_steps if config["classification"] else None,
                        })
                        if config["classification"]:
                            for i in range(scale):
                                wandb.log({
                                    f"val/classification_accuracy_{i}": correct_predictions[i] / total_samples[i] if total_samples[i] > 0 else None
                                })
                            wandb.log({"voting_acc_val":voting_correct/voting_total})
                    else:
                        # TensorBoard 只在有值时 add_scalar
                        if config["classification"]:
                            writer.add_scalar(
                                "val/classification_loss",
                                accumulated_cls_loss / accumulation_steps,
                                step
                            )
                        if config["train_diffusion_outside"]:
                            writer.add_scalar(
                                "val/ddim_loss",
                                accumulated_ddim_loss / accumulation_steps,
                                step
                            )
                        if config["classification"]:
                            for i in range(scale):
                                if total_samples[i] > 0:
                                    writer.add_scalar(
                                        f"val/classification_accuracy_{i}",
                                        correct_predictions[i] / total_samples[i],
                                        step
                                    )
                            writer.add_scalar(
                                "voting_acc_val",
                                voting_correct / voting_total,
                                step
                            )
                    
                    
                    accumulated_cls_loss = 0.0
                    accumulated_ddim_loss = 0.0
                step += 1

            # 计算整体平均损失和准确率
            avg_cls_loss = val_loss / (batch_count // accumulation_steps)
            avg_ddim_loss = val_ddim_loss / (batch_count // accumulation_steps)
            accuracy = 0
            scale_use = 0
            for i in range(scale):
                if total_samples[i] <= 0:
                    continue
                scale_use += 1
                accuracy += correct_predictions[i] / total_samples[i]
            accuracy = accuracy/scale_use
            # 最终日志记录
            if config["use_wandb"]:
                wandb.log({
                    "val/average_classification_loss": avg_cls_loss,
                    "val/final_accuracy": accuracy
                })
            
            print(f"Validation Classification Loss: {avg_cls_loss:.4f}")
            print(f"Validation Accuracy: {accuracy:.4f}")

    # 恢复训练模式
    classification_model.train()
    return step

class trainer_class():
    def __init__(self, config, writer, inner_config):
        self.config = config
        self.writer = writer
        self.device = config.get("device", "cuda")
        self.scale = len(config["save_timestep"])*len(config["idxs"])
        self.step = 0
        self.transfer_mode = False
        self.few_shot_mode = inner_config["few_shot_mode"]

    def validate(self, config, diffusion_extractor, scale_fusion_network, classification_model, val_dataloader, alpha, margin):
        few_shot = True
        classification_model.eval()  # Set classification model to evaluation mode
        val_loss = 0.0
        correct_predictions = [0] *  self.scale
        total_samples = [0] * self.scale
        lambda_weight = config.get("lambda_weight", 0.01)
        accumulation_steps = config.get("accumulation_steps", 4)
        val_ddim_loss=0.0
        voting_total = 0
        voting_correct = 0
        config_classification=config.get('classification',True)
        with torch.no_grad():
            if not few_shot:
                raise ValueError
            else:
                accumulated_cls_loss = 0.0
                accumulated_ddim_loss = 0.0
                batch_count = 0
                #for batch_idx, (support_images, support_labels, query_images, query_labels, _, __) in enumerate(tqdm(val_dataloader)):
                for batch_idx, (support_images, support_labels, query_images, query_labels, support_class_names, query_class_names) in enumerate(tqdm(val_dataloader)):
                    if config["transfer_learning"] and batch_idx >= accumulation_steps + 1:
                        break
                    support_images = support_images.squeeze(0).to(self.device)
                    support_labels = support_labels.to(self.device)
                    query_images = query_images.squeeze(0).to(self.device)
                    query_labels = query_labels.to(self.device)

                    support_images_ava = create_control_images(support_images)
                    #visualize_batch_results(support_images, support_images_ava)
                    query_images_ava = create_control_images(query_images)
                    if self.transfer_mode:
                        control_net_scale = self.val_control_net_scale
                    else:
                        control_net_scale = min(1.0, (self.step - self.config_classification_wait_step) / 10000)

                    if config_classification:
                        with torch.autocast(device_type='cuda'):
                            if False:
                                if config["only_baseline"]:
                                    pred_support = support_images.unsqueeze(1)
                                    pred_query = query_images.unsqueeze(1)
                                elif config["vae_baseline"] or config["vae_decode_baseline"]:
                                    pred_support = diffusion_extractor.forward_only_use_vae(support_images).unsqueeze(1)
                                    pred_query = diffusion_extractor.forward_only_use_vae(query_images).unsqueeze(1)
                                    if config["vae_decode_baseline"]:
                                        pred_support = pred_support.squeeze(1)
                                        pred_query = pred_query.squeeze(1)
                                        pred_support = diffusion_extractor.latents_to_images_for_feats(pred_support).unsqueeze(1)
                                        pred_query = diffusion_extractor.latents_to_images_for_feats(pred_query).unsqueeze(1)
                                else:
                                    raise ValueError("debug_sep_True_without_spc")
                            else:
                                pred_support = get_feat1(config, diffusion_extractor, support_images, transfer_mode=False, step=self.step, validate_mode=False, control_image=support_images_ava, control_net_scale=control_net_scale)
                                torch.cuda.empty_cache()
                                pred_query = get_feat1(config, diffusion_extractor, query_images, transfer_mode=False, step=self.step,validate_mode=False, control_image=query_images_ava, control_net_scale=control_net_scale)
                            classification_result = classification_model(pred_support, support_labels, pred_query)
                            #classification_result = classification_result.unsqueeze(1)
                            classification_loss, loss_list = loss_with_contrastive(classification_result, query_labels,alpha,margin)
                            accumulated_cls_loss += classification_loss
                        if config["use_wandb"]:
                                for query_idx, loss_tensor in enumerate(loss_list):
                                    wandb.log({
                                        f"query_{query_idx}_loss": loss_tensor.item()
                                    }, step=self.step)
                        else:
                            for query_idx, loss_tensor in enumerate(loss_list):
                                self.writer.add_scalar(f"query_{query_idx}_loss", loss_tensor.item(), self.step)

                        with torch.no_grad():
                            predicted_labels_list = []
                            confidence_list = []  # 存储每个预测的置信度分数

                            for i in range(classification_result.size(1)):
                                # 获取距离并转换为概率
                                distances = classification_result[:, i, :]
                                probabilities = F.softmax(-distances, dim=1)  # 负号是因为更小的距离意味着更高的概率
                                confidence, predicted_labels = torch.max(probabilities, dim=1)
                                
                                predicted_labels_list.append(predicted_labels)
                                confidence_list.append(confidence)
                                
                                correct_predictions[i] += (predicted_labels == query_labels).sum().item()
                                total_samples[i] += query_labels.size(1)

                            # 堆叠预测结果和置信度
                            stacked_predictions = torch.stack(predicted_labels_list, dim=0)  # [num_timesteps, batch_size]
                            stacked_confidences = torch.stack(confidence_list, dim=0)  # [num_timesteps, batch_size]
                            
                            # 为每个样本计算加权预测
                            with torch.no_grad():        
                                # 对每个样本进行投票
                                vote_results = []
                                for sample_idx in range(stacked_predictions.shape[1]):
                                    sample_predictions = stacked_predictions[:, sample_idx]  # [num_timesteps]
                                    sample_confidences = stacked_confidences[:, sample_idx]  # [num_timesteps]
                                    
                                    # Create a tensor to store weighted votes for each class
                                    num_classes = config["num_classes"]    # Assuming 5 classes, adjust if different
                                    weighted_votes = torch.zeros(num_classes, device=query_labels.device)
                                    
                                    # Accumulate confidence-weighted votes for each class
                                    for pred, conf in zip(sample_predictions, sample_confidences):
                                        weighted_votes[pred] += conf
                                    
                                    # Get the class with highest weighted votes
                                    final_prediction = torch.argmax(weighted_votes)
                                    vote_results.append(final_prediction)
                                
                                # 转换投票结果为tensor并计算准确率
                                voted_predictions = torch.tensor(vote_results, device=query_labels.device)
                                voting_correct += (voted_predictions == query_labels).sum().item()
                                voting_total += query_labels.size(1)
                        pred_support_control = pred_support.detach()
                        pred_query_control = pred_query.detach()
                        del pred_support, pred_query, classification_result
                        torch.cuda.empty_cache()

                    # DDIM验证损失计算
                    if config["train_diffusion_outside"]:
                        with torch.autocast(device_type='cuda'):
                            if config["use_control_net"] and config_classification:
                                alpha = 0.5 * (1 + np.cos(np.pi * self.step / self.total_steps))
                                query_grad_map = cal_grad(classification_model, scale_fusion_network, pred_support_control, support_labels, pred_query_control, query_labels, stacked_confidences, config["control_net_confidence_threshold"])
                                query_grad_map = scale_fusion_network(query_grad_map)
                                query_grad_map = alpha * query_grad_map + (1-alpha) * query_images_ava
                                support_grad_map = cal_grad(classification_model, scale_fusion_network, pred_support_control, support_labels, pred_support_control, support_labels, stacked_confidences, config["control_net_confidence_threshold"])
                                support_grad_map = scale_fusion_network(support_grad_map)
                                support_grad_map = alpha * support_grad_map + (1-alpha) * support_images_ava

                            else:      
                                    query_grad_map = torch.zeros_like(query_images, dtype=torch.float16)
                                    support_grad_map = torch.zeros_like(support_images, dtype=torch.float16)
                                    control_net_scale = 0.0
                            # 支持集的DDIM损失
                            support_ddim_loss = get_DDIM_loss(config, diffusion_extractor, support_images, support_class_names, control_net_pic=support_grad_map, control_net_scale=control_net_scale)
                            if not (torch.isnan(support_ddim_loss) or torch.isinf(support_ddim_loss)):
                                accumulated_ddim_loss += lambda_weight * support_ddim_loss.item()*config["train_ddim_range"]
                            
                            # 查询集的DDIM损失
                            query_ddim_loss = get_DDIM_loss(config, diffusion_extractor, query_images, query_class_names, control_net_pic = query_grad_map, control_net_scale=control_net_scale)
                            if not (torch.isnan(query_ddim_loss) or torch.isinf(query_ddim_loss)):
                                accumulated_ddim_loss += lambda_weight * query_ddim_loss.item()*config["train_ddim_range"]
                            self.step += 1 

                        del support_ddim_loss, query_ddim_loss
                        #torch.cuda.empty_cache()

                    batch_count += 1
                    
                    # 每accumulation_steps批次记录一次平均损失
                    if (batch_idx + 1) % accumulation_steps == 0:
                        val_loss += accumulated_cls_loss / accumulation_steps
                        val_ddim_loss += accumulated_ddim_loss / accumulation_steps
                        
                        if config["use_wandb"]:
                            wandb.log({
                                "val/classification_loss": accumulated_cls_loss / accumulation_steps if config["classification"] else None,
                                "val/ddim_loss": accumulated_ddim_loss / accumulation_steps if config["train_diffusion_outside"] else None,
                            })
                            if config["classification"]:
                                for i in range(self.scale):
                                    wandb.log({
                                        f"val/classification_accuracy_{i}": correct_predictions[i] / total_samples[i] if total_samples[i] > 0 else None
                                    })
                                wandb.log({"voting_acc_val":voting_correct/voting_total})
                        
                        accumulated_cls_loss = 0.0
                        accumulated_ddim_loss = 0.0
                    else:
                        # TensorBoard 只在有值时 add_scalar
                        if config["classification"]:
                            self.writer.add_scalar(
                                "val/classification_loss",
                                accumulated_cls_loss / accumulation_steps,
                                self.step
                            )
                        if config["train_diffusion_outside"]:
                            self.writer.add_scalar(
                                "val/ddim_loss",
                                accumulated_ddim_loss / accumulation_steps,
                                self.step
                            )
                        if config["classification"]:
                            for i in range(self.scale):
                                if total_samples[i] > 0:
                                    self.writer.add_scalar(
                                        f"val/classification_accuracy_{i}",
                                        correct_predictions[i] / total_samples[i],
                                        self.step
                                    )
                            self.writer.add_scalar(
                                "voting_acc_val",
                                voting_correct / voting_total,
                                self.step
                            )

                # 计算整体平均损失和准确率
                avg_cls_loss = val_loss / (batch_count // accumulation_steps)
                avg_ddim_loss = val_ddim_loss / (batch_count // accumulation_steps)
                accuracy = 0
                scale_use = 0
                for i in range(self.scale):
                    if total_samples[i] <= 0:
                        continue
                    scale_use += 1
                    accuracy += correct_predictions[i] / total_samples[i]
                accuracy = accuracy/scale_use
                # 最终日志记录
                if config["use_wandb"]:
                    wandb.log({
                        "val/average_classification_loss": avg_cls_loss,
                        "val/average_ddim_loss": avg_ddim_loss if config["train_diffusion_outside"] else None,
                        "val/final_accuracy": accuracy
                    })
                else:
                    self.writer.add_scalar(
                        f"val/average_classification_loss",
                        avg_cls_loss
                        )
                    self.writer.add_scalar(
                        f"val/final_accuracy",
                        accuracy
                        )
                
                print(f"Validation Classification Loss: {avg_cls_loss:.4f}")
                if config["train_diffusion_outside"]:
                    print(f"Validation DDIM Loss: {avg_ddim_loss:.4f}")
                print(f"Validation Accuracy: {accuracy:.4f}")

        # 恢复训练模式
        classification_model.train()

    def train_NEU(self, config, diffusion_extractor, scale_fusion_network, classification_model,  optimizer,scheduler, train_dataloader, val_dataloader, transfer_dataloader = None):
        device = config.get("device", "cuda")
        max_epochs = config["max_epochs"]
        accumulation_steps = config.get("accumulation_steps", 4)
        optimizer.zero_grad()
        accumulated_cls_loss = 0
        accumulated_ddim_loss = 0.0  # 用于累积DDIM损失
        lambda_weight=config.get('lambda_weight',0.01)
        config_classification=config.get('classification',True)
        self.config_classification_wait_step = config.get('classification_wait_step', 0)
        flag_begin_classification = False
        config_train_ddim_range=config.get('train_ddim_range',4)
        locked_para = None
        unlocked_para = None
        freeze_ratio = config.get("freeze_ratio", 0.95)
        lambda_cls = config.get("lambda_cls", 1)
        lambda_transfer_weight = config.get("lambda_transfer_weight", 0.01)
        transfer_dataset_domain = config.get("transfer_dataset")
        len_train_dataloader = len(train_dataloader)
        len_val_dataloader = len(val_dataloader)
        self.total_steps = max_epochs * len_train_dataloader + max_epochs // config["val_every_n_epoch"] * len_val_dataloader
        for epoch in range(max_epochs):
            transfer_step = 0
            if transfer_dataloader != None:
                transfer_dataloader_iter = iter(transfer_dataloader)
                transfer_len = len(transfer_dataloader)
            correct_predictions = [0] *  (len(config["save_timestep"])*len(config["idxs"]))
            total_samples = [0] * (len(config["save_timestep"])*len(config["idxs"]))
            voting_total = 0
            voting_correct = 0
            
            if self.few_shot_mode == False:
                raise ValueError
                    
            elif self.few_shot_mode == True:
                accumulated_cls_loss = 0.0
                accumulated_ddim_loss = 0.0
                alpha=config.get("alpha", 0.5)
                margin=config.get("margin",1.0)
                
                for batch_idx, (support_images, support_labels, query_images, query_labels, support_class_names, query_class_names) in enumerate(tqdm(train_dataloader)):
                
                    support_images = support_images.squeeze(0).to(device)
                    support_labels = support_labels.to(device)
                    query_images = query_images.squeeze(0).to(device)
                    query_labels = query_labels.to(device)
                    
                    support_images_ava = create_control_images(support_images).detach()
                    #visualize_batch_results(support_images, support_images_ava)
                    query_images_ava = create_control_images(query_images).detach()
                    
                    if self.step >= self.config_classification_wait_step:
                        flag_begin_classification = True
                    if config_classification and flag_begin_classification:
                        #print_message_if_all_requires_grad_false(diffusion_extractor.unet)
                        control_net_scale = min(1.0, (self.step - self.config_classification_wait_step) / 10000)  # 前10000步线性增长
                        locked_para = lock_non_upblock_gradients_NEU(diffusion_extractor.unet, freeze_ratio=freeze_ratio, step = self.step)
                        # 分类任务部分
                        with torch.autocast(device_type='cuda'):
                            #pred_support, pred_query形状都为[5,384,64,64]
                            #原图为[5,3,200,200]
                            if config["debug_sep"]:
                                if config["only_baseline"]:
                                    pred_support = support_images.unsqueeze(1)
                                    pred_query = query_images.unsqueeze(1)
                                elif config["vae_baseline"] or config["vae_decode_baseline"]:
                                    pred_support = diffusion_extractor.forward_only_use_vae(support_images).unsqueeze(1)
                                    pred_query = diffusion_extractor.forward_only_use_vae(query_images).unsqueeze(1)
                                    if config["vae_decode_baseline"]:
                                        pred_support = pred_support.squeeze(1)
                                        pred_query = pred_query.squeeze(1)
                                        pred_support = diffusion_extractor.latents_to_images_for_feats(pred_support).unsqueeze(1)
                                        pred_query = diffusion_extractor.latents_to_images_for_feats(pred_query).unsqueeze(1)
                                else:
                                    raise ValueError("debug_sep_True_without_spc")
                            else:
                                pred_support = get_feat1(config, diffusion_extractor, support_images, transfer_mode=False, step=self.step, validate_mode=False, control_image=support_images_ava, control_net_scale=control_net_scale)
                                torch.cuda.empty_cache()
                                pred_query = get_feat1(config, diffusion_extractor, query_images, transfer_mode=False, step=self.step,validate_mode=False, control_image=query_images_ava, control_net_scale=control_net_scale)
                            classification_result = classification_model(pred_support, support_labels, pred_query)
                            #classification_result = classification_result.unsqueeze(1)
                            classification_loss, loss_list = loss_with_contrastive(classification_result, query_labels,alpha,margin)
                            # 缩放损失以匹配累积步数
                            scaled_cls_loss = classification_loss * lambda_cls / accumulation_steps
                            # 反向传播
                            scaled_cls_loss.backward()
                            accumulated_cls_loss += classification_loss
                            #print_message_if_all_requires_grad_false(diffusion_extractor.unet)
                            # 记录分类准确率
                            if config["use_wandb"]:
                                for query_idx, loss_tensor in enumerate(loss_list):
                                    wandb.log({
                                        f"query_{query_idx}_loss": loss_tensor.item()
                                    }, step=self.step)
                            else:
                                for query_idx, loss_tensor in enumerate(loss_list):
                                    self.writer.add_scalar(f"query_{query_idx}_loss", loss_tensor.item(), self.step)
                                    
                            with torch.no_grad():
                                predicted_labels_list = [] 
                                confidence_list = []  # Store confidence scores for each prediction
                                
                                for i in range(classification_result.size(1)):
                                    # Get distances and convert to probabilities using softmax
                                    distances = classification_result[:, i, :]
                                    probabilities = F.softmax(-distances, dim=1)  # Negative because smaller distance means higher probability
                                    confidence, predicted_labels = torch.max(probabilities, dim=1)
                                    
                                    predicted_labels_list.append(predicted_labels)
                                    confidence_list.append(confidence)
                                    
                                    correct_predictions[i] += (predicted_labels == query_labels).sum().item()
                                    total_samples[i] += query_labels.size(1)
                                # 使用已计算的predicted_labels进行投票
                                stacked_predictions = torch.stack(predicted_labels_list, dim=0)  # [num_timesteps, batch_size]
                                stacked_confidences = torch.stack(confidence_list, dim=0).detach()  # [num_timesteps, batch_size]
                            with torch.no_grad():        
                                if config["log_train_accuracy"]: 
                                    # 对每个样本进行投票
                                    vote_results = []
                                    for sample_idx in range(stacked_predictions.shape[1]):
                                        sample_predictions = stacked_predictions[:, sample_idx]  # [num_timesteps]
                                        sample_confidences = stacked_confidences[:, sample_idx]  # [num_timesteps]
                                        
                                        # Create a tensor to store weighted votes for each class
                                        num_classes = config["num_classes"]    # Assuming 5 classes, adjust if different
                                        weighted_votes = torch.zeros(num_classes, device=query_labels.device)
                                        
                                        # Accumulate confidence-weighted votes for each class
                                        for pred, conf in zip(sample_predictions, sample_confidences):
                                            weighted_votes[pred] += conf
                                        
                                        # Get the class with highest weighted votes
                                        final_prediction = torch.argmax(weighted_votes)
                                        vote_results.append(final_prediction)
                                    
                                    # 转换投票结果为tensor并计算准确率
                                    voted_predictions = torch.tensor(vote_results, device=query_labels.device)
                                    voting_correct += (voted_predictions == query_labels).sum().item()
                                    voting_total += query_labels.size(1)
                        pred_support_control = pred_support.detach()
                        pred_query_control = pred_query.detach()
                        # 清理内存
                        del classification_result,classification_loss, pred_support, pred_query
                        torch.cuda.empty_cache()

                    # DDIM损失部分（如果需要）
                    if config["train_diffusion_outside"]:

                        if locked_para != None:
                            _ = 0
                            restore_lora_gradients_NEU(diffusion_extractor.unet, locked_para)
                        if unlocked_para != None:
                            relock_gradients_NEU(diffusion_extractor.unet, unlocked_para)
                        ddim_support_images = support_images.detach()
                        ddim_query_images = query_images.detach()
                        ddim_support_class_names = [[""] for i in range(len(support_class_names))]
                        ddim_query_class_names = [[""] for i in range(len(query_class_names))]
                        for i in range(config_train_ddim_range):
                            with torch.autocast(device_type='cuda'):
                                torch.cuda.empty_cache()
                                if config["use_control_net"] and config_classification and flag_begin_classification:
                                    alpha = 0.5 * (1 + np.cos(np.pi * self.step / self.total_steps))
                                    query_grad_map = cal_grad(classification_model, scale_fusion_network, pred_support_control, support_labels, pred_query_control, query_labels, stacked_confidences, config["control_net_confidence_threshold"]).detach()
                                    query_grad_map = scale_fusion_network(query_grad_map)
                                    if self.step in config["visualize_time_step"]:
                                        visualize_grad_cam(ddim_query_images, query_grad_map, save_path='grad_cam_pic')
                                    query_grad_map = alpha * query_grad_map + (1-alpha) * query_images_ava
                                    support_grad_map = cal_grad(classification_model, scale_fusion_network, pred_support_control, support_labels, pred_support_control, support_labels, stacked_confidences, config["control_net_confidence_threshold"]).detach()
                                    support_grad_map = scale_fusion_network(support_grad_map)
                                    support_grad_map = alpha * support_grad_map + (1-alpha) * support_images_ava
                                else:      
                                    query_grad_map = torch.zeros_like(query_images, dtype=torch.float16)
                                    support_grad_map = torch.zeros_like(support_images, dtype=torch.float16)
                                    control_net_scale = 0.0
                                torch.cuda.empty_cache()
                                
                                DDIM_loss = get_DDIM_loss(config, diffusion_extractor, ddim_support_images, support_class_names, control_net_pic=support_grad_map, control_net_scale=control_net_scale)
                                if torch.isnan(DDIM_loss) or torch.isinf(DDIM_loss) or DDIM_loss==0.0:
                                    raise ValueError(f"{DDIM_loss} detected in step {self.step} ddim_range")
                                torch.cuda.empty_cache()
                                scaled_ddim_loss = lambda_weight*DDIM_loss / accumulation_steps
                                scaled_ddim_loss.backward()
                                accumulated_ddim_loss += scaled_ddim_loss.item()
                                torch.cuda.empty_cache()
                                DDIM_loss=get_DDIM_loss(config, diffusion_extractor,ddim_query_images, query_class_names, control_net_pic = query_grad_map, control_net_scale=control_net_scale)
                                if torch.isnan(DDIM_loss) or torch.isinf(DDIM_loss) or DDIM_loss==0.0:
                                    raise ValueError("{DDIM_loss} detected in {batched_idx}th batch {i} ddim_range")
                                torch.cuda.empty_cache()
                                scaled_ddim_loss = lambda_weight*DDIM_loss / accumulation_steps
                                scaled_ddim_loss.backward()
                                accumulated_ddim_loss += scaled_ddim_loss
                                
                                torch.cuda.empty_cache()
                                
                                if transfer_dataloader != None:
                                    try:
                                        transfer_support_images, _, transfer_query_images, _, _, _ = next(transfer_dataloader_iter)
                                    except StopIteration:
                                        transfer_dataloader_iter = iter(transfer_dataloader)
                                        transfer_support_images, _, transfer_query_images, _, _, _ = next(transfer_dataloader_iter)
                                    transfer_support_images = transfer_support_images.squeeze(0).to(self.device)
                                    transfer_query_images = transfer_query_images.squeeze(0).to(self.device)
                                    transfer_class_names = [[transfer_dataset_domain] for i in range(transfer_support_images.size(0))]
                                    DDIM_loss = get_DDIM_loss(config, diffusion_extractor, transfer_support_images, transfer_class_names)
                                    scaled_ddim_loss = lambda_transfer_weight * DDIM_loss / accumulation_steps
                                    scaled_ddim_loss.backward()
                                    accumulated_ddim_loss += scaled_ddim_loss

                                    DDIM_loss = get_DDIM_loss(config, diffusion_extractor, transfer_query_images, transfer_class_names)
                                    scaled_ddim_loss = lambda_transfer_weight*DDIM_loss / accumulation_steps
                                    scaled_ddim_loss.backward()
                                    accumulated_ddim_loss += scaled_ddim_loss
                                    #del DDIM_loss
                                    #torch.cuda.empty_cache()
                            
                            if self.step in config["visualize_time_step"]:
                                with torch.no_grad():
                                    _ = get_feat1(config, diffusion_extractor, query_images, manually_visualize = True, transfer_mode=False, step=self.step,validate_mode=False, control_image=query_images_ava)
                                    del _
                                                
                    # 达到累积步数后更新参数
                    if (batch_idx + 1) % accumulation_steps == 0:
                        # 更新参数
                        if diffusion_extractor is not None:
                            torch.nn.utils.clip_grad_norm_(diffusion_extractor.unet.parameters(), max_norm=10)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                            
                        # 记录日志
                        if config["use_wandb"]:
                            current_lr = optimizer.param_groups[1]['lr']
                            log_dict = {
                                "classification/loss": accumulated_cls_loss / accumulation_steps if config_classification and flag_begin_classification else None,
                                "ddim/loss": accumulated_ddim_loss if config["train_diffusion_outside"] else None,
                                "learning_rate_cls": current_lr
                            }
                            if config_classification and flag_begin_classification:
                                for i in range(len(correct_predictions)):
                                    if total_samples[i] <= 0:
                                        continue
                                    log_dict[f"train/accuracy_{i}"] = correct_predictions[i]/total_samples[i]
                                log_dict["voting_acc"] = voting_correct/voting_total
                            wandb.log(log_dict, step=self.step)
                        else:
                            current_lr = optimizer.param_groups[1]['lr']
                            # 跟 wandb 一样，TensorBoard 只记录非 None 项
                            if config_classification and flag_begin_classification:
                                self.writer.add_scalar("classification/loss", accumulated_cls_loss / accumulation_steps, self.step)
                            if config["train_diffusion_outside"]:
                                self.writer.add_scalar("ddim/loss", accumulated_ddim_loss, self.step)
                            self.writer.add_scalar("learning_rate_cls", current_lr, self.step)
                            if config_classification and flag_begin_classification:
                                for i in range(len(correct_predictions)):
                                    if total_samples[i] <= 0:
                                        continue
                                    self.writer.add_scalar(f"train/accuracy_{i}", correct_predictions[i] / total_samples[i], self.step)
                                self.writer.add_scalar("voting_acc", voting_correct / voting_total, self.step)
                        
                        # 重置累积的损失
                        accumulated_cls_loss = 0.0
                        accumulated_ddim_loss = 0.0
                        voting_total = 0
                        voting_correct = 0

                    # 验证和保存检查点
                    if (self.step == config["val_step_once"] or (self.step > 0 and config["val_every_n_steps"] > 0 and self.step % config["val_every_n_steps"] == 0)) and flag_begin_classification:
                        with torch.no_grad():
                            save_model_NEU(config, diffusion_extractor, scale_fusion_network,
                                        classification_model,  optimizer, self.step, save_reason="step")
                        self.validate(config, diffusion_extractor, scale_fusion_network, classification_model,  val_dataloader, alpha, margin)
                        optimizer.zero_grad()
                    
                    self.step += 1
                    transfer_step += 1
                
                if flag_begin_classification and (config["val_every_n_epoch"] > 0 and epoch % config["val_every_n_epoch"] == 0):
                    with torch.no_grad():
                        extra_epoch = 0 if config["checkpoint_epoch"] == "None" else config["checkpoint_epoch"]
                        
                        
                        save_model_NEU(config, diffusion_extractor, scale_fusion_network,
                                        classification_model,  optimizer, epoch+extra_epoch, save_reason="epoch")
                        
                        self.validate(config, diffusion_extractor, scale_fusion_network, classification_model,  val_dataloader,alpha,margin)
                        optimizer.zero_grad()
            if not config["use_wandb"]:
                self.writer.close()

    def transfer(self, config, diffusion_extractor, scale_fusion_network, classification_model, optimizer,  labeled_dataloader, val_dataloader, unlabeled_dataloader, scheduler, writer):
        self.transfer_mode = True
        max_epochs = config["max_epochs"]
        accumulation_steps = config.get("accumulation_steps", 4)
        optimizer.zero_grad()
        accumulated_cls_loss = 0
        config_classification = config.get('classification', True)
        if config["use_pseudo_labels"]:
            unlabeled_dataloader = iter(unlabeled_dataloader)
        max_voting_acc = 0
        transfer_accumulation = config.get("transfer_accumulation_classification")
        lock_all_unet_gradients(diffusion_extractor.unet)
        freeze_classification(classification_model)
        diffusion_extractor.unet.eval()
        classification_model.eval()  # Set classification model to evaluation mode
        
        for epoch in range(max_epochs):
            correct_predictions = [0] *  (len(config["save_timestep"])*len(config["idxs"]))
            total_samples = [0] * (len(config["save_timestep"])*len(config["idxs"]))
            voting_total = 0
            voting_correct = 0
            accumulated_cls_loss = 0.0
            alpha = config.get("alpha", 0.5)
            margin = config.get("margin", 1.0)
            freeze_ratio = config.get("freeze_ratio", 0.95)
            
            # 修改数据加载逻辑，区分有标签和无标签数据
            for batch_idx, batch_data in enumerate(tqdm(labeled_dataloader)):
                # 原始的数据格式
                support_images, support_labels, query_images_labeled, query_labels, \
                support_class_names, query_class_names = batch_data

                # 处理设备转移和维度调整
                support_images = support_images.squeeze(0).to(self.device)
                support_labels = support_labels.to(self.device)
                query_images_labeled = query_images_labeled.squeeze(0).to(self.device)
                query_labels = query_labels.to(self.device)

                support_images_ava = create_control_images(support_images).detach()
                #visualize_batch_results(support_images, support_images_ava)
                query_images_ava = create_control_images(query_images_labeled).detach()
                self.val_control_net_scale = 0.5
                with torch.autocast(device_type='cuda'):
                    # 处理支持集和有标签查询集
                    pred_support = get_feat1(config, diffusion_extractor, support_images, transfer_mode=True, step=self.step, validate_mode=False, control_image=support_images_ava, control_net_scale=self.val_control_net_scale)
                    torch.cuda.empty_cache()
                    pred_query = get_feat1(config, diffusion_extractor, query_images_labeled, transfer_mode=True, step=self.step, validate_mode=False, control_image=query_images_ava, control_net_scale=self.val_control_net_scale)
                    classification_result = classification_model(pred_support, support_labels, pred_query)
                    #classification_result = classification_result.unsqueeze(1)
                    classification_loss, loss_list = loss_with_contrastive(classification_result, query_labels, alpha, margin)
                    # 缩放损失以匹配累积步数
                    total_loss = classification_loss / accumulation_steps
                    total_loss.backward()
                    accumulated_cls_loss += total_loss
                    del total_loss, pred_query, classification_loss
                    torch.cuda.empty_cache()
                    if config["log_train_accuracy"]:
                        with torch.no_grad():
                            predicted_labels_list = [] 
                            confidence_list = []  # Store confidence scores for each prediction
                            
                            for i in range(classification_result.size(1)):
                                # Get distances and convert to probabilities using softmax
                                distances = classification_result[:, i, :]
                                probabilities = F.softmax(-distances, dim=1)  # Negative because smaller distance means higher probability
                                confidence, predicted_labels = torch.max(probabilities, dim=1)
                                
                                predicted_labels_list.append(predicted_labels)
                                confidence_list.append(confidence)
                                
                                correct_predictions[i] += (predicted_labels == query_labels).sum().item()
                                total_samples[i] += query_labels.size(1)
                                    # 使用已计算的predicted_labels进行投票
                            stacked_predictions = torch.stack(predicted_labels_list, dim=0)  # [num_timesteps, batch_size]
                            stacked_confidences = torch.stack(confidence_list, dim=0)  # [num_timesteps, batch_size]
                                    # 对每个样本进行投票
                            vote_results = []
                            for sample_idx in range(stacked_predictions.shape[1]):
                                sample_predictions = stacked_predictions[:, sample_idx]  # [num_timesteps]
                                sample_confidences = stacked_confidences[:, sample_idx]  # [num_timesteps]
                                
                                # Create a tensor to store weighted votes for each class
                                num_classes = config["num_classes"]    # Assuming 5 classes, adjust if different
                                weighted_votes = torch.zeros(num_classes, device=query_labels.device)
                                
                                # Accumulate confidence-weighted votes for each class
                                for pred, conf in zip(sample_predictions, sample_confidences):
                                    weighted_votes[pred] += conf
                                
                                # Get the class with highest weighted votes
                                final_prediction = torch.argmax(weighted_votes)
                                vote_results.append(final_prediction)
                            
                            # 转换投票结果为tensor并计算准确率
                            voted_predictions = torch.tensor(vote_results, device=query_labels.device)
                            voting_correct += (voted_predictions == query_labels).sum().item()
                            voting_total += query_labels.size(1)
                    # Wandb日志记录
                    if config["use_wandb"]:
                        log_dict = {
                            "classification/total_loss": accumulated_cls_loss / accumulation_steps,
                            "learning_rate": optimizer.param_groups[0]['lr']
                        }
                        if config_classification:
                            for i in range(len(correct_predictions)):
                                if total_samples[i] <= 0:
                                        continue
                                log_dict[f"train/accuracy_{i}"] = correct_predictions[i]/total_samples[i]
                            voting_accuracy = voting_correct/voting_total
                            log_dict["voting_acc"] = voting_accuracy
                            max_voting_acc = max(voting_accuracy, max_voting_acc)
                        wandb.log(log_dict, step=self.step)
                    else:
                        print(f"classification/total_loss: {accumulated_cls_loss / accumulation_steps}")
                        print(f"train/accuracy = {correct_predictions[0]/total_samples[0]}")
                    self.step += 1
                    flag += 1
            if flag % transfer_accumulation == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                accumulated_cls_loss = 0.0
            
            if flag % 3 == 0:
                self.validate(config, diffusion_extractor, scale_fusion_network, classification_model, val_dataloader, alpha, margin)

def train_only_few_shot(config, diffusion_extractor, aggregation_network, classification_model, optimizer,  labeled_dataloader, val_dataloader, unlabeled_dataloader, lr_schduler, writer):
    device = config.get("device", "cuda")
    max_epochs = config["max_epochs"]
    step = 0
    flag = 0
    accumulation_steps = config.get("accumulation_steps", 4)
    optimizer.zero_grad()
    accumulated_cls_loss = 0
    config_classification = config.get('classification', True)
    unlabeled_dataloader = iter(unlabeled_dataloader)
    max_voting_acc = 0
    transfer_accumulation = config.get("transfer_accumulation_classification")
    
    for epoch in range(max_epochs):
        correct_predictions = [0] *  (len(config["save_timestep"])*len(config["idxs"]))
        total_samples = [0] * (len(config["save_timestep"])*len(config["idxs"]))
        voting_total = 0
        voting_correct = 0
        accumulated_cls_loss = 0.0
        alpha = config.get("alpha", 0.5)
        margin = config.get("margin", 1.0)
        freeze_ratio = config.get("freeze_ratio", 0.95)
        # 修改数据加载逻辑，区分有标签和无标签数据
        for batch_idx, batch_data in enumerate(tqdm(labeled_dataloader)):
            # 原始的数据格式
            support_images, support_labels, query_images_labeled, query_labels, \
            support_class_names, query_class_names = batch_data

            # 处理设备转移和维度调整
            support_images = support_images.to(device).squeeze(0)
            support_labels = support_labels.to(device).squeeze(0)
            query_images_labeled = query_images_labeled.to(device).squeeze(0)
            query_labels = query_labels.to(device)
            
            with torch.autocast(device_type='cuda'):
                # 处理支持集和有标签查询集
                support_images = support_images.unsqueeze(1)
                query_images_labeled = query_images_labeled.unsqueeze(1)
                #support_images = diffusion_extractor.forward_only_use_vae(support_images).unsqueeze(1)
                #query_images_labeled = diffusion_extractor.forward_only_use_vae(query_images_labeled).unsqueeze(1)
                classification_result = classification_model(support_images, support_labels, query_images_labeled)
                #classification_result = classification_result.unsqueeze(1)
                classification_loss, loss_list = loss_with_contrastive(classification_result, query_labels, alpha, margin)
                # 缩放损失以匹配累积步数
                total_loss = classification_loss / accumulation_steps
                total_loss.backward()
                accumulated_cls_loss += total_loss
                del total_loss, query_images_labeled, classification_loss
                torch.cuda.empty_cache()
                # 处理无标签样本（如果启用）
                                
                if config["log_train_accuracy"]:
                    with torch.no_grad():
                        predicted_labels_list = [] 
                        confidence_list = []  # Store confidence scores for each prediction
                        
                        for i in range(classification_result.size(1)):
                            # Get distances and convert to probabilities using softmax
                            distances = classification_result[:, i, :]
                            probabilities = F.softmax(-distances, dim=1)  # Negative because smaller distance means higher probability
                            confidence, predicted_labels = torch.max(probabilities, dim=1)
                            
                            predicted_labels_list.append(predicted_labels)
                            confidence_list.append(confidence)
                            
                            correct_predictions[i] += (predicted_labels == query_labels).sum().item()
                            total_samples[i] += query_labels.size(1)
                                # 使用已计算的predicted_labels进行投票
                        stacked_predictions = torch.stack(predicted_labels_list, dim=0)  # [num_timesteps, batch_size]
                        stacked_confidences = torch.stack(confidence_list, dim=0)  # [num_timesteps, batch_size]
                                # 对每个样本进行投票
                        vote_results = []
                        for sample_idx in range(stacked_predictions.shape[1]):
                            sample_predictions = stacked_predictions[:, sample_idx]  # [num_timesteps]
                            sample_confidences = stacked_confidences[:, sample_idx]  # [num_timesteps]
                            
                            # Create a tensor to store weighted votes for each class
                            num_classes = config["num_classes"]    # Assuming 5 classes, adjust if different
                            weighted_votes = torch.zeros(num_classes, device=query_labels.device)
                            
                            # Accumulate confidence-weighted votes for each class
                            for pred, conf in zip(sample_predictions, sample_confidences):
                                weighted_votes[pred] += conf
                            
                            # Get the class with highest weighted votes
                            final_prediction = torch.argmax(weighted_votes)
                            vote_results.append(final_prediction)
                        
                        # 转换投票结果为tensor并计算准确率
                        voted_predictions = torch.tensor(vote_results, device=query_labels.device)
                        voting_correct += (voted_predictions == query_labels).sum().item()
                        voting_total += query_labels.size(1)
                # Wandb日志记录
                if config["use_wandb"]:
                    log_dict = {
                        "classification/total_loss": accumulated_cls_loss / accumulation_steps,
                        "learning_rate_unet": optimizer.param_groups[0]['lr'],
                        "learning_rate_cls": optimizer.param_groups[1]['lr']
                    }
                    
                    
                    if config_classification:
                        for i in range(len(correct_predictions)):
                            if total_samples[i] <= 0:
                                    continue
                            log_dict[f"train/accuracy_{i}"] = correct_predictions[i]/total_samples[i]
                        voting_accuracy = voting_correct/voting_total
                        log_dict["voting_acc"] = voting_accuracy
                        max_voting_acc = max(voting_accuracy, max_voting_acc)
                    wandb.log(log_dict, step=step)
                else:
                    current_lr = optimizer.param_groups[0]['lr']
                    # 跟 wandb 一样，TensorBoard 只记录非 None 项
                    if config_classification:
                        writer.add_scalar("classification/loss", accumulated_cls_loss / accumulation_steps, step)
                    writer.add_scalar("learning_rate_cls", current_lr, step)
                    if config_classification:
                        for i in range(len(correct_predictions)):
                            if total_samples[i] <= 0:
                                continue
                            writer.add_scalar(f"train/accuracy_{i}", correct_predictions[i] / total_samples[i], step)
                        writer.add_scalar("voting_acc", voting_correct / voting_total, step)
                step += 1
                flag += 1
        if flag % transfer_accumulation == 0:
            optimizer.step()
            lr_schduler.step()
            optimizer.zero_grad()
            accumulated_cls_loss = 0.0
        
        if flag % 3 == 0:
            step = validate_only_few_shot(config, diffusion_extractor, classification_model, val_dataloader, alpha, margin, step, writer = writer) + 1


def update_reliable_memory(memory, new_samples, max_size):
    """更新可靠样本池"""
    images, labels = new_samples
    for img, label in zip(images, labels):
        if len(memory) >= max_size:
            memory.pop(0)  # 移除最旧的样本
        memory.append((img, label))

def get_batch_from_memory(memory, batch_size=5):
    """从可靠样本池中随机选取5个样本进行stack"""
    if len(memory) < batch_size:
        return False, [], []
    
    # 随机选择indices
    indices = torch.randperm(len(memory))[:batch_size]
    batch_images = []
    batch_labels = []
    
    # 收集选定样本
    for idx in indices:
        img, label = memory[idx]
        batch_images.append(img)
        batch_labels.append(label)
    
    # 如果没有足够的样本，直接返回收集的样本
    if not batch_images:
        return None, None
    
    # 堆叠样本
    return True, torch.stack(batch_images), torch.stack(batch_labels)

#//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

def main(args):
    inner_config = {"few_shot_mode":True,
                    "few_shot":True,
                    "only_few_shot": False,
                    "debug_sep": False,
                    "only_baseline": False,
                    "vae_baseline": False,
                    "vae_decode_baseline": False,}
    only_few_shot = inner_config["only_few_shot"]
    few_shot_mode = inner_config["few_shot_mode"]
    config, diffusion_extractor, scale_fusion_network ,classification_model = load_models_NEU(args.config_path, inner_config)
    if config["use_wandb"]:                                                                                      
        wandb.init(project=config["wandb_project"], name=config["wandb_run"])
        wandb.run.name = f"{str(wandb.run.id)}_{wandb.run.name}"
        writer = None
    else:
        from torch.utils.tensorboard import SummaryWriter
        import time
        writer = SummaryWriter(log_dir=f'runs/{config["wandb_run"]}/{int(time.time())}')
    weight_decay=config.get("weight_decay", 0.001)
    eps=config.get("eps", 1e-4)
    parameter_groups = []
    if diffusion_extractor is not None:
        parameter_groups.append({"params": diffusion_extractor.unet.parameters(), "lr": config["u_net_lr"]})
        if diffusion_extractor.control_net is not None:
            parameter_groups.append({"params": diffusion_extractor.control_net.parameters(), "lr": config["u_net_lr"]})
    if scale_fusion_network is not None:
        parameter_groups.append({"params": scale_fusion_network.parameters(), "lr": config["lr"]})
    if classification_model is not None:
        parameter_groups.append({"params": classification_model.parameters(), "lr": config["lr"]})

    optimizer = torch.optim.AdamW(  # AdamW通常比Adam更稳定
    parameter_groups,
    eps=eps,
    weight_decay=weight_decay,
    betas=(0.9, 0.97)  # 添加权重衰减以增加稳定性
    )
    unlabeled_dataloader = None
    if config["dataset"] == 'NEU':
        if few_shot_mode == False:
            if config["transfer_learning"]:
                raise ValueError("transfer learning should be False when few_shot is False")
            _, train_dataloader,_,val_dataloader = get_loader_NEU(config, True)   
        elif few_shot_mode == True:
            if not ((config["transfer_learning"]) or only_few_shot):
                train_dataloader,val_dataloader = few_shot_data_loader_NEU_whole(config, True)   
            else:
                train_dataloader, val_dataloader, unlabeled_dataloader = few_shot_data_loader_NEU_transfer(config, True)   
        else:
            raise ValueError("few_shot should be either True or False")
    elif config["dataset"] == 'severstal':
        if few_shot_mode == True:
            if not ((config["transfer_learning"] or config["use_pseudo_labels"]) or only_few_shot):
                train_dataloader,val_dataloader = few_shot_data_loader_severstal_whole(config, True)   
            else:
                train_dataloader, val_dataloader, unlabeled_dataloader = few_shot_data_loader_severstal_transfer(config, True)   
                
    elif config["dataset"] == 'NEU_64':
        if few_shot_mode == True:
            if not ((config["transfer_learning"]) or only_few_shot):
                train_dataloader,val_dataloader = few_shot_data_loader_NEU64_whole(config, True)   
            else:
                train_dataloader, val_dataloader, unlabeled_dataloader = few_shot_data_loader_NEU64_transfer(config, True)   
    elif config["dataset"] == 'X_SDD':
        if not ((config["transfer_learning"]) or only_few_shot):
            train_dataloader,val_dataloader = few_shot_data_loader_X_SDD_whole(config, True)   
        else:
            train_dataloader, val_dataloader, unlabeled_dataloader = few_shot_data_loader_X_SDD_transfer(config, True)
    else:
        raise ValueError('dataset_name_incorrect')
    
    #use target_domain in training ddim
    if config["transfer_dataset"] == 'NEU':
        if few_shot_mode == False:
            if config["transfer_learning"]:
                raise ValueError("transfer learning should be False when few_shot is False")
            _, train_dataloader,_,val_dataloader = get_loader_NEU(config, True)   
        elif few_shot_mode == True:
            if not (config["transfer_learning"] and config["use_pseudo_labels"]):
                transfer_dataloader, _ = few_shot_data_loader_NEU_whole(config, True)   
        else:
            raise ValueError("few_shot should be either True or False")
    elif config["transfer_dataset"] == 'severstal':
        if few_shot_mode == True:
            transfer_dataloader, _ = few_shot_data_loader_severstal_whole(config, True)   
    elif config["transfer_dataset"] == 'X_SDD':
        if not (config["transfer_learning"] and config["use_pseudo_labels"]):
            transfer_dataloader, _ = few_shot_data_loader_X_SDD_whole(config, True)   
    elif config["transfer_dataset"] == 'None':
        transfer_dataloader = None
    else:
        raise ValueError('dataset_name_incorrect')
    if config["checkpoint_path"] != "None":
        load_checkpoint(config, diffusion_extractor, scale_fusion_network, classification_model, optimizer)
    if config["visualialize_generated_images"]:
        get_pic(diffusion_extractor, config["num_generate_steps"], config.get('generate_prompt', None), guidance_scale = config.get('guidance_scale', -1))
    
    trainer = trainer_class(config, writer, inner_config)
    if not (config["transfer_learning"] or only_few_shot):
        lr_scheduler = get_scheduler_NEU(config, optimizer, len(train_dataloader))
        trainer.train_NEU(config, diffusion_extractor, scale_fusion_network, classification_model,  optimizer, lr_scheduler, train_dataloader, val_dataloader, transfer_dataloader = transfer_dataloader)
    elif config["transfer_learning"]:
        lr_scheduler = get_scheduler_NEU(config, optimizer, len(train_dataloader))
        trainer.transfer(config, diffusion_extractor, scale_fusion_network, classification_model,  optimizer,  train_dataloader, val_dataloader, unlabeled_dataloader, lr_scheduler)
    elif only_few_shot:
        lr_scheduler = get_scheduler_NEU(config, optimizer, 20)
        train_only_few_shot(config, diffusion_extractor, scale_fusion_network, classification_model, optimizer,  train_dataloader, val_dataloader, unlabeled_dataloader, lr_scheduler, writer = writer)


if __name__ == "__main__":
    # python3 train_generic.py --config_path configs/train.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, help="Path to yaml config file", default="configs/train_NEU.yaml")
    args = parser.parse_args()
    main(args)