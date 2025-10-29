from PIL import Image
import torch

from diffusers import DDIMScheduler
from archs.stable_diffusion.diffusion import (
    init_models, 
    init_models_with_controlnet,
    get_tokens_embedding,
    generalized_steps,
    generalized_steps_with_xt,
    inversion_and_generation,
    collect_and_resize_feats,
    train_DDIM,
    get_x0
)
from archs.stable_diffusion.resnet import init_resnet_func
from diffusers import DPMSolverMultistepScheduler

class DiffusionExtractor:
    """
    Module for running either the generation or inversion process 
    and extracting intermediate feature maps.
    """
    def __init__(self, config, device):
        self.eta = config["inversion_eta"]
        self.device = device
        self.num_timesteps = config["num_timesteps"]
        self.generator = torch.Generator(self.device).manual_seed(config.get("seed", 0)) #随机数生成器
        self.batch_size = config.get("batch_size", 1)
        self.support = config.get("num_support", 1)
        self.num_classes = config.get("num_classes", 5)
        self.guidance_scale = config.get("guidance_scale", -1)
        if config.get("few_shot",False):
            self.cond_batch = self.batch_size * self.support * self.num_classes
        else:
            self.cond_batch = self.batch_size
        if config['use_control_net'] == False:
            self.unet, self.vae, self.clip, self.clip_tokenizer,self.lora_paramters = \
                init_models(device=self.device, model_id=config["model_id"],freeze=config.get("freeze_weights_in_unet", True),lora_config=config.get("lora", None)) #clip用于将文本转变为embedding，CLIP tokenizer用于分词，形成适合嵌入的文本
            self.control_net = None
        else:
            self.unet, self.vae, self.clip, self.clip_tokenizer, self.lora_paramters, self.control_net, scheduler_config = \
                init_models_with_controlnet(device=self.device, model_id=config["model_id"], control_net_id=config["control_net_id"],\
                                            freeze=config.get("freeze_weights_in_unet", True),lora_config=config.get("lora", None)) #clip用于将文本转变为embedding，CLIP tokenizer用于分词，形成适合嵌入的文本
        scheduler_config_cls = scheduler_config.copy()
        scheduler_config_ddim = scheduler_config.copy()
        scheduler_config_cls["num_train_timesteps"] = config["num_timesteps"]
        scheduler_config_ddim["num_train_timesteps"] = config['train_time_steps']
        self.scheduler = DPMSolverMultistepScheduler.from_config(
        scheduler_config_cls,
        use_karras_sigmas=True,  # Karras 变体
        )
        self.scheduler_for_ddim = DPMSolverMultistepScheduler.from_config(
        scheduler_config_ddim,
        use_karras_sigmas=True,  # Karras 变体
        )
        self.train_time_steps=config['train_time_steps']
        self.train_ddim_batch_size=config['train_ddim_batch_size']
        self.prompt = config.get("prompt", "")
        self.negative_prompt = config.get("negative_prompt", "")
        self.change_cond(self.prompt, "cond")
        self.change_cond(self.negative_prompt, "uncond")
        if config.get("gradient_checkpointing", False):
            self.unet.enable_gradient_checkpointing()
            self.vae.enable_gradient_checkpointing()
            #self.unet.enable_xformers_memory_efficient_attention()
        self.diffusion_mode = config.get("diffusion_mode", "generation")
        if "idxs" in config and config["idxs"] is not None:
            config_idxs = config["idxs"]

            self.idxs = [tuple(i) for i in config_idxs]
        else:
            self.idxs = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        self.freeze_idxs=config.get("freeze_idxs",self.idxs)
        self.output_resolution = config["output_resolution"]
        self.save_timestep = config.get("save_timestep", [])
        self.transfer_hyperfeature_mode = config.get("transfer_hyperfeature_mode", 'inversion')

        # Note that save_timestep is in terms of number of generation steps
        # save_timestep = 0 is noise, save_timestep = T is a clean image
        # generation saves as [0...T], inversion saves as [T...0]
        self.train_diffusion = True
        self.config = config

        self.transfer_inverstion_steps = config["transfer_inversion_steps"]
        self.transfer_generation_steps = config["transfer_generation_steps"]
        
        print(f"idxs: {self.idxs}")
        print(f"prompt: {self.prompt}")
        print(f"negative_prompt: {self.negative_prompt}")

    def change_cond(self, prompt, cond_type="cond"):
        with torch.no_grad():
            with torch.autocast("cuda"):
                _, new_cond = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, prompt)
                
                new_cond = new_cond.expand((self.cond_batch, *new_cond.shape[1:]))
                new_cond = new_cond.to(self.device)
                if cond_type == "cond":
                    self.cond = new_cond
                    self.prompt = prompt
                elif cond_type == "uncond":
                    self.uncond = new_cond
                    self.negative_prompt = prompt
                else:
                    raise NotImplementedError
                
    def get_cond_for_ddim(self, prompt, cond_type="cond"):
        with torch.no_grad():
            with torch.autocast("cuda"):
                new_conds_list = []
                if prompt == "" or prompt == None:
                    i = 'a photo of steel superficial defect '
                    _, new_cond = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, i)
                    new_conds_list.append(new_cond)
                else:
                    for i in prompt:
                        i = 'a photo of steel superficial defect ' + i[0]
                        _, new_cond = get_tokens_embedding(self.clip_tokenizer, self.clip, self.device, i)
                        new_conds_list.append(new_cond)
                    
                new_cond=torch.cat(new_conds_list, dim=0).to(self.device)
                if cond_type == "cond":
                    cond = new_cond
                    prompt = prompt
                elif cond_type == "uncond":
                    uncond = new_cond
                    negative_prompt = prompt
                else:
                    raise NotImplementedError
                return cond
                
    def run_generation(self, latent, guidance_scale=-1, min_i=None, max_i=None):
        xs, xt, time_step_use = generalized_steps_with_xt(
            latent,
            self.unet, 
            self.scheduler, 
            run_inversion=False, 
            guidance_scale=guidance_scale, 
            conditional=self.cond, 
            unconditional=self.uncond, 
            min_i=min_i,
            max_i=max_i,
            save_timestep=self.save_timestep,
            eta=self.eta
        )
        return xs, xt, time_step_use
    
    def run_inversion(self, latent, guidance_scale = -1, min_i=None, max_i=None):
        xs, xt, time_step_use = inversion_and_generation(
            latent, 
            self.unet, 
            self.scheduler, 
            run_inversion=True, 
            guidance_scale = guidance_scale, 
            inversion_steps=self.transfer_inverstion_steps, 
            generation_steps=self.transfer_generation_steps, 
            conditional=self.cond, 
            unconditional=self.uncond,
            min_i=min_i,
            max_i=max_i,
            idxs=self.idxs,
            save_timestep=self.save_timestep,
            transfer_hyperfeature_mode = ""
        )
        return xs, xt, time_step_use #为每一个时间步中对于x0的预测
    
    def run_inversion_generalization(self, latent, guidance_scale = -1, validate=True, min_i=None, max_i=None, control_image = None, control_net_scale = 0.0):
        xs, xt, time_step_use = inversion_and_generation(
            latent, 
            self.unet, 
            self.scheduler, 
            inversion_steps=self.transfer_inverstion_steps, 
            generation_steps=self.transfer_generation_steps, 
            guidance_scale = guidance_scale, 
            validate=validate,
            conditional=self.cond, 
            unconditional=self.uncond,
            idxs=self.idxs,
            save_timestep=self.save_timestep,
            transfer_hyperfeature_mode = self.transfer_hyperfeature_mode,
            control_image = control_image,
            control_net = self.control_net,
            control_net_scale = control_net_scale
        )
        return xs, xt, time_step_use #为每一个时间步中对于x0的预测
    
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
    
    def get_feats(self, latents, extractor_fn, preview_mode=False):
        # returns feats of shape [batch_size, num_timesteps, channels, w, h]
        if not preview_mode:
            init_resnet_func(self.unet, save_hidden=True, reset=True, idxs=self.idxs, save_timestep=self.save_timestep)
        outputs, xt, time_step_use = extractor_fn(latents) #outputs为在每一个时间步中对于x0的预测
        try:
            if not preview_mode:
                feats = []
                for timestep in self.save_timestep:
                    timestep_feats = collect_and_resize_feats(self.unet, self.idxs, timestep, self.output_resolution, xt[timestep], self.scheduler, self.cond, time_step_use[timestep])
                    feats.append(timestep_feats)
                    #torch.cuda.empty_cache()
                feats = torch.cat(feats, dim=1) #为存储的feats #为主要增加显存处
                init_resnet_func(self.unet, reset=True, idxs=self.idxs, save_timestep=self.save_timestep) #重新初始化，消除存储的feats
            else:
                feats = None
        except Exception as e:
            print("error in collect_and_resize_feats")
            print(e)
        return feats, outputs

    def latents_to_images_for_generate_pic(self, latents):
        images=[]
        for i in range(len(latents)):
            latent = latents[i].to(self.device)
            latent = latent/ 0.18215
            image = self.vae.decode(latent.to(self.vae.dtype)).sample
            image = (image/2+0.5).clamp(0, 1)
            image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
            image = (image * 255).round().astype("uint8")
            images.append(image)
        return [Image.fromarray(image[0]) for image in images]

    def latents_to_images(self, latents):
        images=[]
        #for i in range(len(latents)):
        #latent = latents[i].to(self.device)
        latent = latents.to(self.device) / 0.18215
        image = self.vae.decode(latent.to(self.vae.dtype)).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        images.append(image)
        return [Image.fromarray(image[0]) for image in images]
    
    def latents_to_images_all(self, latents):
        images=[]
        #for i in range(len(latents)):
        #latent = latents[i].to(self.device)
        latent = latents.to(self.device) / 0.18215
        image = self.vae.decode(latent.to(self.vae.dtype)).sample
        #image = (image / 2 + 0.5).clamp(0, 1)
        image = image.clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")
        for i in range(len(image)):
            image_array = Image.fromarray(image[i])
            images.append(image_array)
        return images
    
    def latents_to_images_for_feats(self, latents):
        latent = latents.to(self.device) / 0.18215
        image = self.vae.decode(latent.to(self.vae.dtype)).sample
        image = (image * 255)
        return image
    #//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////  
    def get_lora_paramters(self):
        return self.lora_paramters
    
    def calculate_DDIM_loss(self, latent, guidance_scale=-1, cond = None, min_i=None, max_i=None, control_net_pic = None, control_net_scale = 0.0):
        """计算DDIM损失，增加数值稳定性检查"""
        # 1. 初始化和类型转换
        latent = latent.to(torch.float16)

        if cond == None:
            cond = self.cond
        
        # 3. 计算损失
        with torch.amp.autocast('cuda'):
            loss = train_DDIM(
                self.train_ddim_batch_size,
                latent, 
                self.unet, 
                self.control_net,
                self.scheduler_for_ddim, 
                self.config,
                conditional = cond,
                unconditional=self.uncond,
                guidance_scale = guidance_scale,
                min_i=min_i,
                max_i=max_i,
                control_net_pic = control_net_pic,
                control_net_scale = control_net_scale
            )
        return loss
            


    def forward(self, images=None, latents=None, guidance_scale = None, preview_mode=False, transfer_mode=False, validate_mode=True, control_image = None, control_net_scale = 0.0):
        if guidance_scale == None:
            guidance_scale = self.guidance_scale
        if not transfer_mode:
            if images is None:
                if latents is None:
                    latents = torch.randn((self.batch_size, self.unet.in_channels, 512 // 8, 512 // 8), device=self.device, generator=self.generator)
                if self.diffusion_mode == "generation":
                    if preview_mode:
                        extractor_fn = lambda latents: self.run_generation(latents, guidance_scale, max_i=self.end_timestep)
                    else:
                        extractor_fn = lambda latents: self.run_generation(latents, guidance_scale)
                elif self.diffusion_mode == "inversion":
                    raise NotImplementedError
            else:
                images = torch.nn.functional.interpolate(images, size=256, mode="bilinear")
                control_image = torch.nn.functional.interpolate(control_image, size=256, mode="bilinear")
                #with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    latents = self.vae.encode(images).latent_dist.sample(generator=None) * 0.18215   #一般先微调diffusion，vae对于域的适应性较高
                del images
                torch.cuda.empty_cache()

                
                if self.diffusion_mode == "inversion":
                    extractor_fn = lambda latents: self.run_inversion_generalization(latents, guidance_scale, validate=validate_mode, control_image=control_image, control_net_scale=control_net_scale)
                elif self.diffusion_mode == "generation":
                    raise NotImplementedError
                
        else:
            if images is None:
                raise NotImplementedError
            else:
                images = torch.nn.functional.interpolate(images, size=256, mode="bilinear")
                with torch.no_grad():
                    with torch.amp.autocast('cuda'):
                        latents = self.vae.encode(images).latent_dist.sample(generator=None) * 0.18215   #一般先微调diffusion，vae对于域的适应性较高
                    del images
                    torch.cuda.empty_cache()
                extractor_fn = lambda latents: self.run_inversion_generalization(latents, guidance_scale, validate=validate_mode, control_image=control_image, control_net_scale=0.0)     
        
        with torch.autocast("cuda"):
            feats, outputs = self.get_feats(latents, extractor_fn,  preview_mode=preview_mode)
            return feats, outputs, latents
            #return latents.unsqueeze(1), latents
    
    def forward_only_use_vae(self, images=None, latents=None, guidance_scale = None, preview_mode=False, transfer_mode=False):
                if guidance_scale == None:
                    guidance_scale = self.guidance_scale
                if not transfer_mode:
                    if images is None:
                        raise ValueError('with no image')
                    else:
                        images = torch.nn.functional.interpolate(images, size=256, mode="bilinear")
                        #with torch.no_grad():
                        with torch.amp.autocast('cuda'):
                            latents = self.vae.encode(images).latent_dist.sample(generator=None) * 0.18215   #一般先微调diffusion，vae对于域的适应性较高
                        del images
                        torch.cuda.empty_cache()
                with torch.autocast("cuda"):
                    return latents
                
    def forward_for_DDIM_loss(self, config, images=None, labels = None, guidance_scale = -1, control_net_pic=None, control_net_scale = 0.0):
        if images is None:
            raise NotImplementedError

        # 图像预处理
        images = torch.nn.functional.interpolate(images, size=256, mode="bilinear")
        control_net_pic = torch.nn.functional.interpolate(control_net_pic, size=256, mode="bilinear")
        
        # 转换为latents
        with torch.no_grad():
            with torch.autocast("cuda"):
                latents = self.vae.encode(images).latent_dist.sample(generator=None) * 0.18215
                latents = latents.to(torch.float16)
        del images

        cond = self.get_cond_for_ddim(labels, cond_type="cond")

        # 计算DDIM loss
        if self.diffusion_mode == "inversion":
            with torch.amp.autocast('cuda'):
                loss = self.calculate_DDIM_loss(latents, guidance_scale, cond, control_net_pic = control_net_pic, control_net_scale=control_net_scale)

            # 检查loss是否为NaN
            if torch.isnan(loss):
                print("NaN loss detected in forward_for_DDIM_loss")
                return torch.tensor(0.0, device=latents.device, requires_grad=True)
            
            return loss
        
        elif self.diffusion_mode == "generation":
            raise NotImplementedError


    def generate_images(self, prompt=None, negative_prompt=None, guidance_scale = -1, num_inference_steps=50):
        """
        Generates images using the diffusion model with optional classifier-free guidance.
        """
        try:

            # Initialize noise
            latents = torch.randn(
                (self.batch_size, self.unet.in_channels, 64, 64),
                device=self.device,
                generator=self.generator,
                dtype=torch.float16
            )

            # Set timesteps
            self.scheduler_for_ddim.set_timesteps(num_inference_steps)
            timesteps = self.scheduler_for_ddim.timesteps
            
            if prompt != None:
                prompt = self.get_cond_for_ddim(prompt)
            else:
                prompt = self.cond
            
            # Initialize storage
            x0_preds = [latents]
            xs = [latents]

            # Generate
            with torch.no_grad():
                with torch.autocast("cuda"):
                    for i, t in enumerate(timesteps):
                        # Get batch of timesteps
                        t_batch = torch.full((self.batch_size,), t, 
                                        device=self.device, dtype=torch.float16)
                        
                        # Get current latent
                        xt = xs[-1].to(self.device)

                        # Get alphas for current and next timestep
                        at = (1 - self.scheduler_for_ddim.betas.to(self.device)).cumprod(dim=0).index_select(0, t_batch.long())
                        at = at[:, None, None, None]
                        
                        next_t = timesteps[i + 1] if i < len(timesteps) - 1 else -1
                        at_next = torch.ones_like(at) if next_t == -1 else \
                                (1 - self.scheduler_for_ddim.betas.to(self.device)).cumprod(dim=0).index_select(
                                    0, torch.full((self.batch_size,), next_t, device=self.device).long()
                                )[:, None, None, None]

                        # Predict noise with optional guidance
                        if guidance_scale > 0:
                            # Classifier-free guidance
                            et_uncond = self.unet(xt, t_batch, encoder_hidden_states = self.uncond).sample
                            et_cond = self.unet(xt, t_batch, encoder_hidden_states = prompt).sample
                            et = et_uncond + guidance_scale * (et_cond - et_uncond)
                        else:
                            # Regular prediction
                            et = self.unet(xt, t_batch, encoder_hidden_states = self.cond[:xt.size(0)]).sample

                        # Predict x0 and get next latent
                        x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                        
                        # DDIM sampling (eta=0 for deterministic)
                        c2 = ((1 - at_next) - 0).sqrt()
                        xt_next = at_next.sqrt() * x0_t + c2 * et

                        # Save predictions
                        x0_preds.append(x0_t)
                        xs.append(xt_next.to('cpu'))

                        # Clear cache
                        torch.cuda.empty_cache()

                    return xs

        except Exception as e:
            raise ValueError(f"Error in generate_images: {e}")

