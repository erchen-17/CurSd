# CurSd
This repository contains the code for the patent "**Cross-Domain Few Shot Industrial Defect Classification Based on a Progressive Conditional Diffusion Model**"

## Highlights
1. Implements a unified diffusion framework that simultaneously performs domain adaptation, feature extraction, and image generation.
2. Utilizes hyperfeature extraction to learn a more robust model.
3. Introduces a progressive curriculum learningbased control strategy that fuses strong- and weak-supervision images to mitigate domain-shift-induced bias.
4. Achieves few-shot domain adaptation via stable diffusion–based image migration.

## Usage
### Stable Diffusion
- We use Stable-Diffusion-v1-5 as the diffusion model.
- We use lllyasviel/sd-controlnet-depth as the ControlNet.

### Training dataset
- NEU dataset is provided on http://faculty.neu.edu.cn/songkechen/zh_CN/zhym/263269/list/index.htm
- Severstal Dataset is provided on https://github.com/Severstal-AI/DefectNet.

## Related Works
- [Diffusion Hyperfeatures: Searching Through Time and Space for Semantic Correspondence](https://diffusion-hyperfeatures.github.io/)
- [Unsupervised Domain Adaptation via Domain-Adaptive Diffusion](https://ieeexplore.ieee.org/document/10599225)
- [DiffDD: A surface defect detection framework with diffusion probabilistic model](https://www.sciencedirect.com/science/article/pii/S1474034624002854)
- [Denoising Diffusion Implicit Models](https://openreview.net/forum?id=St1giarCHLP)
- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
