import cv2
from PIL import Image
import numpy as np
from omegaconf import OmegaConf
import os
import random

def standardize_image(img_bgr):
    """对BGR图像做自适应直方图均衡化（CLAHE），输出增强后BGR图像"""
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(4,4))
    img_eq = clahe.apply(img_gray)
    img_eq_bgr = cv2.cvtColor(img_eq, cv2.COLOR_GRAY2BGR)
    return img_eq_bgr

def get_canny_image(image):
    low_threshold = 120
    high_threshold = 180

    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    image = Image.fromarray(image)
    return image

def get_hed_image(image, net):
    """
    输入：image 为 numpy.ndarray（BGR），如cv2.imread读入的图片
    输出：PIL.Image（3通道HED边缘图，和输入尺寸一致）
    """
    H, W = image.shape[:2]
    # OpenCV DNN要求尺寸为32的倍数
    newW = int(np.round(W / 32) * 32)
    newH = int(np.round(H / 32) * 32)
    resized = cv2.resize(image, (newW, newH))
    blob = cv2.dnn.blobFromImage(
        resized, scalefactor=1.0, size=(newW, newH),
        mean=(104.00698793, 116.66876762, 122.67891434),
        swapRB=False, crop=False
    )
    net.setInput(blob)
    hed = net.forward()
    hed = hed[0, 0]  # 单通道
    hed = cv2.resize(hed, (W, H))  # 拉回原图尺寸
    hed = (255 * hed).astype("uint8")
    hed = hed[:, :, None]  # [H, W, 1]
    hed = np.concatenate([hed, hed, hed], axis=2)  # 转3通道
    pil_image = Image.fromarray(hed)
    return pil_image

def visualize_image(image):
    image.show()

if __name__ == "__main__":
    '''
    protoPath = "deploy.prototxt"
    modelPath = "hed_pretrained_bsds.caffemodel"
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
    '''
    config_path = "configs/train_NEU.yaml"
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config, resolve=True)
    image_path = config["image_path"]
    all_files = [f for f in os.listdir(image_path) 
             if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    rand_name = random.choice(all_files)
    rand_img_path = os.path.join(image_path, rand_name)

    # 4. 加载为PIL Image
    image = Image.open(rand_img_path).convert('RGB')
    np_img = np.array(image)        # 通道是RGB
    np_img_bgr = np_img[..., ::-1]    # 转为BGR再给cv2用
    std_img_bgr = standardize_image(np_img_bgr)
    # 转回RGB格式再转为PIL Image
    std_img_rgb = std_img_bgr[..., ::-1]              # BGR -> RGB
    std_pil_image = Image.fromarray(std_img_rgb)      # numpy -> PIL
    #hed_img = get_hed_image(np_img_bgr, net)
    canny_image = get_canny_image(std_img_bgr)
    visualize_image(image)
    visualize_image(std_pil_image)
    visualize_image(canny_image)