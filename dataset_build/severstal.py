import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset
import pandas as pd
import numpy as np
import cv2

class CustomDataset_severstal(Dataset):
    def __init__(self, image_folder, brightness_threshold=30, target_size=(64, 64)):
        self.image_folder = image_folder
        self.brightness_threshold = brightness_threshold
        self.target_size = target_size
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        df = pd.read_csv(os.path.join(image_folder, 'train.csv'))
        
        for img_id in df['ImageId'].unique():
            img_path = os.path.join(image_folder, 'train_images', img_id)
            img_defects = df[df['ImageId'] == img_id]['ClassId'].dropna().unique()
            label = 0 if len(img_defects) == 0 else int(min(img_defects))
            img_relative_path=os.path.join('train_images', img_id)
            
            self.images.append((img_relative_path,label))
                
        # 根据标签对图片进行分类
        self.raw_class_to_images = {}
        self.num_len = {}

        # 分类图片
        for image, label in self.images:
            if label not in self.raw_class_to_images:
                self.raw_class_to_images[label] = [image]  # 初始化类别图片列表
                self.num_len[label] = 1  # 初始化图片计数
            else:
                if self.num_len[label] < 300:  # 控制每个类别最多300张图片
                    self.raw_class_to_images[label].append(image)
                    self.num_len[label] += 1  # 更新计数

        # 对不足300的类别进行上采样
        for label, images in self.raw_class_to_images.items():
            if self.num_len[label] < 300:  # 如果某类别图片数量不足300
                extra_images = images * (300 // self.num_len[label]) + images[:300 % self.num_len[label]]  # 重复采样到300
                self.raw_class_to_images[label] = extra_images  # 更新为上采样后的列表
                self.num_len[label] = 300  # 更新计数为300

        # 采样图片集
        self.class_to_images = self.raw_class_to_images

    def crop_low_brightness(self, image_path):
        """
        裁剪掉亮度低于阈值的区域并调整大小为目标尺寸
        """
        # 加载图像
        full_path = os.path.join(self.image_folder, image_path)
        image = Image.open(full_path).convert('RGB')
        
        img_array = np.array(image)
        
        # 计算亮度值
        brightness = np.mean(img_array, axis=2)
        
        # 找出亮度高于阈值的行和列
        bright_rows = np.any(brightness >= self.brightness_threshold, axis=1)
        bright_cols = np.any(brightness >= self.brightness_threshold, axis=0)
        
        # 如果没有亮度足够高的区域，不进行裁剪
        if not np.any(bright_rows) or not np.any(bright_cols):
            cropped_image = image
        else:
            # 获取亮度足够高的区域的边界
            row_indices = np.where(bright_rows)[0]
            col_indices = np.where(bright_cols)[0]
            
            min_row, max_row = np.min(row_indices), np.max(row_indices)
            min_col, max_col = np.min(col_indices), np.max(col_indices)
            
            # 裁剪图像
            cropped_array = img_array[min_row:max_row+1, min_col:max_col+1]
            
            # 将裁剪后的numpy数组转回PIL图像
            cropped_image = Image.fromarray(cropped_array)
        
        # 调整图像大小为目标尺寸
        resized_image = cropped_image.resize(self.target_size, Image.LANCZOS)
        
        return resized_image
    
    def process_all_images(self):
        """
        处理所有采样的图像，但保留原始的sampled_images变量
        """
        self.processed_images = []
        for img_path in self.sampled_images:
            try:
                processed_img = self.crop_low_brightness(img_path)
                self.processed_images.append(processed_img)
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                # 如果处理失败，添加一个空白图像
                blank_img = Image.new('RGB', self.target_size, (0, 0, 0))
                self.processed_images.append(blank_img)

    def __len__(self):
        return 300

    def __getitem__(self, idx):
        image = self.processed_images[idx]
        label = self.sampled_labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class unlabeled_FewShotDataset(IterableDataset):
    def __init__(self, dataset, total_samples, transform):
        self.dataset = dataset
        self.batch_samples = total_samples
        self.all_images_pool = []
        for class_name in self.dataset.class_to_images:
            self.all_images_pool.extend(self.dataset.class_to_images[class_name])
        self.transform = transform
    
    def __iter__(self):
        """返回迭代器自身"""
        return self
    
    def process_image_with_brightness_crop(self, pil_img, brightness_threshold=5, target_size=(512, 512)):
        """
        使用OpenCV处理PIL图像：
        1. 裁剪掉亮度低于阈值的区域
        2. 确保裁剪区域是连续的
        3. 将图像调整为目标尺寸
        
        参数:
            pil_img: PIL.Image - 输入的PIL图像
            brightness_threshold: int - 亮度阈值
            target_size: tuple - 目标尺寸 (width, height)
            
        返回:
            PIL.Image - 处理后的图像，用于进一步转换
        """
        # 将PIL图像转换为numpy数组（RGB格式）
        img_array = np.array(pil_img)
        
        # 转换为OpenCV格式（BGR）
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 转换为灰度图像计算亮度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 创建二值掩码，亮度大于等于阈值的为255，否则为0
        _, mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # 如果没有足够亮的像素，不进行裁剪
        if np.sum(mask) == 0:
            cropped_img = img
        else:
            # 找出所有轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 如果有轮廓
            if contours:
                # 找出最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 获取包围最大轮廓的矩形
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # 裁剪图像
                cropped_img = img[y:y+h, x:x+w]
            else:
                cropped_img = img
        
        # 调整图像大小
        resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # 转换颜色空间从BGR到RGB
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        result_pil_img = Image.fromarray(rgb_img)
        
        return result_pil_img

    def __next__(self):
        """每次调用时返回随机样本"""
        # 完全随机采样
        sampled_images = random.sample(self.all_images_pool, self.batch_samples)
        
        # 转换图像
        processed_images = [
            self.transform(
                self.process_image_with_brightness_crop(
                    Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB')
                )
            ) 
            for img in sampled_images
        ]
        '''
        processed_images = [
            self.dataset.transform(
                
                    Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB')
                
            ) 
            for img in sampled_images
        ]
        '''
        # 堆叠图像
        processed_images = torch.stack(processed_images, dim=0)
        return processed_images
    
class CustomSubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        # 重新构建 class_to_images，只包含indices对应位置的内容
        self.class_to_images = {i: [] for i in range(1, 5)}
        
        # 遍历每个类别的图片列表
        for class_name, images in dataset.class_to_images.items():
            # 只保留indices中包含的索引对应的图片
            filtered_images = [img for i, img in enumerate(images) if i in indices]
            self.class_to_images[class_name] = filtered_images

        self.image_folder = dataset.image_folder  # 继承 image_folder 属性

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.dataset[original_idx]

class FewShotDataset(Dataset):
    def __init__(self, dataset, num_classes, num_support, num_query, max_sample_per_class = None, transform=None):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.all_possible_labels = list(range(self.num_classes))
        self.max_samples_per_class = max_sample_per_class
        if self.max_samples_per_class is not None:
            self.class_used_images = {class_name: set() for class_name in self.dataset.class_to_images.keys()}
        self.transform = transform

    def process_image_with_brightness_crop(self, pil_img, brightness_threshold=5, target_size=(256, 256)):
        """
        使用OpenCV处理PIL图像：
        1. 裁剪掉亮度低于阈值的区域
        2. 确保裁剪区域是连续的
        3. 将图像调整为目标尺寸
        
        参数:
            pil_img: PIL.Image - 输入的PIL图像
            brightness_threshold: int - 亮度阈值
            target_size: tuple - 目标尺寸 (width, height)
            
        返回:
            PIL.Image - 处理后的图像，用于进一步转换
        """
        # 将PIL图像转换为numpy数组（RGB格式）
        img_array = np.array(pil_img)
        
        # 转换为OpenCV格式（BGR）
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # 转换为灰度图像计算亮度
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 创建二值掩码，亮度大于等于阈值的为255，否则为0
        _, mask = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)
        
        # 如果没有足够亮的像素，不进行裁剪
        if np.sum(mask) == 0:
            cropped_img = img
        else:
            # 找出所有轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # 如果有轮廓
            if contours:
                # 找出最大的轮廓
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 获取包围最大轮廓的矩形
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # 裁剪图像
                cropped_img = img[y:y+h, x:x+w]
            else:
                cropped_img = img
        
        # 调整图像大小
        resized_img = cv2.resize(cropped_img, target_size, interpolation=cv2.INTER_LANCZOS4)
        
        # 转换颜色空间从BGR到RGB
        rgb_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        
        # 转换为PIL图像
        result_pil_img = Image.fromarray(rgb_img)
        
        return result_pil_img
    
    def change_class_name_into_prompt(self, class_name):
        for i, prompt_label in enumerate(class_name):
            if prompt_label == 1:
                class_name[i] = 'pitted surface'
            elif prompt_label == 2:
                class_name[i] = 'inclusion'
            elif prompt_label == 3:
                class_name[i] = 'scratches'
            elif prompt_label == 4:
                class_name[i] = 'patches'
            else:
                raise ValueError('class_name_not_in_supported')
        return class_name

    def __getitem__(self, index):
        # 1. 随机采样类别
        sampled_classes = random.sample(list(self.dataset.class_to_images.keys()), self.num_classes)
        
        # 2. 随机分配标签
        random_labels = random.sample(self.all_possible_labels, self.num_classes)
        label_map = {class_name: label for class_name, label in zip(sampled_classes, random_labels)}

        # 3. 为每个类别创建支持集和查询集
        support_images_by_class = []
        support_labels_by_class = []
        support_class_names_by_class = []
        query_images_by_class = []
        query_labels_by_class = []
        query_class_names_by_class = []

        for class_name in sampled_classes:
            # 随机采样该类别的图片
            class_images = random.sample(self.dataset.class_to_images[class_name], 
                                         self.num_support + self.num_query)
            
                        # 如果设置了max_samples_per_class，检查并更新已使用的图片
            if self.max_samples_per_class is not None:
                # 更新已使用的图片集合
                for img in class_images:
                    self.class_used_images[class_name].add(img)
                
                # 检查已使用的图片数量是否超过了限制
                if len(self.class_used_images[class_name]) > self.max_samples_per_class:
                    print(self.class_used_images[class_name])
                    raise ValueError(f"ERROR: Class {class_name} has used {len(self.class_used_images[class_name])} images.")
                
            # 获取该类别对应的标签
            class_label = label_map[class_name]
            
            # 分配 support set
            class_support_images = class_images[:self.num_support]
            support_images_by_class.append(class_support_images)
            support_labels_by_class.append([class_label] * self.num_support)
            support_class_names_by_class.append([class_name] * self.num_support)
            
            # 分配 query set
            class_query_images = class_images[self.num_support:]
            query_images_by_class.append(class_query_images)
            query_labels_by_class.append([class_label] * self.num_query)
            query_class_names_by_class.append([class_name] * self.num_query)

        # 4. 随机打乱类别在batch中的顺序，打破support和query之间的顺序对应关系
        # 对于支持集，我们保持类别的原始顺序
        support_images = []
        support_labels = []
        support_class_names = []
        for i in range(self.num_classes):
            support_images.extend(support_images_by_class[i])
            support_labels.extend(support_labels_by_class[i])
            support_class_names.extend(support_class_names_by_class[i])
        
        # 对于查询集，我们随机打乱类别的顺序
        query_class_indices = list(range(self.num_classes))
        random.shuffle(query_class_indices)  # 随机打乱类别顺序
        
        query_images = []
        query_labels = []
        query_class_names = []
        for i in query_class_indices:
            query_images.extend(query_images_by_class[i])
            query_labels.extend(query_labels_by_class[i])
            query_class_names.extend(query_class_names_by_class[i])
        
        # 打乱query set全部顺序
        query_pairs = list(zip(query_images, query_labels, query_class_names))
        random.shuffle(query_pairs)
        query_images, query_labels, query_class_names = zip(*query_pairs)
        query_class_names = list(query_class_names)

        
        # 5. 转换图像
        support_images = [self.transform(self.process_image_with_brightness_crop(Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB')))
                          for img in support_images]
        query_images = [self.transform(self.process_image_with_brightness_crop(Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB')))
                        for img in query_images]
        '''
        support_images = [self.transform(Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB'))
                          for img in support_images]
        query_images = [self.transform(Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB'))
                        for img in query_images]
        '''
        # 6. 转换为 tensor
        support_images = torch.stack(support_images)
        query_images = torch.stack(query_images)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        # 7. 验证数据的正确性
        assert len(support_images) == self.num_classes * self.num_support
        assert len(query_images) == self.num_classes * self.num_query
        assert len(support_labels) == self.num_classes * self.num_support
        assert len(query_labels) == self.num_classes * self.num_query
        assert len(set(support_labels.tolist())) == self.num_classes
        assert set(support_labels.tolist()) == set(query_labels.tolist())

        support_class_names = self.change_class_name_into_prompt(support_class_names)
        query_class_names = self.change_class_name_into_prompt(query_class_names)
            
        # 返回 support 和 query，同时包含 class_name 信息
        return support_images, support_labels, query_images, query_labels,\
             support_class_names, query_class_names


    def __len__(self):
        return len(self.dataset)

def load_unlabeled_data(dataset, num_classes=5, num_query=15, batch_size=6, num_workers=1, pin_memory=False, shuffle=True, prefetch_factor=4, transform = None):
    few_shot_dataset = unlabeled_FewShotDataset(dataset, num_query, transform=transform)
    data_loader = DataLoader(few_shot_dataset, batch_size=None)
    return data_loader

def check_common_indices(indice1, indice2):
        # 转换为集合后取交集
    common_elements = set(indice1.indices) & set(indice2.indices)

    if common_elements:
        print("有相同的元素:", common_elements)
    else:
        print("没有相同的元素")

def few_shot_data_loader_severstal_transfer(config, shuffle=True):
    """创建带有无标签数据的few-shot数据加载器"""
    transform_base = transforms.Compose([
        transforms.ToTensor(),
        ])

    transform_aug = transforms.Compose([
        # 随机水平和垂直翻转
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        # 随机旋转
        #transforms.RandomRotation(degrees=15),

        # 颜色抖动
        #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])
    
    image_folder = config["image_path"]
    num_workers=config["num_workers"]
    pin_memory=config["pin_memory"]
    train_ratio = 1 - config["validate_ratio"]
    base_dataset = CustomDataset_severstal(image_folder=image_folder)
    labeled_indices, unlabeled_indices, test_indices = random_split(list(range(len(base_dataset))), [config["transfer_n_shot"], int(train_ratio * len(base_dataset))-config["transfer_n_shot"], len(base_dataset) - int(train_ratio * len(base_dataset))])
    test_dataset = CustomSubsetDataset(base_dataset, test_indices)

    #labeled_indices, unlabeled_indices = random_split(train_indices, [config["transfer_n_shot"], len(train_indices) - config["transfer_n_shot"]])
    labeled_dataset = CustomSubsetDataset(base_dataset, labeled_indices)
    unlabeled_dataset = CustomSubsetDataset(base_dataset, unlabeled_indices)
    
    check_common_indices(labeled_indices, test_indices)
    check_common_indices(unlabeled_indices, test_indices)

    labeled_train_loader = load_few_shot_data_check(labeled_dataset, num_classes=config["num_classes"], num_support=config["num_support"], num_query=config["num_query"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4),transform=transform_aug)
    unlabeled_train_loader = load_unlabeled_data(unlabeled_dataset, num_classes=config["num_classes"], num_query=config["num_query"]*config["num_classes"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4),transform=transform_base)
    test_loader = load_few_shot_data(test_dataset, num_classes=config["num_classes"], num_support=config["num_support"], num_query=config["num_query"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4), transform=transform_base)

    return labeled_train_loader, test_loader, unlabeled_train_loader

def load_few_shot_data_check(dataset, num_classes=5, num_support=5, num_query=15, batch_size=6, num_workers=1, pin_memory=False, shuffle=True, prefetch_factor=4,transform=None):
    few_shot_dataset = FewShotDataset(dataset, num_classes, num_support, num_query, num_support+num_query, transform=transform)
    data_loader = DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    return data_loader

def load_few_shot_data(dataset, num_classes=5, num_support=5, num_query=15, batch_size=6, num_workers=1, pin_memory=False, shuffle=True, prefetch_factor=4, transform=None):
    few_shot_dataset = FewShotDataset(dataset, num_classes, num_support, num_query,transform=transform)
    data_loader = DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    return data_loader

def train_test_split_severstal(base_dataset, test_size=0.3):
    train_indices, validate_indices = random_split(list(range(len(base_dataset))), [int((1-test_size) * len(base_dataset)), len(base_dataset) - int((1-test_size) * len(base_dataset))])

    train_dataset = CustomSubsetDataset(base_dataset, train_indices)
    validate_dataset = CustomSubsetDataset(base_dataset, validate_indices)
    return train_dataset, validate_dataset

def few_shot_data_severstal_severstal(train_dataset, test_dataset, num_classes=5, num_support=5, num_query=15, batch_size=6, num_workers=1, pin_memory=False, shuffle=True):
    train_loader = load_few_shot_data(train_dataset, num_classes, num_support, num_query, batch_size, num_workers, shuffle)
    test_loader = load_few_shot_data(test_dataset, num_classes, num_support, num_query, batch_size, num_workers, shuffle)
    return train_loader, test_loader

def few_shot_data_loader_severstal_whole(config,shuffle):
    transform_base = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 转换到 [-1, 1]
        ])

    transform_aug = transforms.Compose([
        # 随机水平和垂直翻转
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        
        # 随机旋转
        transforms.RandomRotation(degrees=15),

        # 颜色抖动
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 转换到 [-1, 1]
    ])
    if config["dataset"] == 'severstal':
        image_folder = config["image_path"]
        num_classes = config["num_classes"]
    elif config["transfer_dataset"] == 'severstal':
        image_folder = config["transfer_image_path"]
        num_classes = config["transfer_num_classes"]
    else:
        raise ValueError
    num_workers=config["num_workers"]
    pin_memory=config["pin_memory"]
    sep = config.get("validate_ratio")
    base_dataset = CustomDataset_severstal(image_folder=image_folder)
    train_indices, test_indices = random_split(list(range(len(base_dataset))), [len(base_dataset) - int(sep * len(base_dataset)), int(sep * len(base_dataset))])

    train_dataset = CustomSubsetDataset(base_dataset, train_indices)
    test_dataset = CustomSubsetDataset(base_dataset, test_indices)

    train_loader = load_few_shot_data(train_dataset, num_classes=num_classes, num_support=config["num_support"], num_query=config["num_query"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, transform=transform_aug)
    test_loader = load_few_shot_data(test_dataset, num_classes=num_classes, num_support=config["num_support"], num_query=config["num_query"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle,transform=transform_base)

    return train_loader, test_loader