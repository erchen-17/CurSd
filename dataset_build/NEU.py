import os
import random
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

'''
if config["few_shot"] == False:
    _, train_dataloader,_,val_dataloader = get_loader_NEU(config, True)
elif config["few_shot"] == True:
    train_dataloader,val_dataloader = few_shot_data_loader_NEU_whole(config, True)
'''

def get_transform():
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ])
    return transform

def get_loader_NEU(config, shuffle):
    image_path = config["image_path"]
    dataset = CustomDataset_NEU(image_path,transform=transforms.ToTensor())
    train_dataset, validate_dataset=train_test_split_NEU(dataset, config["validate_ratio"])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=shuffle,
        batch_size=config["batch_size"],
    )
    validate_dataloader = torch.utils.data.DataLoader(
        validate_dataset,
        shuffle=shuffle,
        batch_size=config["batch_size"],
    )
    return train_dataset, train_dataloader, validate_dataset, validate_dataloader

class CustomDataset_NEU(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.images = []
        self.labels = []
        self.class_to_idx = {}

        # 读取文件夹中的所有图片，并根据文件名提取标签
        for filename in os.listdir(image_folder):
            if filename.endswith(".bmp"):
                label = filename[:2]
                self.images.append((filename, label))
                if label not in self.class_to_idx:
                    self.class_to_idx[label] = len(self.class_to_idx)

        # 根据标签对图片进行分类
        self.raw_class_to_images = {}
        self.num_len = {}
        for image, label in self.images:
            if label not in self.raw_class_to_images:
                self.raw_class_to_images[label] = [image] # 直接在创建时添加第一个值
                self.num_len[label] = 1
            else:
                if len(self.raw_class_to_images[label]) <= 300: # 控制每个类别最多300张图片
                    self.raw_class_to_images[label].append(image)
                    self.num_len[label] += 1

        # 采样图片集
        self.class_to_images = self.raw_class_to_images
        self.transform = transform
    def __len__(self):
        return min(self.num_len.values())

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.sampled_images[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[self.sampled_labels[idx]]
        if self.transform != None:
            image = self.transform(image)
        return image, label

class CustomSubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.class_to_idx = dataset.class_to_idx  # 继承 class_to_idx 属性
        # 重新构建 class_to_images，只包含indices对应位置的内容
        self.class_to_images = {class_name: [] for class_name in dataset.class_to_idx.keys()}
        
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
    def __init__(self, dataset, num_classes, num_support, num_query, transform, max_samples_per_class=None):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.all_possible_labels = list(range(self.num_classes))
        self.max_samples_per_class = max_samples_per_class
        if self.max_samples_per_class is not None:
            self.class_used_images = {class_name: set() for class_name in self.dataset.class_to_images.keys()}
        self.sampled_classes = random.sample(list(self.dataset.class_to_images.keys()), self.num_classes)
        self.transform = transform

    def change_class_name_into_prompt(self, class_name):
        for i, prompt_label in enumerate(class_name):
            if prompt_label == 'PS':
                class_name[i] = 'pitted surface'
            elif prompt_label == 'RS':
                class_name[i] = 'rolled-in scale'
            elif prompt_label == 'Pa':
                class_name[i] = 'patches'
            elif prompt_label == 'Cr':
                class_name[i] = 'crazing'
            elif prompt_label == 'In':
                class_name[i] = 'inclusion'
            elif prompt_label == 'Sc':
                class_name[i] = 'scratches'
            else:
                raise ValueError('class_name_not_in_supported')
        return class_name
    
    def __getitem__(self, index):
        # 1. 随机采样类别
        
        # 2. 随机分配标签
        random_labels = random.sample(self.all_possible_labels, self.num_classes)
        label_map = {class_name: label for class_name, label in zip(self.sampled_classes, random_labels)}

        # 3. 为每个类别创建支持集和查询集
        support_images_by_class = []
        support_labels_by_class = []
        support_class_names_by_class = []
        query_images_by_class = []
        query_labels_by_class = []
        query_class_names_by_class = []

        for class_name in self.sampled_classes:
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
        support_images = [self.transform(Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB')) 
                          for img in support_images]
        query_images = [self.transform(Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB')) 
                        for img in query_images]

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


class unlabeled_FewShotDataset(IterableDataset):
    def __init__(self, dataset, total_samples, transform):
        self.dataset = dataset
        self.total_samples = total_samples
        self.all_images_pool = []
        for class_name in self.dataset.class_to_images:
            self.all_images_pool.extend(self.dataset.class_to_images[class_name])
        self.transform = transform
    
    def __iter__(self):
        """返回迭代器自身"""
        return self
    
    def __next__(self):
        """每次调用时返回随机样本"""
        # 完全随机采样
        sampled_images = random.sample(self.all_images_pool, self.total_samples)
        
        # 转换图像
        processed_images = [
            self.dataset.transform(Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB')) 
            for img in sampled_images
        ]
        
        # 堆叠图像
        processed_images = torch.stack(processed_images, dim=0)
        return processed_images

def check_common_indices(indice1, indice2):
        # 转换为集合后取交集
    common_elements = set(indice1.indices) & set(indice2.indices)

    if common_elements:
        print("有相同的元素:", common_elements)
    else:
        print("没有相同的元素")

def few_shot_data_loader_NEU_transfer(config, shuffle=True):
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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
    ])
    
    if config["dataset"] == "NEU":
        image_folder = config["image_path"]
    elif config["transfer_dataset"] == "NEU":
        image_folder = config["transfer_image_path"]
    num_workers=config["num_workers"]
    pin_memory=config["pin_memory"]
    train_ratio = 1 - config["validate_ratio"]
    base_dataset = CustomDataset_NEU(image_folder=image_folder)
    labeled_indices, unlabeled_indices, test_indices = random_split(list(range(len(base_dataset))), [config["transfer_n_shot"], int(train_ratio * len(base_dataset))-config["transfer_n_shot"], len(base_dataset) - int(train_ratio * len(base_dataset))])
    test_dataset = CustomSubsetDataset(base_dataset, test_indices)
    
    #labeled_indices, unlabeled_indices = random_split(train_indices, [config["transfer_n_shot"], len(train_indices) - config["transfer_n_shot"]])
    labeled_dataset = CustomSubsetDataset(base_dataset, labeled_indices)
    unlabeled_dataset = CustomSubsetDataset(base_dataset, unlabeled_indices)

    check_common_indices(labeled_indices, test_indices)
    check_common_indices(unlabeled_indices, test_indices)

    labeled_train_loader = load_few_shot_data(labeled_dataset, num_classes=config["num_classes"], num_support=config["num_support"], num_query=config["num_query"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4), transform=transform_aug)
    unlabeled_train_loader = load_unlabeled_data(unlabeled_dataset, num_classes=config["num_classes"], num_query=config["num_query"]*config["num_classes"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4), transform=transform_base)
    test_loader = load_test_data(test_dataset, num_classes=config["num_classes"], num_support=config["num_support"], num_query=config["num_query"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4),transform=transform_base)

    return labeled_train_loader, test_loader, unlabeled_train_loader

def load_test_data(dataset, num_classes=5, num_support=5, num_query=15, batch_size=6, num_workers=1, pin_memory=False, shuffle=True, prefetch_factor=4, transform = None):
    few_shot_dataset = FewShotDataset(dataset, num_classes, num_support, num_query,transform = transform)
    data_loader = DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    return data_loader

def load_few_shot_data(dataset, num_classes=5, num_support=5, num_query=15, batch_size=6, num_workers=1, pin_memory=False, shuffle=True, prefetch_factor=4, transform = None):
    few_shot_dataset = FewShotDataset(dataset, num_classes, num_support, num_query, transform=transform, max_samples_per_class=num_support+num_query)
    data_loader = DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    return data_loader

def load_unlabeled_data(dataset, num_classes=5, num_query=15, batch_size=6, num_workers=1, pin_memory=False, shuffle=True, prefetch_factor=4,transform = None):
    few_shot_dataset = unlabeled_FewShotDataset(dataset, num_query, transform=transform)
    data_loader = DataLoader(few_shot_dataset, batch_size=None)
    return data_loader

    
def train_test_split_NEU(base_dataset, test_size=0.3):
    train_indices, validate_indices = random_split(list(range(len(base_dataset))), [int((1-test_size) * len(base_dataset)), len(base_dataset) - int((1-test_size) * len(base_dataset))])

    train_dataset = CustomSubsetDataset(base_dataset, train_indices)
    validate_dataset = CustomSubsetDataset(base_dataset, validate_indices)
    return train_dataset, validate_dataset

def few_shot_data_loader_NEU(train_dataset, test_dataset, num_classes=5, num_support=5, num_query=15, batch_size=6, num_workers=1, pin_memory=False, shuffle=True):
    train_loader = load_few_shot_data(train_dataset, num_classes, num_support, num_query, batch_size, num_workers, shuffle)
    test_loader = load_few_shot_data(test_dataset, num_classes, num_support, num_query, batch_size, num_workers, shuffle)
    return train_loader, test_loader

def few_shot_data_loader_NEU_whole(config,shuffle):

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
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    

    num_workers=config["num_workers"]
    pin_memory=config["pin_memory"]
    train_ratio = 1 - config["validate_ratio"]
    if config["dataset"] == "NEU":
        image_folder = config["image_path"]
    elif config["transfer_dataset"] == "NEU":
        image_folder = config["transfer_image_path"]
    base_dataset = CustomDataset_NEU(image_folder=image_folder)
    train_indices, test_indices = random_split(list(range(len(base_dataset))), [int(train_ratio * len(base_dataset)), len(base_dataset) - int(train_ratio * len(base_dataset))])

    train_dataset = CustomSubsetDataset(base_dataset, train_indices)
    test_dataset = CustomSubsetDataset(base_dataset, test_indices)

    train_loader = load_test_data(train_dataset, num_classes=config["num_classes"], num_support=config["num_support"], num_query=config["num_query"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4) ,transform=transform_aug )
    test_loader = load_test_data(test_dataset, num_classes=config["num_classes"], num_support=config["num_support"], num_query=config["num_query"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4), transform=transform_base)

    return train_loader, test_loader
