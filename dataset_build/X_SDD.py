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
    _, train_dataloader,_,val_dataloader = get_loader_X_SDD(config, True)
elif config["few_shot"] == True:
    train_dataloader,val_dataloader = few_shot_data_loader_X_SDD_whole(config, True)
'''

def get_transform():
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        ToTensorV2(),
    ])
    return transform

def get_loader_X_SDD(config, shuffle):
    image_path = config["image_path"]
    dataset = CustomDataset_X_SDD(image_path,transform=transforms.ToTensor())
    train_dataset, validate_dataset=train_test_split_X_SDD(dataset, config["validate_ratio"])
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

class CustomDataset_X_SDD(Dataset):
    def __init__(self, image_folder, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.image_folder = image_folder 
        
        # 遍历根文件夹下的所有子文件夹
        for label in os.listdir(image_folder):
            folder_path = os.path.join(image_folder, label)
            if os.path.isdir(folder_path):  # 确保是文件夹
                # 将文件夹名作为标签
                if label not in self.class_to_idx:
                    self.class_to_idx[label] = len(self.class_to_idx)
                
                # 读取该文件夹中的所有图片
                for filename in os.listdir(folder_path):
                    if filename.endswith(".png") or filename.endswith(".jpg"):
                        self.images.append((os.path.join(label, filename), label))
        
        # 根据标签对图片进行分类
        self.class_to_images = {}
        for image, label in self.images:
            if label not in self.class_to_images:
                self.class_to_images[label] = []
            self.class_to_images[label].append(image)
        
        # 采样图片集
        self.sampled_images = []
        self.sampled_labels = []
        for label, images in self.class_to_images.items():
            sampled = random.sample(images, min(100, len(images)))
            self.sampled_images.extend(sampled)
            self.sampled_labels.extend([label] * len(sampled))
    
    def __len__(self):
        return len(self.sampled_images)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.sampled_images[idx])
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[self.sampled_labels[idx]]
        if self.transform:
            image = self.transform(image)
        return image, label

class CustomSubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.class_to_idx = dataset.class_to_idx  # 继承 class_to_idx 属性
        self.class_to_images = dataset.class_to_images  # 继承 class_to_images 属性
        self.transform = dataset.transform  # 继承 transform 属性
        self.image_folder = dataset.image_folder  # 继承 image_folder 属性

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return self.dataset[original_idx]
    
class FewShotDataset(Dataset):
    def __init__(self, dataset, num_classes, num_support, num_query):
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_support = num_support
        self.num_query = num_query
        self.all_possible_labels = list(range(self.num_classes))

    def change_class_name_into_prompt(self, class_name):
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

        # 5. 转换图像
        support_images = [self.dataset.transform(Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB')) 
                          for img in support_images]
        query_images = [self.dataset.transform(Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB')) 
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

class FewShotDatasetWithUnlabeled(FewShotDataset):
    def __init__(self, dataset, num_classes, num_support, num_query, shots_per_class=1, use_unlabeled=False):
        """
        dataset: 数据集
        num_classes: 每个episode的类别数
        num_support: 支持集中每个类别的样本数
        num_query: 查询集中每个类别的样本数
        shots_per_class: 每个类别用于标注的样本数量
        """
        super().__init__(dataset, num_classes, num_support, num_query)
        self.shots_per_class = shots_per_class
        self.use_unlabeled = use_unlabeled
        # 划分有标签和无标签数据
        self.split_dataset()
        
    def split_dataset(self):
        """将数据集分为有标签和无标签部分"""
        self.labeled_images = {}
        self.unlabeled_images = {}
        
        for class_name, images in self.dataset.class_to_images.items():
            # 确保每个类别的标注数量不超过可用样本数
            n_labeled = min(self.shots_per_class, len(images))
            labeled_subset = random.sample(images, n_labeled)
            unlabeled_subset = list(set(images) - set(labeled_subset))
            
            self.labeled_images[class_name] = labeled_subset
            self.unlabeled_images[class_name] = unlabeled_subset
            
            print(f"Class {class_name}: {n_labeled} labeled samples, {len(unlabeled_subset)} unlabeled samples")

    def sample_unlabeled_images(self, num_samples):
        """从无标签数据池中采样"""
        all_unlabeled = []
        for images in self.unlabeled_images.values():
            all_unlabeled.extend(images)
        return random.sample(all_unlabeled, min(num_samples, len(all_unlabeled)))

    def __getitem__(self, index):
        # 1. 随机采样类别
        sampled_classes = random.sample(list(self.dataset.class_to_images.keys()), self.num_classes)
        
        # 2. 随机分配标签
        random_labels = random.sample(self.all_possible_labels, self.num_classes)
        label_map = {class_name: label for class_name, label in zip(sampled_classes, random_labels)}

        support_images = []
        support_labels = []
        support_class_names = []
        query_images = []
        query_labels = []
        query_class_names = []

        # 3. 从有标签数据中采样支持集和查询集
        for class_name in sampled_classes:
            # 从有标签数据中采样一个样本，同时用于support和query
            labeled_sample = random.choice(self.labeled_images[class_name])
            
            # 对support set
            support_images.append(labeled_sample)
            support_labels.append(label_map[class_name])
            support_class_names.append(class_name)
            
            # 对query set，使用相同的样本
            query_images.append(labeled_sample)
            query_labels.append(label_map[class_name])
            query_class_names.append(class_name)

        # 4. 转换图像
        support_images = [self.dataset.transform(Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB')) 
                        for img in support_images]
        query_images = [self.dataset.transform(Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB')) 
                        for img in query_images]

        # 5. 转换为tensor
        support_images = torch.stack(support_images)
        query_images = torch.stack(query_images)
        support_labels = torch.tensor(support_labels)
        query_labels = torch.tensor(query_labels)

        # 6. 转换类别名称
        support_class_names = self.change_class_name_into_prompt(support_class_names)
        query_class_names = self.change_class_name_into_prompt(query_class_names)

        if self.use_unlabeled:
            # 采样额外的无标签数据
            num_unlabeled = self.num_query * self.num_classes
            unlabeled_samples = self.sample_unlabeled_images(num_unlabeled)
            unlabeled_images = [self.dataset.transform(Image.open(os.path.join(self.dataset.image_folder, img)).convert('RGB'))
                            for img in unlabeled_samples]
            unlabeled_images = torch.stack(unlabeled_images)
            
            return (
                support_images, support_labels,
                query_images, query_labels,
                unlabeled_images,
                support_class_names, query_class_names
            )
        else:
            return (
                support_images, support_labels,
                query_images, query_labels,
                support_class_names, query_class_names
            )

def few_shot_data_loader_X_SDD_transfer(config, shuffle=True):
    """创建带有无标签数据的few-shot数据加载器"""
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    image_folder = config["image_path"]
    num_workers = config["num_workers"]
    pin_memory = config["pin_memory"]
    train_ratio = 1 - config["validate_ratio"]
    base_dataset = CustomDataset_X_SDD(image_folder=image_folder, transform=transform)
    train_indices, test_indices = random_split(list(range(len(base_dataset))), [int(train_ratio * len(base_dataset)), len(base_dataset) - int(train_ratio * len(base_dataset))])

    train_dataset = CustomSubsetDataset(base_dataset, train_indices)
    test_dataset = CustomSubsetDataset(base_dataset, test_indices)

    labeled_indices, unlabeled_indices = random_split(list(range(len(train_dataset))), [config["transfer_n_shot"], len(train_dataset) - config["transfer_n_shot"]])
    labeled_dataset = CustomSubsetDataset(train_dataset, labeled_indices)
    unlabeled_dataset = CustomSubsetDataset(train_dataset, unlabeled_indices)
    labeled_train_loader = load_few_shot_data(labeled_dataset, num_classes=config["num_classes"], num_support=config["num_support"], num_query=config["num_query"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4))
    unlabeled_train_loader = load_unlabeled_data(unlabeled_dataset, num_classes=config["num_classes"], num_query=config["num_query"]*config["num_classes"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4))
    test_loader = load_few_shot_data(test_dataset, num_classes=config["num_classes"], num_support=config["num_support"], num_query=config["num_query"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4))

    return labeled_train_loader, test_loader, unlabeled_train_loader

def load_few_shot_data(dataset, num_classes=5, num_support=5, num_query=15, batch_size=6, num_workers=1, pin_memory=False, shuffle=True, prefetch_factor=4):
    few_shot_dataset = FewShotDataset(dataset, num_classes, num_support, num_query)
    data_loader = DataLoader(few_shot_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    return data_loader

def load_unlabeled_data(dataset, num_classes=5, num_query=15, batch_size=6, num_workers=1, pin_memory=False, shuffle=True, prefetch_factor=4):
    few_shot_dataset = unlabeled_FewShotDataset(dataset, num_query)
    data_loader = DataLoader(few_shot_dataset, batch_size=None)
    return data_loader

class unlabeled_FewShotDataset(IterableDataset):
    def __init__(self, dataset, total_samples):
        self.dataset = dataset
        self.total_samples = total_samples
        self.all_images_pool = []
        for class_name in self.dataset.class_to_images:
            self.all_images_pool.extend(self.dataset.class_to_images[class_name])
    
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
    
def train_test_split_X_SDD(base_dataset, test_size=0.3):
    train_indices, validate_indices = random_split(list(range(len(base_dataset))), [int((1-test_size) * len(base_dataset)), len(base_dataset) - int((1-test_size) * len(base_dataset))])

    train_dataset = CustomSubsetDataset(base_dataset, train_indices)
    validate_dataset = CustomSubsetDataset(base_dataset, validate_indices)
    return train_dataset, validate_dataset

def few_shot_data_loader_X_SDD(train_dataset, test_dataset, num_classes=5, num_support=5, num_query=15, batch_size=6, num_workers=1, pin_memory=False, shuffle=True):
    train_loader = load_few_shot_data(train_dataset, num_classes, num_support, num_query, batch_size, num_workers, shuffle)
    test_loader = load_few_shot_data(test_dataset, num_classes, num_support, num_query, batch_size, num_workers, shuffle)
    return train_loader, test_loader

def few_shot_data_loader_X_SDD_whole(config,shuffle):
    if config["transfer_learning"]:
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            ])
    else:
        transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  
    ])
    sep = config.get("validate_ratio")
    image_folder = config["image_path"]
    num_workers=config["num_workers"]
    pin_memory=config["pin_memory"]
    base_dataset = CustomDataset_X_SDD(image_folder=image_folder, transform=transform)
    train_indices, test_indices = random_split(list(range(len(base_dataset))), [len(base_dataset) - int(sep * len(base_dataset)), int(sep * len(base_dataset))])

    train_dataset = CustomSubsetDataset(base_dataset, train_indices)
    test_dataset = CustomSubsetDataset(base_dataset, test_indices)

    train_loader = load_few_shot_data(train_dataset, num_classes=config["num_classes"], num_support=config["num_support"], num_query=config["num_query"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4))
    test_loader = load_few_shot_data(test_dataset, num_classes=config["num_classes"], num_support=config["num_support"], num_query=config["num_query"], batch_size=config["batch_size"], num_workers=num_workers, pin_memory=pin_memory, shuffle=shuffle, prefetch_factor=config.get("prefetch_factor",4))

    return train_loader, test_loader
