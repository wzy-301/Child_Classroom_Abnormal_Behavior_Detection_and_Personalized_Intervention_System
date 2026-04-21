import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import random

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {DEVICE}")

# ========== 1. 数据准备 ==========

class BehaviorDataset(Dataset):
    """行为数据集 - 从 images/ 文件夹读取"""
    def __init__(self, root_dir="./images", transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # 读取类别
        if os.path.exists("class_names.txt"):
            class_names = [c.strip() for c in open("class_names.txt", encoding='utf-8') if c.strip()]
        else:
            class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        for idx, class_name in enumerate(class_names):
            self.class_to_idx[class_name] = idx
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.samples.append((os.path.join(class_dir, img_name), idx, class_name))
        
        print(f"数据集加载完成: {len(self.samples)} 张图片, {len(class_names)} 个类别")
        for cls, idx in self.class_to_idx.items():
            count = sum(1 for _, i, _ in self.samples if i == idx)
            print(f"  {cls}: {count} 张")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label, class_name


# 数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ========== 2. Prototypical Network 模型 ==========

class PrototypicalNetwork(nn.Module):
    """原型网络 - 少样本学习核心模型"""
    def __init__(self, num_classes=7, feature_dim=128):
        super().__init__()
        
        # 使用 EfficientNet-B0 作为骨干（轻量且强大）
        backbone = models.efficientnet_b0(weights='IMAGENET1K_V1')
        self.encoder = backbone.features  # 只保留特征提取部分
        
        # 投影头：将特征映射到更适合少样本分类的空间
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, feature_dim)  # 128维特征
        )
        
        self.feature_dim = feature_dim
        
    def forward(self, x):
        """提取特征向量"""
        features = self.encoder(x)
        embeddings = self.projector(features)
        # L2 归一化（对距离计算很重要）
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings


# ========== 3. Episode 采样器 ==========

class EpisodeSampler:
    """Episode 采样器 - 少样本学习的核心"""
    def __init__(self, dataset, n_way=5, k_shot=5, n_query=15):
        self.dataset = dataset
        self.n_way = n_way  # 每轮选几个类别
        self.k_shot = k_shot  # 每类几张 support
        self.n_query = n_query  # 每类几张 query
        
        # 按类别组织数据
        self.class_to_indices = {}
        for idx, (_, label, _) in enumerate(dataset.samples):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
    
    def sample_episode(self):
        """采样一个 episode"""
        # 随机选择 n_way 个类别
        selected_classes = random.sample(list(self.class_to_indices.keys()), self.n_way)
        
        support_images = []
        support_labels = []
        query_images = []
        query_labels = []
        
        for new_label, original_label in enumerate(selected_classes):
            indices = self.class_to_indices[original_label]
            
            # 随机选 k_shot + n_query 张
            selected = random.sample(indices, self.k_shot + self.n_query)
            
            # 前 k_shot 张为 support
            for idx in selected[:self.k_shot]:
                img, _, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(new_label)
            
            # 后 n_query 张为 query
            for idx in selected[self.k_shot:]:
                img, _, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(new_label)
        
        return (torch.stack(support_images), torch.tensor(support_labels),
                torch.stack(query_images), torch.tensor(query_labels))


# ========== 4. 训练逻辑 ==========

def compute_prototypes(embeddings, labels):
    """计算原型（每类特征的平均）"""
    n_way = len(torch.unique(labels))
    prototypes = torch.zeros(n_way, embeddings.shape[1]).to(embeddings.device)
    
    for i in range(n_way):
        mask = labels == i
        prototypes[i] = embeddings[mask].mean(dim=0)
    
    return prototypes

def euclidean_distance(a, b):
    """计算欧氏距离"""
    # a: [n, d], b: [m, d]
    # 返回: [n, m]
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    return torch.pow(a - b, 2).sum(2)

def train_protonet(model, episode_sampler, epochs=1000, lr=0.001):
    """训练原型网络"""
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=300, gamma=0.5)
    
    model.train()
    best_loss = float('inf')
    
    print(f"\\n开始训练: {epochs} episodes, n_way={episode_sampler.n_way}, k_shot={episode_sampler.k_shot}")
    
    for epoch in tqdm(range(epochs)):
        # 采样 episode
        support_x, support_y, query_x, query_y = episode_sampler.sample_episode()
        
        support_x = support_x.to(DEVICE)
        query_x = query_x.to(DEVICE)
        support_y = support_y.to(DEVICE)
        query_y = query_y.to(DEVICE)
        
        # 提取特征
        support_embeddings = model(support_x)
        query_embeddings = model(query_x)
        
        # 计算原型
        prototypes = compute_prototypes(support_embeddings, support_y)
        
        # 计算 query 到原型的距离
        distances = euclidean_distance(query_embeddings, prototypes)
        
        # 距离转概率（负距离越大，概率越高）
        log_probs = -distances
        
        # 计算损失
        loss = nn.functional.cross_entropy(log_probs, query_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 记录最佳模型
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'class_to_idx': episode_sampler.dataset.class_to_idx,
                'feature_dim': model.feature_dim
            }, 'protonet_best.pth')
        
        if (epoch + 1) % 100 == 0:
            print(f"\\nEpoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Best: {best_loss:.4f}")
    
    print(f"\\n训练完成! 最佳损失: {best_loss:.4f}")
    print("模型已保存到: protonet_best.pth")
    
    return model


# ========== 5. 评估 ==========

def evaluate(model, episode_sampler, n_episodes=100):
    """评估模型准确率"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for _ in range(n_episodes):
            support_x, support_y, query_x, query_y = episode_sampler.sample_episode()
            
            support_x = support_x.to(DEVICE)
            query_x = query_x.to(DEVICE)
            query_y = query_y.to(DEVICE)
            
            support_embeddings = model(support_x)
            query_embeddings = model(query_x)
            
            prototypes = compute_prototypes(support_embeddings, support_y)
            distances = euclidean_distance(query_embeddings, prototypes)
            
            predictions = distances.argmin(dim=1)
            correct += (predictions == query_y).sum().item()
            total += query_y.size(0)
    
    accuracy = correct / total * 100
    print(f"\\n评估结果: {correct}/{total} = {accuracy:.2f}%")
    return accuracy


# ========== 6. 生成原型文件（供 GUI 使用） ==========

def generate_prototypes_for_gui(model, dataset, save_path='protonet_prototypes.pth'):
    """为 GUI 生成原型文件"""
    model.eval()
    
    prototypes = {}
    class_names = {}
    
    with torch.no_grad():
        for class_idx in sorted(dataset.class_to_idx.values()):
            # 获取该类所有图片
            indices = [i for i, (_, label, _) in enumerate(dataset.samples) if label == class_idx]
            
            if len(indices) == 0:
                continue
            
            # 提取所有特征并平均
            features = []
            for idx in indices:
                img, _, name = dataset[idx]
                img = img.unsqueeze(0).to(DEVICE)
                feat = model(img)
                features.append(feat.cpu())
            
            # 计算原型
            proto = torch.cat(features).mean(dim=0, keepdim=True)
            proto = nn.functional.normalize(proto, p=2, dim=1)
            
            class_name = [k for k, v in dataset.class_to_idx.items() if v == class_idx][0]
            prototypes[class_name] = proto
            class_names[class_idx] = class_name
    
    # 保存
    torch.save({
        'prototypes': prototypes,
        'class_names': list(class_names.values()),
        'class_to_idx': dataset.class_to_idx,
        'model_config': {
            'feature_dim': model.feature_dim,
            'backbone': 'efficientnet_b0'
        }
    }, save_path)
    
    print(f"\\nGUI 原型文件已保存到: {save_path}")
    print(f"类别: {list(prototypes.keys())}")
    return prototypes


# ========== 7. 主函数 ==========

def main():
    print("="*60)
    print("Prototypical Networks 少样本学习训练")
    print("儿童课堂异常行为检测")
    print("="*60)
    
    # 加载数据
    print("\\n[1/4] 加载数据集...")
    dataset = BehaviorDataset(root_dir="./images", transform=train_transform)
    
    if len(dataset) == 0:
        print("错误: 未找到训练图片，请检查 ./images 目录")
        return
    
    num_classes = len(dataset.class_to_idx)
    print(f"类别数: {num_classes}")
    
    # 创建 episode 采样器
    # n_way: 每轮选几个类别（建议等于总类别数或稍小）
    # k_shot: 每类几张 support（你的数据量决定）
    n_way = min(num_classes, 5)  # 如果类别少，就用全部
    k_shot = 5  # 每类取 5 张作为 support
    
    print(f"\\n[2/4] 创建 Episode 采样器 (n_way={n_way}, k_shot={k_shot})...")
    sampler = EpisodeSampler(dataset, n_way=n_way, k_shot=k_shot, n_query=10)
    
    # 创建模型
    print(f"\\n[3/4] 创建 Prototypical Network...")
    model = PrototypicalNetwork(num_classes=num_classes, feature_dim=128)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")
    
    # 训练
    print(f"\\n[4/4] 开始训练...")
    model = train_protonet(model, sampler, epochs=1000, lr=0.001)
    
    # 评估
    print("\\n" + "="*60)
    print("评估模型性能...")
    print("="*60)
    evaluate(model, sampler, n_episodes=100)
    
    # 生成 GUI 使用的原型文件
    print("\\n" + "="*60)
    print("生成 GUI 原型文件...")
    print("="*60)
    
    # 用测试变换重新加载数据集（不做增强）
    test_dataset = BehaviorDataset(root_dir="./images", transform=test_transform)
    generate_prototypes_for_gui(model, test_dataset, 'protonet_prototypes.pth')
    
    # 同时保存完整模型
    torch.save(model.state_dict(), 'protonet_model.pth')
    print("完整模型已保存到: protonet_model.pth")
    
    print("\\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print("\\n请修改 main_gui.py 以使用新模型:")
    print("1. 加载 protonet_model.pth 作为特征提取器")
    print("2. 加载 protonet_prototypes.pth 作为分类原型")
    print("3. 替换 classify_crop 函数为距离计算方式")


if __name__ == "__main__":
    main()