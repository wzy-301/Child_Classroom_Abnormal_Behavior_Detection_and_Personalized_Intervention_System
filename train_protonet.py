import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    def __init__(self, dataset, indices=None, n_way=7, k_shot=5, n_query=15):
        self.dataset = dataset
        self.n_way = n_way  # 每轮选几个类别
        self.k_shot = k_shot  # 每类几张 support
        self.n_query = n_query  # 每类几张 query
        
        # 按类别组织数据
        self.class_to_indices = {}
        source = indices if indices is not None else range(len(dataset.samples))
        for idx in source:
            _, label, _ = dataset.samples[idx]
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
            n_needed = min(self.k_shot + self.n_query, len(indices))
            selected = random.sample(indices, n_needed)
            
            # 前 k_shot 张为 support
            for idx in selected[:self.k_shot]:
                img, _, _ = self.dataset[idx]
                support_images.append(img)
                support_labels.append(new_label)
            
            # 后 n_query 张为 query
            for idx in selected[self.k_shot:n_needed]:
                img, _, _ = self.dataset[idx]
                query_images.append(img)
                query_labels.append(new_label)
        
        # 如果 query 为空（验证集样本极少时），从 support 复制一张兜底
        if len(query_images) == 0:
            query_images.append(support_images[0].clone())
            query_labels.append(support_labels[0])
        
        return (torch.stack(support_images), torch.tensor(support_labels),
                torch.stack(query_images), torch.tensor(query_labels))


# ========== 4. 训练逻辑 ==========

def compute_prototypes(embeddings, labels):
    """计算原型（每类特征的平均，并重新 L2 归一化）"""
    n_way = len(torch.unique(labels))
    prototypes = torch.zeros(n_way, embeddings.shape[1]).to(embeddings.device)
    
    for i in range(n_way):
        mask = labels == i
        prototypes[i] = embeddings[mask].mean(dim=0)
    
    # 重新归一化，确保原型在单位球面上（与点积/余弦度量一致）
    prototypes = nn.functional.normalize(prototypes, p=2, dim=1)
    return prototypes

def train_protonet(model, train_sampler, val_sampler, epochs=1000, lr=0.001):
    """训练原型网络 - 冻结 backbone，只训练 projector，带验证早停"""
    model = model.to(DEVICE)
    
    # 冻结 backbone，只优化 projector
    for param in model.encoder.parameters():
        param.requires_grad = False
    optimizer = optim.Adam(model.projector.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0
    patience = 300
    no_improve = 0
    
    print(f"\n开始训练: 最大 {epochs} epochs, n_way={train_sampler.n_way}, k_shot={train_sampler.k_shot}")
    print("Backbone 已冻结，仅训练 projector")
    
    for epoch in tqdm(range(epochs)):
        # 混合模式：backbone eval（保持预训练 BN 统计量），projector train
        model.encoder.eval()
        model.projector.train()
        
        # 采样 episode
        support_x, support_y, query_x, query_y = train_sampler.sample_episode()
        
        support_x = support_x.to(DEVICE)
        query_x = query_x.to(DEVICE)
        support_y = support_y.to(DEVICE)
        query_y = query_y.to(DEVICE)
        
        # 提取特征
        support_embeddings = model(support_x)
        query_embeddings = model(query_x)
        
        # 计算原型
        prototypes = compute_prototypes(support_embeddings, support_y)
        
        # 点积作为 logits（对于 L2 归一化向量 = 余弦相似度）
        logits = query_embeddings @ prototypes.T
        
        # 计算损失
        loss = F.cross_entropy(logits, query_y)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # 每 50 个 episode 验证一次
        if (epoch + 1) % 50 == 0:
            val_acc = evaluate(model, val_sampler, n_episodes=50)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                torch.save(model.state_dict(), 'protonet_best.pth')
            else:
                no_improve += 50
            
            print(f"\nEpoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Val Acc: {val_acc:.2f}%, Best: {best_val_acc:.2f}%")
            
            # if no_improve >= patience:
            #     print(f"\n早停于 Epoch {epoch+1}，最佳验证准确率: {best_val_acc:.2f}%")
            #     break
    
    print(f"\n训练完成! 最佳验证准确率: {best_val_acc:.2f}%")
    print("最佳模型已保存到: protonet_best.pth")
    
    return model


# ========== 5. 评估 ==========

def evaluate(model, episode_sampler, n_episodes=100):
    """评估模型准确率 - 使用点积（与训练一致）"""
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
            
            # 与训练一致：点积（对于单位向量 = 余弦相似度）
            logits = query_embeddings @ prototypes.T
            predictions = logits.argmax(dim=1)
            
            correct += (predictions == query_y).sum().item()
            total += query_y.size(0)
    
    accuracy = correct / total * 100
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
            
            # 计算原型（均值后归一化）
            proto = torch.cat(features).mean(dim=0, keepdim=True)
            proto = nn.functional.normalize(proto, p=2, dim=1)
            
            class_name = [k for k, v in dataset.class_to_idx.items() if v == class_idx][0]
            prototypes[class_name] = proto
            class_names[class_idx] = class_name
    
    # 按 class_idx 排序，确保顺序稳定
    sorted_class_names = [class_names[i] for i in sorted(class_names.keys())]
    
    # 保存
    torch.save({
        'prototypes': prototypes,
        'class_names': sorted_class_names,
        'class_to_idx': dataset.class_to_idx,
        'model_config': {
            'feature_dim': model.feature_dim,
            'backbone': 'efficientnet_b0'
        }
    }, save_path)
    
    print(f"\nGUI 原型文件已保存到: {save_path}")
    print(f"类别: {sorted_class_names}")
    return prototypes


# ========== 7. 辅助函数 ==========

def split_dataset(dataset, k_shot=5):
    """
    每类固定抽取 k_shot 张作为验证集，其余为训练集
    保证验证集覆盖全部类别
    """
    class_to_indices = {}
    for idx, (_, label, _) in enumerate(dataset.samples):
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
    
    train_indices = []
    val_indices = []
    
    for class_idx, indices in class_to_indices.items():
        random.shuffle(indices)
        # 确保每类至少有 k_shot 张用于验证
        n_val = min(k_shot, len(indices))
        val_indices.extend(indices[:n_val])
        train_indices.extend(indices[n_val:])
    
    return train_indices, val_indices


# ========== 8. 主函数 ==========

def main():
    print("="*60)
    print("Prototypical Networks 少样本学习训练")
    print("儿童课堂异常行为检测")
    print("="*60)
    
    # 加载数据
    print("\n[1/5] 加载数据集...")
    dataset = BehaviorDataset(root_dir="./images", transform=train_transform)
    
    if len(dataset) == 0:
        print("错误: 未找到训练图片，请检查 ./images 目录")
        return
    
    num_classes = len(dataset.class_to_idx)
    print(f"类别数: {num_classes}")
    
    # 划分训练集/验证集
    print("\n[2/5] 划分训练集/验证集（每类 5 张验证）...")
    train_indices, val_indices = split_dataset(dataset, k_shot=5)
    print(f"训练集: {len(train_indices)} 张, 验证集: {len(val_indices)} 张")
    
    # 创建 episode 采样器
    n_way = num_classes
    k_shot = 5
    
    print(f"\n[3/5] 创建 Episode 采样器...")
    train_sampler = EpisodeSampler(dataset, indices=train_indices, n_way=n_way, k_shot=k_shot, n_query=10)
    # 验证集样本少，k_shot 和 n_query 都设小一点
    val_sampler = EpisodeSampler(dataset, indices=val_indices, n_way=n_way, k_shot=3, n_query=2)
    
    # 创建模型
    print(f"\n[4/5] 创建 Prototypical Network...")
    model = PrototypicalNetwork(num_classes=num_classes, feature_dim=128)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.projector.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数量 (projector): {trainable_params:,}")
    
    # 训练
    print(f"\n[5/5] 开始训练...")
    model = train_protonet(model, train_sampler, val_sampler, epochs=1000, lr=0.001)
    
    # 评估
    print("\n" + "="*60)
    print("最终评估（训练集采样）...")
    print("="*60)
    evaluate(model, train_sampler, n_episodes=100)
    
    # 生成 GUI 使用的原型文件
    print("\n" + "="*60)
    print("生成 GUI 原型文件...")
    print("="*60)
    
    # 用测试变换重新加载数据集（不做增强）
    test_dataset = BehaviorDataset(root_dir="./images", transform=test_transform)
    generate_prototypes_for_gui(model, test_dataset, 'protonet_prototypes.pth')
    
    # 同时保存完整模型（优先用早停保存的最佳模型）
    if os.path.exists('protonet_best.pth'):
        best_state = torch.load('protonet_best.pth', map_location=DEVICE)
        torch.save(best_state, 'protonet_model.pth')
        print("\n已加载最佳验证模型保存为 protonet_model.pth")
    else:
        torch.save(model.state_dict(), 'protonet_model.pth')
    
    print("\n完整模型已保存到: protonet_model.pth")
    
    print("\n" + "="*60)
    print("训练完成！")
    print("="*60)
    print("\n请确保 main_gui.py 已配置为加载:")
    print("1. protonet_model.pth 作为特征提取器")
    print("2. protonet_prototypes.pth 作为分类原型")


if __name__ == "__main__":
    main()