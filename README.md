# 儿童课堂异常行为检测与个性化干预系统

基于 **Prototypical Networks 少样本学习** 与 **CLIP 视觉-语言模型** 的儿童课堂异常行为检测与个性化干预系统。

## 功能特点

### 1. 现代化 UI 界面
- **三栏式布局**：左侧导航 + 中间主区域 + 右侧实时干预面板
- **清新蓝白配色**：圆角卡片式布局，视觉体验优化
- **实时干预建议面板**：右侧显示当前异常行为及干预建议，支持显示异常人数（如 `lie ×3`）
- **顶部参数控制栏**：滑块实时调节置信度(Conf)和交并比(IoU)
- **底部统计信息卡片**：HTML 格式可视化数据，带进度条展示，按**人次**统计异常

### 2. 双模型架构（Prototypical Networks + CLIP）
- **主模型**：Prototypical Networks（少样本学习）
  - 基于 EfficientNet-B0 骨干网络（ImageNet 预训练，**训练时冻结**）
  - Episode 训练方式，支持 n-way k-shot 少样本学习
  - 特征空间 **L2 归一化 + 点积/余弦相似度** 分类
  - **验证集早停**：每类固定抽取 5 张验证，自动保存最佳模型
  - **原型归一化**：生成 GUI 原型时均值后重新 L2 归一化，确保度量一致
- **备用模型**：CLIP (ViT-B/32)
  - 当 Prototypical Networks 未训练/加载失败时自动回退
  - 余弦相似度匹配多原型（每类 3 个聚类原型）
  - 融合多视角文本描述

### 3. 智能手机检测（解决 play_phone 误判）
- **YOLO 手机检测**：检测 COCO 类别 67（cell phone），置信度阈值 **0.30**
- **双重验证机制**：
  - **重叠判断**：手机框与人框的覆盖比例（`cover_ratio = inter_area / phone_area`）> 0.3，适应手持场景
  - **距离判断**：手机中心与人中心距离 < 人体高度 × `phone_dist_multiplier`（默认 1.2），且手机中心位于人框水平范围 ±0.5 倍宽度内，防止邻座误伤
  - **垂直范围**：手机位于人体 20%~90% 高度区域（适应坐姿手持）
- **直接判定**：满足上述条件 → 直接判定 `play_phone`，置信度根据距离动态计算（0.70~0.92）

### 4. 标签防截断绘制
- **智能 Padding**：检测阶段先收集所有标签尺寸，若标签会超出画面上边缘或右边缘，自动通过 `cv2.copyMakeBorder` 扩展画布
- **视频保存兼容**：视频写入时自动裁切回原始尺寸，避免视频编码错误

### 5. 异常行为按人数统计
- **一帧多学生统计**：同一帧中若 3 个学生同时趴桌，统计面板记录 `lie: 3次` 而非 `lie: 1次`
- **实时干预面板**：显示 `lie (×3人)`，弹窗标题显示 `涉及 N 人`
- **CSV 导出**：导出数据包含具体人次

### 6. 增强型 CLIP 原型生成（build_prototype.py）
- **多视角文本描述**：每个类别配备 3 组不同角度的英文描述
- **多原型聚类**：使用 K-Means 对每类图像特征聚类，生成 1-3 个原型（受样本量限制，保证每簇至少 8 张）
- **融合策略**：图像原型 + 文本引导（权重 0.4）

### 7. 实时检测与多源支持
- **摄像头实时检测**：支持 USB 摄像头实时分析
- **图片检测**：单张图片上传分析，支持标签防截断扩展
- **视频处理**：视频文件检测并保存处理结果（自动处理 padding 裁切）
- **实时性能监控**：FPS、检测耗时、目标数量实时显示

### 8. 智能干预系统
- **合并弹窗**：多个异常行为合并显示在一个弹窗中，显示涉及总人数
- **实时干预建议**：右侧面板显示具体干预措施及异常人数
- **异常行为列表**：带时间戳和人数的当前异常记录
- **个性化建议**：按类别分类（lie/stand/play_phone/fight/whispering/looking_around）

### 9. 详细统计与导出
- **会话统计**：当前会话异常**人次**、分布、时间
- **历史统计**：累计异常总**人次**、按行为分类
- **可视化报告**：HTML 格式带进度条
- **数据导出**：CSV 格式统计数据导出
- **会话历史持久化**：自动保存到 `session_history.json`

## 系统要求

- Python 3.8+
- CUDA 11.7+ (推荐，GPU 加速)
- 至少 8GB RAM
- 摄像头（可选，用于实时检测）

### 依赖包
```
torch>=2.0.0
torchvision>=0.15.0
git+https://github.com/openai/CLIP.git
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.24.0
Pillow>=10.0.0
PyQt5>=5.15.0
tqdm>=4.65.0
scikit-learn>=1.3.0
```

> **注意**：`scikit-learn` 用于 `build_prototype.py` 中的 K-Means 多原型聚类，请勿遗漏。

## 安装步骤

1. 克隆或下载本仓库代码
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 准备数据：
   - 创建 `./images` 目录
   - 在 `./images` 下为每个行为类别创建子文件夹
   - 将示例图片放入对应文件夹（**详见下方"数据准备规范"，标签纯净性至关重要**）
4. 生成 CLIP 原型（备用方案）：
   ```bash
   python build_prototype.py
   ```
5. **训练 Prototypical Networks（推荐主方案）**：
   ```bash
   python train_protonet.py
   ```

## 目录结构

```
.
├── build_prototype.py      # CLIP 增强原型生成脚本（备用）
├── train_protonet.py       # Prototypical Networks 训练脚本
├── main_gui.py             # 主 GUI 程序（现代化界面）
├── config_manager.py       # 配置管理模块
├── config.json             # 配置文件（检测阈值、UI 参数等）
├── class_names.txt         # 行为类别定义
├── prototypes.pkl          # CLIP 多原型文件（生成）
├── protonet_model.pth      # Protonet 模型权重（生成）
├── protonet_prototypes.pth # Protonet 分类原型（生成）
├── protonet_best.pth       # Protonet 最佳检查点（生成）
├── yolov8s.pt              # YOLO 模型（自动下载）
├── requirements.txt        # 依赖包
├── session_history.json    # 会话历史（生成）
├── README.md               # 说明文档
└── images/                 # 训练图片目录
    ├── normal/             # 正常行为图片（建议 50+ 张，单人/局部构图）
    ├── lie/                # 趴桌行为图片
    ├── stand/              # 站立行为图片（**严禁混入坐着的人**）
    ├── play_phone/         # 使用手机图片（必须清晰可见手机）
    ├── fight/              # 打闹行为图片（真实场景，勿用 AI 生成图）
    ├── whispering/         # 交头接耳图片（**严禁混入单人正常听课图**）
    └── looking_around/     # 东张西望图片（**严禁混入正常看黑板图**）
```

## 数据准备规范（极其重要）

### 类别定义（class_names.txt）
```
normal
lie
stand
play_phone
fight
whispering
looking_around
```

### 构图一致性要求
- **所有类别的图片必须保持构图尺度一致**：以单人或 1-3 人局部中景为主
- **`normal` 类严禁使用全班全景/广角照片**：必须是"端正坐着、看黑板、手里拿笔"的单人特写，否则会导致 normal 原型与 YOLO 切出的单人框无法匹配，引发大量误判
- **避免跨类别重复图片**：同一张图不能同时出现在两个文件夹中
- **严禁使用 AI 生成图片**：AI 图的光影和纹理分布与真实照片不同，会污染原型特征

### 各类别数量建议
| 类别 | 建议数量 | 特殊要求 |
|------|---------|---------|
| normal | 50-80 | **单人/局部构图**，多种正常听课姿态 |
| lie | 40-60 | 头趴在桌上、手垫头等 |
| stand | 40-60 | **真正站立**，从座位上站起、离开座位 |
| play_phone | 50+ | **必须清晰可见手机/电子设备**，多种持机姿态 |
| fight | 40-60 | 真实场景，两人以上肢体冲突 |
| whispering | 40-60 | **两人交头接耳、身体前倾**，严禁单人图 |
| looking_around | 40-60 | **头部明显转向左右、不看黑板**，严禁正常看讲台 |

### 图片质量要求
- 分辨率建议 300×300 到 1500×1500 之间，避免过大（>2000px）或过小（<200px）
- 删除损坏、模糊、重复的图片
- 尽量覆盖不同角度、光照、服装、场景

## 使用说明

### 1. 生成 CLIP 增强原型（备用方案）

```bash
python build_prototype.py
```

生成逻辑：
- 提取每类图片的 CLIP 视觉特征
- K-Means 聚类生成 1-3 个图像原型（受样本量限制，保证每簇至少 8 张）
- 融合多视角文本描述（权重 0.4）
- 输出 `prototypes.pkl`

### 2. 训练 Prototypical Networks（推荐主方案）

```bash
python train_protonet.py
```

训练参数（可在代码中调整）：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_way | 7 | 每轮 Episode 选择的类别数（等于总类别数） |
| k_shot | 5 | 每类 support 样本数 |
| n_query | 10 | 每类 query 样本数 |
| epochs | 800 | 最大训练轮数（**实际通常 200-400 轮早停**） |
| lr | 0.001 | 学习率 |
| feature_dim | 128 | 特征空间维度 |
| backbone | EfficientNet-B0 | 骨干网络（**冻结，只训练 projector**） |
| patience | 100 | 早停耐心值（连续 100 轮验证不提升则停） |

数据增强策略：
- Resize(224, 224)
- RandomHorizontalFlip(p=0.5)
- RandomRotation(15°)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
- RandomResizedCrop(224, scale=(0.8, 1.0))
- Normalize(ImageNet 均值/方差)

**训练流程**：
1. 每类固定抽取 5 张作为验证集，其余训练
2. Backbone 冻结，只优化 projector（约 72 万参数）
3. 每 50 轮验证一次，保存最佳模型到 `protonet_best.pth`
4. 连续 200 轮验证不提升 → 自动早停
5. 用最佳模型生成 `protonet_prototypes.pth` 和 `protonet_model.pth`

训练完成后生成：
- `protonet_best.pth`：验证集最佳模型检查点
- `protonet_model.pth`：完整模型权重（供 GUI 加载）
- `protonet_prototypes.pth`：分类原型（供 GUI 加载）

### 3. 配置文件说明（config.json）

系统首次运行会自动生成默认配置，可手动调整：

```json
{
  "detection": {
    "conf_thres": 0.35,        // YOLO 人检测置信度
    "sim_thres": 0.25,         // CLIP 相似度阈值（备用模式）
    "cooldown": 4.5,           // 统计冷却时间（秒）
    "iou_thres": 0.5,          // YOLO NMS 交并比阈值
    "class_thresholds": {       // 各类别判定阈值（余弦相似度范围 0~1）
      "lie": 0.35,
      "stand": 0.45,
      "play_phone": 0.35,
      "fight": 0.25,
      "whispering": 0.42,
      "looking_around": 0.42,
      "normal": 0.15
    },
    "protonet_fix": {           // Protonet 保守策略与手机检测参数
      "phone_discount": 0.5,          // 无手机时 play_phone 相似度折扣
      "normal_min_for_fix2": 0.35,      // normal 优先策略的 normal 最低相似度
      "fix2_gap_threshold": 0.1,        // normal 优先的 gap 阈值
      "fix3_gap_threshold": 0.1,        // play_phone 降级 gap 阈值
      "fallback_gap": 0.15,             // 降级到第二类的 gap 阈值（已废弃）
      "phone_iou_threshold": 0.3,       // 手机框与人框的覆盖比例阈值
      "phone_dist_multiplier": 1.2,     // 手机距离倍数（相对人体高度）
      "phone_y_min_ratio": 0.1,         // 手机在人体的 y 范围下限
      "phone_y_max_ratio": 0.9          // 手机在人体的 y 范围上限
    }
  },
  "ui": {
    "window_width": 1854,
    "window_height": 1011,
    "theme": "light"
  },
  "paths": {
    "yolo_weight": "yolov8s.pt",
    "prototype_path": "prototypes.pkl"
  }
}
```

### 4. 启动系统

```bash
python main_gui.py
```

系统会自动检测并加载 Prototypical Networks（需 `protonet_model.pth` + `protonet_prototypes.pth`），如果未找到则回退到 CLIP 方案（需 `prototypes.pkl`）。

### 5. 界面功能

#### 左侧导航栏
| 按钮 | 功能 |
|------|------|
| 📷 摄像头 | 开启实时摄像头检测 |
| 🖼️ 图片检测 | 选择单张图片进行检测 |
| 🎬 视频保存 | 选择视频并保存处理结果 |
| ⏹️ 停止检测 | 停止当前检测任务 |
| 💾 保存图片 | 保存当前检测画面（含扩展后的完整标签） |
| 🔄 重置统计 | 清空会话和全局统计 |
| 📈 统计面板 | 查看详细统计（支持导出 CSV） |
| 📊 会话历史 | 查看所有历史检测记录 |
| ⚙️ 参数配置 | 调整检测参数阈值 |
| ℹ️ 关于系统 | 查看系统信息 |

#### 中间主区域
- **置信度(Conf)滑块**：0.10-0.90，控制 YOLO 目标检测阈值
- **交并比(IoU)滑块**：0.10-0.90，控制非极大值抑制阈值
- **检测信息**：实时显示耗时、目标数、系统状态
- **视频显示**：实时画面 + 目标框 + 类别标签 + 相似度分数，标签靠近边缘时自动扩展画布

#### 右侧面板
- **🚨 实时干预建议**：异常行为卡片式展示，显示异常人数
- **⚠️ 当前检测到的异常**：带时间戳和人数的异常列表
- **FPS 显示**：实时帧率监控

### 6. 快捷键

| 快捷键 | 功能 |
|--------|------|
| `ESC` | 停止检测 |
| `Space` | 开始/停止摄像头 |
| `Ctrl + S` | 保存当前图片 |
| `H` | 查看会话历史 |
| `P` | 打开统计面板 |
| `C` | 打开参数配置对话框 |

## 检测参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| YOLO 置信度阈值 | 0.35 | 目标检测置信度，越高越严格 |
| CLIP 相似度阈值 | 0.25 | 行为识别相似度（CLIP 备用模式） |
| 统计冷却时间 | 4.5 秒 | 同一行为再次统计的最小间隔 |
| 交并比(IoU) | 0.50 | 目标框重叠度阈值 |

### Prototypical Networks 分类阈值（余弦相似度）

| 行为类别 | 阈值 | 说明 |
|----------|------|------|
| normal | 0.15 | 正常行为，阈值最低，优先判定 |
| lie | 0.35 | 趴桌 |
| fight | 0.25 | 打闹 |
| stand | 0.45 | 站立 |
| play_phone | 0.35 | 玩手机（较高阈值） |
| whispering | 0.42 | 交头接耳 |
| looking_around | 0.42 | 东张西望 |

## 保守分类策略（6 层保护逻辑）

系统采用 **6 层保守策略**，核心目标：**宁可漏报，不可误报**，优先保护正常行为不被误判：

1. **手机检测优先**：YOLO 检测到手机物体（class 67）且满足空间约束 → 直接判定 `play_phone`
2. **play_phone 打折**：未检测到手机 → `play_phone` 相似度 × 0.5
3. **强 normal 绝对值保护（核心）**：当 `normal ≥ 0.28`，除非最高异常类显著领先（差距 ≥ 0.12），否则强制判 `normal`
4. **通用模糊保护**：若最高异常类只比 `normal` 高 < 0.10，且 `normal ≥ 0.18` → 返回 `normal`
5. **whispering/looking_around/stand 专属降级**：
   - 若第二名是 `normal` 且差距 < 0.06 → 返回 `normal`
   - 若第二名也是异常但差距 < 0.04 → 模型混乱，返回 `normal`
6. **play_phone 专属降级**：打折后仍最高，检查第二名；若第二名与 `normal` 接近 → 降级为 `normal` 或第二名
7. **阈值过滤**：所有异常类必须超过各自阈值。`whispering`/`looking_around`/`stand` 门槛较高（0.42-0.45），`play_phone` 额外要求超过阈值 + 0.05，否则退回 `normal`

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| 缺少原型文件 | 运行 `python build_prototype.py` 或 `python train_protonet.py` |
| YOLO 模型下载失败 | 检查网络，或手动下载 yolov8s.pt 放到根目录 |
| CUDA 内存不足 | 降低检测阈值或使用 CPU 模式 |
| 摄像头无法打开 | 检查设备连接和权限 |
| Qt 插件错误 | 安装 `libxcb-xinerama0` 等系统依赖 |
| **成片正常行为被判为 whispering/stand** | **1. 立即检查数据集标签纯净性**<br>2. 删除 `whispering` 中的单人正常听课图<br>3. 删除 `stand` 中的坐着写字图<br>4. 删除 `looking_around` 中的正常看黑板图<br>5. 重新运行 `train_protonet.py` 训练 |
| **play_phone 大量误判** | 1. 清理 play_phone 数据集（删除无手机图片）<br>2. 检查 normal 类是否为单人构图（非全景）<br>3. 删除跨类别重复图片<br>4. 删除 AI 生成图片后重新训练 |
| **有手机但判为 normal** | 1. 检查手机检测日志<br>2. 降低手机检测置信度（代码中 `conf=0.30` 可改为 `0.25`）<br>3. 放宽 `phone_dist_multiplier` 或 `phone_iou_threshold` |
| 手机检测不到 | 降低手机检测阈值（`phone_iou_threshold`）或检查 YOLO 模型 |
| 所有人均被判为同一类 | 检查 `normal` 类图片构图是否与其他类一致（必须是单人/局部） |
| Protonet 加载失败 | 确保 `protonet_model.pth` 和 `protonet_prototypes.pth` 同时存在 |
| 标签被截断 | 系统已自动处理：若标签超出画面，会自动扩展画布，无需手动调整 |
| 验证准确率 >80% 但测试仍误判 | **数据集标签污染**，验证集准确率高是因为验证集本身也混了错误标签。必须人工清洗数据 |

## 性能优化建议

1. **GPU 加速**：确保安装 CUDA 版本的 PyTorch
2. **调整阈值**：根据实际场景调节 Conf/IoU 滑块
3. **清理数据集（最重要）**：
   - 确保 `whispering` 里只有两人交头接耳的图，**删除所有单人图**
   - 确保 `stand` 里只有真正站起来的图，**删除所有坐着的图**
   - 确保 `looking_around` 里只有明显转头东张西望的图，**删除正常看黑板的图**
   - 确保 `normal` 为单人/局部构图（严禁全景图）
4. **增加少数类样本**：`play_phone` 和 `whispering` 建议补充到 50+ 张
5. **调整保守策略**：若某场景 normal 误判率仍高，可在 `config.json` 中：
   - 提高 `class_thresholds` 中 `whispering`/`looking_around`/`stand` 的值
   - 或修改 `main_gui.py` 中 `classify_crop_protonet` 的 gap 阈值（0.08/0.10/0.12）

## 更新日志

### v1.5.0 (2026-04-25)
- **训练优化**：
  - Backbone 冻结，只训练 projector，避免 BatchNorm 被小 batch 破坏
  - 训练/评估统一使用**点积/余弦相似度**（替代欧氏距离+高斯核），度量更稳定
  - 验证集早停：每类固定抽取 5 张验证，自动保存最佳模型
  - 原型归一化：生成 GUI 原型时均值后重新 L2 归一化
  - CosineAnnealingLR 替代 StepLR，学习率衰减更平滑
- **分类逻辑重构**：
  - 6 层保守保护策略，强化 normal 行为保护
  - 大幅提高 `whispering`/`looking_around`/`stand` 判定门槛
  - 删除 `detect_and_draw` 中的 play_phone 二次降级，避免重复打压
- **阈值更新**：同步余弦相似度范围，重新校准所有类别阈值
- **数据规范**：README 中新增"标签纯净性"强制要求，明确常见污染类型

### v1.4.0 (2026-04-23)
- **标签防截断**：检测阶段自动计算标签所需空间，通过 `cv2.copyMakeBorder` 扩展画布
- **异常按人数统计**：同一帧多个学生出现同类异常时，统计面板按人次计数
- **视频保存兼容**：视频写入时自动裁切 padding，避免编码错误
- **手机检测优化**：改用 `cover_ratio` 替代标准 IoU，增加水平范围约束
- **5 层保守分类策略**：新增通用模糊保护（gap < 0.08）

### v1.3.0 (2026-04-22)
- **Prototypical Networks 集成**：新增少样本学习主模型
- **增强 CLIP 原型**：多视角文本描述 + 负样本约束 + K-Means 多原型聚类
- **手机检测逻辑**：YOLO 检测手机物体，解决 play_phone 误判
- **保守分类策略**：多层降级机制
- **配置化管理**：新增 `config.json` 与 `config_manager.py`
- **现代化 UI**：三栏式布局、蓝白配色、实时干预面板
- **双模型备份**：Protonet 失败自动回退 CLIP

### v1.2.0 (2026-04-19)
- **全新 UI 界面**：三栏式现代化布局
- **实时干预面板**：右侧新增干预建议面板和异常列表
- **合并弹窗**：多个异常行为合并显示
- **可视化统计**：HTML 格式统计报告
- **性能监控**：实时显示 FPS、检测耗时

### v1.1.0
- 配置文件管理
- 异常处理和日志记录
- 统计功能和数据导出
- 快捷键支持

### v1.0.0
- 初始版本，基础检测功能
- CLIP 少样本原型学习

## 技术架构

### Prototypical Networks
- **骨干网络**：EfficientNet-B0（ImageNet 预训练，**训练时冻结**）
- **投影头**：1280 → 512 → 128 维特征空间
- **训练方式**：Episode 训练（7-way 5-shot），验证集早停
- **分类方式**：**点积/余弦相似度**（对于 L2 归一化向量等价于欧氏距离，但数值范围更稳定）
- **原型生成**：每类样本特征平均 + **重新 L2 归一化**，确保与训练度量一致
- **验证策略**：每类 5 张固定验证，patience=100 早停

### CLIP 增强原型（build_prototype.py）
- **视觉编码器**：ViT-B/32
- **文本编码器**：多视角描述（每类 3 组）
- **原型融合**：图像 K-Means 聚类中心（1-3 个，受样本量限制）+ 0.4×文本特征
- **多原型输出**：每类 1-3 个原型，分类时取最高相似度

### 手机检测增强
- **检测模型**：YOLOv8 COCO 预训练（class 67: cell phone），置信度 0.30
- **重叠判断**：`cover_ratio = inter_area / phone_area > 0.3`
- **距离判断**：手机中心与人中心距离 < 人体高度 × `phone_dist_multiplier`（默认 1.2）
- **水平约束**：手机中心位于人框水平范围 ±0.5 倍宽度内
- **垂直约束**：手机位于人体 20%~90% 高度

### 系统特性
- **线程安全**：所有 UI 操作在主线程执行
- **设备兼容**：自动适配 CPU/GPU
- **错误恢复**：模型加载失败自动降级，检测异常自动捕获
- **标签防截断**：自动扩展画布，确保标签完整显示
- **人次统计**：同一帧多学生同类异常按人数累计

## 注意事项

1. 首次运行自动下载 YOLOv8 模型（约 22MB）
2. **`play_phone` 数据集质量决定系统准确率**：必须清晰可见手机
3. **`normal` 类构图决定系统误判率**：严禁使用全班全景图，必须是单人/局部特写
4. **`whispering`/`looking_around`/`stand` 的标签纯净性决定系统底线**：这三个类混入正常样本会导致**成片误判所有正常行为**
5. 建议在 GPU 环境运行，CPU 较慢
6. 定期清理 `session_history.json`
7. 每个类别至少 25-30 张训练图片，建议 50+ 张
8. `protonet_model.pth` 和 `protonet_prototypes.pth` 必须同时存在才能加载主模型
9. 如果 play_phone 误判严重，优先检查数据集质量（手机是否可见、是否有全景图混入 normal）
10. 统计冷却时间防止同一行为频繁统计
11. 配置文件自动保存，下次启动加载
12. 手机检测依赖 YOLO COCO 预训练，对小目标效果有限
13. 标签靠近画面边缘时，系统会自动扩展画布，保存图片时包含完整扩展区域

## 许可证

MIT License

**© 2026 儿童课堂异常行为检测与干预系统**