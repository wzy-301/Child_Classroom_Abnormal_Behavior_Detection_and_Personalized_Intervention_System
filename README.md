# 儿童课堂异常行为检测与干预系统

基于 **Prototypical Networks 少样本学习** 与 **CLIP 视觉-语言模型** 的儿童课堂异常行为检测与个性化干预系统。

## 功能特点

### 1. 现代化 UI 界面
- **三栏式布局**：左侧导航 + 中间主区域 + 右侧实时干预面板
- **清新蓝白配色**：圆角卡片式布局，视觉体验优化
- **实时干预建议面板**：右侧显示当前异常行为及干预建议
- **顶部参数控制栏**：滑块实时调节置信度(Conf)和交并比(IoU)
- **底部统计信息卡片**：HTML 格式可视化数据，带进度条展示

### 2. 双模型架构（Prototypical Networks + CLIP）
- **主模型**：Prototypical Networks（少样本学习）
  - 基于 EfficientNet-B0 骨干网络（ImageNet 预训练）
  - Episode 训练方式，支持 n-way k-shot 少样本学习
  - 特征空间 L2 归一化，欧氏距离分类
  - 高斯核函数将距离转换为相似度
- **备用模型**：CLIP (ViT-B/32)
  - 当 Prototypical Networks 未训练/加载失败时自动回退
  - 余弦相似度匹配多原型（每类 3-5 个聚类原型）
  - 融合多视角文本描述与负样本约束

### 3. 智能手机检测（解决 play_phone 误判）
- **YOLO 手机检测**：检测 COCO 类别 67（cell phone）
- **双重验证机制**：
  - 检测到手机物体 → 直接判定 play_phone（高置信度）
  - 未检测到手机 → Prototypical Networks 分类 + 保守降级策略
- **保守分类策略**（基于 `config.json` 可调参数）：
  - play_phone 相似度打 5 折（无手机验证时）
  - normal 优先：若 normal ≥ 0.5 且与最高异常类差距 < 0.1 → 返回 normal
  - 降级机制：play_phone 打折后仍最高 → 选第二高类别（如果差距 < 0.15）

### 4. 增强型 CLIP 原型生成（build_prototype.py）
- **多视角文本描述**：每个类别配备 3 组不同角度的英文描述
- **负样本约束**：为每个类别定义"不是..."的否定描述，降低与负样本的相似度
- **多原型聚类**：使用 K-Means 对每类图像特征聚类，生成 3-5 个原型
- **融合策略**：图像原型 + 文本引导（权重 0.4）+ 负样本约束

### 5. 实时检测与多源支持
- **摄像头实时检测**：支持 USB 摄像头实时分析
- **图片检测**：单张图片上传分析
- **视频处理**：视频文件检测并保存处理结果
- **实时性能监控**：FPS、检测耗时、目标数量实时显示

### 6. 智能干预系统
- **合并弹窗**：多个异常行为合并显示在一个弹窗中
- **实时干预建议**：右侧面板显示具体干预措施
- **异常行为列表**：带时间戳的当前异常记录
- **个性化建议**：按类别分类（lie/stand/play_phone/fight/whispering/looking_around）

### 7. 详细统计与导出
- **会话统计**：当前会话异常次数、分布、时间
- **历史统计**：累计异常总数、按行为分类
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
   - 将示例图片放入对应文件夹（详见下方"数据准备规范"）
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
    ├── stand/              # 站立行为图片
    ├── play_phone/         # 使用手机图片（必须清晰可见手机）
    ├── fight/              # 打闹行为图片（真实场景，勿用 AI 生成图）
    ├── whispering/         # 交头接耳图片
    └── looking_around/     # 东张西望图片
```

## 数据准备规范（重要）

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

### 构图一致性要求（直接影响准确率）
- **所有类别的图片必须保持构图尺度一致**：以单人或 1-3 人局部中景为主
- **`normal` 类严禁使用全班全景/广角照片**：必须是"端正坐着、看黑板、手里拿笔"的单人特写，否则会导致 normal 原型与 YOLO 切出的单人框无法匹配，引发大量误判
- **避免跨类别重复图片**：同一张图不能同时出现在两个文件夹中
- **严禁使用 AI 生成图片**：AI 图的光影和纹理分布与真实照片不同，会污染原型特征

### 各类别数量建议
| 类别 | 建议数量 | 特殊要求 |
|------|---------|---------|
| normal | 50-80 | **单人/局部构图**，多种正常听课姿态 |
| lie | 40-60 | 头趴在桌上、手垫头等 |
| stand | 40-60 | 从座位上站起、离开座位 |
| play_phone | 50+ | **必须清晰可见手机/电子设备**，多种持机姿态 |
| fight | 40-60 | 真实场景，两人以上肢体冲突 |
| whispering | 40-60 | 两人交头接耳、身体前倾 |
| looking_around | 40-60 | 头部明显转向左右、不看黑板 |

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
- K-Means 聚类生成 3-5 个图像原型
- 融合多视角文本描述（权重 0.4）
- 负样本约束：若与负描述相似度 > 0.5，则降低权重
- 输出 `prototypes.pkl`

### 2. 训练 Prototypical Networks（推荐主方案）

```bash
python train_protonet.py
```

训练参数（可在代码中调整）：
| 参数 | 默认值 | 说明 |
|------|--------|------|
| n_way | 5 | 每轮 Episode 随机选择的类别数（建议 ≤ 实际类别数） |
| k_shot | 5 | 每类 support 样本数 |
| n_query | 10 | 每类 query 样本数 |
| epochs | 1000 | 训练轮数 |
| lr | 0.001 | 学习率 |
| feature_dim | 128 | 特征空间维度 |
| backbone | EfficientNet-B0 | 骨干网络 |

数据增强策略：
- Resize(224, 224)
- RandomHorizontalFlip(p=0.5)
- RandomRotation(15°)
- ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1)
- RandomResizedCrop(224, scale=(0.8, 1.0))
- Normalize(ImageNet 均值/方差)

训练完成后生成：
- `protonet_best.pth`：最佳模型检查点
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
    "class_thresholds": {       // 各类别判定阈值
      "lie": 0.35,
      "stand": 0.40,
      "play_phone": 0.45,      // 最高阈值，防止误判
      "fight": 0.35,
      "whispering": 0.38,
      "looking_around": 0.38,
      "normal": 0.25
    },
    "protonet_fix": {           // Protonet 保守策略参数
      "phone_discount": 0.5,          // 无手机时 play_phone 相似度折扣
      "normal_min_for_fix2": 0.5,       // normal 优先策略的 normal 最低相似度
      "fix2_gap_threshold": 0.1,        // normal 优先的 gap 阈值
      "fix3_gap_threshold": 0.1,        // play_phone 降级 gap 阈值
      "fallback_gap": 0.15,             // 降级到第二类的 gap 阈值
      "phone_iou_threshold": 0.3,       // 手机框与人框的 IoU 阈值
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
| 💾 保存图片 | 保存当前检测画面 |
| 🔄 重置统计 | 清空会话和全局统计 |
| 📈 统计面板 | 查看详细统计（支持导出 CSV） |
| 📊 会话历史 | 查看所有历史检测记录 |
| ⚙️ 参数配置 | 调整检测参数阈值 |
| ℹ️ 关于系统 | 查看系统信息 |

#### 中间主区域
- **置信度(Conf)滑块**：0.10-0.90，控制 YOLO 目标检测阈值
- **交并比(IoU)滑块**：0.10-0.90，控制非极大值抑制阈值
- **检测信息**：实时显示耗时、目标数、系统状态
- **视频显示**：实时画面 + 目标框 + 类别标签 + 相似度分数

#### 右侧面板
- **🚨 实时干预建议**：异常行为卡片式展示
- **⚠️ 当前检测到的异常**：带时间戳的异常列表
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

### Prototypical Networks 分类阈值

| 行为类别 | 阈值 | 说明 |
|----------|------|------|
| normal | 0.25 | 正常行为，阈值较低，优先判定 |
| lie | 0.35 | 趴桌 |
| stand | 0.40 | 站立 |
| play_phone | 0.45 | 玩手机（最高阈值，防止误判） |
| fight | 0.35 | 打闹 |
| whispering | 0.38 | 交头接耳 |
| looking_around | 0.38 | 东张西望 |

## 保守分类策略（核心逻辑）

系统采用多层保守策略，优先判为 normal，防止课堂场景下的过度告警：

1. **手机检测优先**：YOLO 检测到手机物体（class 67）且与人框 IoU > 0.3 或距离 < 人体高度 × 1.2 → 直接判定 play_phone
2. **play_phone 打折**：未检测到手机 → play_phone 相似度 × 0.5
3. **normal 优先策略**：若 normal 相似度 ≥ 0.5 且最高异常类与 normal 差距 < 0.1 → 返回 normal
4. **降级机制**：play_phone 打折后仍最高，但与第二名差距 < 0.15 → 降级为第二高类别
5. **阈值过滤**：所有异常类必须超过各自阈值，否则退回 normal

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| 缺少原型文件 | 运行 `python build_prototype.py` 或 `python train_protonet.py` |
| YOLO 模型下载失败 | 检查网络，或手动下载 yolov8s.pt 放到根目录 |
| CUDA 内存不足 | 降低检测阈值或使用 CPU 模式 |
| 摄像头无法打开 | 检查设备连接和权限 |
| Qt 插件错误 | 安装 `libxcb-xinerama0` 等系统依赖 |
| **play_phone 大量误判** | 1. 清理 play_phone 数据集（删除无手机图片）<br>2. 检查 normal 类是否为单人构图（非全景）<br>3. 删除跨类别重复图片<br>4. 删除 AI 生成图片后重新训练 |
| 手机检测不到 | 降低手机检测阈值（`phone_iou_threshold`）或检查 YOLO 模型 |
| 所有人均被判为同一类 | 检查 `normal` 类图片构图是否与其他类一致（必须是单人/局部） |
| Protonet 加载失败 | 确保 `protonet_model.pth` 和 `protonet_prototypes.pth` 同时存在 |

## 性能优化建议

1. **GPU 加速**：确保安装 CUDA 版本的 PyTorch
2. **调整阈值**：根据实际场景调节 Conf/IoU 滑块
3. **清理数据集**：
   - 确保 play_phone 图片清晰可见手机
   - 确保 normal 为单人/局部构图（严禁全景图）
   - 删除 AI 生成图片和跨类重复图
4. **增加少数类样本**：play_phone 和 whispering 建议补充到 50+ 张
5. **训练更多轮数**：如果准确率不够，增加 epochs 到 2000-3000
6. **调整保守策略**：若某场景 normal 误判率仍高，可在 `config.json` 中降低 `normal_min_for_fix2` 或增大 `fix2_gap_threshold`

## 更新日志

### v1.3.0 (2026-04-22)
- **Prototypical Networks 集成**：新增少样本学习主模型（EfficientNet-B0 + Episode 训练）
- **增强 CLIP 原型**：多视角文本描述 + 负样本约束 + K-Means 多原型聚类
- **手机检测逻辑**：YOLO 检测手机物体（COCO class 67），解决 play_phone 误判
- **保守分类策略**：多层降级机制（打折 → normal 优先 → 降级 → 阈值过滤）
- **配置化管理**：新增 `config.json` 与 `config_manager.py`，所有阈值可配置
- **现代化 UI**：三栏式布局、蓝白配色、实时干预面板
- **IoU 滑块控制**：新增交并比实时调节
- **双模型备份**：Protonet 失败自动回退 CLIP

### v1.2.0 (2026-04-19)
- **全新 UI 界面**：三栏式现代化布局，蓝白配色方案
- **实时干预面板**：右侧新增干预建议面板和异常列表
- **顶部控制栏**：滑块式参数调节，实时显示检测信息
- **合并弹窗**：多个异常行为合并显示在一个弹窗中
- **可视化统计**：HTML 格式统计报告，带进度条展示
- **增强交互**：优化按钮布局，添加图标和状态指示
- **性能监控**：实时显示 FPS、检测耗时、目标数量

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
- **骨干网络**：EfficientNet-B0（ImageNet 预训练）
- **投影头**：1280 → 512 → 128 维特征空间
- **训练方式**：Episode 训练（5-way 5-shot，可配置）
- **分类方式**：欧氏距离 + 高斯核相似度（σ=0.5）
- **原型生成**：每类样本特征平均 + L2 归一化

### CLIP 增强原型（build_prototype.py）
- **视觉编码器**：ViT-B/32
- **文本编码器**：多视角描述（每类 3 组）+ 负样本描述
- **原型融合**：图像 K-Means 聚类中心 + 0.4×文本特征 - 0.2×负样本特征
- **多原型输出**：每类 3-5 个原型，分类时取最高相似度

### 手机检测增强
- **检测模型**：YOLOv8 COCO 预训练（class 67: cell phone）
- **重叠判断**：手机框与人框 IoU > 0.3
- **距离判断**：手机中心与人中心距离 < 人体高度 × 1.2，且手机位于人体上半身区域

### 系统特性
- **线程安全**：所有 UI 操作在主线程执行
- **设备兼容**：自动适配 CPU/GPU
- **错误恢复**：模型加载失败自动降级，检测异常自动捕获

## 注意事项

1. 首次运行自动下载 YOLOv8 模型（约 22MB）
2. **`play_phone` 数据集质量决定系统准确率**：必须清晰可见手机
3. **`normal` 类构图决定系统误判率**：严禁使用全班全景图，必须是单人/局部特写
4. 建议在 GPU 环境运行，CPU 较慢
5. 定期清理 `session_history.json`
6. 每个类别至少 25-30 张训练图片，建议 50+ 张
7. `protonet_model.pth` 和 `protonet_prototypes.pth` 必须同时存在才能加载主模型
8. 如果 play_phone 误判严重，优先检查数据集质量（手机是否可见、是否有全景图混入 normal）
9. 统计冷却时间防止同一行为频繁统计
10. 配置文件自动保存，下次启动加载
11. 手机检测依赖 YOLO COCO 预训练，对小目标效果有限
12. 训练 Protonet 前务必检查并删除跨类别重复图片和 AI 生成图片

## 许可证

MIT License

**© 2026 儿童课堂异常行为检测与干预系统**