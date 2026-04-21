# 儿童课堂异常行为检测与干预系统

基于 **Prototypical Networks 少样本学习** 的儿童课堂异常行为检测与个性化干预系统。

## 功能特点

### 1. 现代化 UI 界面
- **三栏式布局**：左侧导航 + 中间主区域 + 右侧实时干预面板
- **清新蓝白配色**：圆角卡片式布局，视觉体验优化
- **实时干预建议面板**：右侧显示当前异常行为及干预建议
- **顶部参数控制栏**：滑块实时调节置信度(Conf)和交并比(IoU)
- **底部统计信息卡片**：HTML 格式可视化数据，带进度条展示

### 2. 双模型架构（Prototypical Networks + CLIP）
- **主模型**：Prototypical Networks（少样本学习）
  - 基于 EfficientNet-B0 骨干网络
  - Episode 训练方式，支持 5-way 5-shot 少样本学习
  - 特征空间 L2 归一化，欧氏距离分类
- **备用模型**：CLIP (ViT-B/32)
  - 当 Prototypical Networks 未训练/加载失败时自动回退
  - 余弦相似度匹配原型

### 3. 智能手机检测（解决 play_phone 误判）
- **YOLO 手机检测**：检测 COCO 类别 67（cell phone）
- **双重验证机制**：
  - 检测到手机物体 → 直接判定 play_phone（高置信度）
  - 未检测到手机 → Prototypical Networks 分类 + 保守降级策略
- **保守分类策略**：
  - play_phone 相似度打 5 折（无手机验证时）
  - 强制优先选 normal（课堂场景正常为大多数）
  - 降级机制：play_phone 最高但无手机 → 选第二高类别

### 4. 实时检测与多源支持
- **摄像头实时检测**：支持 USB 摄像头实时分析
- **图片检测**：单张图片上传分析
- **视频处理**：视频文件检测并保存处理结果
- **实时性能监控**：FPS、检测耗时、目标数量实时显示

### 5. 智能干预系统
- **合并弹窗**：多个异常行为合并显示在一个弹窗中
- **实时干预建议**：右侧面板显示具体干预措施
- **异常行为列表**：带时间戳的当前异常记录
- **个性化建议**：按类别分类（lie/stand/play_phone/fight/whispering/looking_around）

### 6. 详细统计与导出
- **会话统计**：当前会话异常次数、分布、时间
- **历史统计**：累计异常总数、按行为分类
- **可视化报告**：HTML 格式带进度条
- **数据导出**：CSV 格式统计数据导出

## 系统要求

- Python 3.8+
- CUDA 11.7+ (推荐，GPU加速)
- 至少 8GB RAM
- 摄像头(可选，用于实时检测)

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
```

## 安装步骤

1. 克隆或下载本仓库代码
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 准备数据：
   - 创建 `./images` 目录
   - 在 `./images` 下为每个行为类别创建子文件夹
   - 将示例图片放入对应文件夹
4. 生成 CLIP 原型（备用）：
   ```bash
   python build_prototype.py
   ```
5. **训练 Prototypical Networks（推荐）**：
   ```bash
   python train_protonet.py
   ```

## 目录结构

```
.
├── build_prototype.py      # CLIP 原型生成脚本（备用）
├── train_protonet.py       # Prototypical Networks 训练脚本
├── main_gui.py             # 主GUI程序（现代化界面）
├── config_manager.py       # 配置管理模块
├── config.json             # 配置文件
├── class_names.txt         # 行为类别定义
├── prototypes.pkl          # CLIP 原型文件(生成)
├── protonet_model.pth      # Protonet 模型权重(生成)
├── protonet_prototypes.pth # Protonet 分类原型(生成)
├── yolov8s.pt              # YOLO模型(自动下载)
├── requirements.txt        # 依赖包
├── session_history.json    # 会话历史(生成)
├── README.md               # 说明文档
└── images/                 # 训练图片目录
    ├── normal/             # 正常行为图片（建议 50+ 张）
    ├── lie/                # 趴桌行为图片
    ├── stand/            # 站立行为图片
    ├── play_phone/       # 使用手机图片（必须清晰可见手机）
    ├── fight/            # 打闹行为图片
    ├── whispering/       # 交头接耳图片
    └── looking_around/   # 东张西望图片
```

## 使用说明

### 1. 数据准备

在 `class_names.txt` 中定义行为类别：
```
normal
lie
stand
play_phone
fight
whispering
looking_around
```

**重要**：`play_phone` 文件夹中的图片必须**清晰可见手机/电子设备**，否则会导致模型误判。

### 2. 训练 Prototypical Networks（推荐）

```bash
python train_protonet.py
```

训练参数：
- **n_way**：每轮选择类别数（默认 5）
- **k_shot**：每类 support 样本数（默认 5）
- **n_query**：每类 query 样本数（默认 10）
- **epochs**：训练轮数（默认 1000）
- **lr**：学习率（默认 0.001）

训练完成后生成：
- `protonet_model.pth`：模型权重
- `protonet_prototypes.pth`：分类原型
- `protonet_best.pth`：最佳模型检查点

### 3. 启动系统

```bash
python main_gui.py
```

系统会自动检测并加载 Prototypical Networks，如果未找到则回退到 CLIP。

### 4. 界面功能

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
- **视频显示**：实时画面 + 目标框 + 类别标签

#### 右侧面板
- **🚨 实时干预建议**：异常行为卡片式展示
- **⚠️ 当前检测到的异常**：带时间戳的异常列表
- **FPS 显示**：实时帧率监控

### 5. 检测标记说明

| 标记 | 含义 |
|------|------|
| `[📱]` | YOLO 检测到手机物体，直接判定 play_phone |
| `[fallback]` | 无手机但 Protonet 判为 play_phone，降级为第二高类别 |
| 无标记 | 模型正常判断结果 |

## 检测参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| YOLO 置信度阈值 | 0.35 | 目标检测置信度，越高越严格 |
| CLIP 相似度阈值 | 0.25 | 行为识别相似度（CLIP 备用模式） |
| 统计冷却时间 | 3.0秒 | 同一行为再次统计的最小间隔 |
| 交并比(IoU) | 0.50 | 目标框重叠度阈值 |

### Prototypical Networks 分类阈值

| 行为类别 | 阈值 | 说明 |
|----------|------|------|
| normal | 0.25 | 正常行为，阈值较低 |
| lie | 0.35 | 趴桌 |
| stand | 0.40 | 站立 |
| play_phone | 0.45 | 玩手机（最高阈值，防止误判） |
| fight | 0.35 | 打闹 |
| whispering | 0.38 | 交头接耳 |
| looking_around | 0.38 | 东张西望 |

## 保守分类策略

系统采用多层保守策略，优先判为 normal：

1. **手机检测优先**：检测到手机物体 → 直接 play_phone
2. **play_phone 打折**：未检测到手机 → play_phone 相似度 × 0.5
3. **normal 优先**：normal ≥ 0.2 且最高异常类差距 < 0.12 → 返回 normal
4. **降级机制**：play_phone 打折后仍最高 → 选第二高类别（如果差距 < 0.15）
5. **阈值过滤**：所有异常类必须超过各自阈值

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| 缺少原型文件 | 运行 `python build_prototype.py` 或 `python train_protonet.py` |
| YOLO 模型下载失败 | 检查网络，或手动下载 yolov8s.pt |
| CUDA 内存不足 | 降低检测阈值或使用 CPU 模式 |
| 摄像头无法打开 | 检查设备连接和权限 |
| Qt 插件错误 | 安装 `libxcb-xinerama0` 等依赖 |
| play_phone 大量误判 | 清理 play_phone 数据集（删除无手机图片），重新训练 |
| 手机检测不到 | 降低手机检测阈值（`conf=0.15`）或检查 YOLO 模型 |

## 性能优化建议

1. **GPU 加速**：确保安装 CUDA 版本的 PyTorch
2. **调整阈值**：根据实际场景调节 Conf/IoU 滑块
3. **清理数据集**：确保 play_phone 图片清晰可见手机
4. **增加 normal 样本**：建议 50+ 张，覆盖各种正常姿势
5. **训练更多轮数**：如果准确率不够，增加 epochs 到 2000-3000

## 更新日志

### v1.3.0 (2026-04-21)
- **Prototypical Networks 集成**：新增少样本学习主模型
- **手机检测逻辑**：YOLO 检测手机物体，解决 play_phone 误判
- **保守分类策略**：多层降级机制，优先判 normal
- **现代化 UI**：三栏式布局、蓝白配色、实时干预面板
- **IoU 滑块控制**：新增交并比实时调节
- **双模型备份**：Protonet 失败自动回退 CLIP

### v1.2.0 (2026-04-19)
- **全新UI界面**：三栏式现代化布局，蓝白配色方案
- **实时干预面板**：右侧新增干预建议面板和异常列表
- **顶部控制栏**：滑块式参数调节，实时显示检测信息
- **合并弹窗**：多个异常行为合并显示在一个弹窗中
- **可视化统计**：HTML格式统计报告，带进度条展示
- **增强交互**：优化按钮布局，添加图标和状态指示
- **性能监控**：实时显示FPS、检测耗时、目标数量

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
- **投影头**：512 → 128 维特征空间
- **训练方式**：Episode 训练（5-way 5-shot）
- **分类方式**：欧氏距离 + 高斯核相似度
- **原型生成**：每类样本特征平均 + L2 归一化

### 手机检测增强
- **检测模型**：YOLOv8 COCO 预训练（class 67: cell phone）
- **距离判断**：手机中心与人体中心距离 < 人体高度 × 0.7
- **位置验证**：手机位于人体上半身区域（手部区域）

### 系统特性
- **线程安全**：所有 UI 操作在主线程执行
- **设备兼容**：自动适配 CPU/GPU
- **错误恢复**：模型加载失败自动降级

## 注意事项

1. 首次运行自动下载 YOLOv8 模型（约 22MB）
2. **play_phone 数据集质量决定系统准确率**：必须清晰可见手机
3. 建议在 GPU 环境运行，CPU 较慢
4. 定期清理 `session_history.json`
5. 每个类别至少 20-30 张训练图片
6. `protonet_model.pth` 和 `protonet_prototypes.pth` 必须同时存在
7. 如果 play_phone 误判严重，重新训练并清理数据集
8. 统计冷却时间防止同一行为频繁统计
9. 配置文件自动保存，下次启动加载
10. 手机检测依赖 YOLO COCO 预训练，对小目标效果有限

## 许可证

MIT License

**© 2026 儿童课堂异常行为检测与干预系统**