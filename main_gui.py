import sys
import cv2
import torch
import clip
import pickle
import numpy as np
import json
import os
import time
import csv
import logging
from PIL import Image
from ultralytics import YOLO
from datetime import datetime
from threading import Lock
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置管理器
from config_manager import ConfigManager
config_manager = ConfigManager()

# 从配置中读取参数
CONF_THRES = config_manager.get("detection.conf_thres", 0.35)
SIM_THRES = config_manager.get("detection.sim_thres", 0.25)
STAT_COOLDOWN = config_manager.get("detection.cooldown", 3.0)
YOLO_WEIGHT = config_manager.get("paths.yolo_weight", "yolov8s.pt")
PROTOTYPE_PATH = config_manager.get("paths.prototype_path", "prototypes.pkl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 干预建议映射
INTERVENTION_MAP = {
    "lie": "【干预建议】学生趴桌，提醒端正坐姿，保持专注",
    "stand": "【注意】检测到学生站立，请确认：\n1. 是否经允许回答问题\n2. 是否擅自离座\n3. 是否小组活动需要",
    "play_phone": "【干预建议】发现使用手机，提醒收起电子设备",
    "fight": "【干预建议】发现打闹行为，立即制止，维持课堂安全",
    "whispering": "【干预建议】检测到交头接耳，请提醒保持安静，专注听讲",
    "looking_around": "【干预建议】检测到东张西望，请引导关注课堂内容",
    "normal": "状态正常，无需干预"
}

# 类别特定阈值
CLASS_THRESHOLDS = {
    "lie": 0.20,          # 趴桌特征明显，可降低阈值
    "stand": 0.30,        # 站立容易误判，提高阈值
    "play_phone": 0.25,   # 保持默认
    "fight": 0.20,        # 打闹特征明显
    "whispering": 0.28,   # 交头接耳
    "looking_around": 0.30, # 东张西望
    "normal": 0.15,       # 正常行为阈值较低
}

class StatisticsManager:
    """专门管理统计数据的类"""
    def __init__(self, class_names):
        self.class_names = class_names
        self.global_stats = {c: 0 for c in class_names}
        self.session_stats = {c: 0 for c in class_names}
        self.last_stat_time = {c: datetime.min for c in class_names}
        self.lock = Lock()
        self.cooldown = STAT_COOLDOWN
        
    def update(self, behavior, cooldown=None):
        """更新统计，包含冷却时间控制"""
        if behavior not in self.class_names or behavior == "normal":
            return False
            
        if cooldown is None:
            cooldown = self.cooldown
            
        with self.lock:
            now = datetime.now()
            if (now - self.last_stat_time[behavior]).total_seconds() > cooldown:
                self.global_stats[behavior] += 1
                self.session_stats[behavior] += 1
                self.last_stat_time[behavior] = now
                return True
        return False
    
    def get_session_summary(self):
        """获取当前会话统计摘要"""
        with self.lock:
            return {
                "total": sum(self.session_stats.values()),
                "by_behavior": {k: v for k, v in self.session_stats.items() if v > 0 and k != "normal"},
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def reset_session(self):
        """重置会话统计"""
        with self.lock:
            self.session_stats = {c: 0 for c in self.class_names}
            self.last_stat_time = {c: datetime.min for c in self.class_names}
    
    def reset_all(self):
        """重置所有统计"""
        with self.lock:
            self.global_stats = {c: 0 for c in self.class_names}
            self.session_stats = {c: 0 for c in self.class_names}
            self.last_stat_time = {c: datetime.min for c in self.class_names}
    
    def get_global_summary(self):
        """获取全局统计摘要"""
        with self.lock:
            global_stats = {k: v for k, v in self.global_stats.items() if v > 0 and k != "normal"}
            return {
                "total": sum(global_stats.values()),
                "by_behavior": global_stats
            }

class PerformanceMonitor:
    """性能监控器"""
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.frame_times = []
        self.fps_history = []
        
    def add_frame_time(self, time_ms):
        """添加帧处理时间"""
        self.frame_times.append(time_ms)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)
            
    def get_fps(self):
        """计算当前FPS"""
        if not self.frame_times:
            return 0
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return 1000 / avg_time if avg_time > 0 else 0
    
    def get_stats(self):
        """获取性能统计"""
        fps = self.get_fps()
        self.fps_history.append(fps)
        if len(self.fps_history) > 100:
            self.fps_history.pop(0)
            
        return {
            "current_fps": fps,
            "avg_fps": sum(self.fps_history[-30:]) / min(30, len(self.fps_history)) if self.fps_history else 0,
            "frame_time_avg": sum(self.frame_times) / len(self.frame_times) if self.frame_times else 0
        }

def check_required_files():
    """检查必要的模型和配置文件"""
    missing_files = []
    required = [PROTOTYPE_PATH, YOLO_WEIGHT]
    
    for file_path in required:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        error_msg = f"缺少必要文件：{', '.join(missing_files)}\n"
        error_msg += "请确保：\n1. 已运行 build_prototype.py 生成 prototypes.pkl\n"
        error_msg += "2. 已下载 yolov8s.pt 模型文件到当前目录"
        return False, error_msg
    
    return True, ""

# 加载模型和原型
try:
    with open(PROTOTYPE_PATH, "rb") as f:
        data = pickle.load(f)
        prototypes = data["prototypes"]
        CLASS_NAMES = data["class_names"]
except Exception as e:
    logger.error(f"加载原型文件失败: {str(e)}")
    QMessageBox.critical(None, "加载失败", f"无法加载原型文件: {str(e)}")
    sys.exit(1)

clip_model, preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
yolo = YOLO(YOLO_WEIGHT)

def classify_crop(frame, box, use_class_specific=True):
    """保守分类策略 - 优先判为 normal"""
    try:
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return "normal", 0
            
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "normal", 0
            
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img = preprocess(pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            feat = clip_model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
        
        # 计算与所有原型的相似度
        similarities = {}
        for cls, proto_tensor in prototypes.items():
            if proto_tensor.dim() == 2 and proto_tensor.shape[0] > 1:
                # 多原型：取最大
                sims = torch.cosine_similarity(feat, proto_tensor.to(DEVICE))
                sim = sims.max().item()
            else:
                sim = torch.cosine_similarity(feat, proto_tensor.to(DEVICE)).item()
            similarities[cls] = sim
        
        # 计算相似度
        normal_sim = similarities.get("normal", 0)
        phone_sim = similarities.get("play_phone", 0)
        max_sim = max(similarities.values())
        max_cls = max(similarities, key=similarities.get)
    
        # 1. 如果 normal 与stand相似度差距 < 0.1，且stand相似度小于0.80,优先判为 normal
        if max_cls == "stand" and max_sim - normal_sim < 0.10 and max_sim < 0.80:
            return "normal", normal_sim
    
        # 2. play_phone 必须比 normal 高至少 0.1，且绝对值 > 0.35
        if max_cls == "play_phone":
            if abs(phone_sim - normal_sim) < 0.08:
                return "normal", normal_sim
            if phone_sim - normal_sim < 0.10:  # 差距不够大
                return "normal", normal_sim
            if phone_sim < 0.35:  # 置信度不够高
                return "normal", normal_sim
        
        # 3. 异常类必须超过各自的高阈值
        thresholds = {
            "normal": 0.15,
            "lie": 0.15,        
            "stand": 0.30,      
            "play_phone": 0.35, 
            "fight": 0.20,      
            "whispering": 0.25, 
            "looking_around": 0.25  
        }
        
        threshold = thresholds.get(max_cls, 0.35)
        
        if max_sim > threshold:
            return max_cls, max_sim
        else:
            return "normal", normal_sim
            
    except Exception as e:
        return "normal", 0

def detect_and_draw(frame, conf_thres=CONF_THRES, sim_thres=SIM_THRES, use_class_specific=True):
    """检测并绘制结果"""
    current_abnormal = set()
    try:
        results = yolo(frame, classes=[0], conf=conf_thres)
        for r in results:
            for box in r.boxes.xyxy:
                cls_name, sim = classify_crop(frame, box, use_class_specific)
                if cls_name not in CLASS_NAMES:
                    continue

                # 判断是否为异常行为
                is_abnormal = cls_name != "normal"
                if is_abnormal:
                    current_abnormal.add(cls_name)
                
                # 绘制边界框和标签
                x1, y1, x2, y2 = map(int, box)
                color = (0, 0, 255) if is_abnormal else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{cls_name} {sim:.2f}",
                            (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    except Exception as e:
        logger.error(f"检测异常: {str(e)}")
    
    return frame, current_abnormal

class VideoThread(QThread):
    """视频处理线程"""
    frame_signal = pyqtSignal(np.ndarray, object)
    error_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)

    def __init__(self, source=0, save_path=None, conf_thres=CONF_THRES, sim_thres=SIM_THRES):
        super().__init__()
        self.source = source
        self.save_path = save_path
        self.conf_thres = conf_thres
        self.sim_thres = sim_thres
        self.running = True
        self.writer = None
        self.cap = None

    def run(self):
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                self.error_signal.emit("无法打开视频源")
                return

            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            if self.save_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.writer = cv2.VideoWriter(self.save_path, fourcc, fps, (w, h))

            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame, ab = detect_and_draw(frame, self.conf_thres, self.sim_thres)
                if self.writer:
                    self.writer.write(frame)
                self.frame_signal.emit(frame, ab)
                frame_count += 1
                if total_frames > 0:
                    self.progress_signal.emit(int(frame_count/total_frames*100))
        except Exception as e:
            logger.error(f"视频处理异常: {str(e)}")
            self.error_signal.emit(f"处理异常：{str(e)}")
        finally:
            self.safe_release()

    def safe_release(self):
        if self.cap:
            self.cap.release()
        if self.writer:
            self.writer.release()

class ConfigDialog(QDialog):
    """参数配置对话框"""
    def __init__(self, parent=None, conf_thres=CONF_THRES, sim_thres=SIM_THRES, cooldown=STAT_COOLDOWN):
        super().__init__(parent)
        self.setWindowTitle("检测参数配置")
        self.setMinimumSize(400, 300)
        self.conf_thres = conf_thres
        self.sim_thres = sim_thres
        self.cooldown = cooldown
        self.init_ui()
        
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        form_layout = QFormLayout()
        
        # YOLO置信度阈值
        self.yolo_conf_slider = QSlider(Qt.Horizontal)
        self.yolo_conf_slider.setRange(10, 90)  # 0.1-0.9
        self.yolo_conf_slider.setValue(int(self.conf_thres * 100))
        self.yolo_conf_label = QLabel(f"{self.conf_thres:.2f}")
        self.yolo_conf_slider.valueChanged.connect(
            lambda v: self.yolo_conf_label.setText(f"{v/100:.2f}"))
        form_layout.addRow("YOLO置信度阈值:", self.yolo_conf_slider)
        form_layout.addRow("当前值:", self.yolo_conf_label)
        
        # CLIP相似度阈值
        self.clip_sim_slider = QSlider(Qt.Horizontal)
        self.clip_sim_slider.setRange(10, 90)  # 0.1-0.9
        self.clip_sim_slider.setValue(int(self.sim_thres * 100))
        self.clip_sim_label = QLabel(f"{self.sim_thres:.2f}")
        self.clip_sim_slider.valueChanged.connect(
            lambda v: self.clip_sim_label.setText(f"{v/100:.2f}"))
        form_layout.addRow("CLIP相似度阈值:", self.clip_sim_slider)
        form_layout.addRow("当前值:", self.clip_sim_label)
        
        # 冷却时间
        self.cooldown_spin = QDoubleSpinBox()
        self.cooldown_spin.setRange(0.5, 10.0)
        self.cooldown_spin.setSingleStep(0.5)
        self.cooldown_spin.setValue(self.cooldown)
        self.cooldown_spin.setSuffix(" 秒")
        form_layout.addRow("统计冷却时间:", self.cooldown_spin)
        
        layout.addLayout(form_layout)
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_apply = QPushButton("应用")
        self.btn_apply.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        
        btn_layout.addWidget(self.btn_apply)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
    
    def get_values(self):
        return {
            'conf_thres': self.yolo_conf_slider.value() / 100,
            'sim_thres': self.clip_sim_slider.value() / 100,
            'cooldown': self.cooldown_spin.value()
        }

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # 检查必要文件
        success, msg = check_required_files()
        if not success:
            QMessageBox.critical(None, "文件缺失错误", msg)
            sys.exit(1)
        
        # 初始化参数
        self.conf_thres = CONF_THRES
        self.sim_thres = SIM_THRES
        self.cooldown = STAT_COOLDOWN
        
        # 统计管理器
        self.stat_manager = StatisticsManager(CLASS_NAMES)
        
        # 性能监控
        self.performance_monitor = PerformanceMonitor()
        
        # 界面初始化
        self.init_window()
        self.initUI()
        self.load_history()
        
    def init_window(self):
        """初始化窗口"""
        width = config_manager.get("ui.window_width", 1300)
        height = config_manager.get("ui.window_height", 850)
        self.setWindowTitle("儿童课堂异常行为检测与干预系统")
        self.setGeometry(100, 100, width, height)
        
    def initUI(self):
        """初始化现代化界面 - 保留所有原有功能"""
        # 设置全局样式
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f4f8;
                font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
            }
            QLabel {
                font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
            }
            QPushButton {
                font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
            }
        """)
        
        # 主窗口
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ========== 左侧导航栏（20%） ==========
        left_panel = QWidget()
        left_panel.setObjectName("left_panel")
        left_panel.setStyleSheet("""
            #left_panel {
                background-color: #ffffff;
                border-right: 1px solid #e0e6ed;
            }
        """)
        left_panel.setMaximumWidth(250)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(12)
        
        # 标题
        nav_title = QLabel("🎓 儿童课堂异常行为检测与干预系统")
        nav_title.setStyleSheet("""
            font-size: 18px;
            font-weight: bold;
            color: #2c3e50;
            padding-bottom: 15px;
            border-bottom: 3px solid #5bc0de;
        """)
        left_layout.addWidget(nav_title)
        
        # 导航按钮 - 保留所有原有功能
        nav_buttons = [
            ("📷 摄像头", self.open_cam),
            ("🖼️ 图片检测", self.open_img),
            ("🎬 视频保存", self.open_video_save),
            ("⏹️ 停止检测", self.stop_all),
            ("💾 保存图片", self.save_img),
            ("🔄 重置统计", self.reset_stats),
            ("📈 统计面板", self.show_statistics_panel),
            ("📊 会话历史", self.show_session_history),
            ("⚙️ 参数配置", self.open_config_dialog),
            ("ℹ️ 关于系统", self.show_about_dialog),
        ]
        
        for text, callback in nav_buttons:
            btn = QPushButton(text)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #5bc0de;
                    color: white;
                    border: none;
                    border-radius: 8px;
                    padding: 10px 18px;
                    font-size: 13px;
                    font-weight: 500;
                    text-align: left;
                    min-height: 40px;
                }
                QPushButton:hover {
                    background-color: #46b8da;
                }
                QPushButton:pressed {
                    background-color: #31b0d5;
                }
            """)
            btn.clicked.connect(callback)
            left_layout.addWidget(btn)
        
        left_layout.addStretch()
        
        # 版本信息
        version_label = QLabel("v1.1.0 | 儿童课堂异常行为检测与干预系统")
        version_label.setStyleSheet("color: #a0aec0; font-size: 11px;")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setWordWrap(True)
        left_layout.addWidget(version_label)
        
        main_layout.addWidget(left_panel, stretch=2)
        
        # ========== 中间主区域（55%） ==========
        center_panel = QWidget()
        center_panel.setStyleSheet("background-color: #f8fafc;")
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(20, 20, 20, 20)
        center_layout.setSpacing(15)
        
        # 顶部控制栏 - 包含原有参数
        control_bar = QWidget()
        control_bar.setStyleSheet("""
            background-color: #ffffff;
            border-radius: 10px;
            border: 1px solid #e0e6ed;
        """)
        control_layout = QHBoxLayout(control_bar)
        control_layout.setContentsMargins(15, 15, 15, 15)
        
        # 置信度滑块（原有功能）
        conf_layout = QHBoxLayout()
        conf_label = QLabel("置信度 (Conf):")
        conf_label.setStyleSheet("font-size: 13px; color: #4a5568; font-weight: 500;")
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(10, 90)
        self.conf_slider.setValue(int(self.conf_thres * 100))
        self.conf_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 6px;
                background: #e0e6ed;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #5bc0de;
                width: 16px;
                height: 16px;
                border-radius: 8px;
                margin: -5px 0;
            }
            QSlider::sub-page:horizontal {
                background: #5bc0de;
                border-radius: 3px;
            }
        """)
        self.conf_value = QLabel(f"{self.conf_thres:.2f}")
        self.conf_value.setStyleSheet("font-size: 13px; color: #5bc0de; font-weight: bold; min-width: 35px;")
        self.conf_slider.valueChanged.connect(lambda v: self.conf_value.setText(f"{v/100:.2f}"))
        
        conf_layout.addWidget(conf_label)
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_value)
        control_layout.addLayout(conf_layout, stretch=2)
        
        # 交并比滑块（新增但有用）
        iou_layout = QHBoxLayout()
        iou_label = QLabel("交并比 (IoU):")
        iou_label.setStyleSheet("font-size: 13px; color: #4a5568; font-weight: 500;")
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(10, 90)
        self.iou_slider.setValue(50)
        self.iou_slider.setStyleSheet(self.conf_slider.styleSheet())
        self.iou_value = QLabel("0.50")
        self.iou_value.setStyleSheet("font-size: 13px; color: #5bc0de; font-weight: bold; min-width: 35px;")
        self.iou_slider.valueChanged.connect(lambda v: self.iou_value.setText(f"{v/100:.2f}"))
        
        iou_layout.addWidget(iou_label)
        iou_layout.addWidget(self.iou_slider)
        iou_layout.addWidget(self.iou_value)
        control_layout.addLayout(iou_layout, stretch=2)
        
        # 检测信息卡片
        info_card = QWidget()
        info_card.setStyleSheet("""
            background-color: #f7fafc;
            border-radius: 8px;
            border: 1px solid #e2e8f0;
        """)
        info_layout = QHBoxLayout(info_card)
        info_layout.setContentsMargins(15, 10, 15, 10)
        
        self.time_label = QLabel("⏱️ 检测耗时: 0.00s")
        self.time_label.setStyleSheet("font-size: 12px; color: #4a5568;")
        self.target_label = QLabel("🎯 检测目标: 0个")
        self.target_label.setStyleSheet("font-size: 12px; color: #4a5568;")
        self.status = QLabel("⏹️ 已停止")
        self.status.setStyleSheet("font-size: 12px; color: #fc8181; font-weight: bold;")
        
        info_layout.addWidget(self.time_label)
        info_layout.addWidget(self.target_label)
        info_layout.addWidget(self.status)
        control_layout.addWidget(info_card, stretch=2)
        
        center_layout.addWidget(control_bar)
        
        # 视频显示区域（原有）
        video_container = QWidget()
        video_container.setStyleSheet("""
            background-color: #ffffff;
            border-radius: 12px;
            border: 2px solid #e0e6ed;
            padding: 10px;
        """)
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(10, 10, 10, 10)
        
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumHeight(400)
        self.label.setStyleSheet("""
            background-color: #f0f4f8;
            border-radius: 8px;
            border: 2px dashed #cbd5e0;
            color: #a0aec0;
            font-size: 16px;
        """)
        self.label.setText("<div align='center'>📹 实时检测画面<br/><br/>请选择图片、视频或开启摄像头</div>")
        video_layout.addWidget(self.label)
        
        center_layout.addWidget(video_container, stretch=1)
        
        # 进度条（原有）
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #e0e6ed;
                border-radius: 5px;
                height: 8px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #5bc0de;
                border-radius: 5px;
            }
        """)
        center_layout.addWidget(self.progress_bar)
        
        # 统计信息标签（原有，移到中间底部）
        self.stat_label = QLabel("📊 异常行为统计：暂无数据")
        self.stat_label.setTextFormat(Qt.RichText)  
        self.stat_label.setWordWrap(True)  # 自动换行
        self.stat_label.setStyleSheet("""
            background-color: #ffffff;
            border-radius: 8px;
            padding: 12px 15px;
            border: 1px solid #e0e6ed;
            font-size: 13px;
            font-weight: bold;
            color: #4a5568;
        """)
        center_layout.addWidget(self.stat_label)
        
        main_layout.addWidget(center_panel, stretch=5)
        
        # ========== 右侧面板（25%） ==========
        right_panel = QWidget()
        right_panel.setStyleSheet("background-color: #ffffff; border-left: 1px solid #e0e6ed;")
        right_panel.setMaximumWidth(350)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(15)
        
        # 系统标题
        system_title = QWidget()
        system_title.setStyleSheet("""
            background-color: #f7fafc;
            border-radius: 10px;
            border: 1px solid #e2e8f0;
        """)
        system_layout = QHBoxLayout(system_title)
        system_layout.setContentsMargins(15, 15, 15, 15)
        
        icon_label = QLabel("🎯")
        icon_label.setStyleSheet("font-size: 32px;")
        title_text = QLabel("YOLO检测系统")
        title_text.setStyleSheet("font-size: 16px; font-weight: bold; color: #2c3e50;")
        
        system_layout.addWidget(icon_label)
        system_layout.addWidget(title_text)
        system_layout.addStretch()
        right_layout.addWidget(system_title)
        
        # 实时干预建议面板
        advice_panel = QWidget()
        advice_panel.setStyleSheet("""
            background-color: #fffaf0;
            border-radius: 10px;
            border: 1px solid #fed7aa;
            padding: 15px;
        """)
        advice_layout = QVBoxLayout(advice_panel)
        advice_layout.setContentsMargins(0, 0, 0, 0)
        advice_layout.setSpacing(10)

        advice_title = QLabel("🚨 实时干预建议")
        advice_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #c05621;")
        advice_layout.addWidget(advice_title)

        self.advice_text = QTextEdit()
        self.advice_text.setReadOnly(True)
        self.advice_text.setMinimumHeight(180)  # 设置最小高度
        self.advice_text.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.advice_text.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.advice_text.setStyleSheet("""
            QTextEdit {
                background-color: #fff5eb;
                border: none;
                border-radius: 5px;
                padding: 10px;
                font-size: 12px;
                line-height: 1.5;
            }
            QScrollBar:vertical {
                background-color: #f0f0f0;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #cbd5e0;
                border-radius: 6px;
                min-height: 40px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #a0aec0;
            }
        """)
        self.advice_text.setPlaceholderText("暂无异常行为...")
        advice_layout.addWidget(self.advice_text, stretch=1)  # 添加 stretch 让它占据更多空间

        # 当前异常列表
        abnormal_title = QLabel("⚠️ 当前检测到的异常")
        abnormal_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #9c4221; margin-top: 5px;")
        advice_layout.addWidget(abnormal_title)

        self.current_abnormal_list = QListWidget()
        self.current_abnormal_list.setMinimumHeight(120)  # 设置最小高度
        self.current_abnormal_list.setStyleSheet("""
            QListWidget {
                background-color: #ffffff;
                border: 1px solid #fed7aa;
                border-radius: 5px;
                padding: 5px;
            }
            QListWidget::item {
                padding: 8px;
                border-bottom: 1px solid #feebc8;
                color: #9c4221;
                font-size: 11px;
            }
            QListWidget::item:selected {
                background-color: #ed8936;
                color: white;
            }
        """)
        advice_layout.addWidget(self.current_abnormal_list, stretch=1)

        right_layout.addWidget(advice_panel, stretch=3)  # 增加 stretch 权重
        
        # FPS显示（原有）
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #a0aec0; font-size: 12px;")
        self.fps_label.setAlignment(Qt.AlignRight)
        right_layout.addWidget(self.fps_label)
        
        right_layout.addStretch()
        
        main_layout.addWidget(right_panel, stretch=3)
        
        # ========== 初始化所有原有变量 ==========
        self.thread = None
        self.current_frame = None
        self.current_session = None
        self.shown_behaviors = set()
        self.session_saved = False
        self.session_history = []
        self.current_frame_abnormal = set()
        self.detection_time = 0
        
        # 加载历史记录（原有功能）
        self.load_history()
        
    def load_history(self):
        """加载历史记录"""
        if os.path.exists("session_history.json"):
            try:
                with open("session_history.json", "r", encoding="utf-8") as f:
                    self.session_history = json.load(f)
            except Exception as e:
                logger.error(f"加载历史记录失败: {str(e)}")
                
    def start_new_session(self, session_type, source_info=""):
        """开始新会话"""
        # 保存当前会话
        if self.current_session and not self.session_saved:
            self.save_current_session()
        
        # 重置会话相关状态
        self.shown_behaviors = set()
        self.stat_manager.reset_session()
        
        # 创建新会话
        self.current_session = {
            "type": session_type,
            "source": source_info,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "behaviors": [],
            "config": {
                "conf_thres": self.conf_thres,
                "sim_thres": self.sim_thres,
                "cooldown": self.cooldown
            }
        }
        self.session_saved = False
        
    def log_behavior(self, behavior):
        """记录行为"""
        if self.current_session is None:
            return
        self.current_session["behaviors"].append({
            "behavior": behavior,
            "time": datetime.now().strftime("%H:%M:%S"),
            "advice": INTERVENTION_MAP.get(behavior, "")
        })

    def save_current_session(self):
        """保存当前会话"""
        if self.current_session and not self.session_saved:
            self.current_session["end_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.session_history.append(self.current_session)
            try:
                with open("session_history.json", "w", encoding="utf-8") as f:
                    json.dump(self.session_history, f, ensure_ascii=False, indent=2)
                self.session_saved = True
            except Exception as e:
                logger.error(f"保存会话失败: {str(e)}")
                
    def show_frame(self, frame, ab_set):
        """显示帧图像 - 更新现代化界面"""
        start_time = time.time()
        
        self.current_frame = frame.copy()
        self.current_frame_abnormal = ab_set
        
        # 更新状态标签
        if ab_set and ab_set != {"normal"}:
            self.status.setText("🔴 检测中 - 发现异常")
            self.status.setStyleSheet("font-size: 12px; color: #fc8181; font-weight: bold;")
        else:
            self.status.setText("🟢 检测中 - 正常")
            self.status.setStyleSheet("font-size: 12px; color: #48bb78; font-weight: bold;")
        
        # 更新目标数
        self.target_label.setText(f"🎯 检测目标: {len(ab_set)}个")
        
        # 原有显示代码...
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bpl = ch * w
        q_img = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888).copy()
        self.label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 更新干预建议面板
        self.update_advice_panel(ab_set)
        
        # 更新统计（原有代码）
        new_abnormal = False
        for ab in ab_set:
            if self.stat_manager.update(ab, self.cooldown):
                new_abnormal = True
                if ab not in self.shown_behaviors:
                    self.shown_behaviors.add(ab)
                    self.log_behavior(ab)
        
        # 显示合并弹窗（原有代码）
        if new_abnormal and ab_set:
            self.show_merged_intervention_dialog(ab_set)
        
        self.update_stat()
        
        # 更新性能监控
        end_time = time.time()
        process_time = (end_time - start_time) * 1000
        self.performance_monitor.add_frame_time(process_time)
        stats = self.performance_monitor.get_stats()
        self.fps_label.setText(f"FPS: {stats['current_fps']:.1f}")
        self.time_label.setText(f"⏱️ 检测耗时: {process_time/1000:.2f}s")


    def update_advice_panel(self, ab_set):
        """更新右侧干预建议面板"""
        # 清空当前异常列表
        self.current_abnormal_list.clear()
        
        if not ab_set or ab_set == {"normal"}:
            # 无异常
            self.advice_text.setHtml("""
                <div style="color: #5cb85c; text-align: center; padding: 20px;">
                    <h3>✅ 课堂秩序良好</h3>
                    <p>当前未检测到异常行为</p>
                    <p>所有学生专注学习</p>
                </div>
            """)
            self.current_abnormal_list.addItem("无异常")
            self.current_abnormal_list.item(0).setForeground(QColor("#5cb85c"))
            return
        
        # 构建干预建议文本
        advice_html = "<div style=\"padding: 10px;\">"
        advice_html += "<h3 style=\"color: #d9534f; margin-bottom: 15px;\">⚠️ 检测到异常行为</h3>"
        
        for ab in sorted(ab_set):
            if ab == "normal":
                continue
                
            # 添加到列表
            item_text = f"{ab} ({datetime.now().strftime('%H:%M:%S')})"
            self.current_abnormal_list.addItem(item_text)
            
            # 添加干预建议到文本区域
            advice = INTERVENTION_MAP.get(ab, f"【注意】检测到 {ab} 行为")
            advice_html += f"""
                <div style="background-color: #f8d7da; border-left: 4px solid #d9534f; 
                            padding: 10px; margin-bottom: 10px; border-radius: 4px;">
                    <strong style="color: #721c24;">{ab}</strong><br/>
                    <span style="color: #856404;">{advice}</span>
                </div>
            """
        
        advice_html += "</div>"
        self.advice_text.setHtml(advice_html)


    def show_merged_intervention_dialog(self, ab_set):
        """显示合并的干预建议弹窗 - 修复布局"""
        if not ab_set or ab_set == {"normal"}:
            return
        
        # 构建合并的弹窗内容
        title = f"干预建议 - 检测到 {len([ab for ab in ab_set if ab != 'normal'])} 项异常"
        
        # 创建自定义对话框而不是 QMessageBox
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.setMinimumWidth(450)
        dialog.setMaximumWidth(500)
        
        # 主布局
        layout = QVBoxLayout(dialog)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 标题区域（带图标）
        title_layout = QHBoxLayout()
        
        # 警告图标
        warning_icon = QLabel("⚠️")
        warning_icon.setStyleSheet("font-size: 24px;")
        title_layout.addWidget(warning_icon)
        
        # 标题文字
        title_label = QLabel("检测到以下异常行为：")
        title_label.setStyleSheet("""
            font-size: 14px;
            font-weight: bold;
            color: #333;
        """)
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        
        layout.addLayout(title_layout)
        
        # 异常行为列表区域
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setMaximumHeight(300)
        scroll_area.setStyleSheet("border: none;")
        
        content_widget = QWidget()
        content_layout = QVBoxLayout(content_widget)
        content_layout.setSpacing(10)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        for ab in sorted(ab_set):
            if ab == "normal":
                continue
            
            advice = INTERVENTION_MAP.get(ab, f"检测到 {ab}")
            short_advice = advice.replace("【干预建议】", "").replace("【注意】", "")
            
            # 每个异常行为的卡片
            card = QWidget()
            card.setStyleSheet("""
                QWidget {
                    background-color: #f8d7da;
                    border-left: 4px solid #d9534f;
                    border-radius: 4px;
                }
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(12, 10, 12, 10)
            card_layout.setSpacing(5)
            
            # 行为类型标签
            behavior_label = QLabel(f"<b>{ab}</b>")
            behavior_label.setStyleSheet("color: #721c24; font-size: 13px;")
            card_layout.addWidget(behavior_label)
            
            # 干预建议内容
            advice_label = QLabel(short_advice)
            advice_label.setStyleSheet("color: #856404; font-size: 12px;")
            advice_label.setWordWrap(True)
            card_layout.addWidget(advice_label)
            
            content_layout.addWidget(card)
        
        content_layout.addStretch()
        scroll_area.setWidget(content_widget)
        layout.addWidget(scroll_area)
        
        # 底部提示
        hint_label = QLabel("请及时关注并处理上述情况")
        hint_label.setStyleSheet("""
            color: #666;
            font-size: 12px;
            margin-top: 10px;
        """)
        hint_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint_label)
        
        # 确定按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        ok_btn = QPushButton("✓ 确定")
        ok_btn.setStyleSheet("""
            QPushButton {
                padding: 10px 30px;
                background-color: #5bc0de;
                color: white;
                border: none;
                border-radius: 5px;
                font-size: 13px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #46b8da;
            }
            QPushButton:pressed {
                background-color: #31b0d5;
            }
        """)
        ok_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(ok_btn)
        btn_layout.addStretch()
        
        layout.addLayout(btn_layout)
        
        # 设置对话框样式
        dialog.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
            }
        """)
        
        dialog.exec_()
        
    def update_stat(self):
        """更新统计显示 - 现代化样式"""
        session_summary = self.stat_manager.get_session_summary()
        global_summary = self.stat_manager.get_global_summary()
        
        # 使用 HTML 格式显示
        html_text = f"""
        <div style="line-height: 1.6; font-size: 13px;">
            <b>📊 统计概览</b><br/>
            当前会话: <span style="color: #5bc0de; font-weight: bold;">{session_summary['total']}</span> 次异常 | 
            历史累计: <span style="color: #5bc0de; font-weight: bold;">{global_summary['total']}</span> 次异常
        """
        
        if session_summary['by_behavior']:
            html_text += "<br/><br/><b>🔍 当前会话异常分布:</b>"
            for behavior, count in session_summary['by_behavior'].items():
                percentage = (count / session_summary['total'] * 100) if session_summary['total'] > 0 else 0
                bar = "█" * int(percentage / 10)  # 进度条
                html_text += f"""
                <div style="margin: 5px 0;">
                    {behavior}: <span style="color: #d9534f; font-weight: bold;">{count}次</span> 
                    <span style="color: #888;">({percentage:.1f}%)</span> {bar}
                </div>
                """
        
        html_text += "</div>"
        self.stat_label.setText(html_text)
        
    def stop_all(self):
        """停止所有处理"""
        if self.thread:
            self.thread.running = False
            if not self.thread.wait(1000):
                self.thread.terminate()
                self.thread.wait()
            self.thread = None
        self.save_current_session()
        self.progress_bar.setVisible(False)
        self.status.setText("⏹️ 已停止")
        self.status.setStyleSheet("font-size: 12px; color: #a0aec0; font-weight: bold;")
        self.fps_label.setText("FPS: --")
        self.time_label.setText("⏱️ 检测耗时: 0.00s")
        
    def open_cam(self):
        """打开摄像头"""
        self.stop_all()
        
        # 检测摄像头
        cap_test = cv2.VideoCapture(0)
        if not cap_test.isOpened():
            QMessageBox.warning(self, "摄像头错误", "无法打开摄像头，请检查设备连接")
            cap_test.release()
            return
        cap_test.release()
        
        self.start_new_session("camera", "摄像头实时检测")
        self.thread = VideoThread(0, conf_thres=self.conf_thres, sim_thres=self.sim_thres)
        self.thread.frame_signal.connect(self.show_frame)
        self.thread.error_signal.connect(lambda m: self.status.setText(f"错误: {m}"))
        self.thread.start()
        self.status.setText("摄像头运行中")
        
    def open_img(self):
        """打开图片检测"""
        self.stop_all()
        try:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择图片", "", 
                "图片文件 (*.jpg *.jpeg *.png *.bmp *.tiff)")
            if not path:
                return
                
            if not os.access(path, os.R_OK):
                QMessageBox.warning(self, "文件错误", f"无法读取文件：{path}")
                return
                
            frame = cv2.imread(path)
            if frame is None:
                QMessageBox.warning(self, "图片读取失败", "无法解码图片文件，请确认文件格式是否正确")
                return
                
            if frame.size == 0:
                QMessageBox.warning(self, "图片错误", "图片为空或损坏")
                return
                
            self.start_new_session("image", os.path.basename(path))
            frame, ab = detect_and_draw(frame, self.conf_thres, self.sim_thres)
            self.show_frame(frame, ab)
            self.status.setText(f"图片检测完成: {os.path.basename(path)}")
            
        except Exception as e:
            logger.error(f"图片处理异常: {str(e)}")
            QMessageBox.critical(self, "处理异常", f"图片处理时发生错误：{str(e)}")
            self.status.setText("图片处理失败")
            
    def open_video_save(self):
        """打开视频保存"""
        self.stop_all()
        src, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv)")
        if not src:
            return
            
        save_path, _ = QFileDialog.getSaveFileName(
            self, "保存处理后的视频", 
            f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4", 
            "MP4文件 (*.mp4)")
        if not save_path:
            return
            
        if not save_path.endswith(".mp4"):
            save_path += ".mp4"
        
        self.start_new_session("video", f"{os.path.basename(src)} -> {os.path.basename(save_path)}")
        self.thread = VideoThread(src, save_path, self.conf_thres, self.sim_thres)
        self.thread.frame_signal.connect(self.show_frame)
        self.thread.progress_signal.connect(self.progress_bar.setValue)
        self.thread.error_signal.connect(lambda m: self.status.setText(f"错误: {m}"))
        self.thread.start()
        self.progress_bar.setVisible(True)
        self.status.setText("视频处理中...")
        
    def save_img(self):
        """保存图片"""
        if self.current_frame is None:
            QMessageBox.warning(self, "提示", "无图像可保存")
            return
        p, _ = QFileDialog.getSaveFileName(
            self, "保存图片", 
            f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            "图片文件 (*.jpg *.jpeg *.png)")
        if p:
            try:
                cv2.imwrite(p, self.current_frame)
                QMessageBox.information(self, "成功", f"图片已保存到: {p}")
            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"无法保存图片: {str(e)}")
                
    def reset_stats(self):
        """重置统计"""
        reply = QMessageBox.question(self, "确认重置", 
                                    "确定要重置所有统计吗？\n（包括当前会话和全局统计）",
                                    QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.stat_manager.reset_all()
            self.shown_behaviors = set()
            self.update_stat()
            self.status.setText("统计已重置")
            
    def show_statistics_panel(self):
        """显示统计面板"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTextEdit, QPushButton, QTabWidget, QWidget
        
        dialog = QDialog(self)
        dialog.setWindowTitle("详细统计分析")
        dialog.setMinimumSize(600, 500)
        
        layout = QVBoxLayout(dialog)
        
        # 创建标签页
        tab_widget = QTabWidget()
        
        # 会话统计标签页
        session_tab = QWidget()
        session_layout = QVBoxLayout(session_tab)
        
        session_text = QTextEdit()
        session_text.setReadOnly(True)
        
        session_summary = self.stat_manager.get_session_summary()
        text = "=" * 40 + "\n"
        text += "当前会话统计\n"
        text += "=" * 40 + "\n"
        text += f"开始时间: {self.current_session['start_time'] if self.current_session else '无会话'}\n"
        text += f"异常总数: {session_summary['total']}\n"
        text += f"检测参数: YOLO置信度={self.conf_thres:.2f}, CLIP相似度={self.sim_thres:.2f}\n"
        text += "-" * 40 + "\n"
        
        if session_summary['by_behavior']:
            text += "按行为分类:\n"
            for behavior, count in session_summary['by_behavior'].items():
                percentage = (count / session_summary['total'] * 100) if session_summary['total'] > 0 else 0
                text += f"  {behavior}: {count} 次 ({percentage:.1f}%)\n"
        else:
            text += "暂无异常行为记录\n"
        
        session_text.setText(text)
        session_layout.addWidget(session_text)
        
        # 全局统计标签页
        global_tab = QWidget()
        global_layout = QVBoxLayout(global_tab)
        
        global_text = QTextEdit()
        global_text.setReadOnly(True)
        
        global_summary = self.stat_manager.get_global_summary()
        text = "=" * 40 + "\n"
        text += "全局历史统计\n"
        text += "=" * 40 + "\n"
        text += f"累计异常总数: {global_summary['total']}\n"
        text += f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        text += "-" * 40 + "\n"
        
        if global_summary['by_behavior']:
            text += "按行为分类:\n"
            for behavior, count in global_summary['by_behavior'].items():
                percentage = (count / global_summary['total'] * 100) if global_summary['total'] > 0 else 0
                text += f"  {behavior}: {count} 次 ({percentage:.1f}%)\n"
        else:
            text += "暂无历史数据\n"
        
        global_text.setText(text)
        global_layout.addWidget(global_text)
        
        # 添加标签页
        tab_widget.addTab(session_tab, "当前会话")
        tab_widget.addTab(global_tab, "历史统计")
        
        layout.addWidget(tab_widget)
        
        # 添加按钮
        btn_layout = QHBoxLayout()
        export_btn = QPushButton("导出为CSV")
        export_btn.clicked.connect(lambda: self.export_statistics_csv())
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.close)
        
        btn_layout.addWidget(export_btn)
        btn_layout.addWidget(close_btn)
        layout.addLayout(btn_layout)
        
        dialog.exec_()
        
    def export_statistics_csv(self):
        """导出统计为CSV"""
        path, _ = QFileDialog.getSaveFileName(
            self, "导出统计", 
            f"behavior_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
            "CSV文件 (*.csv)")
        
        if not path:
            return
            
        try:
            with open(path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.writer(f)
                writer.writerow(['行为类型', '当前会话次数', '全局累计次数', '导出时间'])
                
                session_summary = self.stat_manager.get_session_summary()
                global_summary = self.stat_manager.get_global_summary()
                
                all_behaviors = set(session_summary['by_behavior'].keys()) | set(global_summary['by_behavior'].keys())
                
                for behavior in all_behaviors:
                    if behavior == "normal":
                        continue
                    
                    session_count = session_summary['by_behavior'].get(behavior, 0)
                    global_count = global_summary['by_behavior'].get(behavior, 0)
                    
                    if session_count > 0 or global_count > 0:
                        writer.writerow([behavior, session_count, global_count, 
                                        datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            
            QMessageBox.information(self, "导出成功", f"统计已导出到: {path}")
            
        except Exception as e:
            logger.error(f"导出CSV失败: {str(e)}")
            QMessageBox.critical(self, "导出失败", f"无法导出文件: {str(e)}")
            
    def show_session_history(self):
        """显示会话历史"""
        d = QDialog(self)
        d.setWindowTitle("会话历史")
        d.setMinimumSize(700, 500)
        layout = QVBoxLayout(d)
        text = QTextEdit()
        if self.session_history:
            text.setPlainText(json.dumps(self.session_history, ensure_ascii=False, indent=2))
        else:
            text.setPlainText("暂无会话历史记录")
        text.setReadOnly(True)
        layout.addWidget(text)
        btn = QPushButton("关闭")
        btn.clicked.connect(d.close)
        layout.addWidget(btn)
        d.exec_()
        
    def open_config_dialog(self):
        """打开参数配置对话框"""
        dialog = ConfigDialog(self, self.conf_thres, self.sim_thres, self.cooldown)
        if dialog.exec_() == QDialog.Accepted:
            values = dialog.get_values()
            self.conf_thres = values['conf_thres']
            self.sim_thres = values['sim_thres']
            self.cooldown = values['cooldown']
            self.stat_manager.cooldown = self.cooldown
            
            # 保存到配置
            config_manager.set("detection.conf_thres", self.conf_thres)
            config_manager.set("detection.sim_thres", self.sim_thres)
            config_manager.set("detection.cooldown", self.cooldown)
            
            self.status.setText(f"参数已更新: YOLO置信度={self.conf_thres:.2f}, CLIP相似度={self.sim_thres:.2f}, 冷却时间={self.cooldown}秒")
            
    def show_about_dialog(self):
        """显示关于对话框"""
        about_text = """
        <h3>儿童课堂异常行为检测与干预系统</h3>
        <p><b>版本:</b> 1.1.0</p>
        <p><b>技术架构:</b></p>
        <ul>
            <li>目标检测: YOLOv8</li>
            <li>行为识别: CLIP (Vision Transformer)</li>
            <li>界面框架: PyQt5</li>
            <li>视频处理: OpenCV</li>
        </ul>
        <p><b>支持的行为类型:</b></p>
        <ul>
            <li>normal - 正常行为</li>
            <li>lie - 趴桌</li>
            <li>stand - 离座</li>
            <li>play_phone - 使用手机</li>
            <li>fight - 打闹</li>
            <li>whispering - 交头接耳</li>
            <li>looking_around - 东张西望</li>
        </ul>
        <p><b>核心功能:</b></p>
        <ul>
            <li>实时摄像头检测</li>
            <li>图片/视频上传检测</li>
            <li>异常行为标注与保存</li>
            <li>个性化干预建议</li>
            <li>详细统计分析</li>
        </ul>
        <p><b>快捷键:</b></p>
        <ul>
            <li>ESC: 停止检测</li>
            <li>空格: 开始/停止摄像头</li>
            <li>Ctrl+S: 保存当前图片</li>
            <li>H: 查看历史记录</li>
            <li>P: 打开统计面板</li>
            <li>C: 打开配置对话框</li>
        </ul>
        <p><b>注意事项:</b></p>
        <p>1. 使用前请运行 build_prototype.py 生成原型文件</p>
        <p>2. 首次使用会自动下载YOLOv8模型</p>
        <p>3. 建议在GPU环境下运行以获得最佳性能</p>
        <p><i>© 2026 儿童课堂异常行为检测与干预系统</i></p>
        """
        
        QMessageBox.about(self, "关于", about_text)
        
    def keyPressEvent(self, event):
        """快捷键支持"""
        if event.key() == Qt.Key_Escape:
            self.stop_all()
        elif event.key() == Qt.Key_Space:
            if self.thread and self.thread.running:
                self.stop_all()
            else:
                self.open_cam()
        elif event.key() == Qt.Key_S and event.modifiers() & Qt.ControlModifier:
            self.save_img()
        elif event.key() == Qt.Key_H:
            self.show_session_history()
        elif event.key() == Qt.Key_P:
            self.show_statistics_panel()
        elif event.key() == Qt.Key_C:
            self.open_config_dialog()
        else:
            super().keyPressEvent(event)
            
    def closeEvent(self, event):
        """关闭事件"""
        self.stop_all()
        
        # 保存窗口尺寸
        config_manager.set("ui.window_width", self.width())
        config_manager.set("ui.window_height", self.height())
        
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 设置应用样式
    app.setStyle("Fusion")
    
    # 设置调色板
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(240, 240, 240))
    palette.setColor(QPalette.WindowText, QColor(0, 0, 0))
    palette.setColor(QPalette.Base, QColor(255, 255, 255))
    palette.setColor(QPalette.AlternateBase, QColor(245, 245, 245))
    palette.setColor(QPalette.ToolTipBase, QColor(255, 255, 220))
    palette.setColor(QPalette.ToolTipText, QColor(0, 0, 0))
    palette.setColor(QPalette.Text, QColor(0, 0, 0))
    palette.setColor(QPalette.Button, QColor(240, 240, 240))
    palette.setColor(QPalette.ButtonText, QColor(0, 0, 0))
    palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
    palette.setColor(QPalette.Highlight, QColor(76, 163, 224))
    palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    app.setPalette(palette)
    
    # 设置字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)
    
    w = MainWindow()
    w.show()
    
    sys.exit(app.exec_())