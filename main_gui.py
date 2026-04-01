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
    "stand": "【干预建议】学生离座，引导回到座位，遵守课堂秩序",
    "play_phone": "【干预建议】发现使用手机，提醒收起电子设备",
    "fight": "【干预建议】发现打闹行为，立即制止，维持课堂安全",
    "normal": "状态正常，无需干预"
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

def classify_crop(frame, box):
    """对裁剪区域进行分类"""
    try:
        x1, y1, x2, y2 = map(int, box)
        
        # 确保边界在图像范围内
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2 <= x1 or y2 <= y1:
            return "invalid_crop", 0
            
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "empty_crop", 0
            
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        img = preprocess(pil).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            feat = clip_model.encode_image(img)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            
        max_sim = -1
        pred = "unknown"
        for cls, proto in prototypes.items():
            sim = torch.cosine_similarity(feat, proto.to(DEVICE)).item()
            if sim > max_sim:
                max_sim = sim
                pred = cls
                
        return pred, max_sim
        
    except Exception as e:
        logger.error(f"分类异常: {str(e)}")
        return "error", 0

def detect_and_draw(frame, conf_thres=CONF_THRES, sim_thres=SIM_THRES):
    """检测并绘制结果"""
    current_abnormal = set()
    try:
        results = yolo(frame, classes=[0], conf=conf_thres)
        for r in results:
            for box in r.boxes.xyxy:
                cls_name, sim = classify_crop(frame, box)
                if cls_name not in CLASS_NAMES:
                    continue
                if sim > sim_thres and cls_name != "normal":
                    current_abnormal.add(cls_name)
                x1, y1, x2, y2 = map(int, box)
                is_ab = cls_name != "normal"
                color = (0, 0, 255) if is_ab else (0, 255, 0)
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
        self.setWindowTitle("儿童课堂异常行为检测系统")
        self.setGeometry(100, 100, width, height)
        
    def initUI(self):
        """初始化界面"""
        main = QWidget()
        self.setCentralWidget(main)
        layout = QVBoxLayout(main)

        # 视频显示区域
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setMinimumHeight(500)
        self.label.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
        layout.addWidget(self.label)

        # 按钮区域
        btn_layout = QHBoxLayout()
        self.btn_cam = QPushButton("📷 摄像头")
        self.btn_img = QPushButton("🖼️ 图片检测")
        self.btn_video = QPushButton("🎬 视频保存")
        self.btn_stop = QPushButton("⏹️ 停止")
        self.btn_save = QPushButton("💾 保存图片")
        self.btn_reset = QPushButton("🔄 重置统计")
        self.btn_stats_panel = QPushButton("📈 统计面板")
        self.btn_history = QPushButton("📊 会话历史")
        self.btn_config = QPushButton("⚙️ 参数配置")
        self.btn_about = QPushButton("ℹ️ 关于")
        
        # 设置按钮样式
        for btn in [self.btn_cam, self.btn_img, self.btn_video, self.btn_stop, 
                    self.btn_save, self.btn_reset, self.btn_stats_panel, 
                    self.btn_history, self.btn_config, self.btn_about]:
            btn.setMinimumHeight(35)
        
        btn_layout.addWidget(self.btn_cam)
        btn_layout.addWidget(self.btn_img)
        btn_layout.addWidget(self.btn_video)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_save)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addWidget(self.btn_stats_panel)
        btn_layout.addWidget(self.btn_history)
        btn_layout.addWidget(self.btn_config)
        btn_layout.addWidget(self.btn_about)
        layout.addLayout(btn_layout)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # 统计信息
        self.stat_label = QLabel("📊 异常行为统计：暂无数据")
        self.stat_label.setStyleSheet("font-weight: bold; padding: 5px;")
        layout.addWidget(self.stat_label)

        # 状态栏
        status_layout = QHBoxLayout()
        self.status = QLabel("就绪")
        self.fps_label = QLabel("FPS: --")
        self.fps_label.setStyleSheet("color: #666;")
        status_layout.addWidget(self.status)
        status_layout.addStretch()
        status_layout.addWidget(self.fps_label)
        layout.addLayout(status_layout)

        # 连接信号
        self.btn_cam.clicked.connect(self.open_cam)
        self.btn_img.clicked.connect(self.open_img)
        self.btn_video.clicked.connect(self.open_video_save)
        self.btn_stop.clicked.connect(self.stop_all)
        self.btn_save.clicked.connect(self.save_img)
        self.btn_reset.clicked.connect(self.reset_stats)
        self.btn_stats_panel.clicked.connect(self.show_statistics_panel)
        self.btn_history.clicked.connect(self.show_session_history)
        self.btn_config.clicked.connect(self.open_config_dialog)
        self.btn_about.clicked.connect(self.show_about_dialog)
        
        # 初始化变量
        self.thread = None
        self.current_frame = None
        self.current_session = None
        self.shown_behaviors = set()
        self.session_saved = False
        self.session_history = []
        
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
        """显示帧图像"""
        start_time = time.time()
        
        self.current_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bpl = ch * w
        q_img = QImage(rgb.data, w, h, bpl, QImage.Format_RGB888).copy()
        self.label.setPixmap(QPixmap.fromImage(q_img).scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

        # 更新统计
        for ab in ab_set:
            if self.stat_manager.update(ab, self.cooldown):
                if ab not in self.shown_behaviors:
                    self.shown_behaviors.add(ab)
                    self.log_behavior(ab)
                    QMessageBox.information(self, "干预建议", INTERVENTION_MAP[ab])
        
        self.update_stat()
        
        # 更新性能监控
        end_time = time.time()
        process_time = (end_time - start_time) * 1000
        self.performance_monitor.add_frame_time(process_time)
        stats = self.performance_monitor.get_stats()
        self.fps_label.setText(f"FPS: {stats['current_fps']:.1f}")
        
    def update_stat(self):
        """更新统计显示"""
        session_summary = self.stat_manager.get_session_summary()
        global_summary = self.stat_manager.get_global_summary()
        
        # 创建统计文本
        stat_text = "📊 行为统计\n"
        stat_text += f"当前会话: {session_summary['total']} 次异常 | "
        stat_text += f"历史累计: {global_summary['total']} 次异常\n"
        
        if session_summary['by_behavior']:
            stat_text += "\n当前会话异常分布:\n"
            for behavior, count in session_summary['by_behavior'].items():
                percentage = (count / session_summary['total'] * 100) if session_summary['total'] > 0 else 0
                stat_text += f"  • {behavior}: {count} 次 ({percentage:.1f}%)\n"
        
        self.stat_label.setText(stat_text)
        
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
        self.status.setText("已停止")
        self.fps_label.setText("FPS: --")
        
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
        <h3>儿童课堂异常行为检测系统</h3>
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
        <p><i>© 2024 儿童课堂异常行为检测系统</i></p>
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