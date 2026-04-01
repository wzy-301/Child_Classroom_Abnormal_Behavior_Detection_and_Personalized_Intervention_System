import os
import clip
import torch
import pickle
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def check_image_folder():
    """检查图片文件夹结构"""
    root = "./images"
    if not os.path.exists(root):
        logger.error(f"图片目录不存在: {root}")
        print("❌ 错误: 请创建 ./images 目录，并在其中为每个行为类别创建子文件夹")
        return False
    
    # 读取类别名称
    if not os.path.exists("class_names.txt"):
        logger.error("class_names.txt 文件不存在")
        print("❌ 错误: 请创建 class_names.txt 文件，每行一个行为类别")
        return False
    
    class_names = [c.strip() for c in open("class_names.txt", encoding='utf-8') if c.strip()]
    if not class_names:
        logger.error("class_names.txt 文件为空")
        print("❌ 错误: class_names.txt 文件为空，请添加行为类别")
        return False
    
    # 检查每个类别是否有对应的文件夹
    missing_folders = []
    for c in class_names:
        path = os.path.join(root, c)
        if not os.path.exists(path):
            missing_folders.append(c)
        elif not os.listdir(path):
            logger.warning(f"类别 '{c}' 的文件夹为空，请添加样本图片")
    
    if missing_folders:
        logger.error(f"缺少文件夹: {missing_folders}")
        print(f"❌ 错误: 请在 ./images 目录下创建以下文件夹: {', '.join(missing_folders)}")
        return False
    
    return True, class_names

def build_prototypes():
    """生成少样本原型"""
    try:
        # 检查文件夹结构
        success, class_names = check_image_folder()
        if not success:
            return False
        
        logger.info(f"开始为 {len(class_names)} 个类别生成原型: {class_names}")
        
        # 加载CLIP模型
        model, pre = clip.load("ViT-B/32", device=DEVICE)
        model.eval()
        
        # 生成图片原型
        root = "./images"
        proto = {}
        
        for c in class_names:
            path = os.path.join(root, c)
            feats = []
            image_count = 0
            
            for f in os.listdir(path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    try:
                        img_path = os.path.join(path, f)
                        img = pre(Image.open(img_path)).unsqueeze(0).to(DEVICE)
                        with torch.no_grad():
                            feats.append(model.encode_image(img))
                        image_count += 1
                    except Exception as e:
                        logger.warning(f"处理图片 {f} 失败: {str(e)}")
                        continue
            
            if image_count == 0:
                logger.warning(f"类别 '{c}' 没有有效图片，跳过")
                continue
            
            if feats:
                proto[c] = torch.mean(torch.cat(feats), dim=0, keepdim=True)
                logger.info(f"类别 '{c}': 使用 {image_count} 张图片生成原型")
            else:
                logger.warning(f"类别 '{c}' 无法生成原型")
        
        if not proto:
            logger.error("没有成功生成任何原型")
            return False
        
        # 保存原型
        with open("prototypes.pkl", "wb") as f:
            pickle.dump({"prototypes": proto, "class_names": class_names}, f)
        
        print(f"✅ 少样本原型生成完成！共生成 {len(proto)} 个类别的原型")
        print(f"📁 保存到: prototypes.pkl")
        return True
        
    except Exception as e:
        logger.error(f"原型生成失败: {str(e)}")
        print(f"❌ 错误: {str(e)}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("儿童课堂异常行为检测系统 - 原型生成工具")
    print("=" * 50)
    
    success = build_prototypes()
    if not success:
        print("请检查错误信息并修复后重试")
        exit(1)