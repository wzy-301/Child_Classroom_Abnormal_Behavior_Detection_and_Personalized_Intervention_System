import os
import clip
import torch
import pickle
from PIL import Image
import logging
import numpy as np
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 策略1: 多视角描述 - 每个类别用多个不同角度描述
CLASS_DESCRIPTIONS_MULTI = {
    "normal": [
        "student sitting properly, hands on desk or holding pen, looking at teacher or blackboard, no electronic device",
        "attentive student, upright posture, studying, listening to lesson, hands empty or with book",
        "good student behavior, focused on class, no phone visible, paying attention to teacher"
    ],
    "lie": [
        "student head down on desk, lying on table, sleeping in class",
        "student collapsed on desk, head resting on arms, not paying attention",
        "student bending over desk, head down, lazy posture"
    ],
    "stand": [
        "student standing up from chair, out of seat, rising from desk",
        "student on feet, standing in classroom, not seated",
        "student upright standing, away from desk, out of chair"
    ],
    "play_phone": [
        "student looking down at smartphone, holding mobile phone with both hands, phone screen visible",
        "student using cellphone in class, head down staring at phone, distracted by device",
        "student hiding phone, looking at electronic device, not paying attention to teacher"
    ],
    "fight": [
        "two students fighting, hitting each other, physical conflict",
        "students pushing and shoving, aggressive physical contact",
        "students in physical altercation, fighting in classroom"
    ],
    "whispering": [
        "two students talking secretly, heads close together, whispering",
        "students chatting quietly, leaning toward each other, private conversation",
        "students speaking softly, close interaction, not listening to teacher"
    ],
    "looking_around": [
        "student looking left and right, turning head around, distracted",
        "student gazing around classroom, not looking at teacher, wandering eyes",
        "student head turning, looking at classmates, distracted attention"
    ]
}

# 策略2: 负样本描述 - 明确说明不是什么
# NEGATIVE_DESCRIPTIONS = {
#     "normal": "student not sleeping, not using phone, NO mobile device, NO electronic gadget, not talking",
#     "lie": "student not sitting upright, not paying attention, head down",
#     "stand": "student not seated, not in chair, standing on feet",
#     "play_phone": "student not listening, not looking at teacher, using phone",
#     "fight": "students not peaceful, physical conflict, aggressive",
#     "whispering": "students not silent, talking to each other, not listening",
#     "looking_around": "student not focused, distracted gaze, wandering attention"
# }

def build_enhanced_prototypes():
    """
    增强版原型构建：
    1. 多文本描述融合
    2. 负样本约束
    3. 多原型聚类
    """
    
    class_names = [c.strip() for c in open("class_names.txt", encoding='utf-8') if c.strip()]
    logger.info(f"处理类别: {class_names}")
    
    model, preprocess = clip.load("ViT-B/32", device=DEVICE)
    model.eval()
    
    root = "./images"
    proto = {}
    
    for c in class_names:
        path = os.path.join(root, c)
        if not os.path.exists(path):
            continue
        
        # 1. 提取图像特征
        img_feats = []
        valid_files = []
        for f in os.listdir(path):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                try:
                    img = preprocess(Image.open(os.path.join(path, f))).unsqueeze(0).to(DEVICE)
                    with torch.no_grad():
                        feat = model.encode_image(img)
                        feat = feat / feat.norm(dim=-1, keepdim=True)
                        img_feats.append(feat.cpu())
                        valid_files.append(f)
                except:
                    continue
        
        if len(img_feats) == 0:
            continue
            
        img_feats = torch.cat(img_feats)
        logger.info(f"{c}: {len(img_feats)} 张图片")
        
        # 2. 多文本描述特征（取平均）
        descs = CLASS_DESCRIPTIONS_MULTI.get(c, [f"a student {c} in classroom"])
        text_feats = []
        for desc in descs:
            tokens = clip.tokenize([desc]).to(DEVICE)
            with torch.no_grad():
                tf = model.encode_text(tokens)
                tf = tf / tf.norm(dim=-1, keepdim=True)
                text_feats.append(tf.cpu())
        text_feat = torch.cat(text_feats).mean(dim=0, keepdim=True)
        
        # 3. 负样本约束（降低与负描述的相似度）
        # neg_desc = NEGATIVE_DESCRIPTIONS.get(c, "")
        # if neg_desc:
        #     neg_tokens = clip.tokenize([neg_desc]).to(DEVICE)
        #     with torch.no_grad():
        #         neg_feat = model.encode_text(neg_tokens)
        #         neg_feat = neg_feat / neg_feat.norm(dim=-1, keepdim=True)
        #         neg_feat = neg_feat.cpu()
        # else:
        #     neg_feat = None
        
        # 4. 图像聚类生成多原型（每个类别3-5个原型）
        n_clusters = min(3, len(img_feats) // 8 + 1)
        if n_clusters < 2:
            centers = img_feats.mean(dim=0, keepdim=True)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(img_feats.numpy())
            centers = torch.from_numpy(kmeans.cluster_centers_).float()

        centers = centers / centers.norm(dim=1, keepdim=True)
        
        # 5. 融合策略：图像原型 + 文本引导 + 负样本约束
        # 每个图像原型与文本特征融合
        enhanced_protos = []
        for i in range(centers.shape[0]):
            img_p = centers[i:i+1]
            
            # 基础融合：图像 + 文本
            combined = img_p + 0.4 * text_feat
            
            # 负样本约束：远离负描述（如果相似度高，则调整）
            # if neg_feat is not None:
            #     neg_sim = torch.cosine_similarity(combined, neg_feat).item()
            #     if neg_sim > 0.5:  # 如果与负描述太像，降低权重
            #         combined = combined - 0.2 * neg_feat
            
            combined = combined / combined.norm(dim=-1, keepdim=True)
            enhanced_protos.append(combined)
        
        # 保存为多个原型（列表形式）
        proto[c] = torch.cat(enhanced_protos).to(DEVICE)
        logger.info(f"  生成 {len(enhanced_protos)} 个增强原型")
    
    # 保存
    with open("prototypes.pkl", "wb") as f:
        pickle.dump({"prototypes": proto, "class_names": class_names}, f)
    
    logger.info("✅ 增强原型生成完成")

if __name__ == "__main__":
    build_enhanced_prototypes()