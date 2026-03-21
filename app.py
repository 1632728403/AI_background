# ==============================================
# U2NetP 轻量版人像分割工具 | 支持三色背景切换
# 西安电子科技大学 大模型应用创新赛作品
# ==============================================
import os
import numpy as np
from PIL import Image
import onnxruntime as ort
import streamlit as st

# 云端环境兼容配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------- 加载轻量版U2NetP ONNX模型 --------------------------
@st.cache_resource
def load_model():
    session = ort.InferenceSession("u2netp.onnx")
    return session

# -------------------------- 图片预处理 --------------------------
def preprocess_image(image):
    img_resize = image.resize((320, 320))
    img_np = np.array(img_resize).astype(np.float32) / 255.0
    input_tensor = img_np.transpose(2, 0, 1)[np.newaxis, :]
    return input_tensor

# -------------------------- ONNX模型推理 --------------------------
def infer_mask(image, session):
    input_tensor = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    output = session.run([output_name], {input_tensor})[0]
    pred = output[0, 0, :, :]
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask).resize(image.size, Image.NEAREST)
    return mask_pil

# -------------------------- 核心功能：切换背景颜色（白/红/蓝） --------------------------
def generate_result(image, mask_pil, bg_color):
    mask_np = np.array(mask_pil)
    img_np = np.array(image)
    
    # 定义三种背景色
    color_map = {
        "白色": (255, 255, 255),
        "红色": (255, 0, 0),
        "蓝色": (0, 0, 255)
    }
    bg_r, bg_g, bg_b = color_map[bg_color]
    
    # 创建彩色背景
    result = np.zeros_like(img_np)
    result[:, :, 0] = np.where(mask_np > 127, img_np[:, :, 0], bg_r)
    result[:, :, 1] = np.where(mask_np > 127, img_np[:, :, 1], bg_g)
    result[:, :, 2] = np.where(mask_np > 127, img_np[:, :, 2], bg_b)
    
    return Image.fromarray(result.astype(np.uint8))

# -------------------------- Streamlit网页界面 --------------------------
st.set_page_config(page_title="西电AI人像证件照工具", page_icon="🎨", layout="wide")
st.title("🎨 西电专属 AI 人像分割 & 证件照背景替换工具")

# 双栏布局：原图 + 结果图
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 📸 上传人像照片")
    uploaded_file = st.file_uploader("支持 JPG / PNG 格式", type=["jpg", "jpeg", "png"])

with col2:
    st.markdown("### ✨ 处理完成效果")

# 新增：背景颜色选择器
st.subheader("🎨 选择证件照背景颜色")
bg_color = st.radio(
    label="",
    options=["白色", "红色", "蓝色"],
    horizontal=True,
    index=0
)

# 处理图片逻辑
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    col1.image(img, caption="原始图片", use_column_width=True)
    
    session = load_model()
    with st.spinner("🔍 AI 正在处理中..."):
        mask = infer_mask(img, session)
        result_img = generate_result(img, mask, bg_color)
    
    col2.image(result_img, caption=f"已切换为{bg_color}背景", use_column_width=True)
    
    # 下载按钮
    st.divider()
    st.download_button(
        label=f"💾 下载{bg_color}背景证件照",
        data=result_img.tobytes(),
        file_name=f"XDU证件照_{bg_color}.png",
        mime="image/png"
    )

# -------------------------- 页面底部：开发者署名（核心要求） --------------------------
st.markdown("---")
st.markdown("<h4 style='text-align: center; color: #666;'>本页面由 XDU 陈宥廷 刘家瑄 开发</h4>", unsafe_allow_html=True)
