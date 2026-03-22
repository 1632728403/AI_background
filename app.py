# ==============================================
# U2NetP 轻量版人像分割工具 | 三色证件照背景
# 西安电子科技大学 大模型应用创新赛作品
# 西电本地图标版 | 下载无破损 | 开发者：陈宥廷 刘家瑄
# ==============================================
import os
import numpy as np
from PIL import Image, ImageEnhance
import onnxruntime as ort
import streamlit as st
from io import BytesIO

# 云端环境兼容配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== 仅保留亮度调节函数 ====================
def adjust_brightness(img, value):
    return ImageEnhance.Brightness(img).enhance(value)
# ============================================================

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
    
    output = session.run([output_name], {input_name: input_tensor})[0]
    pred = output[0, 0, :, :]
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask).resize(image.size, Image.NEAREST)
    return mask_pil

# -------------------------- 切换背景颜色（白/红/蓝） --------------------------
def generate_result(image, mask_pil, bg_color):
    mask_np = np.array(mask_pil)
    img_np = np.array(image)
    
    color_map = {
        "白色": (255, 255, 255),
        "红色": (255, 0, 0),
        "蓝色": (0, 0, 255)
    }
    bg_r, bg_g, bg_b = color_map[bg_color]
    
    result = np.zeros_like(img_np)
    result[:, :, 0] = np.where(mask_np > 127, img_np[:, :, 0], bg_r)
    result[:, :, 1] = np.where(mask_np > 127, img_np[:, :, 1], bg_g)
    result[:, :, 2] = np.where(mask_np > 127, img_np[:, :, 2], bg_b)
    
    return Image.fromarray(result.astype(np.uint8))

# -------------------------- 网页界面配置 --------------------------
st.set_page_config(
    page_title="西电AI人像证件照工具", 
    page_icon="./xidian_logo.png",
    layout="wide"
)

# -------------------------- 顶部标题 --------------------------
col1, col2 = st.columns([1, 10])
with col1:
    st.image("./xidian_logo.png", width=70)
with col2:
    st.title("🎨 西电专属 AI 人像分割工具")

st.divider()

# -------------------------- 图片展示区（双栏 无框体） --------------------------
col_img1, col_img2 = st.columns(2)

# 左侧原图
with col_img1:
    st.subheader("📸 上传照片")
    uploaded_file = st.file_uploader("支持 JPG / PNG", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        # 自适应大小，电脑端不超大，手机端自动适配
        st.image(img, caption="原始图片", use_column_width=True)

# 右侧结果图
with col_img2:
    st.subheader("✨ 处理效果")
    if uploaded_file:
        st.image("https://placehold.co/400x560?text=处理中", caption="等待处理", use_column_width=True)

# -------------------------- 调节区（极简 无框体） --------------------------
st.subheader("⚙️ 参数调节")
brightness = st.slider("亮度调节", 0.5, 1.5, 1.0, 0.05)

# 背景颜色选择
st.subheader("🎨 背景颜色")
bg_color = st.radio("", ["白色", "红色", "蓝色"], horizontal=True)

# -------------------------- 处理逻辑 --------------------------
if uploaded_file is not None:
    session = load_model()
    with st.spinner("AI 处理中..."):
        mask = infer_mask(img, session)
        img_final = adjust_brightness(img, brightness)
        result_img = generate_result(img_final, mask, bg_color)
    
    # 显示结果
    with col_img2:
        st.image(result_img, caption=f"{bg_color}背景 | 亮度:{brightness}", use_column_width=True)
    
    # 下载按钮
    st.divider()
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    st.download_button(
        label=f"下载 {bg_color} 背景证件照",
        data=buf.getvalue(),
        file_name=f"西电证件照_{bg_color}.png",
        mime="image/png",
        use_container_width=True
    )

# -------------------------- 底部署名 --------------------------
st.divider()
st.caption("本工具由 XDU 陈宥廷 刘家瑄 开发 🐱")
st.caption("反馈邮箱：1632728403@qq.com")
