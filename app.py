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

# ==================== 【全局自定义CSS优化：框架感+响应式】 ====================
st.markdown("""
<style>
/* 全局基础样式 */
.main .block-container {
    max-width: 1200px;
    padding-top: 2rem;
}
/* 卡片容器样式（核心：增加框架感） */
.card {
    border: 1px solid #e0e0e0;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
/* 图片限制最大宽度，防止电脑端过大，手机端自适应 */
img {
    max-width: 400px !important;
    border-radius: 8px;
    margin: 0 auto;
}
/* 标题间距优化 */
h1, h3 {
    margin-bottom: 1rem !important;
}
/* 滑块/单选框样式优化 */
.stSlider, .stRadio {
    padding: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)
# ============================================================================

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

# -------------------------- 顶部标题栏（优化布局） --------------------------
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.image("./xidian_logo.png", width=70)
with col_title:
    st.title("🎨 西电专属 AI 人像分割工具")
    st.caption("人像分割 | 三色背景 | 亮度调节 | 无尺寸修改版")

st.divider()

# -------------------------- 核心功能区（卡片式双栏布局） --------------------------
col_upload, col_result = st.columns(2)

# 左侧：上传区（卡片框架）
with col_upload:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### 📸 原始人像照片")
    uploaded_file = st.file_uploader("支持 JPG / PNG 格式", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="上传原图", width=380) # 固定宽度，不超大
    st.markdown('</div>', unsafe_allow_html=True)

# 右侧：结果区（卡片框架）
with col_result:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### ✨ 处理后效果")
    if uploaded_file:
        # 占位图，处理后替换
        st.image("https://placehold.co/380x530?text=处理中...", caption="等待处理", width=380)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- 调节控制面板（卡片框架） --------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### ⚙️ 参数调节")
col_bright, col_bg = st.columns([1, 1])
with col_bright:
    brightness = st.slider("💡 亮度调节", 0.5, 1.5, 1.0, 0.05)
with col_bg:
    st.markdown("##### 🎨 背景颜色")
    bg_color = st.radio("", options=["白色", "红色", "蓝色"], horizontal=True)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- 处理逻辑 --------------------------
if uploaded_file is not None:
    session = load_model()
    with st.spinner("🔍 AI 正在分割人像..."):
        mask = infer_mask(img, session)
        # 仅调节亮度，无尺寸修改
        img_final = adjust_brightness(img, brightness)
        result_img = generate_result(img_final, mask, bg_color)
    
    # 刷新右侧结果图
    with col_result:
        st.image(result_img, caption=f"{bg_color}背景 | 亮度:{brightness}", width=380)
    
    # 下载按钮（卡片样式）
    st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    byte_data = buf.getvalue()
    
    st.download_button(
        label=f"💾 下载{bg_color}背景证件照",
        data=byte_data,
        file_name=f"XDU证件照_{bg_color}.png",
        mime="image/png",
        use_container_width=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------- 页面底部 --------------------------
st.divider()
st.markdown("<p style='text-align: center; color: #666;'>本工具由 XDU 陈宥廷 刘家瑄 开发 🐱</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #666;'>反馈邮箱：1632728403@qq.com</p>", unsafe_allow_html=True)
