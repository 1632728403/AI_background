import os
import numpy as np
from PIL import Image, ImageEnhance
import onnxruntime as ort
import streamlit as st
from io import BytesIO

# 云端环境兼容配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# -------------------------- 核心函数保持不变 --------------------------
@st.cache_resource
def load_model():
    session = ort.InferenceSession("u2netp.onnx")
    return session

def preprocess_image(image):
    img_resize = image.resize((320, 320))
    img_np = np.array(img_resize).astype(np.float32) / 255.0
    input_tensor = img_np.transpose(2, 0, 1)[np.newaxis, :]
    return input_tensor

def infer_mask(image, session):
    input_tensor = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    output = session.run([output_name], {input_name: input_tensor})[0]
    pred = output[0, 0, :, :]
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask).resize(image.size, Image.NEAREST)
    return mask_pil

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

# -------------------------- 界面代码 (已调整顺序) --------------------------

# 1. 页面配置
st.set_page_config(
    page_title="西电AI证件照", 
    page_icon="./xidian_logo.png", 
    layout="centered"
)

# 2. 顶部标题栏
with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        try:
            st.image("./xidian_logo.png", width=50)
        except:
            st.empty()
    with col2:
        st.markdown("<h1 style='text-align: center; font-size: 1.5em;'>西电专属 AI 证件照</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: grey; font-size: 0.9em;'>人像分割 & 背景替换</p>", unsafe_allow_html=True)
    with col3:
        st.empty()

st.divider()

# 3. 核心控制区 (顺序已调整：1.上传 -> 2.背景 -> 3.亮度)

# 【第一步：上传】
st.markdown("### 📸 第一步：上传照片")
uploaded_file = st.file_uploader("支持 JPG / PNG 格式", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# 只有上传了照片，才显示后面的选项，界面更干净
if uploaded_file is not None:
    
    # 【第二步：背景颜色】
    st.markdown("### 🎨 第二步：选择背景颜色")
    bg_color = st.radio(
        label="",
        options=["白色", "红色", "蓝色"],
        horizontal=True,
        index=0
    )

    # 【第三步：亮度调节】
    st.markdown("### 💡 第三步：调节照片亮度")
    brightness_factor = st.slider(
        "亮度系数 (1.0 为原图)",
        min_value=0.5,
        max_value=1.8,
        value=1.0,
        step=0.05
    )

    # 4. 图片处理逻辑
    img = Image.open(uploaded_file).convert("RGB")
    session = load_model()
    
    with st.spinner("🔍 AI 正在处理中，请稍候..."):
        mask = infer_mask(img, session)
        result_img = generate_result(img, mask, bg_color)
        
        # 应用亮度调节
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(result_img)
            result_img = enhancer.enhance(brightness_factor)
    
    st.divider()
    st.markdown("### ✨ 效果预览 & 下载")
    
    # 响应式图片展示
    c1, c2 = st.columns(2)
    
    with c1:
        st.caption("原始图片")
        st.image(img, use_container_width=True)
    
    with c2:
        st.caption(f"处理结果 ({bg_color})")
        st.image(result_img, use_container_width=True)

    # 下载区域
    st.divider()
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    byte_data = buf.getvalue()
    
    st.download_button(
        label=f"📥 下载 {bg_color} 背景证件照",
        data=byte_data,
        file_name=f"XDU证件照_{bg_color}.png",
        mime="image/png",
        type="primary",
        use_container_width=True
    )

else:
    # 空状态提示
    st.markdown("""
    <div style="text-align: center; padding: 30px; color: #aaa;">
        <br>
        请上传一张正面人像照片以开始制作
    </div>
    """, unsafe_allow_html=True)

# 5. 底部信息
with st.expander("关于本工具"):
    st.markdown("---")
    st.markdown("**开发者**: 西电 陈宥廷 刘家瑄")
    st.markdown("**技术栈**: U2NetP + ONNX + Streamlit")
    st.markdown("**反馈邮箱**: 1632728403@qq.com")
