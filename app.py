import os
import numpy as np
from PIL import Image
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

# -------------------------- 界面优化代码 --------------------------

# 1. 页面配置
st.set_page_config(
    page_title="西电AI人像证件照工具", 
    page_icon="./xidian_logo.png", 
    layout="centered"  # 改为居中布局，显得更聚焦
)

# 2. 侧边栏 (Sidebar) - 用于放置控件和说明
with st.sidebar:
    # 侧边栏顶部Logo
    try:
        st.image("./xidian_logo.png", width=100)
    except:
        pass # 防止图标找不到报错
    
    st.title("🎨 西电专属证件照")
    st.markdown("---")
    
    st.markdown("### 1. 参数设置")
    bg_color = st.radio(
        "选择背景颜色",
        options=["白色", "红色", "蓝色"],
        horizontal=True,
        index=0
    )
    
    st.markdown("---")
    st.markdown("### 2. 使用说明")
    st.info("""
    1. 上传一张清晰的人像照片。
    2. 等待 AI 自动抠图。
    3. 在右侧查看效果并下载。
    """)
    
    st.markdown("---")
    st.markdown("### 开发者")
    st.caption("XDU 陈宥廷 刘家瑄")
    st.caption("反馈：1632728403@qq.com")

# 3. 主界面 (Main Area) - 用于展示图片
# 顶部标题栏
col1, col2 = st.columns([1, 10])
with col1:
    try:
        st.image("./xidian_logo.png", width=60)
    except:
        st.empty()
with col2:
    st.header("AI 人像分割 & 背景替换")
    st.write("基于 U2NetP 轻量模型 | 支持三色证件照一键生成")

st.divider()

# 图片上传与展示区域
uploaded_file = st.file_uploader("📸 点击上传人像照片 (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    
    # 加载模型
    session = load_model()
    
    # 两栏布局：左原图，右结果
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("原始图片")
        # 限制宽度为 400px，避免图片过大
        st.image(img, width=400) 
    
    with col2:
        st.subheader("处理结果")
        with st.spinner("AI 正在智能抠图中..."):
            mask = infer_mask(img, session)
            result_img = generate_result(img, mask, bg_color)
        
        # 限制宽度为 400px
        st.image(result_img, width=400)

    # 底部下载区域
    st.divider()
    st.subheader("💾 下载成品")
    
    # 修复下载破损
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    byte_data = buf.getvalue()

    # 使用更醒目的按钮样式
    st.download_button(
        label=f"下载 {bg_color} 背景证件照",
        data=byte_data,
        file_name=f"XDU证件照_{bg_color}.png",
        mime="image/png",
        type="primary",
        use_container_width=True
    )
else:
    # 占位提示，让界面不空白
    st.markdown("""
    <div style="text-align: center; color: #888; padding: 50px;">
        <br><br>
        请在上方上传图片以开始
    </div>
    """, unsafe_allow_html=True)
