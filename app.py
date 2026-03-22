# ==============================================
# U2NetP 轻量版人像分割工具 | 三色证件照背景
# 西安电子科技大学 大模型应用创新赛作品
# 西电本地图标版 | 下载无破损 | 开发者：陈宥廷 刘家瑄
# ==============================================
import os
import numpy as np
from PIL import Image, ImageEnhance  # 仅新增：亮度调节依赖库
import onnxruntime as ort
import streamlit as st
from io import BytesIO

# 云端环境兼容配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== 【新增功能函数：无修改原有代码】 ====================
# 1. 调整人像到证件照标准比例（自动居中）
def resize_person(img, mask, target_ratio=1.0):
    coords = np.argwhere(np.array(mask) > 127)
    if len(coords) == 0:
        return img, mask
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # 裁剪人像区域
    person_crop = img.crop((x_min, y_min, x_max, y_max))
    mask_crop = mask.crop((x_min, y_min, x_max, y_max))
    
    orig_w, orig_h = img.size
    # 根据目标比例计算证件照画布尺寸
    if target_ratio < 1:  # 竖版证件照（宽 < 高）
        new_w = int(orig_h * target_ratio)
        new_h = orig_h
    elif target_ratio > 1:  # 横版证件照（宽 > 高）
        new_w = orig_w
        new_h = int(orig_w / target_ratio)
    else:  # 正方形证件照
        new_size = min(orig_w, orig_h)
        new_w = new_size
        new_h = new_size
    
    # 等比缩放人像至画布的90%大小（避免贴边，更美观）
    person_w, person_h = person_crop.size
    max_person_w = int(new_w * 0.9)
    max_person_h = int(new_h * 0.9)
    scale = min(max_person_w / person_w, max_person_h / person_h)
    scaled_person_w = int(person_w * scale)
    scaled_person_h = int(person_h * scale)
    person_resized = person_crop.resize((scaled_person_w, scaled_person_h), Image.LANCZOS)
    mask_resized = mask_crop.resize((scaled_person_w, scaled_person_h), Image.NEAREST)
    
    # 创建新画布并居中放置人像
    new_img = Image.new("RGB", (new_w, new_h), (0, 0, 0))
    new_mask = Image.new("L", (new_w, new_h), 0)
    x_offset = (new_w - scaled_person_w) // 2
    y_offset = (new_h - scaled_person_h) // 2
    new_img.paste(person_resized, (x_offset, y_offset))
    new_mask.paste(mask_resized, (x_offset, y_offset))
    return new_img, new_mask

# 2. 调整图片亮度（保留原功能）
def adjust_brightness(img, value):
    return ImageEnhance.Brightness(img).enhance(value)
# ========================================================================

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
# 本地西电图标（核心修改）
st.set_page_config(
    page_title="西电AI人像证件照工具", 
    page_icon="./xidian_logo.png",  # 调用你上传的本地图标
    layout="wide"
)

# 顶部Logo + 标题
col1, col2 = st.columns([1, 10])
with col1:
    st.image("./xidian_logo.png", width=80)  # 本地校徽
with col2:
    st.title("🎨 西电专属 AI 人像分割 & 证件照背景替换工具")

# 双栏布局
col_img1, col_img2 = st.columns(2)
with col_img1:
    st.markdown("### 📸 上传人像照片")
    uploaded_file = st.file_uploader("支持 JPG / PNG 格式", type=["jpg", "jpeg", "png"])

with col_img2:
    st.markdown("### ✨ 处理完成效果")

# ==================== 【修改：比例模块 → 证件照标准尺寸选择】 ====================
st.markdown("---")
st.subheader("⚙️ 自定义调节")
col_bright, col_ratio = st.columns(2)
with col_bright:
    brightness = st.slider("💡 亮度调节", 0.5, 1.5, 1.0, 0.05)
with col_ratio:
    ratio_option = st.selectbox(
        "📏 证件照尺寸比例",
        [
            "一寸证件照 (2.5:3.5)",
            "小两寸证件照 (3.3:4.8)",
            "两寸证件照 (3.5:5.3)",
            "正方形 (1:1)",
            "常用竖版 (3:4)",
            "常用横版 (4:3)"
        ],
        index=0,
        help="选择常见证件照的标准比例，自动调整人像位置与尺寸"
    )
    # 映射选项到「宽高比」（w/h）
    ratio_map = {
        "一寸证件照 (2.5:3.5)": 2.5/3.5,
        "小两寸证件照 (3.3:4.8)": 3.3/4.8,
        "两寸证件照 (3.5:5.3)": 3.5/5.3,
        "正方形 (1:1)": 1.0,
        "常用竖版 (3:4)": 3/4,
        "常用横版 (4:3)": 4/3
    }
    target_ratio = ratio_map[ratio_option]
# ========================================================================

# 背景颜色选择
st.subheader("🎨 选择证件照背景颜色")
bg_color = st.radio(
    label="",
    options=["白色", "红色", "蓝色"],
    horizontal=True,
    index=0
)

# 处理逻辑
if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")
    col_img1.image(img, caption="原始图片", use_column_width=True)
    
    session = load_model()
    with st.spinner("🔍 AI 正在处理中..."):
        mask = infer_mask(img, session)
        # ==================== 【调用证件照比例函数】 ====================
        img_resized, mask_resized = resize_person(img, mask, target_ratio)
        img_final = adjust_brightness(img_resized, brightness)
        # ========================================================================
        result_img = generate_result(img_final, mask_resized, bg_color)
    
    col_img2.image(result_img, caption=f"{ratio_option} | {bg_color}背景 | 亮度:{brightness}", use_column_width=True)
    
    # 修复下载破损（标准PNG输出）
    st.divider()
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    byte_data = buf.getvalue()

    st.download_button(
        label=f"💾 下载{bg_color}背景证件照",
        data=byte_data,
        file_name=f"XDU证件照_{ratio_option}_{bg_color}.png",
        mime="image/png"
    )
# -------------------------- 页面底部：开发者署名（核心要求） --------------------------
st.markdown("---")
st.markdown("<h5 style='text-align: center; color: #666;'>本页面由 XDU 陈宥廷 刘家瑄 开发喵🐱</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; color: #666;'>反馈：1632728403@qq.com</h5>", unsafe_allow_html=True)
