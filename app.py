# ==============================================
# U2NetP 轻量版人像分割工具 | ONNX版 | 无CV2依赖
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
    # 关键修改：模型文件名改为 u2netp.onnx（轻量版）
    session = ort.InferenceSession("u2netp.onnx")
    return session

# -------------------------- 图片预处理（适配U2NetP输入要求） --------------------------
def preprocess_image(image):
    # U2NetP 轻量版标准输入尺寸：320x320（如果你的模型是256x256，修改这里即可）
    img_resize = image.resize((320, 320))
    img_np = np.array(img_resize).astype(np.float32) / 255.0  # 归一化到[0,1]
    # 转换为模型输入格式：[batch, channel, height, width]
    input_tensor = img_np.transpose(2, 0, 1)[np.newaxis, :]
    return input_tensor

# -------------------------- ONNX模型推理 --------------------------
def infer_mask(image, session):
    input_tensor = preprocess_image(image)
    # 获取模型输入输出名称（U2NetP和U2Net结构一致，取第一个输出层d0）
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name  # 轻量版输出层和原版一致
    
    # 执行推理
    output = session.run([output_name], {input_name: input_tensor})[0]
    # 后处理：sigmoid概率图 → 二值mask
    pred = output[0, 0, :, :]  # 取batch=0、输出层=0的结果
    mask = (pred > 0.5).astype(np.uint8) * 255  # 阈值分割，转成0/255二值图
    # 还原为原图尺寸
    mask_pil = Image.fromarray(mask).resize(image.size, Image.NEAREST)
    return mask_pil

# -------------------------- 生成最终结果（白底人像） --------------------------
def generate_result(image, mask_pil):
    mask_np = np.array(mask_pil)
    img_np = np.array(image)
    # 把人像保留，背景换成白色
    result = np.where(mask_np[..., None] > 127, img_np, 255)
    return Image.fromarray(result.astype(np.uint8))

# -------------------------- Streamlit网页界面 --------------------------
st.set_page_config(page_title="U2NetP轻量版人像分割", page_icon="🎨", layout="wide")
st.title("🎨 U2NetP 轻量版人像分割工具（ONNX预训练版）")

# 双栏布局：原图 + 结果图
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 输入图片")
    uploaded_file = st.file_uploader("上传 JPG / PNG 图片", type=["jpg", "jpeg", "png"])
with col2:
    st.markdown("### 分割结果")

# 处理图片逻辑
if uploaded_file is not None:
    # 打开并显示原图
    img = Image.open(uploaded_file).convert("RGB")
    col1.image(img, caption="输入图片", use_column_width=True)
    
    # 加载模型并推理
    session = load_model()
    with st.spinner("🔍 AI 正在分割人像..."):
        mask = infer_mask(img, session)
        result = generate_result(img, mask)
    
    # 显示结果
    col2.image(result, caption="分割完成", use_column_width=True)
    
    # 下载按钮
    st.divider()
    st.download_button(
        label="💾 下载无背景人像",
        data=result.tobytes(),
        file_name="u2netp_portrait.png",
        mime="image/png"
    )
