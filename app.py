import os
import numpy as np
from PIL import Image, ImageEnhance
import onnxruntime as ort
import streamlit as st
from io import BytesIO

# 云端环境兼容配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== 【新增：浏览量 + 点赞功能 核心代码】 ====================
# 初始化会话状态，防止重复计数
def init_session():
    if "visited" not in st.session_state:
        st.session_state.visited = False
    if "liked" not in st.session_state:
        st.session_state.liked = False

# 浏览量统计（文件持久化）
def get_visit_count():
    count_file = "visit_count.txt"
    if not os.path.exists(count_file):
        with open(count_file, "w") as f:
            f.write("0")
    with open(count_file, "r") as f:
        count = int(f.read().strip())
    # 仅首次加载计数
    if not st.session_state.visited:
        count += 1
        with open(count_file, "w") as f:
            f.write(str(count))
        st.session_state.visited = True
    return count

# 点赞功能（文件持久化 + 防重复点赞）
def get_like_count():
    like_file = "like_count.txt"
    if not os.path.exists(like_file):
        with open(like_file, "w") as f:
            f.write("0")
    with open(like_file, "r") as f:
        return int(f.read().strip())

def add_like():
    if not st.session_state.liked:
        like_file = "like_count.txt"
        current = get_like_count() + 1
        with open(like_file, "w") as f:
            f.write(str(current))
        st.session_state.liked = True

# 初始化
init_session()
visit_num = get_visit_count()
like_num = get_like_count()
# ==========================================================================

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

# -------------------------- 界面代码 (含格式选择) --------------------------

# 1. 页面配置
st.set_page_config(
    page_title="西电AI证件照", 
    page_icon="./xidian_logo.png", 
    layout="centered"
)

# 2. 顶部标题栏 + 统计数据
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
        # 【显示浏览量+点赞数】
        st.markdown(f"<p style='text-align: center; color: #666; font-size: 0.8em;'>👀 总访问量：{visit_num}  |  👍 点赞数：{like_num}</p>", unsafe_allow_html=True)
    with col3:
        st.empty()

st.divider()

# 3. 核心控制区
st.markdown("### 📸 第一步：上传照片")
uploaded_file = st.file_uploader("支持 JPG / PNG 格式", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

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
    
    # 【第四步：导出格式选择】(新增)
    st.markdown("### 📄 第四步：选择导出格式")
    export_format = st.radio(
        label="推荐使用 PNG 以获得最佳画质",
        options=["PNG (无损高质量)", "JPG (较小文件)"],
        horizontal=True,
        index=0
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

    # 下载区域 (根据格式动态调整)
    st.divider()
    buf = BytesIO()
    
    # 解析用户选择
    if "JPG" in export_format:
        # 保存为 JPG
        result_img.save(buf, format="JPEG", quality=95) # quality=95 兼顾体积和画质
        ext = "jpg"
        mime_type = "image/jpeg"
    else:
        # 保存为 PNG
        result_img.save(buf, format="PNG")
        ext = "png"
        mime_type = "image/png"
        
    byte_data = buf.getvalue()
    
    st.download_button(
        label=f"📥 下载 {bg_color} 背景证件照 (.{ext})",
        data=byte_data,
        file_name=f"XDU证件照_{bg_color}.{ext}",
        mime=mime_type,
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

# ==================== 【新增：点赞按钮】 ====================
st.divider()
col_like, _ = st.columns([1, 3])
with col_like:
    if st.session_state.liked:
        st.button("👍 谢谢喵", disabled=True, use_container_width=True)
    else:
        if st.button("👍 觉得好用点个赞喵", on_click=add_like, use_container_width=True):
            st.rerun()
# ====================== 评论区传送门（独立跳转，不影响主功能） ======================
# 跳转按钮 - 点击直接打开你的独立评论区网站
st.link_button(
    label="💬 前往评论区 | 发布留言、点赞、互动",
    url="https://aibackground-comments.streamlit.app/",
    type="secondary",
    use_container_width=True
)
st.markdown("---")
# 5. 底部信息
with st.expander("关于本工具"):
    st.markdown("**技术栈**: U2NetP + ONNX + Streamlit")
    st.markdown("**反馈邮箱**: 1632728403@qq.com")
    st.markdown("卡拉彼丘好玩喵")

st.markdown("<p style='text-align:center; color:#888; font-size:12px;'>开发者：XDU 陈宥廷 刘家瑄 江奥</p>", unsafe_allow_html=True)
