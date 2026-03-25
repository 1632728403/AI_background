import os
import numpy as np
from PIL import Image, ImageEnhance
import onnxruntime as ort
import streamlit as st
from io import BytesIO
import requests

# 云端环境兼容配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== 【永久存储：浏览量 + 点赞功能 核心代码（带调试）】 ====================
# 读取GitHub Gist配置
try:
    GITHUB_TOKEN = st.secrets["GITHUB_TOKEN"]
    VISIT_GIST_ID = st.secrets["VISIT_GIST_ID"]
    LIKE_GIST_ID = st.secrets["LIKE_GIST_ID"]
    st.session_state.secrets_ok = True
except Exception as e:
    st.session_state.secrets_ok = False
    st.warning(f"Secrets配置错误: {e}")
    GITHUB_TOKEN = ""
    VISIT_GIST_ID = ""
    LIKE_GIST_ID = ""

# 初始化会话状态
def init_session():
    if "visited" not in st.session_state:
        st.session_state.visited = False
    if "liked" not in st.session_state:
        st.session_state.liked = False

# Gist获取访问量（先读初始值，再+1）
def get_visit_count():
    if not st.session_state.secrets_ok:
        return 0
    try:
        url = f"https://api.github.com/gists/{VISIT_GIST_ID}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        res = requests.get(url, headers=headers)
        res.raise_for_status()  # 主动抛出HTTP错误
        count = int(res.json()["files"]["visit_count.txt"]["content"])
        # 仅首次加载时+1
        if not st.session_state.visited:
            new_count = count + 1
            save_visit_count(new_count)
            st.session_state.visited = True
            return new_count
        return count
    except Exception as e:
        st.error(f"读取访问量失败: {e}")
        return 0

# Gist保存访问量
def save_visit_count(count):
    if not st.session_state.secrets_ok:
        return
    try:
        url = f"https://api.github.com/gists/{VISIT_GIST_ID}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        data = {"files": {"visit_count.txt": {"content": str(count)}}}
        res = requests.patch(url, headers=headers, json=data)
        res.raise_for_status()
    except Exception as e:
        st.error(f"保存访问量失败: {e}")

# Gist获取点赞数（适配文件名: like）
def get_like_count():
    if not st.session_state.secrets_ok:
        return 0
    try:
        url = f"https://api.github.com/gists/{LIKE_GIST_ID}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        res = requests.get(url, headers=headers)
        res.raise_for_status()
        return int(res.json()["files"]["like"]["content"])
    except Exception as e:
        st.error(f"读取点赞数失败: {e}")
        return 0

# Gist增加点赞（防重复）
def add_like():
    if not st.session_state.secrets_ok or st.session_state.liked:
        return
    try:
        current = get_like_count()
        new_count = current + 1
        url = f"https://api.github.com/gists/{LIKE_GIST_ID}"
        headers = {"Authorization": f"token {GITHUB_TOKEN}"}
        data = {"files": {"like": {"content": str(new_count)}}}
        res = requests.patch(url, headers=headers, json=data)
        res.raise_for_status()
        st.session_state.liked = True
    except Exception as e:
        st.error(f"点赞失败: {e}")

# 初始化会话
init_session()
visit_num = get_visit_count()
like_num = get_like_count()
# ==========================================================================

# -------------------------- 核心证件照处理函数（不变） --------------------------
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

# -------------------------- 界面代码（不变） --------------------------
st.set_page_config(
    page_title="西电AI证件照", 
    page_icon="./xidian_logo.png", 
    layout="centered"
)

# 顶部标题栏 + 统计数据
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
        st.markdown(f"<p style='text-align: center; color: #666; font-size: 0.8em;'>👀 总访问量：{visit_num}  |  👍 点赞数：{like_num}</p>", unsafe_allow_html=True)
    with col3:
        st.empty()

st.divider()

# 核心控制区
st.markdown("### 📸 第一步：上传照片")
uploaded_file = st.file_uploader("支持 JPG / PNG 格式", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    st.markdown("### 🎨 第二步：选择背景颜色")
    bg_color = st.radio("", options=["白色", "红色", "蓝色"], horizontal=True, index=0)

    st.markdown("### 💡 第三步：调节照片亮度")
    brightness_factor = st.slider("亮度系数 (1.0 为原图)", min_value=0.5, max_value=1.8, value=1.0, step=0.05)
    
    st.markdown("### 📄 第四步：选择导出格式")
    export_format = st.radio("推荐使用 PNG 以获得最佳画质", options=["PNG (无损高质量)", "JPG (较小文件)"], horizontal=True, index=0)

    img = Image.open(uploaded_file).convert("RGB")
    session = load_model()
    
    with st.spinner("🔍 AI 正在处理中，请稍候..."):
        mask = infer_mask(img, session)
        result_img = generate_result(img, mask, bg_color)
        if brightness_factor != 1.0:
            enhancer = ImageEnhance.Brightness(result_img)
            result_img = enhancer.enhance(brightness_factor)
    
    st.divider()
    st.markdown("### ✨ 效果预览 & 下载")
    c1, c2 = st.columns(2)
    with c1:
        st.caption("原始图片")
        st.image(img, use_container_width=True)
    with c2:
        st.caption(f"处理结果 ({bg_color})")
        st.image(result_img, use_container_width=True)

    st.divider()
    buf = BytesIO()
    if "JPG" in export_format:
        result_img.save(buf, format="JPEG", quality=95)
        ext = "jpg"
        mime_type = "image/jpeg"
    else:
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
    st.markdown("""
    <div style="text-align: center; padding: 30px; color: #aaa;">
        <br>请上传一张正面人像照片以开始制作
    </div>
    """, unsafe_allow_html=True)

# 点赞按钮
st.divider()
col_like, _ = st.columns([1, 3])
with col_like:
    if st.session_state.liked:
        st.button("👍 谢谢喵", disabled=True, use_container_width=True)
    else:
        if st.button("👍 觉得好用点个赞喵", on_click=add_like, use_container_width=True):
            st.rerun()

# 评论区传送门
st.link_button(
    label="💬 前往评论区 | 发布留言、点赞、互动",
    url="https://aibackground-comments.streamlit.app/",
    type="secondary",
    use_container_width=True
)

from streamlit import runtime
ctx = runtime.get_instance()
user_agent = ""
if ctx and hasattr(ctx, "_session_manager") and ctx._session_manager:
    sess = next(iter(ctx._session_manager._sessions.values()), None)
    if sess:
        user_agent = sess.client.request.headers.get("User-Agent", "")
is_wechat = "MicroMessenger" in user_agent
if is_wechat:
    st.markdown("""
    <div style='text-align:center; color:#ffcc00; font-size:13px; margin-top:5px;'>
    ⚠️ 微信内无法直接打开，请复制链接到浏览器访问
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

with st.expander("关于本工具"):
    st.markdown("**技术栈**: U2NetP + ONNX + Streamlit")
    st.markdown("**反馈邮箱**: 1632728403@qq.com")
    st.markdown("卡拉彼丘好玩喵")

st.markdown("<p style='text-align:center; color:#888; font-size:12px;'>开发者：XDU 陈宥廷 刘家瑄 江奥</p>", unsafe_allow_html=True)
