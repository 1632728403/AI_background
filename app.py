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
# ==========================================================

# ====================== 【底部评论区】紧凑版 | 含时间+点赞 (不影响主功能) ======================
import json
import os
from datetime import datetime

# 独立存储评论数据（点赞数+时间+昵称+内容）
COMMENT_FILE = "comments.json"

# 初始化JSON文件结构
if not os.path.exists(COMMENT_FILE):
    with open(COMMENT_FILE, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)

# --- 样式微调：让电脑端更紧凑 ---
st.markdown("""
<style>
/* 压缩评论区整体间距 */
div[data-testid="stVerticalBlock"] > div:has(> h4) {
    margin-top: 0.5rem !important;
    margin-bottom: 0.5rem !important;
}
/* 紧凑输入框 */
.stTextInput>div>div>input {
    padding: 0.4rem 0.75rem !important;
    font-size: 0.9rem !important;
}
/* 紧凑文本框 */
.stTextArea>div>div>textarea {
    padding: 0.4rem 0.75rem !important;
    font-size: 0.9rem !important;
    height: 70px !important;
}
/* 紧凑按钮 */
.stButton>button {
    padding: 0.4rem 1.2rem !important;
    font-size: 0.9rem !important;
    margin-top: 1.8rem !important; /* 让按钮和输入框对齐 */
}
/* 评论卡片样式 */
.comment-card {
    background-color: #2a2a2a;
    padding: 10px 15px;
    border-radius: 8px;
    margin-bottom: 8px;
}
/* 评论时间样式 */
.comment-time {
    font-size: 0.8rem;
    color: #888;
}
/* 点赞按钮样式 */
.like-btn {
    background-color: #444;
    color: white;
    border: none;
    padding: 3px 10px;
    border-radius: 12px;
    cursor: pointer;
    font-size: 0.85rem;
}
.like-btn:hover {
    background-color: #666;
}
</style>
""", unsafe_allow_html=True)

# --- 1. 评论区标题 (缩小字体，更紧凑) ---
st.markdown("<h4 style='margin-bottom: 1rem;'>💬 用户评论区 & 反馈建议</h4>", unsafe_allow_html=True)

# --- 2. 发布评论表单 (两列布局，更紧凑) ---
with st.form("compact_comment_form", clear_on_submit=True):
    col_input, col_btn = st.columns([3, 1])
    
    with col_input:
        nickname = st.text_input("", placeholder="留下你的昵称~", label_visibility="collapsed")
        comment_content = st.text_area("", placeholder="分享你的使用体验或建议...", label_visibility="collapsed")
    
    with col_btn:
        # 占位符对齐
        st.write("")
        st.write("")
        submit_btn = st.form_submit_button("发布")

# --- 3. 发布逻辑 (带时间戳) ---
if submit_btn and comment_content.strip():
    # 读取现有评论
    with open(COMMENT_FILE, "r", encoding="utf-8") as f:
        comments = json.load(f)
    
    # 组装新评论（包含时间和初始点赞数）
    new_comment = {
        "nickname": nickname.strip() if nickname.strip() else "匿名同学",
        "content": comment_content.strip(),
        "time": datetime.now().strftime("%m-%d %H:%M"),  # 简洁时间格式
        "likes": 0  # 初始点赞数
    }
    comments.append(new_comment)
    
    # 保存
    with open(COMMENT_FILE, "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=2)
    
    st.success("✅ 评论发布成功！")

# --- 4. 展示评论 (带时间+点赞功能) ---
st.markdown("<h5 style='margin-top: 1.5rem; margin-bottom: 0.5rem;'>📄 最新评论</h5>", unsafe_allow_html=True)

# 读取评论
with open(COMMENT_FILE, "r", encoding="utf-8") as f:
    comments = json.load(f)

if not comments:
    st.info("暂无评论，快来留下你的第一条反馈吧～")
else:
    # 倒序展示（最新的在上面）
    for comment in reversed(comments):
        # 用卡片包裹，更美观
        st.markdown(f'<div class="comment-card">', unsafe_allow_html=True)
        
        # 昵称 + 时间
        col1, col2 = st.columns([4, 1])
        with col1:
            st.markdown(f"**{comment['nickname']}** <span class='comment-time'>{comment['time']}</span>", unsafe_allow_html=True)
        with col2:
            # 点赞按钮
            like_btn = st.button(f"👍 {comment['likes']}", key=f"like_{comment['time']}_{comment['content'][:5]}")
        
        # 评论内容
        st.write(comment['content'])
        
        # 点赞逻辑
        if like_btn:
            # 找到对应评论并+1
            for c in comments:
                if c["time"] == comment["time"] and c["content"] == comment["content"]:
                    c["likes"] += 1
                    break
            # 保存
            with open(COMMENT_FILE, "w", encoding="utf-8") as f:
                json.dump(comments, f, ensure_ascii=False, indent=2)
            # 强制刷新
            st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.divider()

# 5. 底部信息
with st.expander("关于本工具"):
    st.markdown("---")
    st.markdown("**开发者**: XDU 陈宥廷 刘家瑄 江奥")
    st.markdown("**技术栈**: U2NetP + ONNX + Streamlit")
    st.markdown("**反馈邮箱**: 1632728403@qq.com")
    st.markdown("卡拉彼丘好玩喵")
