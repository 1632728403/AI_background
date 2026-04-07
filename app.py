import os
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw, ImageFont
import onnxruntime as ort
import streamlit as st
from io import BytesIO

# 云端环境兼容配置
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ==================== 【原版：浏览量+点赞（强制初始保底）】 ====================
def init_session():
    if "visited" not in st.session_state:
        st.session_state.visited = False
    if "liked" not in st.session_state:
        st.session_state.liked = False

def get_visit_count():
    count_file = "visit_count.txt"
    INIT_VISIT = 506
    if not os.path.exists(count_file):
        with open(count_file, "w") as f:
            f.write(str(INIT_VISIT))
    with open(count_file, "r") as f:
        count = int(f.read().strip())
    if count < INIT_VISIT:
        count = INIT_VISIT
    if not st.session_state.visited:
        count += 1
        with open(count_file, "w") as f:
            f.write(str(count))
        st.session_state.visited = True
    return count

def get_like_count():
    like_file = "like_count.txt"
    INIT_LIKE = 50
    if not os.path.exists(like_file):
        with open(like_file, "w") as f:
            f.write(str(INIT_LIKE))
    with open(like_file, "r") as f:
        count = int(f.read().strip())
    if count < INIT_LIKE:
        count = INIT_LIKE
    return count

def add_like():
    if not st.session_state.liked:
        current = get_like_count() + 1
        like_file = "like_count.txt"
        with open(like_file, "w") as f:
            f.write(str(current))
        st.session_state.liked = True

init_session()
visit_num = get_visit_count()
like_num = get_like_count()

# ==========================================================================
# -------------------------- 核心模型函数 --------------------------
@st.cache_resource
def load_model():
    session = ort.InferenceSession("u2netp.onnx")
    return session

def preprocess_image(image):
    img_resize = image.resize((320, 320))
    img_np = np.array(img_resize).astype(np.float32) / 255.0
    input_tensor = img_np.transpose(2, 0, 1)[np.newaxis, :]
    return input_tensor

def infer_mask(image, session, feather_radius=0):
    input_tensor = preprocess_image(image)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_tensor})[0]
    pred = output[0, 0, :, :]
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask).resize(image.size, Image.NEAREST)
    
    if feather_radius > 0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(feather_radius))
        
    return mask_pil

# 修改点：直接接收 rgb 颜色元组，支持任意颜色
def generate_result(image, mask_pil, bg_rgb):
    mask_np = np.array(mask_pil)
    img_np = np.array(image)
    result = np.zeros_like(img_np)
    
    alpha = mask_np.astype(float) / 255.0
    for c in range(3):
        bg_channel = np.full(img_np[:,:,c].shape, bg_rgb[c])
        result[:, :, c] = (img_np[:, :, c] * alpha + bg_channel * (1 - alpha)).astype(np.uint8)
        
    return Image.fromarray(result)

# -------------------------- 界面代码 --------------------------
st.set_page_config(
    page_title="西电AI证件照",
    page_icon="🎓", 
    layout="centered"
)

# ==================== 【侧边栏：高级功能区】 ====================
with st.sidebar:
    st.header("🧰 高级工具箱")
    st.markdown("上传照片后，在这里进行精修哦~")
    
    with st.expander("✂️ 画面裁剪", expanded=False):
        st.caption("通过滑块切除四周多余的部分 (百分比)")
        crop_top = st.slider("上边缘裁切 %", 0, 40, 0, 1)
        crop_bottom = st.slider("下边缘裁切 %", 0, 40, 0, 1)
        crop_left = st.slider("左边缘裁切 %", 0, 40, 0, 1)
        crop_right = st.slider("右边缘裁切 %", 0, 40, 0, 1)

    with st.expander("🪄 画质与光线优化", expanded=True):
        brightness_factor = st.slider("💡 亮度", 0.5, 1.8, 1.0, 0.05)
        contrast_factor = st.slider("🌓 对比度", 0.5, 1.8, 1.0, 0.05)
        color_factor = st.slider("🎨 饱和度", 0.0, 2.0, 1.0, 0.05)
        sharpness_factor = st.slider("🗡️ 锐化程度", 0.0, 3.0, 1.0, 0.1)
        
    with st.expander("🎭 边缘与人像处理", expanded=False):
        feather_radius = st.slider("🌫️ 边缘柔化 (防锯齿)", 0, 5, 1, 1)
        smooth_skin = st.checkbox("✨ 开启轻度磨皮 (柔焦)")
        
    with st.expander("🪪 添加专属水印", expanded=False):
        watermark_text = st.text_input("水印内容", placeholder="例如：仅供西电选课使用")
        watermark_color = st.color_picker("水印颜色", "#FFFFFF")
        watermark_opacity = st.slider("水印透明度", 0, 255, 128)

# ==================== 【主界面区】 ====================
with st.container():
    col1, col2, col3 = st.columns([1, 6, 1])
    with col1:
        try:
            st.image("./xidian_logo.png", width=50)
        except:
            st.empty()
    with col2:
        st.markdown("<h1 style='text-align: center; font-size: 1.5em;'>西电专属 AI 证件照</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: grey; font-size: 0.9em;'>人像分割 & 背景替换 & 专业精修</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='text-align: center; color: #666; font-size: 0.8em;'>👀 总访问量：{visit_num} | 👍 点赞数：{like_num}</p>", unsafe_allow_html=True)
    with col3:
        st.empty()
st.divider()

st.markdown("### 📸 第一步：上传照片")
uploaded_file = st.file_uploader("支持 JPG / PNG 格式", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    # 【第二步：背景颜色 (恢复调色盘任意选色)】
    st.markdown("### 🎨 第二步：选择背景颜色")
    bg_option = st.radio(
        label="",
        options=["白色", "红色", "蓝色", "西电蓝", "🎨 任意选色 (调色盘)"],
        horizontal=True,
        index=0
    )
    
    # 根据选项解析颜色
    if bg_option == "🎨 任意选色 (调色盘)":
        custom_hex = st.color_picker("👉 请在调色盘中自由选择您的专属背景色", "#87CEEB")
        # 将 HEX 转换为 RGB 元组
        bg_rgb = tuple(int(custom_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        bg_name = "自定义配色"
    else:
        color_map = {
            "白色": (255, 255, 255),
            "红色": (255, 0, 0),
            "蓝色": (0, 0, 255),
            "西电蓝": (0, 65, 130)
        }
        bg_rgb = color_map[bg_option]
        bg_name = bg_option

    # 【第三步：导出格式选择】
    st.markdown("### 📄 第三步：选择导出格式")
    export_format = st.radio(
        label="推荐使用 PNG 以获得最佳画质",
        options=["PNG (无损高质量)", "JPG (较小文件)"],
        horizontal=True,
        index=0
    )

    img = Image.open(uploaded_file).convert("RGB")
    
    w, h = img.size
    left = int(w * crop_left / 100)
    right = int(w * (100 - crop_right) / 100)
    top = int(h * crop_top / 100)
    bottom = int(h * (100 - crop_bottom) / 100)
    if left < right and top < bottom: 
        img = img.crop((left, top, right, bottom))
    
    session = load_model()
    with st.spinner("🔍 AI 正在处理并应用您的设置，请稍候..."):
        if smooth_skin:
            img = img.filter(ImageFilter.SMOOTH_MORE)
            
        mask = infer_mask(img, session, feather_radius)
        # 传入解析后的 RGB 颜色
        result_img = generate_result(img, mask, bg_rgb)
        
        if brightness_factor != 1.0:
            result_img = ImageEnhance.Brightness(result_img).enhance(brightness_factor)
        if contrast_factor != 1.0:
            result_img = ImageEnhance.Contrast(result_img).enhance(contrast_factor)
        if color_factor != 1.0:
            result_img = ImageEnhance.Color(result_img).enhance(color_factor)
        if sharpness_factor != 1.0:
            result_img = ImageEnhance.Sharpness(result_img).enhance(sharpness_factor)
            
        if watermark_text:
            draw = ImageDraw.Draw(result_img)
            try:
                font = ImageFont.load_default() 
            except:
                font = None
            h_color = watermark_color.lstrip('#')
            rgb_color = tuple(int(h_color[i:i+2], 16) for i in (0, 2, 4))
            rgba_color = rgb_color + (watermark_opacity,)
            img_w, img_h = result_img.size
            draw.text((img_w*0.05, img_h*0.9), watermark_text, fill=rgba_color, font=font)

    st.divider()
    st.markdown("### ✨ 效果预览 & 下载")
    st.info("💡 提示：可以在左侧边栏调整对比度、饱和度和裁剪画面哦！")
    
    c1, c2 = st.columns(2)
    with c1:
        st.caption("原始图片 (应用裁剪后)")
        st.image(img, use_container_width=True)
    with c2:
        st.caption(f"处理结果 ({bg_name})")
        st.image(result_img, use_container_width=True)

    st.divider()
    buf = BytesIO()
    
    if "JPG" in export_format:
        result_img.convert("RGB").save(buf, format="JPEG", quality=95)
        ext = "jpg"
        mime_type = "image/jpeg"
    else:
        result_img.save(buf, format="PNG")
        ext = "png"
        mime_type = "image/png"
        
    byte_data = buf.getvalue()
    
    st.download_button(
        label=f"📥 下载 {bg_name} 背景证件照 (.{ext})",
        data=byte_data,
        file_name=f"XDU证件照_{bg_name}.{ext}",
        mime=mime_type,
        type="primary",
        use_container_width=True
    )

else:
    st.markdown("""
    <div style="text-align: center; padding: 30px; color: #aaa;">
        <br>
        请上传一张正面人像照片以开始制作
    </div>
    """, unsafe_allow_html=True)

# ==================== 【点赞按钮】 ====================
st.divider()
col_like, _ = st.columns([1, 3])
with col_like:
    if st.session_state.liked:
        st.button("👍 谢谢喵", disabled=True, use_container_width=True)
    else:
        if st.button("👍 觉得好用点个赞喵", on_click=add_like, use_container_width=True):
            st.rerun()

# ====================== 评论区传送门 ======================
st.link_button(
    label="💬 前往评论区 | 发布留言、点赞、互动",
    url="https://aibackground-comments.streamlit.app/",
    type="secondary",
    use_container_width=True
)
st.markdown("<p style='text-align:center; color:#888; font-size:15px;'>若打不开请移步浏览器</p>", unsafe_allow_html=True)

st.markdown("---")

with st.expander("关于本工具"):
    st.markdown("技术栈: U2NetP + ONNX + Streamlit + PIL")
    st.markdown("反馈邮箱: 1632728403@qq.com")
    st.markdown("卡拉彼丘好玩喵")
    st.markdown("<p style='text-align:center; color:#888; font-size:12px;'>开发者：XDU 陈宥廷 刘家瑄 江奥</p>", unsafe_allow_html=True)
