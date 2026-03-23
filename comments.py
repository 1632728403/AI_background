# 独立评论区网站 | 匿名发布 + 极小点赞(防重复) + 回复 + 点赞排序 + 北京时间 + 紧凑间距
import streamlit as st
import json
import os
from datetime import datetime, timedelta
import uuid

# ===================== 基础配置 =====================
DATA_FILE = "comment_data.json"
ADMIN_PWD = "admin123"  # 你的管理员密码
st.set_page_config(page_title="通用评论区", page_icon="💬", layout="wide")

# 初始化：记录当前用户已点赞的评论ID（防重复点赞）
if "liked_comments" not in st.session_state:
    st.session_state.liked_comments = set()

# ===================== 北京时间获取函数 =====================
def get_beijing_time():
    # UTC时间 +8小时 = 北京时间
    utc_now = datetime.utcnow()
    beijing_now = utc_now + timedelta(hours=8)
    return beijing_now.strftime("%m-%d %H:%M")

# ===================== 数据初始化 =====================
def init_data():
    if not os.path.exists(DATA_FILE):
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)

def load_data():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data):
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

init_data()

# ===================== 核心样式（点赞图标极小 + 紧凑间距） =====================
st.markdown("""
<style>
/* 全局紧凑排版 */
.main > div {padding: 1rem 2rem;}
/* 超小点赞按钮 🔥 核心：极小尺寸，不占排版 */
.like-btn {
    background: transparent !important;
    border: none !important;
    font-size: 12px !important;
    color: #888 !important;
    padding: 0 4px !important;
    margin: 0 4px !important;
    cursor: pointer;
}
.like-btn:hover {color: #ff4444 !important;}
/* 已点赞样式 */
.liked {color: #ff4444 !important;}
/* 评论卡片紧凑 - 减小底部间距 */
.comment-box {
    padding: 8px 12px;
    margin: 4px 0;  /* 从6px改为4px，缩小上下间距 */
    border-radius: 6px;
    background: #1a1a1a;
}
/* 回复区域缩进 - 减小间距 */
.reply-box {
    margin-left: 20px;
    padding: 4px 8px;  /* 从6px 10px改为4px 8px */
    margin-top: 4px;   /* 新增，减小回复与主评论的间距 */
    background: #222;
    border-radius: 4px;
}
/* 文字大小统一 */
.text-sm {font-size: 13px; color: #aaa;}
/* 减小分隔线的上下间距 */
hr {
    margin: 8px 0 !important;  /* 从默认改为8px 0 */
    border-color: #333 !important;
}
/* 减小按钮区域的间距 */
.stButton > button {
    margin: 2px 2px 4px 0 !important;  /* 缩小按钮上下间距 */
}
</style>
""", unsafe_allow_html=True)

# ===================== 页面标题 =====================
st.title("💬 独立评论区")
st.markdown("---")

# ===================== 管理员权限 =====================
admin_mode = st.sidebar.text_input("管理员密码", type="password", label_visibility="collapsed") == ADMIN_PWD
if admin_mode:
    st.sidebar.success("✅ 管理员已登录")

# ===================== 发布评论模块 =====================
st.subheader("✍️ 发布评论")
with st.form("publish_comment", clear_on_submit=True):
    nickname = st.text_input("昵称（可不填，默认匿名）", placeholder="匿名用户")
    content = st.text_area("评论内容", height=80)
    submit = st.form_submit_button("发布评论")

# 发布逻辑
if submit and content.strip():
    data = load_data()
    new_comment = {
        "id": str(uuid.uuid4()),
        "nickname": nickname.strip() or "匿名用户",
        "content": content.strip(),
        "time": get_beijing_time(),  # 🔥 固定北京时间
        "likes": 0,
        "replys": []
    }
    data.append(new_comment)
    save_data(data)
    st.success("发布成功！")
    st.rerun()

st.markdown("---")

# ===================== 评论展示 + 防重复点赞 + 回复 + 排序 =====================
st.subheader("📄 全部评论（按点赞数排序）")
comments = load_data()
# 按点赞数降序排列
comments_sorted = sorted(comments, key=lambda x: x["likes"], reverse=True)

if not comments_sorted:
    st.info("暂无评论，快来发布第一条吧～")
else:
    for idx, c in enumerate(comments_sorted):
        comment_id = c["id"]
        has_liked = comment_id in st.session_state.liked_comments

        # 主评论
        st.markdown(f"""
        <div class="comment-box">
            <b>{c['nickname']}</b> <span class="text-sm">{c['time']}</span>
            <p>{c['content']}</p>
        </div>
        """, unsafe_allow_html=True)

        # 极小点赞按钮 + 防重复点赞
        col1, col2, col3 = st.columns([1, 3, 20])
        with col1:
            btn_label = f"👍 {c['likes']}"
            if has_liked:
                st.button(btn_label, key=f"like_{comment_id}", disabled=True, help="已点赞")
            else:
                if st.button(btn_label, key=f"like_{comment_id}", help="点赞"):
                    c["likes"] += 1
                    st.session_state.liked_comments.add(comment_id)
                    save_data(comments)
                    st.rerun()
        # 回复按钮
        with col2:
            show_reply = st.button("回复", key=f"reply_{comment_id}")
        # 管理员删除
        with col3:
            if admin_mode and st.button("🗑️ 删评", key=f"del_{comment_id}"):
                comments.remove(c)
                save_data(comments)
                st.rerun()

        # 回复发布框
        if show_reply:
            with st.form(f"reply_form_{comment_id}", clear_on_submit=True):
                reply_nick = st.text_input("回复昵称", placeholder="匿名")
                reply_content = st.text_input("回复内容", label_visibility="collapsed")
                reply_submit = st.form_submit_button("发送回复")
                if reply_submit and reply_content.strip():
                    new_reply = {
                        "nickname": reply_nick.strip() or "匿名用户",
                        "content": reply_content.strip(),
                        "time": get_beijing_time()  # 🔥 回复也用北京时间
                    }
                    c["replys"].append(new_reply)
                    save_data(comments)
                    st.rerun()

        # 展示回复
        if c["replys"]:
            for reply in c["replys"]:
                st.markdown(f"""
                <div class="reply-box">
                    <b>{reply['nickname']}</b> <span class='text-sm'>{reply['time']}</span>
                    <p>{reply['content']}</p>
                </div>
                """, unsafe_allow_html=True)

        # 分隔线（区分不同评论）
        st.divider()
