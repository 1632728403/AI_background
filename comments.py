# 独立评论区网站 | 匿名发布 + 极小点赞 + 回复 + 点赞排序
import streamlit as st
import json
import os
from datetime import datetime
import uuid

# ===================== 基础配置 =====================
# 数据存储文件（独立存储，永不丢失）
DATA_FILE = "comment_data.json"
# 管理员密码（删评专用）
ADMIN_PWD = "admin123"
# 页面配置
st.set_page_config(page_title="评论区", page_icon="💬", layout="wide")

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

# ===================== 核心样式（点赞图标极小！） =====================
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
/* 评论卡片紧凑 */
.comment-box {
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 6px;
    background: #1a1a1a;
}
/* 回复区域缩进 */
.reply-box {
    margin-left: 20px;
    padding: 6px 10px;
    background: #222;
    border-radius: 4px;
}
/* 文字大小统一 */
.text-sm {font-size: 13px; color: #aaa;}
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
        "time": datetime.now().strftime("%m-%d %H:%M"),
        "likes": 0,
        "replys": []
    }
    data.append(new_comment)
    save_data(data)
    st.success("发布成功！")
    st.rerun()

st.markdown("---")

# ===================== 评论展示 + 点赞 + 回复 + 排序 =====================
st.subheader("📄 全部评论")
comments = load_data()

# 🔥 核心：按点赞数降序排列
comments_sorted = sorted(comments, key=lambda x: x["likes"], reverse=True)

if not comments_sorted:
    st.info("暂无评论，快来发布第一条吧～")
else:
    for idx, c in enumerate(comments_sorted):
        # 主评论
        st.markdown(f"""
        <div class="comment-box">
            <b>{c['nickname']}</b> <span class="text-sm">{c['time']}</span>
            <p>{c['content']}</p>
        </div>
        """, unsafe_allow_html=True)

        # 🔥 极小点赞按钮（完全不影响排版）
        col1, col2, col3 = st.columns([1, 3, 20])
        with col1:
            if st.button(f"👍 {c['likes']}", key=f"like_{c['id']}", help="点赞"):
                c["likes"] += 1
                save_data(comments)
                st.rerun()
        # 回复按钮
        with col2:
            show_reply = st.button("回复", key=f"reply_{c['id']}")
        # 管理员删除
        with col3:
            if admin_mode and st.button("🗑️ 删评", key=f"del_{c['id']}"):
                comments.remove(c)
                save_data(comments)
                st.rerun()

        # 回复发布框
        if show_reply:
            with st.form(f"reply_form_{c['id']}", clear_on_submit=True):
                reply_nick = st.text_input("回复昵称", placeholder="匿名")
                reply_content = st.text_input("回复内容", label_visibility="collapsed")
                reply_submit = st.form_submit_button("发送回复")
                if reply_submit and reply_content.strip():
                    new_reply = {
                        "nickname": reply_nick.strip() or "匿名用户",
                        "content": reply_content.strip(),
                        "time": datetime.now().strftime("%m-%d %H:%M")
                    }
                    c["replys"].append(new_reply)
                    save_data(comments)
                    st.rerun()

        # 展示回复
        if c["replys"]:
            for reply in c["replys"]:
                st.markdown(f"""
                <div class="reply-box">
                    <b>{reply['nickname']}</b> <span class="text-sm">{reply['time']}</span>
                    <p>{reply['content']}</p>
                </div>
                """, unsafe_allow_html=True)

        st.divider()
