"""
AmazonHelp Customer Support Agent — Streamlit Frontend
Styled with Gemini + Claude aesthetics: glassmorphism, dark mode, ambient glows, streaming.
"""

import uuid
import time
import sys
import os
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AmazonHelp AI Agent",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark / Light mode toggle stored in session ────────────────────────────────
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

# ── CSS Theming ────────────────────────────────────────────────────────────────
def inject_css(dark: bool):
    if dark:
        bg = "#0d0e12"
        sidebar_bg = "#16171d"
        card_bg = "#1f2028"
        card_border = "#2e303b"
        text = "#f3f4f6"
        sub_text = "#9ca3af"
        accent = "#3b82f6"
        accent_hover = "#2563eb"
        accent_glow = "rgba(59, 130, 246, 0.35)"
        user_bubble = "linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%)"
        user_text = "#ffffff"
        ai_bubble = "#16171d"
        ai_border = "#2b2d36"
        ai_text = "#f3f4f6"
        input_bg = "#16171d"
        input_border = "#374151"
        input_text = "#f8fafc"
        escalation_bg = "rgba(234, 88, 12, 0.15)"
        escalation_border = "#f97316"
        escalation_text = "#fdba74"
        scrollbar_track = "#18181b"
        scrollbar_thumb = "#3f3f46"
        sidebar_btn_bg = "#1f2028"
        sidebar_btn_border = "#2e303b"
        sidebar_btn_active = "#2d303e"
        sidebar_btn_active_border = "#3b82f6"
        delete_btn_bg = "#1f2028"
        delete_btn_border = "#2e303b"
        delete_btn_color = "#9ca3af"
        delete_btn_hover_bg = "rgba(239, 68, 68, 0.2)"
        delete_btn_hover_color = "#f87171"
        node_tag_bg = "rgba(59, 130, 246, 0.15)"
        node_tag_color = "#60a5fa"
        glow_top = "radial-gradient(circle, rgba(59, 130, 246, 0.14) 0%, rgba(147, 51, 234, 0.07) 45%, transparent 70%)"
        glow_bottom = "radial-gradient(circle, rgba(14, 165, 233, 0.12) 0%, rgba(99, 102, 241, 0.07) 45%, transparent 70%)"
    else:
        bg = "#f4f7fc"
        sidebar_bg = "#ffffff"
        card_bg = "#ffffff"
        card_border = "#cbd5e1"
        text = "#0f172a"
        sub_text = "#475569"
        accent = "#0b57d0"
        accent_hover = "#0842a0"
        accent_glow = "rgba(11, 87, 208, 0.25)"
        user_bubble = "linear-gradient(135deg, #0b57d0 0%, #0842a0 100%)"
        user_text = "#ffffff"
        ai_bubble = "#ffffff"
        ai_border = "#cbd5e1"
        ai_text = "#0f172a"
        input_bg = "#ffffff"
        input_border = "#94a3b8"
        input_text = "#0f172a"
        escalation_bg = "#fff7ed"
        escalation_border = "#f97316"
        escalation_text = "#9a3412"
        scrollbar_track = "#f1f5f9"
        scrollbar_thumb = "#cbd5e1"
        sidebar_btn_bg = "#f8fafc"
        sidebar_btn_border = "#e2e8f0"
        sidebar_btn_active = "#e8f0fe"
        sidebar_btn_active_border = "#0b57d0"
        delete_btn_bg = "#f8fafc"
        delete_btn_border = "#e2e8f0"
        delete_btn_color = "#475569"
        delete_btn_hover_bg = "#fee2e2"
        delete_btn_hover_color = "#dc2626"
        node_tag_bg = "#e8f0fe"
        node_tag_color = "#0b57d0"
        glow_top = "radial-gradient(circle, rgba(11, 87, 208, 0.12) 0%, rgba(124, 77, 255, 0.08) 45%, transparent 70%)"
        glow_bottom = "radial-gradient(circle, rgba(14, 165, 233, 0.10) 0%, rgba(59, 130, 246, 0.06) 45%, transparent 70%)"

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --bg: {bg};
        --sidebar-bg: {sidebar_bg};
        --card-bg: {card_bg};
        --card-border: {card_border};
        --text: {text};
        --sub-text: {sub_text};
        --accent: {accent};
        --accent-hover: {accent_hover};
        --accent-glow: {accent_glow};
        --user-bubble: {user_bubble};
        --user-text: {user_text};
        --ai-bubble: {ai_bubble};
        --ai-border: {ai_border};
        --ai-text: {ai_text};
        --input-bg: {input_bg};
        --input-border: {input_border};
        --input-text: {input_text};
        --escalation-bg: {escalation_bg};
        --escalation-border: {escalation_border};
        --escalation-text: {escalation_text};
        --scrollbar-track: {scrollbar_track};
        --scrollbar-thumb: {scrollbar_thumb};
        --sidebar-btn-bg: {sidebar_btn_bg};
        --sidebar-btn-border: {sidebar_btn_border};
        --sidebar-btn-active: {sidebar_btn_active};
        --sidebar-btn-active-border: {sidebar_btn_active_border};
        --delete-btn-bg: {delete_btn_bg};
        --delete-btn-border: {delete_btn_border};
        --delete-btn-color: {delete_btn_color};
        --delete-btn-hover-bg: {delete_btn_hover_bg};
        --delete-btn-hover-color: {delete_btn_hover_color};
        --node-tag-bg: {node_tag_bg};
        --node-tag-color: {node_tag_color};
    }}

    html, body, [class*="stApp"] {{
        background-color: var(--bg) !important;
        color: var(--text) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        position: relative;
    }}

    /* Ambient Corner Gradient Glow Effect */
    [class*="stApp"]::before {{
        content: "";
        position: fixed;
        top: -120px;
        right: -120px;
        width: 600px;
        height: 600px;
        background: {glow_top};
        pointer-events: none;
        z-index: 0;
        filter: blur(50px);
    }}
    [class*="stApp"]::after {{
        content: "";
        position: fixed;
        bottom: -120px;
        left: -120px;
        width: 600px;
        height: 600px;
        background: {glow_bottom};
        pointer-events: none;
        z-index: 0;
        filter: blur(50px);
    }}

    /* Collapse Streamlit default header/footer completely (FIXES TOP GAP) */
    #MainMenu, footer, header, [data-testid="stHeader"] {{
        display: none !important;
        height: 0 !important;
        padding: 0 !important;
        margin: 0 !important;
    }}
    .stDeployButton {{ display: none !important; }}
    .stDecoration {{ display: none !important; }}

    /* Sidebar container */
    [data-testid="stSidebar"] {{
        background: var(--sidebar-bg) !important;
        border-right: 1px solid var(--card-border) !important;
        z-index: 1;
        -ms-overflow-style: none;
        scrollbar-width: none;
    }}
    [data-testid="stSidebar"]::-webkit-scrollbar {{
        display: none !important;
    }}
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] div,
    [data-testid="stSidebar"] span {{
        color: var(--text) !important;
    }}

    /* Main content container (ELIMINATES TOP MARGIN GAP) */
    .main .block-container, [data-testid="stMainBlockContainer"] {{
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        max-width: 860px !important;
        margin: 0 auto !important;
        position: relative;
        z-index: 1;
    }}

    /* Custom Scrollbars (SINGLE CLEAN SCROLLBAR) */
    html, body {{ overflow-x: hidden !important; }}
    ::-webkit-scrollbar {{ width: 5px !important; height: 5px !important; }}
    ::-webkit-scrollbar-track {{ background: transparent !important; }}
    ::-webkit-scrollbar-thumb {{ background: var(--scrollbar-thumb) !important; border-radius: 10px !important; }}

    /* User Chat Bubble */
    .user-bubble {{
        background: var(--user-bubble);
        color: var(--user-text) !important;
        border-radius: 20px 20px 4px 20px;
        padding: 14px 20px;
        margin: 10px 0 10px auto;
        max-width: 78%;
        width: fit-content;
        float: right;
        clear: both;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 4px 14px var(--accent-glow);
        animation: slideInRight 0.22s ease;
    }}

    /* AI Chat Bubble */
    .ai-bubble {{
        background: var(--ai-bubble);
        border: 1.5px solid var(--ai-border);
        color: var(--ai-text) !important;
        border-radius: 20px 20px 20px 4px;
        padding: 16px 22px;
        margin: 10px auto 10px 0;
        max-width: 82%;
        width: fit-content;
        float: left;
        clear: both;
        font-size: 0.95rem;
        line-height: 1.65;
        box-shadow: 0 3px 12px rgba(0,0,0,0.04);
        animation: slideInLeft 0.22s ease;
    }}

    /* Escalation Alert Bubble */
    .escalation-bubble {{
        background: var(--escalation-bg);
        border: 1.5px solid var(--escalation-border);
        color: var(--escalation-text) !important;
        border-radius: 16px;
        padding: 18px 22px;
        margin: 12px auto 12px 0;
        max-width: 86%;
        float: left;
        clear: both;
        font-size: 0.93rem;
        line-height: 1.65;
        box-shadow: 0 4px 16px rgba(249,115,22,0.1);
        animation: slideInLeft 0.22s ease;
    }}

    .clearfix {{ clear: both; }}

    @keyframes slideInRight {{
        from {{ opacity: 0; transform: translateX(16px); }}
        to   {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes slideInLeft {{
        from {{ opacity: 0; transform: translateX(-16px); }}
        to   {{ opacity: 1; transform: translateX(0); }}
    }}
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; transform: scale(1); }}
        50%       {{ opacity: 0.4; transform: scale(0.85); }}
    }}

    /* Typing indicator */
    .typing-indicator {{
        display: flex;
        gap: 6px;
        align-items: center;
        padding: 14px 20px;
        background: var(--ai-bubble);
        border: 1px solid var(--ai-border);
        border-radius: 20px 20px 20px 4px;
        width: fit-content;
        margin: 10px 0;
        float: left;
        clear: both;
    }}
    .typing-dot {{
        width: 8px; height: 8px;
        border-radius: 50%;
        background: var(--accent);
        animation: pulse 1.2s infinite ease-in-out;
    }}
    .typing-dot:nth-child(2) {{ animation-delay: 0.2s; }}
    .typing-dot:nth-child(3) {{ animation-delay: 0.4s; }}

    /* CRAG Node Tags */
    .node-tag {{
        display: inline-block;
        background: var(--node-tag-bg);
        color: var(--node-tag-color) !important;
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 0.76rem;
        font-weight: 500;
        font-family: 'JetBrains Mono', monospace;
        margin: 3px 3px 3px 0;
    }}

    /* Intent Badge */
    .intent-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--card-bg);
        border: 1px solid var(--card-border);
        border-radius: 10px;
        padding: 4px 12px;
        font-size: 0.78rem;
        color: var(--sub-text) !important;
        margin: 4px 0;
    }}

    /* ── Form & Button overrides for Light / Dark mode parity ───────────────── */

    /* Input Form Text Area - HIGH CONTRAST IN LIGHT & DARK MODE */
    [data-testid="stForm"] textarea,
    .stTextArea textarea {{
        background: var(--input-bg) !important;
        border: 1.5px solid var(--input-border) !important;
        border-radius: 16px !important;
        color: var(--input-text) !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.96rem !important;
        font-weight: 500 !important;
        padding: 14px 18px !important;
        box-shadow: 0 2px 10px rgba(0,0,0,0.03) !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stForm"] textarea::placeholder,
    .stTextArea textarea::placeholder {{
        color: var(--sub-text) !important;
        opacity: 0.85 !important;
        font-weight: 400 !important;
    }}
    [data-testid="stForm"] textarea:focus,
    .stTextArea textarea:focus {{
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px var(--accent-glow) !important;
    }}

    /* Form Submit Send Button */
    [data-testid="stFormSubmitButton"] button,
    .stFormSubmitButton button {{
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 10px 18px !important;
        min-height: 44px !important;
        box-shadow: 0 4px 14px var(--accent-glow) !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stFormSubmitButton"] button:hover,
    .stFormSubmitButton button:hover {{
        opacity: 0.95 !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px var(--accent-glow) !important;
    }}
    [data-testid="stFormSubmitButton"] button *,
    .stFormSubmitButton button * {{
        color: #ffffff !important;
        font-weight: 600 !important;
    }}

    /* Sidebar Secondary Buttons (FIXES THEME TOGGLE BLACK-ON-BLACK ISSUE) */
    [data-testid="stSidebar"] button:not([kind="primary"]) {{
        background: var(--sidebar-btn-bg) !important;
        border: 1.5px solid var(--sidebar-btn-border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02) !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stSidebar"] button:not([kind="primary"]):hover {{
        background: var(--sidebar-btn-active) !important;
        border-color: var(--sidebar-btn-active-border) !important;
    }}
    [data-testid="stSidebar"] button:not([kind="primary"]) p,
    [data-testid="stSidebar"] button:not([kind="primary"]) span,
    [data-testid="stSidebar"] button:not([kind="primary"]) * {{
        color: var(--text) !important;
        font-weight: 600 !important;
    }}

    /* Sidebar Chat Threads (Column 1) */
    [data-testid="stSidebar"] [data-testid="column"]:nth-child(1) button {{
        background: var(--sidebar-btn-bg) !important;
        border: 1.5px solid var(--sidebar-btn-border) !important;
        color: var(--text) !important;
        border-radius: 10px !important;
        text-align: left !important;
        justify-content: flex-start !important;
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        min-height: 38px !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stSidebar"] [data-testid="column"]:nth-child(1) button:hover {{
        background: var(--sidebar-btn-active) !important;
        border-color: var(--sidebar-btn-active-border) !important;
    }}
    [data-testid="stSidebar"] [data-testid="column"]:nth-child(1) button p {{
        color: var(--text) !important;
        font-weight: 500 !important;
    }}

    /* Sidebar Delete Buttons (Column 2 - FIXES TRASH ICON BLACK BLOCK ISSUE) */
    [data-testid="stSidebar"] [data-testid="column"]:nth-child(2) button {{
        background: var(--delete-btn-bg) !important;
        border: 1.5px solid var(--delete-btn-border) !important;
        color: var(--delete-btn-color) !important;
        border-radius: 10px !important;
        padding: 4px 6px !important;
        min-height: 38px !important;
        width: 100% !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stSidebar"] [data-testid="column"]:nth-child(2) button:hover {{
        background: var(--delete-btn-hover-bg) !important;
        color: var(--delete-btn-hover-color) !important;
        border-color: var(--delete-btn-hover-color) !important;
    }}
    [data-testid="stSidebar"] [data-testid="column"]:nth-child(2) button p,
    [data-testid="stSidebar"] [data-testid="column"]:nth-child(2) button span,
    [data-testid="stSidebar"] [data-testid="column"]:nth-child(2) button * {{
        color: inherit !important;
    }}

    /* Sidebar New Chat Button */
    [data-testid="stSidebar"] button[kind="primary"] {{
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%) !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 10px 16px !important;
        box-shadow: 0 4px 14px var(--accent-glow) !important;
        transition: all 0.2s ease !important;
    }}
    [data-testid="stSidebar"] button[kind="primary"] * {{
        color: #ffffff !important;
        font-weight: 600 !important;
    }}

    /* Sidebar Expander Box */
    [data-testid="stExpander"] {{
        background: var(--card-bg) !important;
        border: 1.5px solid var(--card-border) !important;
        border-radius: 12px !important;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02) !important;
    }}
    [data-testid="stExpander"] * {{
        color: var(--text) !important;
    }}
    [data-testid="stMetricValue"] {{
        color: var(--accent) !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: var(--sub-text) !important;
    }}

    /* Agent header */
    .agent-header {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 16px 0 14px 0;
        border-bottom: 1.5px solid var(--card-border);
        margin-bottom: 20px;
    }}
    .agent-logo {{
        width: 42px; height: 42px;
        background: linear-gradient(135deg,#ff9900 0%,#ff6600 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        box-shadow: 0 4px 12px rgba(255,153,0,0.25);
    }}
    .agent-name {{
        font-size: 1.2rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.01em;
    }}
    .agent-status {{
        font-size: 0.78rem;
        color: var(--sub-text);
    }}

    /* Separator */
    hr {{ border-color: var(--card-border) !important; opacity: 0.8; }}

    /* Welcome screen */
    .welcome-container {{
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 55vh;
        text-align: center;
        gap: 18px;
    }}
    .welcome-icon {{
        font-size: 3.5rem;
        animation: pulse 2.4s infinite ease-in-out;
    }}
    .welcome-title {{
        font-size: 2.2rem;
        font-weight: 700;
        color: var(--text);
        letter-spacing: -0.02em;
    }}
    .welcome-subtitle {{
        font-size: 1.02rem;
        color: var(--sub-text);
        max-width: 480px;
        line-height: 1.65;
    }}
    </style>
    """, unsafe_allow_html=True)


# ── Session state initialization ──────────────────────────────────────────────
def init_session():
    if "current_thread_id" not in st.session_state:
        st.session_state.current_thread_id = None
    if "chat_display" not in st.session_state:
        st.session_state.chat_display = []  # list of {role, content, escalated, intent, node_path}
    if "processing" not in st.session_state:
        st.session_state.processing = False
    if "threads_cache" not in st.session_state:
        st.session_state.threads_cache = []


def load_threads():
    try:
        from src.rag.memory import list_threads
        return list_threads()
    except Exception:
        return []


def start_new_chat():
    from src.rag.memory import create_thread
    thread_id = str(uuid.uuid4())
    create_thread(thread_id)
    st.session_state.current_thread_id = thread_id
    st.session_state.chat_display = []


def load_chat(thread_id: str):
    from src.rag.memory import get_thread
    thread = get_thread(thread_id)
    if not thread:
        return
    st.session_state.current_thread_id = thread_id
    # Rebuild display from messages
    display = []
    messages = thread.get("messages", [])
    for msg in messages:
        role = msg.get("role", "human")
        content = msg.get("content", "")
        display.append({
            "role": role,
            "content": content,
            "escalated": "🚨" in content,
            "intent": None,
            "node_path": [],
        })
    st.session_state.chat_display = display


def delete_chat(thread_id: str):
    from src.rag.memory import delete_thread
    delete_thread(thread_id)
    if st.session_state.current_thread_id == thread_id:
        st.session_state.current_thread_id = None
        st.session_state.chat_display = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;padding:8px 0 16px 0;">
          <div style="width:36px;height:36px;background:linear-gradient(135deg,#ff9900,#ff6600);border-radius:10px;display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 2px 8px rgba(255,153,0,0.3);">🛒</div>
          <div>
            <div style="font-weight:700;font-size:1.05rem;line-height:1.2;">AmazonHelp</div>
            <div style="font-size:0.75rem;opacity:0.75;">AI Support Agent</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("✨ New Chat", use_container_width=True, type="primary"):
            start_new_chat()
            st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:0.75rem;margin-bottom:8px;font-weight:700;letter-spacing:0.06em;'>CONVERSATIONS</div>", unsafe_allow_html=True)

        threads = load_threads()
        st.session_state.threads_cache = threads

        if not threads:
            st.markdown("<div style='font-size:0.84rem;opacity:0.6;padding:8px 0;'>No conversations yet</div>", unsafe_allow_html=True)
        else:
            for thread in threads:
                tid = thread["thread_id"]
                title = thread.get("title", "New Chat")[:30]
                is_active = tid == st.session_state.current_thread_id

                col1, col2 = st.columns([5, 1])
                with col1:
                    btn_style = "primary" if is_active else "secondary"
                    if st.button(
                        f"{'● ' if is_active else ''}{title}",
                        key=f"chat_btn_{tid}",
                        use_container_width=True,
                        type=btn_style,
                    ):
                        load_chat(tid)
                        st.rerun()
                with col2:
                    if st.button("🗑", key=f"del_{tid}", help="Delete chat"):
                        delete_chat(tid)
                        st.rerun()

        st.markdown("<hr>", unsafe_allow_html=True)
        with st.expander("📊 Feedback & Analytics", expanded=False):
            from src.rag.memory import get_feedback_summary
            fb = get_feedback_summary()
            col_f1, col_f2 = st.columns(2)
            col_f1.metric("👍 Helpful", fb["positive"])
            col_f2.metric("👎 Unhelpful", fb["negative"])
            if fb["total"] > 0:
                ratio = int((fb["positive"] / fb["total"]) * 100)
                st.caption(f"Satisfaction Rate: {ratio}% ({fb['total']} total ratings)")
            else:
                st.caption("No user ratings collected yet")

        # Theme toggle
        dm_label = "☀️ Light Mode" if st.session_state.dark_mode else "🌙 Dark Mode"
        if st.button(dm_label, use_container_width=True, key="theme_toggle_btn"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            st.rerun()

        st.markdown("""
        <div style="margin-top:16px;margin-bottom:12px;text-align:center;font-size:0.72rem;color:var(--sub-text);opacity:0.75;">
          Powered by LangGraph + Qdrant<br>BAAI/bge · Qwen3 · DeepSeek
        </div>
        """, unsafe_allow_html=True)


# ── Chat message renderers ────────────────────────────────────────────────────
def render_message(msg_idx: int, msg: dict):
    role = msg.get("role", "human")
    content = msg.get("content", "")
    escalated = msg.get("escalated", False)
    intent = msg.get("intent")
    node_path = msg.get("node_path", [])

    if role == "human":
        st.markdown(f'<div class="user-bubble">{content}</div><div class="clearfix"></div>', unsafe_allow_html=True)
    else:
        if escalated:
            st.markdown(
                f'<div class="escalation-bubble">{content}</div><div class="clearfix"></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="ai-bubble">{content}</div><div class="clearfix"></div>',
                unsafe_allow_html=True,
            )

        # Show intent + node path
        meta_parts = []
        if intent:
            meta_parts.append(f'<span class="intent-badge">🎯 {intent}</span>')
        if node_path:
            tags = "".join(f'<span class="node-tag">{n}</span>' for n in node_path)
            meta_parts.append(tags)

        if meta_parts:
            st.markdown(
                f'<div style="float:left;margin-bottom:6px;">{"".join(meta_parts)}</div><div class="clearfix"></div>',
                unsafe_allow_html=True,
            )

        # Feedback buttons for AI message
        col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 10])
        with col_fb1:
            if st.button("👍", key=f"fb_pos_{msg_idx}", help="Helpful"):
                from src.rag.memory import save_feedback
                save_feedback(st.session_state.current_thread_id, msg_idx, "positive")
                st.toast("Thank you for your feedback! 👍")
        with col_fb2:
            if st.button("👎", key=f"fb_neg_{msg_idx}", help="Not helpful"):
                from src.rag.memory import save_feedback
                save_feedback(st.session_state.current_thread_id, msg_idx, "negative")
                st.toast("Feedback recorded! 👎 We'll use this to improve.")


def render_typing_indicator():
    return st.markdown(
        '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div><div class="clearfix"></div>',
        unsafe_allow_html=True,
    )


# ── Main chat area ────────────────────────────────────────────────────────────
def render_chat_area():
    # Header
    st.markdown("""
    <div class="agent-header">
      <div class="agent-logo">🤖</div>
      <div>
        <div class="agent-name">AmazonHelp AI Agent</div>
        <div class="agent-status">🟢 Online · Powered by RAG + CRAG Pipeline</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Welcome screen
    if not st.session_state.current_thread_id:
        st.markdown("""
        <div class="welcome-container">
          <div class="welcome-icon">🛒</div>
          <div class="welcome-title">How can I help you today?</div>
          <div class="welcome-subtitle">
            I'm your Amazon AI support assistant. I can help with orders, deliveries,
            returns, account issues, and more. Start a new chat to get help!
          </div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        suggestions = [
            ("📦", "Where is my package?", "Track a lost or delayed order"),
            ("🔄", "Return an item", "Start a return or get a refund"),
            ("🔐", "Account security", "Report unauthorized account access"),
        ]
        for col, (icon, title, desc) in zip([col1, col2, col3], suggestions):
            with col:
                if st.button(f"{icon} {title}\n{desc}", use_container_width=True, key=f"sug_{title}"):
                    start_new_chat()
                    handle_query(title)
                    st.rerun()
        return

    # Render message history
    chat_container = st.container()
    with chat_container:
        for idx, msg in enumerate(st.session_state.chat_display):
            render_message(idx, msg)


def handle_query(user_input: str):
    """Process user query through the CRAG pipeline with streaming status."""
    from src.rag.graph import stream_graph
    from src.rag.memory import get_thread

    thread_id = st.session_state.current_thread_id
    if not thread_id:
        start_new_chat()
        thread_id = st.session_state.current_thread_id

    # Add user message to display immediately
    st.session_state.chat_display.append({
        "role": "human",
        "content": user_input,
        "escalated": False,
        "intent": None,
        "node_path": [],
    })

    # Load thread state
    thread_data = get_thread(thread_id) or {}
    messages = thread_data.get("messages", [])
    summary_list = thread_data.get("summary_list", [])
    iteration_count = thread_data.get("iteration_count", 0)

    # Stream through graph
    response_text = ""
    intent_detected = ""
    nodes_visited = []
    escalated = False

    status_placeholder = st.empty()
    typing_placeholder = st.empty()

    typing_placeholder.markdown(
        '<div class="typing-indicator"><div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div></div><div class="clearfix"></div>',
        unsafe_allow_html=True,
    )

    try:
        for node_name, state_update in stream_graph(
            query=user_input,
            thread_id=thread_id,
            messages=messages,
            summary_list=summary_list,
            iteration_count=iteration_count,
        ):
            nodes_visited.append(node_name)

            # Update status tag
            node_labels = {
                "get_intent": "🔍 Classifying intent",
                "intent_evaluator": "⚖️ Evaluating severity",
                "rewrite_query": "✏️ Enhancing query",
                "retriever": "📚 Searching knowledge base",
                "eval_retriever": "🎯 Evaluating results",
                "correct": "✅ Refining context",
                "ambiguous": "🔄 Refining search",
                "ambiguous_retriever": "📚 Searching again",
                "recompose_ambiguous_strips": "🔧 Recomposing context",
                "brain_node": "🧠 Generating answer",
                "incorrect": "⚠️ Insufficient data",
                "human_node": "🚨 Escalating to human",
            }
            label = node_labels.get(node_name, f"⚙️ {node_name}")
            status_placeholder.markdown(
                f'<div class="intent-badge" style="float:left;margin:4px 0;">{label}</div><div class="clearfix"></div>',
                unsafe_allow_html=True,
            )

            # Capture key state
            if "intent" in state_update and state_update["intent"]:
                intent_detected = state_update["intent"]
            if "response" in state_update and state_update["response"]:
                response_text = state_update["response"]
            if "human_escalated" in state_update:
                escalated = state_update["human_escalated"]

    except Exception as e:
        response_text = (
            f"I encountered a technical error: {str(e)[:200]}\n\n"
            "Please try again or contact Amazon support directly at amazon.com/help."
        )

    # Clear typing and status
    typing_placeholder.empty()
    status_placeholder.empty()

    if not response_text:
        response_text = "I'm sorry, I wasn't able to generate a response. Please try again."

    # Progressive token-level streaming rendering
    def token_stream_gen(text: str):
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.012)

    with st.container():
        st.write_stream(token_stream_gen(response_text))

    # Add AI response to display
    st.session_state.chat_display.append({
        "role": "ai",
        "content": response_text,
        "escalated": escalated,
        "intent": intent_detected,
        "node_path": nodes_visited,
    })


# ── Input area ────────────────────────────────────────────────────────────────
def render_input_area():
    if not st.session_state.current_thread_id:
        return

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    with st.form("chat_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 1])
        with col1:
            user_input = st.text_area(
                "Message",
                placeholder="Describe your Amazon issue… (Shift+Enter for new line)",
                label_visibility="collapsed",
                height=80,
            )
        with col2:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Send ➤", use_container_width=True)

        if submitted and user_input.strip():
            handle_query(user_input.strip())
            st.rerun()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    init_session()
    inject_css(st.session_state.dark_mode)
    render_sidebar()

    # Main layout
    render_chat_area()
    render_input_area()


if __name__ == "__main__":
    main()
