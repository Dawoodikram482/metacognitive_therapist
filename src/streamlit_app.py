"""
Advanced Streamlit Frontend for Metacognitive Therapy Chatbot

This module creates a professional, user-friendly web interface with
advanced features like progress tracking, session management, and analytics.
"""

import streamlit as st
import requests
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import uuid
import time

# Page configuration
st.set_page_config(
    page_title="Metacognitive Therapy Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",  # Hide sidebar by default
    menu_items={
        'Get Help': 'https://github.com/Dawoodikram482/metacognitive_therapist',
        'Report a bug': 'https://github.com/Dawoodikram482/metacognitive_therapist/issues',
        'About': "Advanced AI-powered metacognitive therapy assistant using RAG and local LLMs"
    }
)# Custom CSS for professional styling with dark mode support
st.markdown("""
<style>
    /* Light mode styles (default) */
    .main-header {
        font-size: 2.5rem;
        color: #2E86AB;
        font-weight: bold;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #2E86AB;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
    }
    
    .user-message {
        background-color: #E3F2FD;
        border-left-color: #1976D2;
        color: #1A1A1A;
    }
    
    .assistant-message {
        background-color: #F3E5F5;
        border-left-color: #7B1FA2;
        color: #1A1A1A;
    }
    
    .info-box {
        background-color: #E8F5E8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        color: #1A1A1A;
    }
    
    .warning-box {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF9800;
        margin: 1rem 0;
        color: #1A1A1A;
    }
    
    .metric-card {
        background-color: #FAFAFA;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        color: #1A1A1A;
    }
    
    .sidebar-section {
        background-color: #F8F9FA;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #1A1A1A;
    }
    
    .footer {
        text-align: center;
        color: #666;
        font-size: 0.9rem;
        padding: 2rem 0;
        border-top: 1px solid #ddd;
        margin-top: 3rem;
    }

    /* Dark mode styles - activated when Streamlit dark theme is enabled */
    [data-theme="dark"] .main-header,
    .stApp[data-theme="dark"] .main-header {
        color: #64B5F6;
        border-bottom-color: #64B5F6;
    }
    
    [data-theme="dark"] .user-message,
    .stApp[data-theme="dark"] .user-message {
        background-color: #1E3A5F;
        border-left-color: #42A5F5;
        color: #E3F2FD;
    }
    
    [data-theme="dark"] .assistant-message,
    .stApp[data-theme="dark"] .assistant-message {
        background-color: #2D1B3D;
        border-left-color: #AB47BC;
        color: #F3E5F5;
    }
    
    [data-theme="dark"] .info-box,
    .stApp[data-theme="dark"] .info-box {
        background-color: #1B3B1B;
        border-left-color: #66BB6A;
        color: #E8F5E8;
    }
    
    [data-theme="dark"] .warning-box,
    .stApp[data-theme="dark"] .warning-box {
        background-color: #3D2F1A;
        border-left-color: #FFB74D;
        color: #FFF3E0;
    }
    
    [data-theme="dark"] .metric-card,
    .stApp[data-theme="dark"] .metric-card {
        background-color: #262730;
        box-shadow: 0 2px 4px rgba(255,255,255,0.1);
        color: #FAFAFA;
    }
    
    [data-theme="dark"] .sidebar-section,
    .stApp[data-theme="dark"] .sidebar-section {
        background-color: #262730;
        color: #F8F9FA;
    }
    
    [data-theme="dark"] .footer,
    .stApp[data-theme="dark"] .footer {
        color: #999;
        border-top-color: #333;
    }

    /* Enhanced contrast for better readability in dark mode */
    [data-theme="dark"] .stMarkdown,
    .stApp[data-theme="dark"] .stMarkdown {
        color: #FFFFFF !important;
    }
    
    [data-theme="dark"] .stText,
    .stApp[data-theme="dark"] .stText {
        color: #FFFFFF !important;
    }
    
    /* Ensure chat messages are clearly visible */
    [data-theme="dark"] .chat-message p,
    .stApp[data-theme="dark"] .chat-message p {
        color: inherit !important;
    }
    
    /* Style for code blocks in dark mode */
    [data-theme="dark"] pre,
    .stApp[data-theme="dark"] pre {
        background-color: #1E1E1E !important;
        color: #FFFFFF !important;
        border: 1px solid #404040;
    }
    
    [data-theme="dark"] code,
    .stApp[data-theme="dark"] code {
        background-color: #2D2D2D !important;
        color: #FFB74D !important;
        padding: 2px 4px;
        border-radius: 3px;
    }
    
    /* Additional dark mode selectors for better compatibility */
    @media (prefers-color-scheme: dark) {
        .main-header {
            color: #64B5F6 !important;
            border-bottom-color: #64B5F6 !important;
        }
        
        .user-message {
            background-color: #1E3A5F !important;
            border-left-color: #42A5F5 !important;
            color: #E3F2FD !important;
        }
        
        .assistant-message {
            background-color: #2D1B3D !important;
            border-left-color: #AB47BC !important;
            color: #F3E5F5 !important;
        }
        
        .info-box {
            background-color: #1B3B1B !important;
            border-left-color: #66BB6A !important;
            color: #E8F5E8 !important;
        }
        
        .warning-box {
            background-color: #3D2F1A !important;
            border-left-color: #FFB74D !important;
            color: #FFF3E0 !important;
        }
        
        .metric-card {
            background-color: #262730 !important;
            box-shadow: 0 2px 4px rgba(255,255,255,0.1) !important;
            color: #FAFAFA !important;
        }
        
        .sidebar-section {
            background-color: #262730 !important;
            color: #F8F9FA !important;
        }
        
        .footer {
            color: #999 !important;
            border-top-color: #333 !important;
        }
    }
</style>

<script>
// Enhanced dark mode detection and application
function applyDarkModeStyles() {
    const isDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    const streamlitTheme = document.querySelector('[data-theme]');
    
    if (isDark || (streamlitTheme && streamlitTheme.getAttribute('data-theme') === 'dark')) {
        document.body.style.setProperty('--text-color', '#FFFFFF', 'important');
        document.body.style.setProperty('--bg-color', '#0E1117', 'important');
    }
}

// Apply styles when page loads and when theme changes
document.addEventListener('DOMContentLoaded', applyDarkModeStyles);
window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', applyDarkModeStyles);

// Streamlit-specific theme detection
setInterval(() => {
    const streamlitContainer = document.querySelector('.stApp');
    if (streamlitContainer) {
        const computedStyle = window.getComputedStyle(streamlitContainer);
        const bgColor = computedStyle.backgroundColor;
        
        // If background is dark, ensure text is light
        if (bgColor === 'rgb(14, 17, 23)' || bgColor === 'rgb(38, 39, 48)') {
            document.body.classList.add('streamlit-dark-mode');
        }
    }
}, 1000);
</script>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"
MAX_MESSAGE_LENGTH = 2000

# Initialize session state
def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'user_id' not in st.session_state:
        st.session_state.user_id = str(uuid.uuid4())
    
    if 'conversation_started' not in st.session_state:
        st.session_state.conversation_started = False
    
    if 'last_activity' not in st.session_state:
        st.session_state.last_activity = datetime.now()
    
    if 'session_stats' not in st.session_state:
        st.session_state.session_stats = {
            'messages_sent': 0,
            'session_duration': 0,
            'topics_discussed': set()
        }

# API Functions
@st.cache_data(ttl=10)  # Cache for 10 seconds to avoid repeated calls
def check_api_health() -> Dict[str, Any]:
    """Check if API is healthy and responsive"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=8)  # Longer timeout for initial connection
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "unhealthy", "error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"status": "unreachable", "error": str(e)}

def send_chat_message(message: str, session_id: Optional[str] = None) -> Dict[str, Any]:
    """Send chat message to API"""
    try:
        payload = {
            "message": message,
            "session_id": session_id,
            "user_id": st.session_state.user_id,
            "include_sources": True,
            "max_response_length": 500
        }
        
        response = requests.post(
            f"{API_BASE_URL}/chat", 
            json=payload,
            timeout=600  # 10 minutes - let the model run as long as needed
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API Error: HTTP {response.status_code}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": f"Connection Error: {str(e)}"}

def get_session_info(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session information from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/session/{session_id}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def get_system_stats() -> Optional[Dict[str, Any]]:
    """Get system statistics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/admin/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

# UI Components
def render_header():
    """Render application header"""
    st.markdown('<div class="main-header">🧠 Metacognitive Therapy Assistant</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <strong>Welcome to your AI-powered metacognitive therapy companion.</strong><br>
        This assistant uses advanced AI and evidence-based therapy techniques to provide supportive guidance. 
        Please remember that this is not a replacement for professional mental health care.
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with controls and information"""
    with st.sidebar:
        st.header("🎛️ Session Controls")
        
        # API Health Status
        health_status = check_api_health()
        if health_status.get("status") == "healthy":
            st.success("✅ API Connected")
        else:
            st.error(f"❌ API Issue: {health_status.get('error', 'Unknown')}")
        
        st.markdown("---")
        
        # Session Information
        st.subheader("📊 Session Info")
        if st.session_state.session_id:
            session_info = get_session_info(st.session_state.session_id)
            if session_info:
                st.info(f"**Session ID:** {st.session_state.session_id[:8]}...")
                st.info(f"**Messages:** {session_info.get('message_count', 0)}")
                
                created_at = datetime.fromisoformat(session_info.get('created_at', '').replace('Z', '+00:00'))
                duration = datetime.now() - created_at.replace(tzinfo=None)
                st.info(f"**Duration:** {str(duration).split('.')[0]}")
        else:
            st.info("No active session")
        
        # Session Controls
        st.markdown("---")
        if st.button("🔄 New Session", help="Start a fresh conversation"):
            st.session_state.session_id = None
            st.session_state.chat_history = []
            st.session_state.conversation_started = False
            st.session_state.session_stats = {
                'messages_sent': 0,
                'session_duration': 0,
                'topics_discussed': set()
            }
            st.rerun()
        
        if st.button("📤 Export Chat", help="Download conversation history"):
            if st.session_state.chat_history:
                export_data = {
                    "session_id": st.session_state.session_id,
                    "export_date": datetime.now().isoformat(),
                    "messages": st.session_state.chat_history
                }
                st.download_button(
                    label="💾 Download JSON",
                    data=json.dumps(export_data, indent=2),
                    file_name=f"therapy_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        # Quick Actions
        st.markdown("---")
        st.subheader("⚡ Quick Actions")
        
        quick_prompts = [
            "I'm feeling anxious about work",
            "Can you teach me a relaxation technique?",
            "How can I stop worrying so much?",
            "What is metacognitive therapy?",
            "I'm having trouble concentrating"
        ]
        
        for prompt in quick_prompts:
            if st.button(f"💭 {prompt}", key=f"quick_{prompt}"):
                st.session_state.quick_prompt = prompt
                st.rerun()
        
        # Crisis Resources
        st.markdown("---")
        st.subheader("🆘 Crisis Resources")
        st.markdown("""
        <div class="warning-box">
            <strong>If you're in crisis:</strong><br>
            • Netherlands: 113 (Suicide Prevention) | 0800-0113 (24/7)<br>
            • Netherlands: 0900-0767 (Mental Health Crisis Line)<br>
            • EU Emergency: 112<br>
            • International: 116 123 (Befrienders Worldwide)<br>
            <br>
            <em>This AI assistant cannot replace professional help in emergencies.</em>
        </div>
        """, unsafe_allow_html=True)

def render_chat_interface():
    """Render main chat interface"""
    st.header("💬 Therapy Conversation")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            
            elif message["role"] == "assistant":
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>AI Therapist:</strong><br>
                    {message["content"]}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available
                if message.get("sources") and st.checkbox(f"Show sources for message {i+1}", key=f"sources_{i}"):
                    with st.expander("📚 Sources Used"):
                        for j, source in enumerate(message["sources"]):
                            st.markdown(f"""
                            **Source {j+1}:** {source.get('collection', 'Unknown')}  
                            **Relevance:** {source.get('relevance_score', 0):.2f}  
                            **Preview:** {source.get('content_preview', 'No preview available')}
                            """)
                
                # Show suggestions if available
                if message.get("suggestions"):
                    st.markdown("**💡 Suggested follow-ups:**")
                    cols = st.columns(len(message["suggestions"]))
                    for idx, suggestion in enumerate(message["suggestions"]):
                        with cols[idx]:
                            if st.button(f"💭 {suggestion}", key=f"suggestion_{i}_{idx}"):
                                st.session_state.suggestion_prompt = suggestion
                                st.rerun()

def render_message_input():
    """Render message input area"""
    st.markdown("---")
    
    # Handle quick prompts and suggestions
    initial_message = ""
    if hasattr(st.session_state, 'quick_prompt'):
        initial_message = st.session_state.quick_prompt
        del st.session_state.quick_prompt
    elif hasattr(st.session_state, 'suggestion_prompt'):
        initial_message = st.session_state.suggestion_prompt
        del st.session_state.suggestion_prompt
    
    # Message input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_message = st.text_area(
            "💭 Share your thoughts or ask a question:",
            value=initial_message,
            height=100,
            max_chars=MAX_MESSAGE_LENGTH,
            placeholder="I've been feeling anxious lately and would like some guidance...",
            help="Describe what you're experiencing or ask about therapy techniques"
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        send_button = st.button("🚀 Send", type="primary", use_container_width=True)
        
        char_count = len(user_message) if user_message else 0
        st.caption(f"{char_count}/{MAX_MESSAGE_LENGTH} characters")
    
    # Handle message sending
    if send_button and user_message.strip():
        if char_count > MAX_MESSAGE_LENGTH:
            st.error(f"Message too long! Please keep it under {MAX_MESSAGE_LENGTH} characters.")
            return
        
        # Add user message to history
        user_msg = {
            "role": "user",
            "content": user_message.strip(),
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.chat_history.append(user_msg)
        
        # Show loading spinner
        with st.spinner("🤔 AI Therapist is thinking..."):
            # Send to API
            response = send_chat_message(user_message.strip(), st.session_state.session_id)
            
            if "error" in response:
                # Add error message to history without displaying system error
                error_msg = {
                    "role": "assistant", 
                    "content": "I apologize, but I'm experiencing technical difficulties. Please try again or consider speaking with a mental health professional if you need immediate support.",
                    "timestamp": datetime.now().isoformat(),
                    "error": True
                }
                st.session_state.chat_history.append(error_msg)
            else:
                # Update session ID if new
                if not st.session_state.session_id:
                    st.session_state.session_id = response.get("session_id")
                    st.session_state.conversation_started = True
                
                # Add assistant response to history
                assistant_msg = {
                    "role": "assistant",
                    "content": response.get("response", "No response received"),
                    "timestamp": response.get("timestamp", datetime.now().isoformat()),
                    "sources": response.get("sources"),
                    "suggestions": response.get("suggestions"),
                    "metadata": response.get("metadata")
                }
                st.session_state.chat_history.append(assistant_msg)
                
                # Update session stats
                st.session_state.session_stats['messages_sent'] += 1
                st.session_state.last_activity = datetime.now()
        
        st.rerun()

def render_analytics_dashboard():
    """Render analytics and progress dashboard"""
    st.header("📈 Progress & Analytics")
    
    # Get system stats
    system_stats = get_system_stats()
    
    if system_stats:
        # System metrics
        st.subheader("🔧 System Status")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Sessions", system_stats.get("active_sessions", 0))
        
        with col2:
            st.metric("Total Messages", system_stats.get("total_messages", 0))
        
        with col3:
            model_perf = system_stats.get("model_performance", {})
            success_rate = model_perf.get("generation_stats", {}).get("successful_generations", 0)
            total_gen = model_perf.get("generation_stats", {}).get("total_generations", 1)
            success_pct = (success_rate / total_gen * 100) if total_gen > 0 else 0
            st.metric("Model Success Rate", f"{success_pct:.1f}%")
        
        with col4:
            uptime = system_stats.get("uptime", "0:00:00")
            st.metric("System Uptime", uptime.split('.')[0])
        
        # Database statistics
        st.subheader("🗄️ Knowledge Base")
        db_stats = system_stats.get("database_stats", {})
        
        if db_stats:
            cols = st.columns(len(db_stats))
            for i, (collection, stats) in enumerate(db_stats.items()):
                with cols[i % len(cols)]:
                    st.metric(
                        collection.replace("_", " ").title(),
                        stats.get("document_count", 0)
                    )
    
    # Session-specific analytics
    if st.session_state.chat_history:
        st.subheader("💬 Your Session")
        
        # Message timeline
        messages_df = pd.DataFrame([
            {
                "time": datetime.fromisoformat(msg["timestamp"].replace('Z', '+00:00')).replace(tzinfo=None),
                "role": msg["role"],
                "length": len(msg["content"])
            }
            for msg in st.session_state.chat_history
        ])
        
        if not messages_df.empty:
            fig = px.line(
                messages_df, 
                x="time", 
                y="length",
                color="role",
                title="Message Length Over Time",
                labels={"length": "Message Length (chars)", "time": "Time"}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Conversation insights
            total_user_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])
            total_assistant_messages = len([m for m in st.session_state.chat_history if m["role"] == "assistant"])
            avg_user_length = messages_df[messages_df["role"] == "user"]["length"].mean() if total_user_messages > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Your Messages", total_user_messages)
            with col2:
                st.metric("AI Responses", total_assistant_messages)
            with col3:
                st.metric("Avg Message Length", f"{avg_user_length:.0f} chars")

def render_resources_page():
    """Render therapy resources and information page"""
    st.header("📚 Therapy Resources")
    
    # Metacognitive Therapy Information
    st.subheader("🧠 About Metacognitive Therapy")
    st.markdown("""
    <div class="info-box">
        <strong>Metacognitive Therapy (MCT)</strong> is a form of psychotherapy developed by Adrian Wells. 
        It focuses on changing the way people think about their thoughts, particularly worry and rumination.
        
        <br><br><strong>Key Concepts:</strong>
        <ul>
            <li><strong>Meta-worry:</strong> Worrying about worrying</li>
            <li><strong>Cognitive Attentional Syndrome (CAS):</strong> Patterns of worry, rumination, and threat monitoring</li>
            <li><strong>Detached Mindfulness:</strong> Observing thoughts without engaging with them</li>
            <li><strong>Attention Training:</strong> Developing flexible attention control</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Techniques
    st.subheader("🎯 Therapy Techniques")
    
    techniques = [
        {
            "name": "Attention Training Technique (ATT)",
            "description": "A 12-minute audio exercise that helps develop flexible attention control and reduces self-focused attention.",
            "when_to_use": "Daily practice, especially when feeling overwhelmed or anxious"
        },
        {
            "name": "Detached Mindfulness",
            "description": "Learning to observe thoughts and feelings without getting caught up in them or trying to control them.",
            "when_to_use": "When experiencing intrusive thoughts or strong emotions"
        },
        {
            "name": "Worry Postponement",
            "description": "Setting aside a specific 'worry time' rather than worrying throughout the day.",
            "when_to_use": "For managing excessive worry and rumination"
        },
        {
            "name": "Situational Attentional Refocusing (SAR)",
            "description": "Redirecting attention away from internal thoughts to external environment in anxiety-provoking situations.",
            "when_to_use": "During anxiety-provoking situations or panic attacks"
        }
    ]
    
    for technique in techniques:
        with st.expander(f"🔧 {technique['name']}"):
            st.markdown(f"**Description:** {technique['description']}")
            st.markdown(f"**When to use:** {technique['when_to_use']}")
            
            if st.button(f"Ask about {technique['name']}", key=f"ask_{technique['name']}"):
                st.session_state.quick_prompt = f"Can you tell me more about {technique['name']} and how to practice it?"
                st.switch_page("Home")
    
    # Self-Help Resources
    st.subheader("🔗 Additional Resources")
    st.markdown("""
    **Professional Help:**
    - Find a therapist in Netherlands: [NIP Psychologen Register](https://www.nip.nl)
    - Find a therapist in Europe: [EuroPsy Directory](https://www.europsy-efpa.eu)
    - MCT-trained therapists: [MCT Institute](https://www.mct-institute.co.uk)
    
    **Crisis Support:**
    - Netherlands Suicide Prevention: 113 or 0800-0113 (24/7 free)
    - Mental Health Crisis Line NL: 0900-0767 
    - European Emergency: 112
    - Befrienders Worldwide: 116 123 (International)
    - International Association for Suicide Prevention: [IASP](https://www.iasp.info/resources/Crisis_Centres/)
    
    **Educational:**
    - Books on Metacognitive Therapy
    - Research articles and studies
    - Online courses and workshops
    """)

def render_footer():
    """Render application footer"""
    st.markdown("""
    <div class="footer">
        <p><strong>Metacognitive Therapy Assistant</strong> | 
        Built with ❤️</p>
        <p><em>This AI assistant is for educational and supportive purposes only. 
        It does not replace professional mental health care.</em></p>
        <p>© 2024 | <a href="https://github.com/Dawoodikram482/metacognitive_therapist" target="_blank">View Source Code</a></p>
    </div>
    """, unsafe_allow_html=True)

# Main Application
def main():
    """Main application function"""
    initialize_session_state()
    
    # Navigation
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📈 Analytics", "📚 Resources"])
    
    with tab1:
        render_header()
        render_chat_interface()
        render_message_input()
    
    with tab2:
        render_analytics_dashboard()
    
    with tab3:
        render_resources_page()
    
    render_footer()

if __name__ == "__main__":
    main()