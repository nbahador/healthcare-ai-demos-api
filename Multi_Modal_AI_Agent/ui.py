import streamlit as st
import requests
import json
from PIL import Image
import io
import os
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="Multi-Modal AI Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .report-container {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'reports' not in st.session_state:
    st.session_state.reports = []

def main():
    # Header
    st.markdown('<h1 class="main-header">🤖 Multi-Modal AI Agent</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        report_type = st.selectbox(
            "Report Type",
            ["comprehensive", "technical", "executive", "research"],
            index=0
        )
        
        st.header("📊 Statistics")
        st.metric("Total Reports", len(st.session_state.reports))
        
        if st.button("🗑️ Clear History"):
            st.session_state.reports = []
            st.rerun()
    
    # Main interface
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input")
        
        # Text input
        prompt = st.text_area(
            "Enter your prompt or question:",
            height=150,
            placeholder="Describe what you want to analyze or ask about..."
        )
        
        # Image upload
        uploaded_images = st.file_uploader(
            "Upload images (optional):",
            type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
            accept_multiple_files=True
        )
        
        # Display uploaded images
        if uploaded_images:
            st.subheader("📷 Uploaded Images")
            cols = st.columns(min(len(uploaded_images), 3))
            for idx, img_file in enumerate(uploaded_images):
                with cols[idx % 3]:
                    image = Image.open(img_file)
                    st.image(image, caption=img_file.name, use_column_width=True)
        
        # Submit button
        if st.button("🚀 Generate Report", type="primary", disabled=not prompt):
            with st.spinner("Generating report..."):
                try:
                    # Prepare request
                    if os.getenv("MODAL_ENVIRONMENT"):
                        # Running in Modal - use the agent function directly
                        from agent import MultiModalAgent
                        agent = MultiModalAgent(os.getenv("NEBIUS_API_KEY"))
                        
                        images_data = []
                        if uploaded_images:
                            for img_file in uploaded_images:
                                images_data.append(img_file.getvalue())
                        
                        result = agent.process_request(
                            prompt=prompt,
                            images=images_data if images_data else None,
                            report_type=report_type
                        )
                    else:
                        # Local development - make API call
                        # This would call your Modal API endpoint
                        result = {"success": False, "error": "Local development mode - configure API endpoint"}
                    
                    if result.get("success"):
                        st.session_state.reports.append({
                            "timestamp": datetime.now(),
                            "prompt": prompt,
                            "report": result["report"],
                            "metadata": result["metadata"],
                            "images": len(uploaded_images) if uploaded_images else 0
                        })
                        
                        st.markdown('<div class="success-message">✅ Report generated successfully!</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="error-message">❌ Error: {result.get("error", "Unknown error")}</div>', 
                                  unsafe_allow_html=True)
                        
                except Exception as e:
                    st.markdown(f'<div class="error-message">❌ Error: {str(e)}</div>', 
                              unsafe_allow_html=True)
    
    with col2:
        st.header("📋 Reports")
        
        if st.session_state.reports:
            # Show latest report first
            for idx, report in enumerate(reversed(st.session_state.reports)):
                with st.expander(f"Report {len(st.session_state.reports) - idx} - {report['timestamp'].strftime('%H:%M:%S')}", 
                               expanded=(idx == 0)):
                    
                    # Metadata
                    meta = report['metadata']
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Type", meta['report_type'])
                    with col_b:
                        st.metric("Images", report['images'])
                    with col_c:
                        st.metric("Model", meta.get('model', 'N/A').split('/')[-1])
                    
                    # Original prompt
                    st.subheader("Original Prompt")
                    st.text(report['prompt'])
                    
                    # Report content
                    st.subheader("Generated Report")
                    st.markdown(f'<div class="report-container">{report["report"]}</div>', 
                              unsafe_allow_html=True)
                    
                    # Download button
                    report_text = f"""
# Multi-Modal AI Report
Generated: {report['timestamp']}
Type: {meta['report_type']}
Model: {meta.get('model', 'N/A')}

## Original Prompt
{report['prompt']}

## Report
{report['report']}
"""
                    st.download_button(
                        "💾 Download Report",
                        report_text,
                        f"report_{report['timestamp'].strftime('%Y%m%d_%H%M%S')}.md",
                        "text/markdown"
                    )
        else:
            st.info("No reports generated yet. Enter a prompt and click 'Generate Report' to start!")
