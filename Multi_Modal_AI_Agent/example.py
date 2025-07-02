import sys
import os
import requests
import json
from datetime import datetime
import base64

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_markdown_to_html(text):
    """Convert simple markdown-like formatting to HTML"""
    # Convert **bold** to <strong>
    text = text.replace('**', '<strong>', 1)
    text = text.replace('**', '</strong>', 1)
    
    # Handle multiple bold sections
    while '**' in text:
        text = text.replace('**', '<strong>', 1)
        if '**' in text:
            text = text.replace('**', '</strong>', 1)
    
    # Convert emojis and section headers to proper HTML
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        line = line.strip()
        if line.startswith('📋') or line.startswith('📝') or line.startswith('🔍') or \
           line.startswith('🩺') or line.startswith('💡') or line.startswith('📊') or \
           line.startswith('⚠️') or line.startswith('🧠') or line.startswith('📌'):
            # Make section headers more prominent
            line = f'<h3>{line}</h3>'
        elif line and not line.startswith('<'):
            # Regular paragraphs
            line = f'<p>{line}</p>'
        
        formatted_lines.append(line)
    
    return '\n'.join(formatted_lines)

def format_user_friendly_response(response_json):
    """Convert JSON response to user-friendly readable text"""
    if isinstance(response_json, dict):
        formatted_text = ""
        
        # Handle common response fields
        if 'analysis' in response_json:
            formatted_text += f"📋 **Analysis Results:**\n{response_json['analysis']}\n\n"
        
        if 'summary' in response_json:
            formatted_text += f"📝 **Summary:**\n{response_json['summary']}\n\n"
        
        if 'findings' in response_json:
            formatted_text += f"🔍 **Key Findings:**\n"
            findings = response_json['findings']
            if isinstance(findings, list):
                for i, finding in enumerate(findings, 1):
                    formatted_text += f"{i}. {finding}\n"
            else:
                formatted_text += f"{findings}\n"
            formatted_text += "\n"
        
        if 'diagnosis' in response_json:
            formatted_text += f"🩺 **Diagnosis:**\n{response_json['diagnosis']}\n\n"
        
        if 'recommendations' in response_json:
            formatted_text += f"💡 **Recommendations:**\n"
            recommendations = response_json['recommendations']
            if isinstance(recommendations, list):
                for i, rec in enumerate(recommendations, 1):
                    formatted_text += f"{i}. {rec}\n"
            else:
                formatted_text += f"{recommendations}\n"
            formatted_text += "\n"
        
        if 'confidence' in response_json:
            confidence = response_json['confidence']
            formatted_text += f"📊 **Confidence Score:** {confidence}%\n\n"
        
        if 'abnormalities' in response_json:
            formatted_text += f"⚠️ **Abnormalities Detected:**\n{response_json['abnormalities']}\n\n"
        
        if 'anatomical_structures' in response_json:
            formatted_text += f"🧠 **Anatomical Structures Identified:**\n{response_json['anatomical_structures']}\n\n"
        
        # Handle any other fields
        handled_fields = {'analysis', 'summary', 'findings', 'diagnosis', 'recommendations', 
                         'confidence', 'abnormalities', 'anatomical_structures'}
        
        for key, value in response_json.items():
            if key not in handled_fields:
                formatted_text += f"📌 **{key.replace('_', ' ').title()}:**\n{value}\n\n"
        
        return formatted_text.strip()
    
    else:
        return str(response_json)

def generate_html_response(prompt, response_data, status_code, file_path, report_type):
    """Generate a professional HTML report with the API response"""
    
    # Try to encode image as base64 for embedding
    image_base64 = ""
    try:
        with open(file_path, 'rb') as img_file:
            image_base64 = base64.b64encode(img_file.read()).decode('utf-8')
            image_mime = "image/jpeg" if file_path.lower().endswith(('.jpg', '.jpeg')) else "image/png"
    except:
        image_base64 = None

    # Parse and format response for user-friendly display
    try:
        response_json = json.loads(response_data)
        formatted_response = format_user_friendly_response(response_json)
        is_json = True
    except:
        formatted_response = response_data
        is_json = False

    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical AI Analysis Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background-color: #0a0e1a;
            color: #e2e8f0;
            line-height: 1.6;
            padding: 24px;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}

        .header {{
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 16px;
            border: 1px solid #334155;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            padding: 32px;
            margin-bottom: 24px;
        }}

        .header h1 {{
            font-size: 28px;
            font-weight: 600;
            color: #f8fafc;
            margin-bottom: 8px;
        }}

        .header .timestamp {{
            color: #94a3b8;
            font-size: 14px;
        }}

        .status-badge {{
            display: inline-block;
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
            margin-top: 16px;
        }}

        .status-success {{
            background: rgba(16, 185, 129, 0.2);
            color: #10b981;
            border: 1px solid #10b981;
        }}

        .status-error {{
            background: rgba(239, 68, 68, 0.2);
            color: #ef4444;
            border: 1px solid #ef4444;
        }}

        .main-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}

        .panel {{
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 16px;
            border: 1px solid #334155;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            padding: 24px;
        }}

        .panel-header {{
            font-size: 18px;
            font-weight: 600;
            color: #f8fafc;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .panel-icon {{
            font-size: 20px;
        }}

        .info-grid {{
            display: grid;
            gap: 12px;
        }}

        .info-item {{
            display: flex;
            justify-content: space-between;
            padding: 12px 16px;
            background: #0f172a;
            border-radius: 8px;
            border: 1px solid #334155;
        }}

        .info-label {{
            color: #94a3b8;
            font-weight: 500;
        }}

        .info-value {{
            color: #f8fafc;
            font-weight: 500;
        }}

        .image-container {{
            text-align: center;
            margin-bottom: 16px;
        }}

        .analysis-image {{
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            border: 1px solid #334155;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
        }}

        .no-image {{
            background: #0f172a;
            border: 2px dashed #475569;
            border-radius: 12px;
            padding: 40px;
            color: #94a3b8;
            text-align: center;
        }}

        .response-panel {{
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 16px;
            border: 1px solid #334155;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            padding: 24px;
        }}

        .response-content {{
            background: #0f172a;
            border-radius: 12px;
            border: 1px solid #334155;
            padding: 24px;
            margin-top: 16px;
        }}

        .response-text {{
            color: #e2e8f0;
            white-space: pre-wrap;
            font-family: 'SF Mono', Monaco, 'Cascadia Code', 'Roboto Mono', Consolas, 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
        }}

        .prompt-panel {{
            background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
            border-radius: 16px;
            border: 1px solid #334155;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            padding: 24px;
            margin-bottom: 24px;
        }}

        .prompt-content {{
            background: #0f172a;
            border-radius: 8px;
            border: 1px solid #334155;
            padding: 16px;
            margin-top: 12px;
        }}

        .prompt-text {{
            color: #3b82f6;
            font-weight: 500;
            font-style: italic;
        }}

        .metadata {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 24px;
        }}

        .metadata-card {{
            background: #0f172a;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #334155;
        }}

        .metadata-label {{
            color: #94a3b8;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 8px;
        }}

        .metadata-value {{
            color: #f8fafc;
            font-size: 18px;
            font-weight: 600;
        }}

        .copy-btn {{
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            border: none;
            border-radius: 6px;
            color: #f8fafc;
            padding: 8px 12px;
            font-size: 12px;
            cursor: pointer;
            float: right;
            margin-bottom: 12px;
            transition: all 0.2s ease;
        }}

        .copy-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }}

        @media (max-width: 768px) {{
            .main-grid {{
                grid-template-columns: 1fr;
            }}
            
            .container {{
                padding: 16px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>Medical AI Analysis Report</h1>
            <div class="timestamp">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
            <div class="status-badge {'status-success' if status_code == 200 else 'status-error'}">
                Status: {status_code} {'✓ Success' if status_code == 200 else '✗ Error'}
            </div>
        </div>

        <!-- Prompt Panel -->
        <div class="prompt-panel">
            <div class="panel-header">
                <span class="panel-icon">💬</span>
                Analysis Prompt
            </div>
            <div class="prompt-content">
                <div class="prompt-text">"{prompt}"</div>
            </div>
        </div>

        <!-- Main Content Grid -->
        <div class="main-grid">
            <!-- Image Panel -->
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-icon">🖼️</span>
                    Input Image
                </div>
                <div class="image-container">
                    {f'<img src="data:{image_mime};base64,{image_base64}" alt="Analysis Image" class="analysis-image">' if image_base64 else '<div class="no-image">Image not available for display</div>'}
                </div>
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">File Path:</span>
                        <span class="info-value">{os.path.basename(file_path)}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Report Type:</span>
                        <span class="info-value">{report_type.title()}</span>
                    </div>
                </div>
            </div>

            <!-- Request Info Panel -->
            <div class="panel">
                <div class="panel-header">
                    <span class="panel-icon">⚙️</span>
                    Request Details
                </div>
                <div class="info-grid">
                    <div class="info-item">
                        <span class="info-label">API Endpoint:</span>
                        <span class="info-value">Modal API</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Method:</span>
                        <span class="info-value">POST</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Status Code:</span>
                        <span class="info-value">{status_code}</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Response Format:</span>
                        <span class="info-value">{'JSON' if is_json else 'Text'}</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Response Panel -->
        <div class="response-panel">
            <div class="panel-header">
                <span class="panel-icon">📋</span>
                AI Analysis Response
                <button class="copy-btn" onclick="copyResponse()">Copy Response</button>
            </div>
            <div class="response-content">
                <div class="response-text" id="responseText">{convert_markdown_to_html(formatted_response)}</div>
            </div>
        </div>

        <!-- Metadata -->
        <div class="metadata">
            <div class="metadata-card">
                <div class="metadata-label">Response Length</div>
                <div class="metadata-value">{len(response_data)} chars</div>
            </div>
            <div class="metadata-card">
                <div class="metadata-label">Processing Status</div>
                <div class="metadata-value">{'Complete' if status_code == 200 else 'Failed'}</div>
            </div>
            <div class="metadata-card">
                <div class="metadata-label">Model Type</div>
                <div class="metadata-value">Multimodal AI</div>
            </div>
            <div class="metadata-card">
                <div class="metadata-label">Report ID</div>
                <div class="metadata-value">{datetime.now().strftime('%Y%m%d-%H%M%S')}</div>
            </div>
        </div>
    </div>

    <script>
        function copyResponse() {{
            const responseText = document.getElementById('responseText').textContent;
            navigator.clipboard.writeText(responseText).then(() => {{
                const btn = document.querySelector('.copy-btn');
                const originalText = btn.textContent;
                btn.textContent = 'Copied!';
                btn.style.background = '#10b981';
                setTimeout(() => {{
                    btn.textContent = originalText;
                    btn.style.background = '';
                }}, 2000);
            }});
        }}
    </script>
</body>
</html>"""
    
    return html_template

def main():
    # API configuration
    url = "https://nbahador--multimodal-ai-agent-serve-api.modal.run/analyze"
    file_path = "img.jpg"
    prompt = 'In which region of the spectrogram is the chirp pattern observed, and does it exhibit an upward or downward frequency sweep? Please explain your reasoning based on the time-frequency characteristics.'
    report_type = 'comprehensive'
    
    print("🔄 Sending request to Medical AI API...")
    print(f"📁 File: {file_path}")
    print(f"💬 Prompt: {prompt}")
    print(f"📊 Report Type: {report_type}")
    print("-" * 50)
    
    try:
        # Make the API request
        with open(file_path, 'rb') as f:
            response = requests.post(
                url,
                files={'images': f},
                data={
                    'prompt': prompt,
                    'report_type': report_type
                }
            )
        
        # Print basic response info
        print(f"📡 Response Status Code: {response.status_code}")
        print(f"📄 Response Length: {len(response.text)} characters")
        
        # Generate HTML report
        html_content = generate_html_response(
            prompt=prompt,
            response_data=response.text,
            status_code=response.status_code,
            file_path=file_path,
            report_type=report_type
        )
        
        # Save HTML report
        output_filename = f"medical_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ HTML report generated: {output_filename}")
        print(f"🌐 Open the file in your browser to view the formatted report")
        
        # Also print raw response for debugging
        print("\n" + "="*50)
        print("RAW API RESPONSE:")
        print("="*50)
        print(response.text)
        
    except FileNotFoundError:
        print(f"❌ Error: File '{file_path}' not found!")
        print("Please make sure the image file exists in the current directory.")
    except requests.exceptions.RequestException as e:
        print(f"❌ Network Error: {e}")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    main()