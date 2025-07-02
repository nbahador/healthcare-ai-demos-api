import os
import requests
import json
from datetime import datetime
import base64
import mimetypes
import socket
import time
import re
from bs4 import BeautifulSoup
from PIL import Image
import io

# Configuration
API_KEY = os.getenv("NEBIUS_API_KEY")
if not API_KEY:
    raise ValueError("Please set NEBIUS_API_KEY environment variable")
BASE_URL = "https://api.studio.nebius.com/v1"
TEXT_MODEL = "meta-llama/Meta-Llama-3.1-70B-Instruct"
VISION_MODEL = "Qwen/Qwen2-VL-72B-Instruct"
MAX_TEXT_LENGTH = 15000
MAX_IMAGES = 5
REQUEST_TIMEOUT = 60

class NebiusAPI:
    def __init__(self):
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
    
    def call(self, prompt, model, is_vision=False, image_data=None):
        payload = {
            "model": model,
            "messages": [],
            "temperature": 0.7,
            "max_tokens": 2000,
            "response_format": {"type": "json_object"} if not is_vision else None
        }
        
        try:
            if is_vision and image_data:
                payload["messages"] = [{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_data['format']};base64,{image_data['base64']}",
                                "detail": "auto"
                            }
                        }
                    ]
                }]
            else:
                payload["messages"] = [{
                    "role": "user",
                    "content": prompt
                }]
            
            endpoint = f"{BASE_URL}/chat/completions"
            
            for attempt in range(3):
                try:
                    response = requests.post(
                        endpoint,
                        json=payload,
                        headers=self.headers,
                        timeout=REQUEST_TIMEOUT
                    )
                    response.raise_for_status()
                    
                    content = response.json()['choices'][0]['message']['content']
                    try:
                        return json.loads(content)
                    except json.JSONDecodeError:
                        return content
                        
                except requests.exceptions.RequestException as e:
                    if attempt == 2:
                        return None
                    time.sleep((attempt + 1) * 5)
            
        except Exception as e:
            return None

class MedicalCaseParser:
    def __init__(self):
        self.api = NebiusAPI()
    
    def parse_case(self, case_text):
        if not case_text:
            return None
        
        # First pass - extract structured data
        structure_prompt = """Extract medical case details as JSON with:
        - patient_info (age, gender, medical_history, bmi)
        - timeline (list of events with timestamps)
        - symptoms (list with duration)
        - tests (list with results)
        - treatments (list with dosages)
        - diagnosis
        - outcome
        
        Case: """ + case_text[:MAX_TEXT_LENGTH]
        
        structured_data = self.api.call(structure_prompt, TEXT_MODEL)
        if not structured_data:
            return None
        
        # Second pass - create knowledge graph relationships
        graph_prompt = """Create a knowledge graph from this medical case as JSON with:
        - nodes (id, label, type [patient, symptom, test, treatment, diagnosis, outcome])
        - links (source, target, relationship)
        
        Data: """ + json.dumps(structured_data)
        
        graph_data = self.api.call(graph_prompt, TEXT_MODEL)
        
        # Third pass - identify potential issues
        analysis_prompt = """Analyze this medical case for:
        - potential malpractice issues
        - diagnostic accuracy
        - treatment appropriateness
        - timeline concerns
        
        Return as JSON with analysis sections.
        
        Data: """ + json.dumps(structured_data)
        
        analysis_data = self.api.call(analysis_prompt, TEXT_MODEL)
        
        return {
            'structured_data': structured_data,
            'graph_data': graph_data,
            'analysis': analysis_data
        }

class HTMLReportGenerator:
    @staticmethod
    def generate_report(parsed_data):
        if not parsed_data:
            return "<html><body>Error: Failed to parse case</body></html>"
        
        structured = parsed_data.get('structured_data', {})
        graph = parsed_data.get('graph_data', {})
        analysis = parsed_data.get('analysis', {})
        
        # Generate timeline HTML
        timeline_html = ""
        if structured.get('timeline'):
            timeline_html = """
            <div class="timeline">
                <div class="timeline-line"></div>
            """
            positions = [5, 25, 45, 65, 85]
            colors = ["#e74c3c", "#f39c12", "#9b59b6", "#3498db", "#27ae60"]
            
            for i, event in enumerate(structured['timeline'][:5]):
                pos = positions[i % len(positions)]
                color = colors[i % len(colors)]
                time_part = f"{event.get('time', '')}<br>" if event.get('time') else ""
                timeline_html += f"""
                <div class="timeline-event" style="left: {pos}%; top: 50%; background: {color};">
                    {time_part}{event.get('event', '')}
                </div>
                """
            timeline_html += "</div>"
        
        # Generate entities HTML
        entities_html = ""
        entity_types = [
            ('patient_info', 'Patient', '#e74c3c'),
            ('symptoms', 'Symptoms', '#f39c12'),
            ('tests', 'Tests', '#3498db'),
            ('treatments', 'Treatments', '#9b59b6'),
            ('outcome', 'Outcome', '#27ae60')
        ]
        
        for key, label, color in entity_types:
            if key in structured:
                entities_html += f"""
                <div class="entity-card" style="border-left-color: {color}">
                    <h4>{label}</h4>
                    <div class="entity-type">{key.replace('_', ' ').title()}</div>
                """
                if isinstance(structured[key], dict):
                    for k, v in structured[key].items():
                        entities_html += f'<div class="attribute">• {k}: {v}</div>'
                elif isinstance(structured[key], list):
                    for item in structured[key]:
                        entities_html += f'<div class="attribute">• {item}</div>'
                entities_html += "</div>"
        
        # Generate analysis HTML
        analysis_html = ""
        if isinstance(analysis, dict):
            for section, content in analysis.items():
                analysis_html += f"""
                <div class="analysis-section">
                    <h3>{section.replace('_', ' ').title()}</h3>
                    <div class="analysis-content">
                """
                if isinstance(content, list):
                    for item in content:
                        analysis_html += f'<p>• {item}</p>'
                elif isinstance(content, dict):
                    for k, v in content.items():
                        analysis_html += f'<p><strong>{k}:</strong> {v}</p>'
                else:
                    analysis_html += f'<p>{content}</p>'
                analysis_html += "</div></div>"
        
        # Combine into full HTML
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Medical Case Knowledge Graph Report</title>
            <style>
                body {{
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                    background: rgba(255, 255, 255, 0.95);
                    border-radius: 20px;
                    padding: 30px;
                    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                }}
                h1 {{
                    text-align: center;
                    color: #2c3e50;
                    margin-bottom: 30px;
                    font-size: 2.5em;
                    font-weight: 300;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    border-radius: 12px;
                    background: #f8f9fa;
                    border-left: 4px solid #3498db;
                }}
                .section h2 {{
                    color: #2c3e50;
                    margin-top: 0;
                    font-size: 1.5em;
                }}
                .timeline {{
                    position: relative;
                    height: 200px;
                    margin: 20px 0;
                }}
                .timeline-line {{
                    position: absolute;
                    top: 50%;
                    width: 100%;
                    height: 2px;
                    background: #bdc3c7;
                }}
                .timeline-event {{
                    position: absolute;
                    background: #3498db;
                    color: white;
                    padding: 8px 12px;
                    border-radius: 6px;
                    font-size: 12px;
                    white-space: nowrap;
                    transform: translateY(-50%);
                    box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                }}
                .entity-list {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 15px;
                    margin-top: 15px;
                }}
                .entity-card {{
                    background: white;
                    padding: 15px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .entity-card h4 {{
                    margin: 0 0 10px 0;
                    color: #2c3e50;
                }}
                .entity-type {{
                    font-size: 12px;
                    color: #7f8c8d;
                    margin-bottom: 10px;
                }}
                .attribute {{
                    font-size: 14px;
                    margin: 5px 0;
                    color: #555;
                }}
                .analysis-section {{
                    margin-bottom: 20px;
                }}
                .analysis-section h3 {{
                    color: #2c3e50;
                    border-bottom: 1px solid #eee;
                    padding-bottom: 5px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Medical Case Knowledge Graph Report</h1>
                
                <div class="section">
                    <h2>⏰ Timeline Extraction</h2>
                    <div class="timeline-container">
                        {timeline_html}
                    </div>
                </div>
                
                <div class="section">
                    <h2>🏥 Extracted Entities</h2>
                    <div class="entity-list">
                        {entities_html}
                    </div>
                </div>
                
                <div class="section">
                    <h2>🔍 Case Analysis</h2>
                    {analysis_html}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

def main():
    print("Medical Case Knowledge Graph Parser")
    print("Enter medical case text or 'quit' to exit\n")
    
    while True:
        case_text = input("Enter case text > ").strip()
        
        if case_text.lower() in ('quit', 'exit', 'q'):
            print("\nGoodbye!")
            break
        
        if not case_text:
            print("Please enter some text")
            continue
        
        print("\n[1/2] Parsing case...")
        parser = MedicalCaseParser()
        parsed_data = parser.parse_case(case_text)
        
        print("\n[2/2] Generating report...")
        report_html = HTMLReportGenerator.generate_report(parsed_data)
        
        filename = f"medical_case_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        print(f"\nReport generated: {filename}")
        print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    main()