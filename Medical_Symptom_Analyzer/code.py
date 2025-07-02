import os
import re
import requests
from openai import OpenAI
from flask import Flask, request, jsonify, render_template
from datetime import datetime
import uuid

app = Flask(__name__)

# Initialize Nebius AI client
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

def generate_prompt(age, gender, symptoms, duration, conversation_history=None):
    """Generate a comprehensive prompt for medical analysis"""
    prompt = f"""You are a medical triage assistant analyzing symptoms for a {age}-year-old {gender} patient. 
The patient reports symptoms lasting {duration}:

**Patient Symptoms:**
{symptoms}

Please provide a detailed analysis with the following structure:

**URGENCY LEVEL:** [emergency/urgent/monitor] - Brief explanation of urgency determination

**IMMEDIATE ACTION PLAN:**
- Do this NOW: [immediate critical actions]
- Within 24 hours: [urgent but not immediately life-threatening actions]
- Monitor for changes: [ongoing monitoring instructions]

**POSSIBLE CONDITIONS:**
- [List 3-5 most likely conditions based on symptoms]
- [Include likelihood percentages if possible]

**CLINICAL SIGNIFICANCE:**
[2-3 paragraphs explaining the medical significance of these symptoms]
[Discuss potential complications if untreated]
[Explain why the recommended urgency level was chosen]

**EXPECTED EVALUATION:**
[Detailed description of what medical evaluation might involve]
[Possible diagnostic tests that may be ordered]
[Typical treatment approaches for these symptoms]

**FOLLOW-UP QUESTIONS:**
[List 3-5 specific questions to clarify symptoms]
[Focus on details that would change urgency or diagnosis]

**WARNING SIGNS:**
[List specific symptoms that would indicate worsening condition]
[Clear indicators that should prompt immediate medical attention]

Base your analysis on current medical guidelines. Provide clear, actionable information while avoiding alarmist language.
"""
    return prompt

def research_symptoms(age, gender, symptoms, duration, conversation_history=None):
    """Analyze symptoms using medical AI with enhanced prompt engineering"""
    try:
        messages = [
            {
                "role": "system",
                "content": """You are an experienced medical triage assistant with access to current clinical guidelines. 
                Provide thorough but concise analysis of patient symptoms with:
                - Accurate urgency assessment
                - Evidence-based possible conditions
                - Clear action items
                - Anticipatory guidance
                Avoid definitive diagnoses; focus on triage and next steps.
                
                IMPORTANT: Use the exact section headers requested and ensure each section has content."""
            }
        ]
        
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({
            "role": "user",
            "content": generate_prompt(age, gender, symptoms, duration)
        })
        
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            messages=messages,
            max_tokens=2000,
            temperature=0.2,
            top_p=0.9,
        )
        
        return format_response(response.choices[0].message.content, age, gender, symptoms, duration)
    except Exception as e:
        return f"Error analyzing symptoms: {str(e)}"

def get_urgency_color(urgency_text):
    """Return appropriate color based on urgency level"""
    if not urgency_text:
        return '#6c757d'
    
    urgency_lower = urgency_text.lower()
    if 'emergency' in urgency_lower:
        return '#dc3545'
    elif 'urgent' in urgency_lower:
        return '#fd7e14'
    elif 'monitor' in urgency_lower:
        return '#28a745'
    else:
        return '#6c757d'

def format_list_items(text):
    """Convert bullet points to proper HTML list items"""
    if not text:
        return "<p>No information provided</p>"
    
    lines = text.split('\n')
    formatted_lines = []
    in_list = False
    
    for line in lines:
        line = line.strip()
        if line.startswith('- ') or line.startswith('* '):
            if not in_list:
                formatted_lines.append('<ul>')
                in_list = True
            formatted_lines.append(f"<li>{line[2:]}</li>")
        elif line and in_list:
            formatted_lines.append('</ul>')
            formatted_lines.append(f"<p>{line}</p>")
            in_list = False
        elif line:
            formatted_lines.append(f"<p>{line}</p>")
    
    if in_list:
        formatted_lines.append('</ul>')
    
    return ''.join(formatted_lines)

def extract_section_content(text, section_header):
    """Extract content for a specific section from the response text"""
    # Enhanced patterns to match the new format
    patterns = [
        rf"\*\*{section_header}:\*\*(.*?)(?=\*\*[A-Z][A-Z\s]*:\*\*|\Z)",
        rf"{section_header}:(.*?)(?=\n\*\*[A-Z][A-Z\s]*:\*\*|\Z)",
        rf"\*\*{section_header}\*\*(.*?)(?=\*\*[A-Z][A-Z\s]*\*\*|\Z)",
        rf"{section_header}\*\*(.*?)(?=\*\*[A-Z][A-Z\s]*\*\*|\Z)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            content = match.group(1).strip()
            # Clean up the content
            content = re.sub(r'^\s*-\s*', '', content, flags=re.MULTILINE)
            content = re.sub(r'^\s*\*\s*', '', content, flags=re.MULTILINE)
            return content
    
    return ""

def format_response(raw_response, age, gender, symptoms, duration):
    """Format the raw response into structured sections"""
    # Clean up the raw response
    cleaned_response = raw_response.replace("**", "").replace("*", "")
    
    # Define section mappings
    section_mappings = {
        "urgency": ["URGENCY LEVEL", "Urgency Level", "URGENCY", "Urgency"],
        "action_plan": ["IMMEDIATE ACTION PLAN", "Action Timeline", "ACTION PLAN", "Immediate Action Plan"],
        "conditions": ["POSSIBLE CONDITIONS", "Possible Conditions", "CONDITIONS", "Likely Conditions"],
        "significance": ["CLINICAL SIGNIFICANCE", "Why This Matters", "SIGNIFICANCE", "Clinical Significance"],
        "evaluation": ["EXPECTED EVALUATION", "What to Expect", "EVALUATION", "Expected Evaluation"],
        "questions": ["FOLLOW-UP QUESTIONS", "Follow-up Questions", "QUESTIONS", "Follow-up"],
        "warnings": ["WARNING SIGNS", "Warning Signs", "WARNINGS", "Red Flags"]
    }
    
    # Extract sections
    sections = {}
    for key, headers in section_mappings.items():
        sections[key] = ""
        for header in headers:
            content = extract_section_content(raw_response, header)
            if content:
                sections[key] = content
                break
    
    # Fallback: if no sections found, try to parse by looking for key phrases
    if all(not content for content in sections.values()):
        # Simple fallback parsing
        paragraphs = [p.strip() for p in cleaned_response.split('\n\n') if p.strip()]
        if paragraphs:
            sections["urgency"] = paragraphs[0] if len(paragraphs) > 0 else "Assessment not available"
            sections["action_plan"] = paragraphs[1] if len(paragraphs) > 1 else "No specific actions identified"
            sections["conditions"] = paragraphs[2] if len(paragraphs) > 2 else "Conditions not specified"
            sections["significance"] = paragraphs[3] if len(paragraphs) > 3 else "Clinical significance not detailed"
            sections["evaluation"] = paragraphs[4] if len(paragraphs) > 4 else "Evaluation process not specified"
            sections["questions"] = paragraphs[5] if len(paragraphs) > 5 else "No follow-up questions provided"
            sections["warnings"] = paragraphs[6] if len(paragraphs) > 6 else "No specific warning signs identified"
    
    # Generate HTML report with improved structure
    timestamp = datetime.now().strftime("%B %d, %Y at %I:%M %p")
    urgency_color = get_urgency_color(sections['urgency'])
    
    html_response = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Medical Triage Report - {timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .report-container {{
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 25px;
        }}
        .header {{
            text-align: center;
            margin-bottom: 25px;
            padding-bottom: 15px;
            border-bottom: 2px solid #007bff;
        }}
        .header h1 {{
            color: #2c3e50;
            margin-bottom: 5px;
        }}
        .header p {{
            color: #6c757d;
            font-size: 1.1em;
        }}
        .timestamp {{
            text-align: center;
            font-size: 0.9em;
            color: #6c757d;
            margin-bottom: 20px;
        }}
        .patient-info {{
            background-color: #e9f5ff;
            padding: 15px;
            border-radius: 6px;
            margin-bottom: 20px;
        }}
        .info-grid {{
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 15px;
        }}
        .info-item {{
            margin-bottom: 8px;
        }}
        .info-item strong {{
            display: block;
            color: #007bff;
            margin-bottom: 3px;
            font-size: 0.9em;
        }}
        .urgency-badge {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            margin: 15px 0;
            background-color: {urgency_color};
            color: white;
            font-size: 1em;
        }}
        .section {{
            margin-bottom: 25px;
            padding-bottom: 20px;
            border-bottom: 1px solid #e9ecef;
        }}
        .section:last-child {{
            border-bottom: none;
        }}
        .section h2 {{
            color: #2c3e50;
            font-size: 1.3em;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }}
        .section h2::before {{
            content: '';
            display: inline-block;
            width: 8px;
            height: 8px;
            background-color: #007bff;
            border-radius: 50%;
            margin-right: 10px;
        }}
        .warning-section {{
            background-color: #fff3f3;
            padding: 15px;
            border-radius: 6px;
            border-left: 4px solid #dc3545;
        }}
        .warning-section h2::before {{
            background-color: #dc3545;
        }}
        ul, ol {{
            padding-left: 20px;
            margin: 10px 0;
        }}
        li {{
            margin-bottom: 8px;
        }}
        p {{
            margin-bottom: 12px;
        }}
        .disclaimer {{
            background-color: #fff8e6;
            padding: 15px;
            border-radius: 6px;
            margin-top: 25px;
            font-size: 0.9em;
            border-left: 4px solid #ffc107;
        }}
        @media (max-width: 600px) {{
            .info-grid {{
                grid-template-columns: 1fr;
            }}
        }}
    </style>
</head>
<body>
    <div class="report-container">
        <div class="header">
            <h1>Medical Symptom Analysis Report</h1>
            <p>Professional health assessment powered by AI</p>
        </div>
        
        <div class="timestamp">
            Report generated: {timestamp}
        </div>
        
        <div class="patient-info">
            <h2 style="margin-bottom: 15px; font-size: 1.2em;">Patient Information</h2>
            <div class="info-grid">
                <div class="info-item">
                    <strong>Age</strong>
                    {age} years
                </div>
                <div class="info-item">
                    <strong>Gender</strong>
                    {gender}
                </div>
                <div class="info-item">
                    <strong>Symptoms</strong>
                    {symptoms}
                </div>
                <div class="info-item">
                    <strong>Duration</strong>
                    {duration}
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Urgency Assessment</h2>
            <div class="urgency-badge">
                {sections['urgency'].split(' - ')[0].split(':')[-1].strip() if sections['urgency'] else 'Not determined'}
            </div>
            <div>{format_list_items(sections['urgency'])}</div>
        </div>
        
        <div class="section">
            <h2>Immediate Action Plan</h2>
            <div>{format_list_items(sections['action_plan'])}</div>
        </div>
        
        <div class="section">
            <h2>Possible Conditions</h2>
            <div>{format_list_items(sections['conditions'])}</div>
        </div>
        
        <div class="section">
            <h2>Clinical Significance</h2>
            <div>{format_list_items(sections['significance'])}</div>
        </div>
        
        <div class="section">
            <h2>Expected Medical Evaluation</h2>
            <div>{format_list_items(sections['evaluation'])}</div>
        </div>
        
        <div class="section">
            <h2>Follow-up Questions</h2>
            <div>{format_list_items(sections['questions'])}</div>
        </div>
        
        <div class="section warning-section">
            <h2>Warning Signs - Seek Immediate Care If You Experience:</h2>
            <div>{format_list_items(sections['warnings'])}</div>
        </div>
        
        <div class="disclaimer">
            <p><strong>Important Disclaimer:</strong> This analysis is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition. Never disregard professional medical advice or delay seeking it because of information from this report.</p>
        </div>
    </div>
</body>
</html>
    """
    
    return {
        "raw": raw_response,
        "html": html_response,
        "sections": sections
    }

def save_html_report(html_content, patient_info):
    """Save the HTML report to a file in the same directory as the script"""
    try:
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create reports directory if it doesn't exist
        reports_dir = os.path.join(script_dir, "medical_reports")
        if not os.path.exists(reports_dir):
            os.makedirs(reports_dir)
        
        # Generate filename with timestamp and patient info
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_gender = "".join(c for c in patient_info['gender'] if c.isalnum())
        filename = f"report_{patient_info['age']}yo_{safe_gender}_{timestamp}.html"
        filepath = os.path.join(reports_dir, filename)
        
        # Save the HTML file
        with open(filepath, 'w', encoding='utf-8') as file:
            file.write(html_content)
        
        return {
            "success": True,
            "filepath": filepath,
            "filename": filename
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_symptoms', methods=['POST'])
def analyze_symptoms():
    try:
        data = request.json
        age = data.get('age', '')
        gender = data.get('gender', '')
        symptoms = data.get('symptoms', '')
        duration = data.get('duration', '')
        conversation_history = data.get('conversation_history', [])
        save_report = data.get('save_report', True)
        
        if not all([age, gender, symptoms, duration]):
            return jsonify({"error": "All fields are required"}), 400
        
        result = research_symptoms(age, gender, symptoms, duration, conversation_history)
        
        if isinstance(result, str) and result.startswith("Error"):
            return jsonify({"error": result}), 500
        
        response_data = {
            "analysis": result["raw"],
            "html": result["html"],
            "sections": result["sections"],
            "timestamp": datetime.now().isoformat()
        }
        
        # Save HTML report to local file if requested
        if save_report:
            patient_info = {
                "age": age,
                "gender": gender,
                "symptoms": symptoms,
                "duration": duration
            }
            save_result = save_html_report(result["html"], patient_info)
            response_data["save_result"] = save_result
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)