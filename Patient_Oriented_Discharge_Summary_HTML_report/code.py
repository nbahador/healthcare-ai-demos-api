import os
import requests
from bs4 import BeautifulSoup
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import base64
from urllib.parse import urljoin, urlparse
import time
import traceback

# Nebius AI Configuration
API_KEY = os.getenv("NEBIUS_API_KEY")
if not API_KEY:
    raise ValueError("Please set NEBIUS_API_KEY environment variable")
BASE_URL = "https://api.studio.nebius.com/v1"

@dataclass
class ClinicalFindings:
    symptoms: List[Dict[str, str]] = field(default_factory=list)
    vital_signs: List[Dict[str, str]] = field(default_factory=list)
    laboratory_results: List[Dict[str, str]] = field(default_factory=list)
    imaging_findings: List[Dict[str, str]] = field(default_factory=list)
    physical_exam: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class PatientDemographics:
    age: str = ""
    gender: str = ""
    medical_history: List[str] = field(default_factory=list)
    allergies: List[str] = field(default_factory=list)
    current_medications: List[str] = field(default_factory=list)
    social_history: List[str] = field(default_factory=list)

@dataclass
class ClinicalDecisions:
    diagnosis: List[Dict[str, str]] = field(default_factory=list)
    treatment_rationale: List[Dict[str, str]] = field(default_factory=list)
    risk_factors: List[Dict[str, str]] = field(default_factory=list)
    prognosis: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class DetailedAppointments:
    scheduled_visits: List[Dict[str, str]] = field(default_factory=list)
    urgent_care_indicators: List[Dict[str, str]] = field(default_factory=list)
    monitoring_schedule: List[Dict[str, str]] = field(default_factory=list)

@dataclass
class ComprehensivePODS:
    primary_diagnosis: str = ""
    secondary_diagnoses: List[str] = field(default_factory=list)
    hospital_course: str = ""
    
    clinical_findings: ClinicalFindings = field(default_factory=ClinicalFindings)
    patient_demographics: PatientDemographics = field(default_factory=PatientDemographics)
    clinical_decisions: ClinicalDecisions = field(default_factory=ClinicalDecisions)
    detailed_appointments: DetailedAppointments = field(default_factory=DetailedAppointments)
    
    discharge_medications: List[Dict[str, str]] = field(default_factory=list)
    medication_changes: List[Dict[str, str]] = field(default_factory=list)
    
    activity_restrictions: List[Dict[str, str]] = field(default_factory=list)
    dietary_modifications: List[Dict[str, str]] = field(default_factory=list)
    wound_care: List[Dict[str, str]] = field(default_factory=list)
    
    red_flag_symptoms: List[Dict[str, str]] = field(default_factory=list)
    emergency_contacts: List[Dict[str, str]] = field(default_factory=list)
    healthcare_team: List[Dict[str, str]] = field(default_factory=list)
    
    original_content: str = ""
    extracted_images: List[Dict[str, str]] = field(default_factory=list)
    source_url: str = ""

class AdvancedNebiusAIClient:
    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
    def generate_text(self, prompt: str, model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct", temperature: float = 0.2) -> str:
        """Generate text with enhanced parameters for medical analysis"""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are an expert medical AI assistant specializing in clinical documentation and patient care. Provide detailed, accurate medical information based on the provided content. Only extract information that is explicitly mentioned or can be reasonably inferred from the clinical context."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 6000,
            "temperature": temperature,
            "top_p": 0.9,
            "frequency_penalty": 0.1
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error calling Nebius AI: {e}")
            return ""
        except json.JSONDecodeError as e:
            print(f"Error decoding Nebius AI response: {e}")
            return ""
    
    def analyze_image(self, image_data: str, prompt: str, model: str = "meta-llama/Meta-Llama-3.2-90B-Vision-Instruct") -> str:
        """Analyze medical images using vision model"""
        url = f"{self.base_url}/chat/completions"
        
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"As a medical imaging specialist, analyze this image and {prompt}. Only describe what you can clearly observe in the image."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 2000,
            "temperature": 0.1
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            print(f"Error analyzing image: {e}")
            return ""

class ComprehensiveWebsiteParser:
    def __init__(self, nebius_client: AdvancedNebiusAIClient):
        self.nebius_client = nebius_client
        
    def fetch_website_content(self, url: str) -> Tuple[str, List[str]]:
        """Fetch website content and extract images"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract images
            image_urls = []
            for img in soup.find_all('img'):
                src = img.get('src')
                if src:
                    full_url = urljoin(url, src)
                    if any(keyword in src.lower() for keyword in ['xray', 'scan', 'chart', 'graph', 'ecg', 'ekg', 'ct', 'mri', 'ultrasound', 'lab', 'result']):
                        image_urls.append(full_url)
                    elif img.get('alt') and any(keyword in img.get('alt').lower() for keyword in ['patient', 'medical', 'clinical', 'diagnosis', 'test', 'result']):
                        image_urls.append(full_url)
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content with better structure preservation
            text_content = []
            
            # Extract structured content
            for element in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                text_content.append(f"HEADING: {element.get_text().strip()}")
            
            for element in soup.find_all(['p', 'div', 'span', 'li']):
                text = element.get_text().strip()
                if text and len(text) > 10:
                    text_content.append(text)
            
            # Look for structured data (tables, lists)
            for table in soup.find_all('table'):
                text_content.append("TABLE DATA:")
                for row in table.find_all('tr'):
                    cells = [cell.get_text().strip() for cell in row.find_all(['td', 'th'])]
                    if cells:
                        text_content.append(" | ".join(cells))
            
            full_text = "\n".join(text_content)
            return full_text[:20000], image_urls
            
        except Exception as e:
            print(f"Error fetching website: {e}")
            return "", []
    
    def download_and_encode_image(self, image_url: str) -> Optional[str]:
        """Download image and encode to base64"""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(image_url, headers=headers, timeout=15)
            response.raise_for_status()
            
            content_type = response.headers.get('content-type', '')
            if not content_type.startswith('image/'):
                return None
                
            return base64.b64encode(response.content).decode('utf-8')
        except Exception as e:
            print(f"Error downloading image {image_url}: {e}")
            return None
    
    def extract_comprehensive_medical_data(self, content: str, source_url: str) -> ComprehensivePODS:
        """Extract comprehensive medical data using advanced AI analysis"""
    
        comprehensive_pods = ComprehensivePODS(
            original_content=content,
            source_url=source_url
        )

        # Phase 1: Initial medical content analysis
        initial_prompt = f"""
        Analyze this medical content from {source_url} and extract comprehensive clinical information.
    
        Content:
        {content[:15000]}  # Truncate to avoid token limits
    
        Provide detailed JSON analysis with the following structure:
        {{
            "primary_diagnosis": "Main diagnosis or chief complaint",
            "secondary_diagnoses": ["Additional diagnoses"],
            "hospital_course": "Summary of hospital stay or clinical course",
        
            "clinical_findings": {{
                "symptoms": [
                    {{"symptom": "symptom name", "severity": "severity level", "onset": "when started", "duration": "how long", "characteristics": "description"}}
                ],
                "vital_signs": [
                    {{"parameter": "vital sign", "value": "measurement", "normal_range": "normal values", "significance": "clinical meaning"}}
                ],
                "laboratory_results": [
                    {{"test": "lab test name", "value": "result", "reference": "normal range", "interpretation": "clinical significance"}}
                ],
                "imaging_findings": [
                    {{"type": "imaging type", "findings": "what was found", "impression": "radiologist interpretation"}}
                ],
                "physical_exam": [
                    {{"system": "body system", "findings": "examination findings", "significance": "clinical relevance"}}
                ]
            }},
        
            "patient_demographics": {{
                "age": "patient age or age range",
                "gender": "patient gender",
                "medical_history": ["relevant past medical history"],
                "allergies": ["known allergies"],
                "current_medications": ["medications patient was taking"],
                "social_history": ["relevant social factors"]
            }},
        
            "clinical_decisions": {{
                "diagnosis": [
                    {{"condition": "diagnosed condition", "certainty": "confidence level", "evidence": "supporting evidence"}}
                ],
                "treatment_rationale": [
                    {{"decision": "treatment decision", "reasoning": "why chosen", "evidence": "supporting literature/guidelines"}}
                ],
                "risk_factors": [
                    {{"factor": "risk factor", "severity": "risk level", "mitigation": "how to address"}}
                ],
                "prognosis": [
                    {{"outlook": "expected outcome", "timeframe": "recovery timeline", "factors": "influencing factors"}}
                ]
            }}
        }}
    
        IMPORTANT: Return ONLY valid JSON. Do not include any additional commentary, explanations, or markdown formatting.
        The response must be parseable with json.loads() and must match the exact structure specified above.
        """

        response1 = self.nebius_client.generate_text(initial_prompt)
    
        # Phase 2: Detailed care instructions and follow-up
        followup_prompt = f"""
        Based on this medical content, extract detailed discharge planning and follow-up information:
    
        Content:
        {content[:15000]}  # Truncate to avoid token limits
    
        Provide JSON with:
        {{
            "detailed_appointments": {{
                "scheduled_visits": [
                    {{"type": "appointment type", "date": "specific date", "time": "time", "location": "where", "provider": "who", "purpose": "why"}}
                ],
                "urgent_care_indicators": [
                    {{"scenario": "when to seek urgent care", "action": "what to do", "timeframe": "how quickly"}}
                ],
                "monitoring_schedule": [
                    {{"parameter": "what to monitor", "frequency": "how often", "method": "how to check"}}
                ]
            }},
        
            "discharge_medications": [
                {{"name": "medication name", "dose": "dosage", "frequency": "how often", "duration": "how long", "purpose": "why prescribed", "side_effects": "potential side effects", "special_instructions": "additional notes"}}
            ],
        
            "medication_changes": [
                {{"type": "started/stopped/modified", "medication": "drug name", "change": "what changed", "reason": "why changed"}}
            ],
        
            "activity_restrictions": [
                {{"activity": "restricted activity", "restriction": "type of restriction", "duration": "how long", "reason": "why restricted"}}
            ],
        
            "dietary_modifications": [
                {{"modification": "dietary change", "reason": "medical reason", "duration": "how long", "specifics": "detailed instructions"}}
            ],
        
            "wound_care": [
                {{"site": "wound location", "instructions": "care instructions", "frequency": "how often", "signs_to_watch": "warning signs"}}
            ],
        
            "red_flag_symptoms": [
                {{"symptom": "warning symptom", "severity": "how serious", "action": "what to do", "timeframe": "how urgent"}}
            ],
        
            "emergency_contacts": [
                {{"type": "contact type", "contact": "phone/details", "when_to_call": "circumstances"}}
            ],
        
            "healthcare_team": [
                {{"role": "provider role", "name": "provider name", "contact": "contact info", "specialty": "medical specialty"}}
            ]
        }}
    
        IMPORTANT: Return ONLY valid JSON. Do not include any additional commentary, explanations, or markdown formatting.
        The response must be parseable with json.loads() and must match the exact structure specified above.
        """
    
        response2 = self.nebius_client.generate_text(followup_prompt)
    
        def parse_ai_response(response: str) -> Optional[dict]:
            """Helper function to parse AI responses with robust error handling"""
            if not response:
                return None
            
            try:
                # First try to parse directly as JSON
                return json.loads(response)
            except json.JSONDecodeError:
                try:
                    # If that fails, try to extract JSON from markdown code blocks
                    json_str = response.strip()
                    if json_str.startswith("```json"):
                        json_str = json_str[7:-3].strip()
                    elif json_str.startswith("```"):
                        json_str = json_str[3:-3].strip()
                    return json.loads(json_str)
                except (json.JSONDecodeError, AttributeError) as e:
                    print(f"Error parsing AI response: {e}")
                    print("Problematic response content:")
                    print(response[:1000])  # Print first 1000 chars for debugging
                    return None
    
        # Parse responses with error handling
        try:
            data1 = parse_ai_response(response1)
            if data1:
                comprehensive_pods.primary_diagnosis = data1.get("primary_diagnosis", "")
                comprehensive_pods.secondary_diagnoses = data1.get("secondary_diagnoses", [])
                comprehensive_pods.hospital_course = data1.get("hospital_course", "")
            
                if "clinical_findings" in data1:
                    cf = data1["clinical_findings"]
                    comprehensive_pods.clinical_findings = ClinicalFindings(
                        symptoms=cf.get("symptoms", []),
                        vital_signs=cf.get("vital_signs", []),
                        laboratory_results=cf.get("laboratory_results", []),
                        imaging_findings=cf.get("imaging_findings", []),
                        physical_exam=cf.get("physical_exam", [])
                    )
            
                if "patient_demographics" in data1:
                    pd = data1["patient_demographics"]
                    comprehensive_pods.patient_demographics = PatientDemographics(
                        age=pd.get("age", ""),
                        gender=pd.get("gender", ""),
                        medical_history=pd.get("medical_history", []),
                        allergies=pd.get("allergies", []),
                        current_medications=pd.get("current_medications", []),
                        social_history=pd.get("social_history", [])
                    )
            
                if "clinical_decisions" in data1:
                    cd = data1["clinical_decisions"]
                    comprehensive_pods.clinical_decisions = ClinicalDecisions(
                        diagnosis=cd.get("diagnosis", []),
                        treatment_rationale=cd.get("treatment_rationale", []),
                        risk_factors=cd.get("risk_factors", []),
                        prognosis=cd.get("prognosis", [])
                    )
        
            data2 = parse_ai_response(response2)
            if data2:
                if "detailed_appointments" in data2:
                    da = data2["detailed_appointments"]
                    comprehensive_pods.detailed_appointments = DetailedAppointments(
                        scheduled_visits=da.get("scheduled_visits", []),
                        urgent_care_indicators=da.get("urgent_care_indicators", []),
                        monitoring_schedule=da.get("monitoring_schedule", [])
                    )
            
                comprehensive_pods.discharge_medications = data2.get("discharge_medications", [])
                comprehensive_pods.medication_changes = data2.get("medication_changes", [])
                comprehensive_pods.activity_restrictions = data2.get("activity_restrictions", [])
                comprehensive_pods.dietary_modifications = data2.get("dietary_modifications", [])
                comprehensive_pods.wound_care = data2.get("wound_care", [])
                comprehensive_pods.red_flag_symptoms = data2.get("red_flag_symptoms", [])
                comprehensive_pods.emergency_contacts = data2.get("emergency_contacts", [])
                comprehensive_pods.healthcare_team = data2.get("healthcare_team", [])
            
        except Exception as e:
            print(f"Unexpected error parsing AI responses: {e}")
            traceback.print_exc()
    
        return comprehensive_pods
    
    def analyze_medical_images(self, image_urls: List[str]) -> List[Dict[str, str]]:
        """Analyze medical images using vision model"""
        analyzed_images = []
        
        for url in image_urls[:5]:  # Limit to 5 images to avoid excessive API calls
            print(f"Analyzing image: {url}")
            
            image_data = self.download_and_encode_image(url)
            if not image_data:
                continue
                
            analysis_prompt = """
            analyze this medical image and provide:
            1. Type of medical image/chart/graph
            2. Key clinical findings visible
            3. Relevant measurements or values shown
            4. Clinical significance of findings
            5. Any abnormalities or notable features
            
            Be specific about what you observe and avoid speculation beyond what's clearly visible.
            """
            
            analysis = self.nebius_client.analyze_image(image_data, analysis_prompt)
            
            if analysis:
                analyzed_images.append({
                    "url": url,
                    "analysis": analysis,
                    "type": "medical_image"
                })
                
            time.sleep(1)  # Rate limiting
        
        return analyzed_images

class AdvancedPODSHTMLGenerator:
    def generate_comprehensive_html_report(self, pods_data: ComprehensivePODS) -> str:
        """Generate comprehensive HTML report with advanced medical sections"""
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title> Patient Oriented Discharge Summary (PODS)</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1A1F2B; /* Dark gunmetal background */
            color: #E0E0E0; /* Metallic silver text */
            min-height: 100vh;
            padding: 20px;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: #2A2F3B;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            overflow: hidden;
            border: 1px solid #3A3F4B;
        }}
        
        .header {{
            background: linear-gradient(135deg, #1A1F2B 0%, #2A2F3B 100%);
            color: #E0E0E0;
            padding: 40px;
            text-align: center;
            position: relative;
            border-bottom: 1px solid #3A3F4B;
        }}
        
        .header h1 {{
            font-size: 2.8rem;
            margin-bottom: 15px;
            color: #FFFFFF;
        }}
        
        .header .subtitle {{
            font-size: 1.2rem;
            opacity: 0.8;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .overview-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        
        .overview-card {{
            background: #2A2F3B;
            border-radius: 10px;
            padding: 20px;
            border-left: 4px solid #FF7F50; /* Soft coral accent */
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }}
        
        .overview-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 8px 20px rgba(0,0,0,0.3);
        }}
        
        .overview-card h3 {{
            color: #FF7F50; /* Soft coral */
            margin-bottom: 15px;
            font-size: 1.3rem;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .section {{
            margin-bottom: 30px;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            border: 1px solid #3A3F4B;
        }}
        
        .section-header {{
            background: linear-gradient(135deg, #2A2F3B 0%, #3A3F4B 100%);
            color: #E0E0E0;
            padding: 20px;
            font-size: 1.3rem;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 15px;
            cursor: pointer;
            border-bottom: 1px solid #3A3F4B;
        }}
        
        .section-header.clinical {{
            border-left: 5px solid #FF7F50; /* Soft coral */
        }}
        
        .section-header.demographics {{
            border-left: 5px solid #9ABF88; /* Muted sage */
        }}
        
        .section-header.decisions {{
            border-left: 5px solid #6B8E9B; /* Muted teal */
        }}
        
        .section-header.medications {{
            border-left: 5px solid #D4A59A; /* Muted coral */
        }}
        
        .section-header.appointments {{
            border-left: 5px solid #8FBC8F; /* Muted sage green */
        }}
        
        .section-content {{
            padding: 25px;
            background: #2A2F3B;
        }}
        
        .clinical-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
        }}
        
        .clinical-subsection {{
            background: #3A3F4B;
            border-radius: 8px;
            padding: 18px;
            border-left: 4px solid #FF7F50; /* Soft coral */
        }}
        
        .clinical-subsection h4 {{
            color: #E0E0E0;
            margin-bottom: 15px;
            font-size: 1.1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .finding-item {{
            background: #3A3F4B;
            border-radius: 6px;
            padding: 15px;
            margin-bottom: 12px;
            border-left: 3px solid #9ABF88; /* Muted sage */
            transition: all 0.3s ease;
        }}
        
        .finding-item:hover {{
            background: #4A4F5B;
            transform: translateX(5px);
        }}
        
        .finding-label {{
            font-weight: bold;
            color: #FFFFFF;
            margin-bottom: 5px;
        }}
        
        .finding-details {{
            color: #C0C0C0;
            font-size: 0.95rem;
        }}
        
        .severity-high {{
            border-left-color: #FF6B6B;
        }}
        
        .severity-moderate {{
            border-left-color: #FFA500;
        }}
        
        .severity-low {{
            border-left-color: #9ABF88; /* Muted sage */
        }}
        
        .medication-card {{
            background: #3A3F4B;
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 18px;
            border-left: 4px solid #D4A59A; /* Muted coral */
        }}
        
        .medication-name {{
            font-size: 1.1rem;
            font-weight: bold;
            color: #FFFFFF;
            margin-bottom: 10px;
        }}
        
        .medication-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-top: 10px;
        }}
        
        .medication-detail {{
            background: #4A4F5B;
            padding: 10px;
            border-radius: 5px;
            border-left: 2px solid #D4A59A; /* Muted coral */
        }}
        
        .appointment-timeline {{
            position: relative;
            padding-left: 25px;
        }}
        
        .appointment-timeline::before {{
            content: '';
            position: absolute;
            left: 12px;
            top: 0;
            bottom: 0;
            width: 2px;
            background: linear-gradient(to bottom, #6B8E9B, #8FBC8F);
        }}
        
        .appointment-item {{
            position: relative;
            background: #3A3F4B;
            border-radius: 8px;
            padding: 18px;
            margin-bottom: 18px;
            border-left: 4px solid #6B8E9B; /* Muted teal */
        }}
        
        .appointment-item::before {{
            content: '';
            position: absolute;
            left: -25px;
            top: 50%;
            transform: translateY(-50%);
            width: 10px;
            height: 10px;
            background: #6B8E9B; /* Muted teal */
            border-radius: 50%;
            border: 2px solid #2A2F3B;
        }}
        
        .urgent-indicator {{
            background: linear-gradient(135deg, #FF6B6B, #D44D4D);
            color: white;
            padding: 6px 12px;
            border-radius: 15px;
            font-size: 0.85rem;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 8px;
        }}
        
        .warning-box {{
            background: linear-gradient(135deg, #4A4F5B, #5A5F6B);
            border: 1px solid #FFA500;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #FFA500;
        }}
        
        .emergency-box {{
            background: linear-gradient(135deg, #5A4F5B, #6A5F6B);
            border: 1px solid #FF6B6B;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #FF6B6B;
        }}
        
        .image-gallery {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 15px;
            margin: 15px 0;
        }}
        
        .image-analysis {{
            background: #3A3F4B;
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
        }}
        
        .image-analysis img {{
            width: 100%;
            height: 180px;
            object-fit: cover;
            border-radius: 5px;
            margin-bottom: 12px;
        }}
        
        .collapsible {{
            cursor: pointer;
            transition: all 0.3s ease;
        }}
        
        .collapsible:hover {{
            background: rgba(255,255,255,0.05);
        }}
        
        .content-collapsed {{
            display: none;
        }}
        
        .print-button {{
            position: fixed;
            bottom: 25px;
            right: 25px;
            background: linear-gradient(135deg, #FF7F50, #D46B50);
            color: white;
            border: none;
            border-radius: 50px;
            padding: 12px 20px;
            font-size: 0.95rem;
            cursor: pointer;
            box-shadow: 0 5px 15px rgba(255,127,80,0.3);
            transition: all 0.3s ease;
            z-index: 1000;
        }}
        
        .print-button:hover {{
            transform: translateY(-3px);
            box-shadow: 0 8px 20px rgba(255,127,80,0.4);
        }}
        
        /* Additional styling for better dark theme readability */
        a {{
            color: #FF7F50;
            text-decoration: none;
        }}
        
        a:hover {{
            text-decoration: underline;
        }}
        
        ul, ol {{
            padding-left: 20px;
            color: #C0C0C0;
        }}
        
        li {{
            margin-bottom: 5px;
        }}
        
        strong {{
            color: #FFFFFF;
        }}
        
        @media print {{
            body {{ 
                background: white; 
                color: black;
                padding: 0; 
            }}
            .container {{ 
                box-shadow: none;
                background: white;
                border: none;
            }}
            .print-button {{ display: none; }}
            .section {{ 
                page-break-inside: avoid;
                border: 1px solid #DDD;
            }}
            .section-header {{
                background: #F5F5F5 !important;
                color: #333 !important;
                border-bottom: 1px solid #DDD !important;
            }}
            .section-content {{
                background: white !important;
            }}
            .finding-item, .medication-card, .appointment-item, .clinical-subsection {{
                background: #F9F9F9 !important;
                border-color: #DDD !important;
            }}
            .finding-label, .medication-name {{
                color: #333 !important;
            }}
            .finding-details {{
                color: #666 !important;
            }}
        }}
        
        @media (max-width: 768px) {{
            .container {{ 
                margin: 10px; 
                border-radius: 8px; 
            }}
            .content {{ padding: 15px; }}
            .header {{ padding: 20px; }}
            .header h1 {{ font-size: 2rem; }}
            .clinical-grid, .overview-grid {{ grid-template-columns: 1fr; }}
            .medication-grid {{ grid-template-columns: 1fr 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1><i class="fas fa-hospital-alt"></i> Patient Oriented Discharge Summary </h1>
            <p style="margin-top: 10px; opacity: 0.8;"><i class="fas fa-calendar"></i> Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        
        <div class="content">
            <!-- Overview Cards -->
            <div class="overview-grid">
                <div class="overview-card">
                    <h3><i class="fas fa-stethoscope"></i> Primary Diagnosis</h3>
                    <p><strong>{pods_data.primary_diagnosis or 'Not specified'}</strong></p>
                </div>
                
                {f'''<div class="overview-card">
                    <h3><i class="fas fa-list-ul"></i> Secondary Diagnoses</h3>
                    <ul>{"".join([f"<li>{diag}</li>" for diag in pods_data.secondary_diagnoses])}</ul>
                </div>''' if pods_data.secondary_diagnoses else ''}
                
                {f'''<div class="overview-card">
                    <h3><i class="fas fa-user"></i> Patient Demographics</h3>
                    <p><strong>Age:</strong> {pods_data.patient_demographics.age or 'Not specified'}</p>
                    <p><strong>Gender:</strong> {pods_data.patient_demographics.gender or 'Not specified'}</p>
                </div>''' if pods_data.patient_demographics.age or pods_data.patient_demographics.gender else ''}
            </div>
            
            {f'''<div class="overview-card" style="grid-column: 1 / -1;">
                <h3><i class="fas fa-hospital"></i> Hospital Course Summary</h3>
                <p>{pods_data.hospital_course}</p>
            </div>''' if pods_data.hospital_course else ''}
            
            <!-- Clinical Findings Section -->
            {self._generate_clinical_findings_section(pods_data.clinical_findings) if self._has_clinical_findings(pods_data.clinical_findings) else ''}
            
            <!-- Patient Demographics Section -->
            {self._generate_demographics_section(pods_data.patient_demographics) if self._has_demographics(pods_data.patient_demographics) else ''}
            
            <!-- Clinical Decisions Section -->
            {self._generate_clinical_decisions_section(pods_data.clinical_decisions) if self._has_clinical_decisions(pods_data.clinical_decisions) else ''}
            
            <!-- Medications Section -->
            {self._generate_medications_section(pods_data) if pods_data.discharge_medications or pods_data.medication_changes else ''}
            
            <!-- Appointments and Follow-up Section -->
            {self._generate_appointments_section(pods_data.detailed_appointments) if self._has_appointments(pods_data.detailed_appointments) else ''}
            
            <!-- Care Instructions Section -->
            {self._generate_care_instructions_section(pods_data) if self._has_care_instructions(pods_data) else ''}
            
            <!-- Warning Signs and Emergency Contacts -->
            {self._generate_safety_section(pods_data) if pods_data.red_flag_symptoms or pods_data.emergency_contacts else ''}
            
            <!-- Healthcare Team Section -->
            {self._generate_healthcare_team_section(pods_data.healthcare_team) if pods_data.healthcare_team else ''}
            
            <!-- Medical Images Analysis -->
            {self._generate_images_section(pods_data.extracted_images) if pods_data.extracted_images else ''}
            
            <!-- Original Content Section -->
            <div class="section">
                <div class="section-header collapsible">
                    <i class="fas fa-code"></i>
                    Original Content Reference
                    <i class="fas fa-chevron-down" style="margin-left: auto;"></i>
                </div>
                <div class="section-content content-collapsed">
                    <div style="max-height: 400px; overflow-y: auto; background: #3A3F4B; padding: 15px; border-radius: 5px;">
                        <pre style="white-space: pre-wrap; font-family: monospace; color: #C0C0C0;">{pods_data.original_content[:10000]}</pre>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <button class="print-button" onclick="window.print()">
        <i class="fas fa-print"></i> Print Report
    </button>
    
    <script>
        // Collapsible sections
        document.querySelectorAll('.section-header').forEach(header => {{
            header.addEventListener('click', () => {{
                const content = header.nextElementSibling;
                const icon = header.querySelector('i:last-child');
                
                content.classList.toggle('content-collapsed');
                icon.classList.toggle('fa-chevron-down');
                icon.classList.toggle('fa-chevron-up');
            }});
        }});
        
        // Smooth scrolling for internal links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {{
            anchor.addEventListener('click', function (e) {{
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({{
                    behavior: 'smooth'
                }});
            }});
        }});
    </script>
</body>
</html>
        """
        
        return html_content
    
    def _has_clinical_findings(self, findings: ClinicalFindings) -> bool:
        return bool(findings.symptoms or findings.vital_signs or findings.laboratory_results or 
                   findings.imaging_findings or findings.physical_exam)
    
    def _has_demographics(self, demographics: PatientDemographics) -> bool:
        return bool(demographics.medical_history or demographics.allergies or 
                   demographics.current_medications or demographics.social_history)
    
    def _has_clinical_decisions(self, decisions: ClinicalDecisions) -> bool:
        return bool(decisions.diagnosis or decisions.treatment_rationale or 
                   decisions.risk_factors or decisions.prognosis)
    
    def _has_appointments(self, appointments: DetailedAppointments) -> bool:
        return bool(appointments.scheduled_visits or appointments.urgent_care_indicators or 
                   appointments.monitoring_schedule)
    
    def _has_care_instructions(self, pods_data: ComprehensivePODS) -> bool:
        return bool(pods_data.activity_restrictions or pods_data.dietary_modifications or 
                   pods_data.wound_care)
    
    def _generate_clinical_findings_section(self, findings: ClinicalFindings) -> str:
        return f'''
            <div class="section">
                <div class="section-header clinical collapsible">
                    <i class="fas fa-microscope"></i>
                    Clinical Findings & Assessment
                    <i class="fas fa-chevron-down" style="margin-left: auto;"></i>
                </div>
                <div class="section-content">
                    <div class="clinical-grid">
                        {self._generate_symptoms_subsection(findings.symptoms)}
                        {self._generate_vital_signs_subsection(findings.vital_signs)}
                        {self._generate_lab_results_subsection(findings.laboratory_results)}
                        {self._generate_imaging_subsection(findings.imaging_findings)}
                        {self._generate_physical_exam_subsection(findings.physical_exam)}
                    </div>
                </div>
            </div>
        '''
    
    def _generate_symptoms_subsection(self, symptoms: List[Dict[str, str]]) -> str:
        if not symptoms:
            return ""
        
        symptoms_html = ""
        for symptom in symptoms:
            severity_class = self._get_severity_class(symptom.get('severity', ''))
            symptoms_html += f'''
                <div class="finding-item {severity_class}">
                    <div class="finding-label">{symptom.get('symptom', 'Unknown symptom')}</div>
                    <div class="finding-details">
                        <strong>Severity:</strong> {symptom.get('severity', 'Not specified')}<br>
                        <strong>Onset:</strong> {symptom.get('onset', 'Not specified')}<br>
                        <strong>Duration:</strong> {symptom.get('duration', 'Not specified')}<br>
                        {f"<strong>Characteristics:</strong> {symptom.get('characteristics', '')}<br>" if symptom.get('characteristics') else ''}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="clinical-subsection">
                <h4><i class="fas fa-thermometer-half"></i> Symptoms</h4>
                {symptoms_html}
            </div>
        '''
    
    def _generate_vital_signs_subsection(self, vital_signs: List[Dict[str, str]]) -> str:
        if not vital_signs:
            return ""
        
        vitals_html = ""
        for vital in vital_signs:
            vitals_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{vital.get('parameter', 'Unknown parameter')}</div>
                    <div class="finding-details">
                        <strong>Value:</strong> {vital.get('value', 'Not recorded')}<br>
                        <strong>Normal Range:</strong> {vital.get('normal_range', 'Not specified')}<br>
                        <strong>Significance:</strong> {vital.get('significance', 'Not specified')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="clinical-subsection">
                <h4><i class="fas fa-heartbeat"></i> Vital Signs</h4>
                {vitals_html}
            </div>
        '''
    
    def _generate_lab_results_subsection(self, lab_results: List[Dict[str, str]]) -> str:
        if not lab_results:
            return ""
        
        labs_html = ""
        for lab in lab_results:
            labs_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{lab.get('test', 'Unknown test')}</div>
                    <div class="finding-details">
                        <strong>Result:</strong> {lab.get('value', 'Not available')}<br>
                        <strong>Reference Range:</strong> {lab.get('reference', 'Not specified')}<br>
                        <strong>Interpretation:</strong> {lab.get('interpretation', 'Not provided')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="clinical-subsection">
                <h4><i class="fas fa-flask"></i> Laboratory Results</h4>
                {labs_html}
            </div>
        '''
    
    def _generate_imaging_subsection(self, imaging: List[Dict[str, str]]) -> str:
        if not imaging:
            return ""
        
        imaging_html = ""
        for img in imaging:
            imaging_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{img.get('type', 'Unknown imaging')}</div>
                    <div class="finding-details">
                        <strong>Findings:</strong> {img.get('findings', 'Not specified')}<br>
                        <strong>Impression:</strong> {img.get('impression', 'Not provided')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="clinical-subsection">
                <h4><i class="fas fa-x-ray"></i> Imaging Studies</h4>
                {imaging_html}
            </div>
        '''
    
    def _generate_physical_exam_subsection(self, physical_exam: List[Dict[str, str]]) -> str:
        if not physical_exam:
            return ""
        
        exam_html = ""
        for exam in physical_exam:
            exam_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{exam.get('system', 'Unknown system')}</div>
                    <div class="finding-details">
                        <strong>Findings:</strong> {exam.get('findings', 'Not documented')}<br>
                        <strong>Significance:</strong> {exam.get('significance', 'Not specified')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="clinical-subsection">
                <h4><i class="fas fa-user-md"></i> Physical Examination</h4>
                {exam_html}
            </div>
        '''
    
    def _generate_demographics_section(self, demographics: PatientDemographics) -> str:
        return f'''
            <div class="section">
                <div class="section-header demographics collapsible">
                    <i class="fas fa-user-circle"></i>
                    Patient Demographics & History
                    <i class="fas fa-chevron-down" style="margin-left: auto;"></i>
                </div>
                <div class="section-content">
                    <div class="clinical-grid">
                        {self._generate_list_subsection("Medical History", demographics.medical_history, "fas fa-history")}
                        {self._generate_list_subsection("Allergies", demographics.allergies, "fas fa-exclamation-triangle")}
                        {self._generate_list_subsection("Current Medications", demographics.current_medications, "fas fa-pills")}
                        {self._generate_list_subsection("Social History", demographics.social_history, "fas fa-users")}
                    </div>
                </div>
            </div>
        '''
    
    def _generate_clinical_decisions_section(self, decisions: ClinicalDecisions) -> str:
        return f'''
            <div class="section">
                <div class="section-header decisions collapsible">
                    <i class="fas fa-brain"></i>
                    Clinical Decision Making
                    <i class="fas fa-chevron-down" style="margin-left: auto;"></i>
                </div>
                <div class="section-content">
                    <div class="clinical-grid">
                        {self._generate_diagnosis_subsection(decisions.diagnosis)}
                        {self._generate_treatment_rationale_subsection(decisions.treatment_rationale)}
                        {self._generate_risk_factors_subsection(decisions.risk_factors)}
                        {self._generate_prognosis_subsection(decisions.prognosis)}
                    </div>
                </div>
            </div>
        '''
    
    def _generate_medications_section(self, pods_data: ComprehensivePODS) -> str:
        medications_html = ""
        
        for med in pods_data.discharge_medications:
            medications_html += f'''
                <div class="medication-card">
                    <div class="medication-name">{med.get('name', 'Unknown medication')}</div>
                    <div class="medication-grid">
                        <div class="medication-detail">
                            <strong>Dose:</strong><br>{med.get('dose', 'Not specified')}
                        </div>
                        <div class="medication-detail">
                            <strong>Frequency:</strong><br>{med.get('frequency', 'Not specified')}
                        </div>
                        <div class="medication-detail">
                            <strong>Duration:</strong><br>{med.get('duration', 'Not specified')}
                        </div>
                        <div class="medication-detail">
                            <strong>Purpose:</strong><br>{med.get('purpose', 'Not specified')}
                        </div>
                    </div>
                    {f'<div class="warning-box" style="margin-top: 15px;"><strong>Side Effects:</strong> {med.get("side_effects")}</div>' if med.get('side_effects') else ''}
                    {f'<div class="medication-detail" style="margin-top: 10px;"><strong>Special Instructions:</strong> {med.get("special_instructions")}</div>' if med.get('special_instructions') else ''}
                </div>
            '''
        
        changes_html = ""
        for change in pods_data.medication_changes:
            changes_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{change.get('type', 'Unknown change').title()}: {change.get('medication', 'Unknown medication')}</div>
                    <div class="finding-details">
                        <strong>Change:</strong> {change.get('change', 'Not specified')}<br>
                        <strong>Reason:</strong> {change.get('reason', 'Not provided')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="section">
                <div class="section-header medications collapsible">
                    <i class="fas fa-prescription-bottle-alt"></i>
                    Medications & Changes
                    <i class="fas fa-chevron-down" style="margin-left: auto;"></i>
                </div>
                <div class="section-content">
                    {f'<h3 style="margin-bottom: 20px;"><i class="fas fa-pills"></i> Discharge Medications</h3>{medications_html}' if medications_html else ''}
                    {f'<h3 style="margin-bottom: 20px; margin-top: 30px;"><i class="fas fa-exchange-alt"></i> Medication Changes</h3>{changes_html}' if changes_html else ''}
                </div>
            </div>
        '''
    
    def _generate_appointments_section(self, appointments: DetailedAppointments) -> str:
        visits_html = ""
        for visit in appointments.scheduled_visits:
            visits_html += f'''
                <div class="appointment-item">
                    <h4><i class="fas fa-calendar-check"></i> {visit.get('type', 'Appointment')}</h4>
                    <p><strong>Date & Time:</strong> {visit.get('date', 'TBD')} at {visit.get('time', 'TBD')}</p>
                    <p><strong>Location:</strong> {visit.get('location', 'Not specified')}</p>
                    <p><strong>Provider:</strong> {visit.get('provider', 'Not specified')}</p>
                    <p><strong>Purpose:</strong> {visit.get('purpose', 'Not specified')}</p>
                </div>
            '''
        
        urgent_html = ""
        for urgent in appointments.urgent_care_indicators:
            urgent_html += f'''
                <div class="emergency-box">
                    <div class="urgent-indicator">URGENT CARE NEEDED</div>
                    <p><strong>Scenario:</strong> {urgent.get('scenario', 'Not specified')}</p>
                    <p><strong>Action:</strong> {urgent.get('action', 'Contact healthcare provider')}</p>
                    <p><strong>Timeframe:</strong> {urgent.get('timeframe', 'Immediately')}</p>
                </div>
            '''
        
        monitoring_html = ""
        for monitor in appointments.monitoring_schedule:
            monitoring_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{monitor.get('parameter', 'Unknown parameter')}</div>
                    <div class="finding-details">
                        <strong>Frequency:</strong> {monitor.get('frequency', 'Not specified')}<br>
                        <strong>Method:</strong> {monitor.get('method', 'Not specified')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="section">
                <div class="section-header appointments collapsible">
                    <i class="fas fa-calendar-alt"></i>
                    Appointments & Follow-up Care
                    <i class="fas fa-chevron-down" style="margin-left: auto;"></i>
                </div>
                <div class="section-content">
                    {f'<div class="appointment-timeline">{visits_html}</div>' if visits_html else ''}
                    {urgent_html}
                    {f'<h3 style="margin: 30px 0 20px 0;"><i class="fas fa-chart-line"></i> Monitoring Schedule</h3>{monitoring_html}' if monitoring_html else ''}
                </div>
            </div>
        '''
    
    def _generate_care_instructions_section(self, pods_data: ComprehensivePODS) -> str:
        activities_html = ""
        for activity in pods_data.activity_restrictions:
            activities_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{activity.get('activity', 'Unknown activity')}</div>
                    <div class="finding-details">
                        <strong>Restriction:</strong> {activity.get('restriction', 'Not specified')}<br>
                        <strong>Duration:</strong> {activity.get('duration', 'Not specified')}<br>
                        <strong>Reason:</strong> {activity.get('reason', 'Not provided')}
                    </div>
                </div>
            '''
        
        diet_html = ""
        for diet in pods_data.dietary_modifications:
            diet_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{diet.get('modification', 'Dietary change')}</div>
                    <div class="finding-details">
                        <strong>Reason:</strong> {diet.get('reason', 'Not specified')}<br>
                        <strong>Duration:</strong> {diet.get('duration', 'Not specified')}<br>
                        {f"<strong>Specifics:</strong> {diet.get('specifics')}<br>" if diet.get('specifics') else ''}
                    </div>
                </div>
            '''
        
        wound_html = ""
        for wound in pods_data.wound_care:
            wound_html += f'''
                <div class="finding-item">
                    <div class="finding-label">Wound Care - {wound.get('site', 'Unknown location')}</div>
                    <div class="finding-details">
                        <strong>Instructions:</strong> {wound.get('instructions', 'Not specified')}<br>
                        <strong>Frequency:</strong> {wound.get('frequency', 'Not specified')}<br>
                        <strong>Signs to Watch:</strong> {wound.get('signs_to_watch', 'Not specified')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="section">
                <div class="section-header collapsible" style="border-left: 5px solid #9ABF88;">
                    <i class="fas fa-tasks"></i>
                    Care Instructions
                    <i class="fas fa-chevron-down" style="margin-left: auto;"></i>
                </div>
                <div class="section-content">
                    <div class="clinical-grid">
                        {f'<div class="clinical-subsection"><h4><i class="fas fa-running"></i> Activity Restrictions</h4>{activities_html}</div>' if activities_html else ''}
                        {f'<div class="clinical-subsection"><h4><i class="fas fa-utensils"></i> Dietary Modifications</h4>{diet_html}</div>' if diet_html else ''}
                        {f'<div class="clinical-subsection"><h4><i class="fas fa-band-aid"></i> Wound Care</h4>{wound_html}</div>' if wound_html else ''}
                    </div>
                </div>
            </div>
        '''
    
    def _generate_safety_section(self, pods_data: ComprehensivePODS) -> str:
        red_flags_html = ""
        for symptom in pods_data.red_flag_symptoms:
            severity_class = self._get_severity_class(symptom.get('severity', 'high'))
            red_flags_html += f'''
                <div class="emergency-box">
                    <div class="urgent-indicator">⚠️ WARNING SIGN</div>
                    <p><strong>Symptom:</strong> {symptom.get('symptom', 'Not specified')}</p>
                    <p><strong>Severity:</strong> {symptom.get('severity', 'High concern')}</p>
                    <p><strong>Action:</strong> {symptom.get('action', 'Seek immediate medical attention')}</p>
                    <p><strong>Timeframe:</strong> {symptom.get('timeframe', 'Immediately')}</p>
                </div>
            '''
        
        contacts_html = ""
        for contact in pods_data.emergency_contacts:
            contacts_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{contact.get('type', 'Emergency Contact')}</div>
                    <div class="finding-details">
                        <strong>Contact:</strong> {contact.get('contact', 'Not provided')}<br>
                        <strong>When to Call:</strong> {contact.get('when_to_call', 'In case of emergency')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="section">
                <div class="section-header collapsible" style="border-left: 5px solid #FF6B6B;">
                    <i class="fas fa-exclamation-triangle"></i>
                    Warning Signs & Emergency Contacts
                    <i class="fas fa-chevron-down" style="margin-left: auto;"></i>
                </div>
                <div class="section-content">
                    {red_flags_html}
                    {f'<h3 style="margin: 30px 0 20px 0;"><i class="fas fa-phone-alt"></i> Emergency Contacts</h3>{contacts_html}' if contacts_html else ''}
                </div>
            </div>
        '''
    
    def _generate_healthcare_team_section(self, healthcare_team: List[Dict[str, str]]) -> str:
        team_html = ""
        for member in healthcare_team:
            team_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{member.get('role', 'Healthcare Provider')} - {member.get('name', 'Not specified')}</div>
                    <div class="finding-details">
                        <strong>Specialty:</strong> {member.get('specialty', 'Not specified')}<br>
                        <strong>Contact:</strong> {member.get('contact', 'Not provided')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="section">
                <div class="section-header collapsible" style="border-left: 5px solid #6B8E9B;">
                    <i class="fas fa-user-friends"></i>
                    Healthcare Team
                    <i class="fas fa-chevron-down" style="margin-left: auto;"></i>
                </div>
                <div class="section-content">
                    {team_html}
                </div>
            </div>
        '''
    
    def _generate_images_section(self, images: List[Dict[str, str]]) -> str:
        images_html = ""
        for img in images:
            images_html += f'''
                <div class="image-analysis">
                    <img src="{img['url']}" alt="Medical image">
                    <h4>Image Analysis</h4>
                    <div class="finding-details">
                        {img['analysis'].replace('\n', '<br>')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="section">
                <div class="section-header collapsible" style="border-left: 5px solid #8FBC8F;">
                    <i class="fas fa-images"></i>
                    Medical Image Analysis
                    <i class="fas fa-chevron-down" style="margin-left: auto;"></i>
                </div>
                <div class="section-content">
                    <div class="image-gallery">
                        {images_html}
                    </div>
                </div>
            </div>
        '''
    
    def _generate_list_subsection(self, title: str, items: List[str], icon: str) -> str:
        if not items:
            return ""
        
        items_html = "".join([f"<li>{item}</li>" for item in items])
        return f'''
            <div class="clinical-subsection">
                <h4><i class="{icon}"></i> {title}</h4>
                <ul>{items_html}</ul>
            </div>
        '''
    
    def _generate_diagnosis_subsection(self, diagnoses: List[Dict[str, str]]) -> str:
        if not diagnoses:
            return ""
        
        diagnosis_html = ""
        for diag in diagnoses:
            diagnosis_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{diag.get('condition', 'Unknown condition')}</div>
                    <div class="finding-details">
                        <strong>Certainty:</strong> {diag.get('certainty', 'Not specified')}<br>
                        <strong>Evidence:</strong> {diag.get('evidence', 'Not provided')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="clinical-subsection">
                <h4><i class="fas fa-diagnoses"></i> Diagnosis</h4>
                {diagnosis_html}
            </div>
        '''
    
    def _generate_treatment_rationale_subsection(self, treatments: List[Dict[str, str]]) -> str:
        if not treatments:
            return ""
        
        treatment_html = ""
        for tx in treatments:
            treatment_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{tx.get('decision', 'Treatment decision')}</div>
                    <div class="finding-details">
                        <strong>Reasoning:</strong> {tx.get('reasoning', 'Not specified')}<br>
                        <strong>Evidence:</strong> {tx.get('evidence', 'Not provided')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="clinical-subsection">
                <h4><i class="fas fa-procedures"></i> Treatment Rationale</h4>
                {treatment_html}
            </div>
        '''
    
    def _generate_risk_factors_subsection(self, risks: List[Dict[str, str]]) -> str:
        if not risks:
            return ""
        
        risk_html = ""
        for risk in risks:
            severity_class = self._get_severity_class(risk.get('severity', ''))
            risk_html += f'''
                <div class="finding-item {severity_class}">
                    <div class="finding-label">{risk.get('factor', 'Risk factor')}</div>
                    <div class="finding-details">
                        <strong>Severity:</strong> {risk.get('severity', 'Not specified')}<br>
                        <strong>Mitigation:</strong> {risk.get('mitigation', 'Not specified')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="clinical-subsection">
                <h4><i class="fas fa-exclamation-circle"></i> Risk Factors</h4>
                {risk_html}
            </div>
        '''
    
    def _generate_prognosis_subsection(self, prognoses: List[Dict[str, str]]) -> str:
        if not prognoses:
            return ""
        
        prognosis_html = ""
        for prog in prognoses:
            prognosis_html += f'''
                <div class="finding-item">
                    <div class="finding-label">{prog.get('outlook', 'Prognosis')}</div>
                    <div class="finding-details">
                        <strong>Timeframe:</strong> {prog.get('timeframe', 'Not specified')}<br>
                        <strong>Influencing Factors:</strong> {prog.get('factors', 'Not specified')}
                    </div>
                </div>
            '''
        
        return f'''
            <div class="clinical-subsection">
                <h4><i class="fas fa-chart-line"></i> Prognosis</h4>
                {prognosis_html}
            </div>
        '''
    
    def _get_severity_class(self, severity: str) -> str:
        severity = severity.lower()
        if 'high' in severity or 'severe' in severity or 'critical' in severity:
            return "severity-high"
        elif 'moderate' in severity or 'medium' in severity:
            return "severity-moderate"
        elif 'low' in severity or 'mild' in severity:
            return "severity-low"
        return ""

class PODSGenerator:
    def __init__(self):
        self.nebius_client = AdvancedNebiusAIClient(API_KEY, BASE_URL)
        self.website_parser = ComprehensiveWebsiteParser(self.nebius_client)
        self.html_generator = AdvancedPODSHTMLGenerator()
    
    def generate_pods_from_url(self, url: str) -> str:
        """Generate comprehensive PODS from a given URL"""
        print(f"Fetching content from: {url}")
        content, image_urls = self.website_parser.fetch_website_content(url)
        
        if not content:
            return "<h1>Error: Could not fetch content from the provided URL</h1>"
        
        print("Extracting medical data...")
        pods_data = self.website_parser.extract_comprehensive_medical_data(content, url)
        
        print("Analyzing medical images...")
        pods_data.extracted_images = self.website_parser.analyze_medical_images(image_urls)
        
        print("Generating HTML report...")
        html_report = self.html_generator.generate_comprehensive_html_report(pods_data)
        
        return html_report

# Example usage
if __name__ == "__main__":
    generator = PODSGenerator()
    url = "https://www.medmalreviewer.com/sob-firstpost/"  # Replace with actual medical content URL
    html_report = generator.generate_pods_from_url(url)
    
    # Save to file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pods_report_{timestamp}.html"
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_report)
    print(f"PODS report generated successfully: {filename}")
    print("Open the file in your web browser to view the interactive report.")