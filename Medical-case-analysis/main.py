import modal
import os
import json
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random
from dataclasses import dataclass, asdict
from openai import OpenAI
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import base64
from io import BytesIO
from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import uvicorn

# Modal app setup
app = modal.App("medical-case-agent")

# Define the Modal image with required dependencies
image = modal.Image.debian_slim().pip_install([
    "openai>=1.0.0",
    "numpy>=1.21.0",
    "pandas>=1.3.0",
    "scikit-learn>=1.0.0",
    "plotly>=5.0.0",
    "fastapi>=0.68.0",
    "uvicorn>=0.15.0",
    "jinja2>=3.0.0",
    "python-multipart>=0.0.5"
])

@dataclass
class PatientCase:
    """Patient case data structure"""
    case_id: str
    age: int
    gender: str
    chief_complaint: str
    symptoms: List[str]
    diagnosis: str
    treatment: str
    outcome: str
    duration_days: int
    severity: str
    comorbidities: List[str]
    lab_values: Dict[str, float]
    admission_date: str
    discharge_date: str
    cost: float
    
    def to_text(self) -> str:
        """Convert case to searchable text"""
        return f"""
        Patient: {self.age}yo {self.gender}
        Chief Complaint: {self.chief_complaint}
        Symptoms: {', '.join(self.symptoms)}
        Diagnosis: {self.diagnosis}
        Treatment: {self.treatment}
        Outcome: {self.outcome}
        Severity: {self.severity}
        Comorbidities: {', '.join(self.comorbidities)}
        Duration: {self.duration_days} days
        """

class SyntheticDataGenerator:
    """Generate synthetic patient case data"""
    
    def __init__(self):
        self.diseases = [
            "Pneumonia", "Diabetes Type 2", "Hypertension", "Heart Failure",
            "COPD", "Stroke", "Myocardial Infarction", "Sepsis", "Kidney Disease",
            "Cancer - Lung", "Cancer - Breast", "Appendicitis", "Gallstones",
            "Asthma", "Depression", "Anxiety", "Arthritis", "Osteoporosis"
        ]
        
        self.symptoms_map = {
            "Pneumonia": ["cough", "fever", "shortness of breath", "chest pain", "fatigue"],
            "Diabetes Type 2": ["increased thirst", "frequent urination", "blurred vision", "fatigue", "slow healing"],
            "Hypertension": ["headache", "dizziness", "chest pain", "shortness of breath", "nosebleeds"],
            "Heart Failure": ["shortness of breath", "fatigue", "swelling", "rapid heartbeat", "cough"],
            "COPD": ["chronic cough", "shortness of breath", "wheezing", "chest tightness", "fatigue"],
            "Stroke": ["sudden weakness", "confusion", "difficulty speaking", "severe headache", "vision problems"],
            "Myocardial Infarction": ["chest pain", "shortness of breath", "nausea", "sweating", "fatigue"],
            "Sepsis": ["fever", "rapid heart rate", "difficulty breathing", "confusion", "extreme pain"],
            "Kidney Disease": ["fatigue", "swelling", "changes in urination", "nausea", "muscle cramps"],
            "Cancer - Lung": ["persistent cough", "chest pain", "shortness of breath", "weight loss", "fatigue"],
            "Cancer - Breast": ["breast lump", "breast pain", "skin changes", "nipple discharge", "fatigue"],
            "Appendicitis": ["abdominal pain", "nausea", "vomiting", "fever", "loss of appetite"],
            "Gallstones": ["abdominal pain", "nausea", "vomiting", "fever", "jaundice"],
            "Asthma": ["wheezing", "shortness of breath", "chest tightness", "coughing", "fatigue"],
            "Depression": ["persistent sadness", "loss of interest", "fatigue", "sleep changes", "appetite changes"],
            "Anxiety": ["excessive worry", "restlessness", "fatigue", "difficulty concentrating", "muscle tension"],
            "Arthritis": ["joint pain", "stiffness", "swelling", "reduced range of motion", "fatigue"],
            "Osteoporosis": ["back pain", "loss of height", "stooped posture", "bone fractures", "joint pain"]
        }
        
        self.treatments_map = {
            "Pneumonia": ["Antibiotics", "Oxygen therapy", "Rest", "Fluid management"],
            "Diabetes Type 2": ["Metformin", "Insulin", "Diet modification", "Exercise program"],
            "Hypertension": ["ACE inhibitors", "Beta blockers", "Diuretics", "Lifestyle changes"],
            "Heart Failure": ["ACE inhibitors", "Beta blockers", "Diuretics", "Cardiac monitoring"],
            "COPD": ["Bronchodilators", "Corticosteroids", "Oxygen therapy", "Pulmonary rehabilitation"],
            "Stroke": ["Anticoagulants", "Physical therapy", "Speech therapy", "Occupational therapy"],
            "Myocardial Infarction": ["Antiplatelet therapy", "Beta blockers", "Statins", "Cardiac catheterization"],
            "Sepsis": ["IV antibiotics", "Fluid resuscitation", "Vasopressors", "Supportive care"],
            "Kidney Disease": ["ACE inhibitors", "Diuretics", "Dialysis", "Diet modification"],
            "Cancer - Lung": ["Chemotherapy", "Radiation therapy", "Surgery", "Targeted therapy"],
            "Cancer - Breast": ["Surgery", "Chemotherapy", "Radiation therapy", "Hormone therapy"],
            "Appendicitis": ["Appendectomy", "Antibiotics", "Pain management", "IV fluids"],
            "Gallstones": ["Cholecystectomy", "Pain management", "Dietary changes", "Medications"],
            "Asthma": ["Bronchodilators", "Corticosteroids", "Allergy management", "Education"],
            "Depression": ["Antidepressants", "Psychotherapy", "Lifestyle changes", "Support groups"],
            "Anxiety": ["Anxiolytics", "Cognitive behavioral therapy", "Relaxation techniques", "Lifestyle changes"],
            "Arthritis": ["NSAIDs", "Physical therapy", "Exercise", "Joint injections"],
            "Osteoporosis": ["Bisphosphonates", "Calcium supplements", "Vitamin D", "Exercise"]
        }
        
        self.comorbidities = [
            "Diabetes", "Hypertension", "Heart Disease", "COPD", "Kidney Disease",
            "Depression", "Anxiety", "Arthritis", "Obesity", "Sleep Apnea"
        ]
        
    def generate_case(self, case_id: str) -> PatientCase:
        """Generate a single synthetic patient case"""
        age = random.randint(18, 95)
        gender = random.choice(["Male", "Female"])
        diagnosis = random.choice(self.diseases)
        
        symptoms = random.sample(
            self.symptoms_map.get(diagnosis, ["fatigue", "pain"]), 
            random.randint(2, 4)
        )
        
        treatment = random.choice(self.treatments_map.get(diagnosis, ["Supportive care"]))
        
        severity = random.choice(["Mild", "Moderate", "Severe"])
        outcome = random.choices(
            ["Full Recovery", "Partial Recovery", "Stable", "Deteriorated", "Deceased"],
            weights=[50, 25, 15, 8, 2]
        )[0]
        
        duration = random.randint(1, 30) if severity == "Mild" else random.randint(3, 60)
        
        # Generate comorbidities based on age
        num_comorbidities = 0 if age < 40 else random.randint(0, 3)
        patient_comorbidities = random.sample(self.comorbidities, num_comorbidities)
        
        # Generate lab values
        lab_values = {
            "hemoglobin": round(random.uniform(10.0, 16.0), 1),
            "white_blood_cells": round(random.uniform(4.0, 12.0), 1),
            "glucose": round(random.uniform(70, 200), 0),
            "creatinine": round(random.uniform(0.6, 2.0), 2),
            "sodium": round(random.uniform(135, 145), 0)
        }
        
        admission_date = datetime.now() - timedelta(days=random.randint(1, 365))
        discharge_date = admission_date + timedelta(days=duration)
        
        cost = random.uniform(5000, 50000) * (1.5 if severity == "Severe" else 1)
        
        return PatientCase(
            case_id=case_id,
            age=age,
            gender=gender,
            chief_complaint=f"Patient presents with {symptoms[0]}",
            symptoms=symptoms,
            diagnosis=diagnosis,
            treatment=treatment,
            outcome=outcome,
            duration_days=duration,
            severity=severity,
            comorbidities=patient_comorbidities,
            lab_values=lab_values,
            admission_date=admission_date.strftime("%Y-%m-%d"),
            discharge_date=discharge_date.strftime("%Y-%m-%d"),
            cost=round(cost, 2)
        )
    
    def generate_dataset(self, num_cases: int = 1000) -> List[PatientCase]:
        """Generate a dataset of synthetic patient cases"""
        return [self.generate_case(f"CASE_{str(i).zfill(4)}") 
                for i in range(1, num_cases + 1)]

class EmbeddingService:
    """Service for generating embeddings using Nebius API"""
    
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=api_key
        )
        self.model = "intfloat/e5-mistral-7b-instruct"
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text"""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return [0.0] * 4096  # Return zero vector as fallback
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple texts"""
        embeddings = []
        for text in texts:
            embedding = self.get_embedding(text)
            embeddings.append(embedding)
        return embeddings

class CaseSimilarityAnalyzer:
    """Analyze similarity between patient cases"""
    
    def __init__(self, embedding_service: EmbeddingService):
        self.embedding_service = embedding_service
        self.case_embeddings = {}
        self.cases = []
    
    def load_cases(self, cases: List[PatientCase]):
        """Load and embed patient cases"""
        self.cases = cases
        texts = [case.to_text() for case in cases]
        
        print(f"Generating embeddings for {len(cases)} cases...")
        embeddings = self.embedding_service.get_embeddings_batch(texts)
        
        for case, embedding in zip(cases, embeddings):
            self.case_embeddings[case.case_id] = embedding
    
    def find_similar_cases(self, new_case: PatientCase, top_k: int = 10) -> List[Dict[str, Any]]:
        """Find similar cases to a new patient case"""
        new_embedding = self.embedding_service.get_embedding(new_case.to_text())
        
        similarities = []
        for case in self.cases:
            if case.case_id in self.case_embeddings:
                similarity = cosine_similarity(
                    [new_embedding], 
                    [self.case_embeddings[case.case_id]]
                )[0][0]
                
                similarities.append({
                    'case': case,
                    'similarity': similarity
                })
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

class VisualizationGenerator:
    """Generate interactive visualizations for case analysis"""
    
    def create_similarity_chart(self, similar_cases: List[Dict[str, Any]]) -> str:
        """Create similarity score chart"""
        cases = [item['case'] for item in similar_cases]
        similarities = [item['similarity'] for item in similar_cases]
        case_ids = [case.case_id for case in cases]
        
        fig = go.Figure(data=go.Bar(
            x=case_ids,
            y=similarities,
            marker_color='lightblue',
            text=[f"{s:.3f}" for s in similarities],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Case Similarity Scores",
            xaxis_title="Case ID",
            yaxis_title="Similarity Score",
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def create_outcome_distribution(self, similar_cases: List[Dict[str, Any]]) -> str:
        """Create outcome distribution pie chart"""
        cases = [item['case'] for item in similar_cases]
        outcomes = [case.outcome for case in cases]
        
        outcome_counts = pd.Series(outcomes).value_counts()
        
        fig = go.Figure(data=go.Pie(
            labels=outcome_counts.index,
            values=outcome_counts.values,
            hole=0.3
        ))
        
        fig.update_layout(
            title="Outcome Distribution in Similar Cases",
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def create_treatment_analysis(self, similar_cases: List[Dict[str, Any]]) -> str:
        """Create treatment effectiveness analysis"""
        cases = [item['case'] for item in similar_cases]
        
        treatment_outcomes = {}
        for case in cases:
            if case.treatment not in treatment_outcomes:
                treatment_outcomes[case.treatment] = []
            treatment_outcomes[case.treatment].append(case.outcome)
        
        treatments = list(treatment_outcomes.keys())
        success_rates = []
        
        for treatment in treatments:
            outcomes = treatment_outcomes[treatment]
            success_count = sum(1 for outcome in outcomes 
                              if outcome in ['Full Recovery', 'Partial Recovery', 'Stable'])
            success_rate = (success_count / len(outcomes)) * 100
            success_rates.append(success_rate)
        
        fig = go.Figure(data=go.Bar(
            x=treatments,
            y=success_rates,
            marker_color='lightgreen',
            text=[f"{r:.1f}%" for r in success_rates],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Treatment Success Rates in Similar Cases",
            xaxis_title="Treatment",
            yaxis_title="Success Rate (%)",
            height=400
        )
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')
    
    def create_demographics_chart(self, similar_cases: List[Dict[str, Any]]) -> str:
        """Create demographics analysis"""
        cases = [item['case'] for item in similar_cases]
        
        ages = [case.age for case in cases]
        genders = [case.gender for case in cases]
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Age Distribution', 'Gender Distribution'),
            specs=[[{'type': 'histogram'}, {'type': 'pie'}]]
        )
        
        # Age histogram
        fig.add_trace(
            go.Histogram(x=ages, nbinsx=10, name="Age", marker_color='lightcoral'),
            row=1, col=1
        )
        
        # Gender pie chart
        gender_counts = pd.Series(genders).value_counts()
        fig.add_trace(
            go.Pie(labels=gender_counts.index, values=gender_counts.values, name="Gender"),
            row=1, col=2
        )
        
        fig.update_layout(height=400, title_text="Demographics of Similar Cases")
        
        return fig.to_html(full_html=False, include_plotlyjs='cdn')

class ReportGenerator:
    """Generate comprehensive HTML reports"""
    
    def __init__(self, viz_generator: VisualizationGenerator):
        self.viz_generator = viz_generator
    
    def generate_case_analysis_report(self, 
                                   new_case: PatientCase, 
                                   similar_cases: List[Dict[str, Any]]) -> str:
        """Generate comprehensive case analysis report"""
        
        # Generate visualizations
        similarity_chart = self.viz_generator.create_similarity_chart(similar_cases)
        outcome_chart = self.viz_generator.create_outcome_distribution(similar_cases)
        treatment_chart = self.viz_generator.create_treatment_analysis(similar_cases)
        demographics_chart = self.viz_generator.create_demographics_chart(similar_cases)
        
        # Calculate key insights
        cases = [item['case'] for item in similar_cases]
        avg_similarity = np.mean([item['similarity'] for item in similar_cases])
        
        outcomes = [case.outcome for case in cases]
        most_common_outcome = pd.Series(outcomes).mode()[0] if outcomes else "Unknown"
        
        treatments = [case.treatment for case in cases]
        most_common_treatment = pd.Series(treatments).mode()[0] if treatments else "Unknown"
        
        avg_duration = np.mean([case.duration_days for case in cases])
        avg_cost = np.mean([case.cost for case in cases])
        
        # Generate reasoning
        reasoning = self._generate_reasoning(new_case, similar_cases, {
            'avg_similarity': avg_similarity,
            'most_common_outcome': most_common_outcome,
            'most_common_treatment': most_common_treatment,
            'avg_duration': avg_duration,
            'avg_cost': avg_cost
        })
        
        html_template = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Medical Case Analysis Report</title>
            <style>
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 20px rgba(0,0,0,0.1);
                }}
                .header {{
                    text-align: center;
                    color: #2c3e50;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 20px;
                    margin-bottom: 30px;
                }}
                .case-summary {{
                    background-color: #ecf0f1;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
                .insights {{
                    background-color: #e8f5e8;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
                .reasoning {{
                    background-color: #fff3cd;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
                .chart-container {{
                    margin: 30px 0;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 8px;
                }}
                .metric {{
                    display: inline-block;
                    margin: 10px 15px;
                    padding: 10px;
                    background-color: #3498db;
                    color: white;
                    border-radius: 5px;
                    font-weight: bold;
                }}
                .similar-cases {{
                    margin-top: 30px;
                }}
                .case-card {{
                    border: 1px solid #ddd;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    background-color: #fafafa;
                }}
                .timestamp {{
                    text-align: right;
                    color: #7f8c8d;
                    font-size: 0.9em;
                    margin-top: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🏥 Medical Case Analysis Report</h1>
                    <h2>AI-Powered Similar Case Analysis</h2>
                </div>
                
                <div class="case-summary">
                    <h3>📋 New Patient Case Summary</h3>
                    <p><strong>Case ID:</strong> {new_case.case_id}</p>
                    <p><strong>Patient:</strong> {new_case.age}-year-old {new_case.gender}</p>
                    <p><strong>Chief Complaint:</strong> {new_case.chief_complaint}</p>
                    <p><strong>Symptoms:</strong> {', '.join(new_case.symptoms)}</p>
                    <p><strong>Diagnosis:</strong> {new_case.diagnosis}</p>
                    <p><strong>Severity:</strong> {new_case.severity}</p>
                    <p><strong>Comorbidities:</strong> {', '.join(new_case.comorbidities) if new_case.comorbidities else 'None'}</p>
                </div>
                
                <div class="insights">
                    <h3>🔍 Key Insights from Similar Cases</h3>
                    <div class="metric">Avg Similarity: {avg_similarity:.3f}</div>
                    <div class="metric">Most Common Outcome: {most_common_outcome}</div>
                    <div class="metric">Recommended Treatment: {most_common_treatment}</div>
                    <div class="metric">Expected Duration: {avg_duration:.1f} days</div>
                    <div class="metric">Estimated Cost: ${avg_cost:,.2f}</div>
                </div>
                
                <div class="reasoning">
                    <h3>🧠 AI Reasoning & Recommendations</h3>
                    <p>{reasoning}</p>
                </div>
                
                <div class="chart-container">
                    <h3>📊 Case Similarity Analysis</h3>
                    {similarity_chart}
                </div>
                
                <div class="chart-container">
                    <h3>📈 Outcome Distribution</h3>
                    {outcome_chart}
                </div>
                
                <div class="chart-container">
                    <h3>💊 Treatment Effectiveness</h3>
                    {treatment_chart}
                </div>
                
                <div class="chart-container">
                    <h3>👥 Patient Demographics</h3>
                    {demographics_chart}
                </div>
                
                <div class="similar-cases">
                    <h3>📚 Top 5 Most Similar Cases</h3>
                    {self._generate_similar_cases_table(similar_cases[:5])}
                </div>
                
                <div class="timestamp">
                    Report generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html_template
    
    def _generate_reasoning(self, new_case: PatientCase, similar_cases: List[Dict[str, Any]], insights: Dict) -> str:
        """Generate AI reasoning for the case analysis"""
        cases = [item['case'] for item in similar_cases]
        
        reasoning_parts = []

        patient_cases = []
        embedding_service = None
        case_analyzer = None
        
        # Similarity analysis
        reasoning_parts.append(f"Based on semantic analysis of {len(similar_cases)} similar cases, "
                             f"the average similarity score is {insights['avg_similarity']:.3f}, indicating "
                             f"{'strong' if insights['avg_similarity'] > 0.8 else 'moderate' if insights['avg_similarity'] > 0.6 else 'weak'} "
                             f"correlation with historical cases.")
        
        # Outcome prediction
        outcomes = [case.outcome for case in cases]
        positive_outcomes = sum(1 for outcome in outcomes if outcome in ['Full Recovery', 'Partial Recovery', 'Stable'])
        success_rate = (positive_outcomes / len(outcomes)) * 100
        
        reasoning_parts.append(f"Historical data shows a {success_rate:.1f}% success rate for similar cases, "
                             f"with '{insights['most_common_outcome']}' being the most frequent outcome.")
        
        # Treatment recommendation
        reasoning_parts.append(f"The most commonly used treatment '{insights['most_common_treatment']}' "
                             f"appears in similar cases and should be considered for this patient.")
        
        # Duration and cost estimation
        reasoning_parts.append(f"Expected treatment duration is approximately {insights['avg_duration']:.1f} days "
                             f"with an estimated cost of ${insights['avg_cost']:,.2f} based on similar case patterns.")
        
        # Risk factors
        if new_case.age > 65:
            reasoning_parts.append("Patient age (>65) may increase complexity and recovery time.")
        
        if new_case.comorbidities:
            reasoning_parts.append(f"Presence of comorbidities ({', '.join(new_case.comorbidities)}) "
                                 f"requires careful monitoring and may affect treatment approach.")
        
        return " ".join(reasoning_parts)
    
    def _generate_similar_cases_table(self, similar_cases: List[Dict[str, Any]]) -> str:
        """Generate HTML table for similar cases"""
        table_html = "<table style='width:100%; border-collapse: collapse;'>"
        table_html += """
        <tr style='background-color: #3498db; color: white;'>
            <th style='padding: 10px; border: 1px solid #ddd;'>Case ID</th>
            <th style='padding: 10px; border: 1px solid #ddd;'>Similarity</th>
            <th style='padding: 10px; border: 1px solid #ddd;'>Diagnosis</th>
            <th style='padding: 10px; border: 1px solid #ddd;'>Treatment</th>
            <th style='padding: 10px; border: 1px solid #ddd;'>Outcome</th>
            <th style='padding: 10px; border: 1px solid #ddd;'>Duration</th>
        </tr>
        """
        
        for item in similar_cases:
            case = item['case']
            similarity = item['similarity']
            table_html += f"""
            <tr>
                <td style='padding: 8px; border: 1px solid #ddd;'>{case.case_id}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{similarity:.3f}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{case.diagnosis}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{case.treatment}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{case.outcome}</td>
                <td style='padding: 8px; border: 1px solid #ddd;'>{case.duration_days} days</td>
            </tr>
            """
        
        table_html += "</table>"
        return table_html

# Global variables for the Modal app
synthetic_data_generator = SyntheticDataGenerator()
embedding_service = None
case_analyzer = None
viz_generator = VisualizationGenerator()
report_generator = ReportGenerator(viz_generator)
patient_cases = []

@app.function(image=image, secrets=[modal.Secret.from_name("nebius-api-key")])
def initialize_system():
    """Initialize the medical case analysis system"""
    global embedding_service, case_analyzer, patient_cases
    
    # Get API key from environment
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError("NEBIUS_API_KEY environment variable not set")
    
    # Initialize services
    embedding_service = EmbeddingService(api_key)
    case_analyzer = CaseSimilarityAnalyzer(embedding_service)
    
    # Generate synthetic dataset
    print("Generating synthetic patient cases...")
    patient_cases = synthetic_data_generator.generate_dataset(500)  # Reduced for demo
    
    # Load cases into analyzer
    print("Loading cases and generating embeddings...")
    case_analyzer.load_cases(patient_cases)
    
    return {"status": "System initialized successfully", "cases_loaded": len(patient_cases)}

@app.function(image=image, secrets=[modal.Secret.from_name("nebius-api-key")])
def analyze_new_case(case_data: Dict[str, Any]) -> str:
    """Analyze a new patient case and generate report"""
    global case_analyzer, report_generator
    global embedding_service, case_analyzer, patient_cases
    
    # Initialize system if not already done
    if case_analyzer is None:
        api_key = os.environ.get("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("NEBIUS_API_KEY environment variable not set")
    
        # Initialize services locally
        embedding_service = EmbeddingService(api_key)
        case_analyzer = CaseSimilarityAnalyzer(embedding_service)
    
        # Generate synthetic dataset if not already done
        if not patient_cases:
            patient_cases = synthetic_data_generator.generate_dataset(500)
    
        # Load cases into analyzer
        case_analyzer.load_cases(patient_cases)

    new_case = PatientCase(**case_data)
    similar_cases = case_analyzer.find_similar_cases(new_case, top_k=10)
    
    # Generate comprehensive report
    print("Generating analysis report...")
    html_report = report_generator.generate_case_analysis_report(new_case, similar_cases)
    
    return html_report

@app.function(image=image, secrets=[modal.Secret.from_name("nebius-api-key")])
def get_case_statistics() -> Dict[str, Any]:
    """Get statistics about the loaded cases"""
    global patient_cases
    global patient_cases, embedding_service
    
    if not patient_cases:
        initialize_system()
    
    stats = {
        "total_cases": len(patient_cases),
        "diagnoses": {},
        "outcomes": {},
        "severity_distribution": {},
        "age_stats": {
            "min": min(case.age for case in patient_cases),
            "max": max(case.age for case in patient_cases),
            "avg": sum(case.age for case in patient_cases) / len(patient_cases)
        }
    }
    
    for case in patient_cases:
        # Count diagnoses
        if case.diagnosis not in stats["diagnoses"]:
            stats["diagnoses"][case.diagnosis] = 0
        stats["diagnoses"][case.diagnosis] += 1
        
        # Count outcomes
        if case.outcome not in stats["outcomes"]:
            stats["outcomes"][case.outcome] = 0
        stats["outcomes"][case.outcome] += 1
        
        # Count severity
        if case.severity not in stats["severity_distribution"]:
            stats["severity_distribution"][case.severity] = 0
        stats["severity_distribution"][case.severity] += 1
    
    return stats

@app.function(image=image, secrets=[modal.Secret.from_name("nebius-api-key")])
def search_cases_by_diagnosis(diagnosis: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Search for cases by diagnosis"""
    global patient_cases
    global patient_cases, embedding_service
    
    if not patient_cases:
        initialize_system()
    
    matching_cases = [
        asdict(case) for case in patient_cases 
        if diagnosis.lower() in case.diagnosis.lower()
    ]
    
    return matching_cases[:limit]

# FastAPI web interface


# Create FastAPI app that will be served by Modal
web_app = FastAPI(title="Medical Case Analysis System")

# HTML templates
#templates = Jinja2Templates(directory="templates") # Remove this line since we're not using template files


@web_app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page with case input form"""
    return HTMLResponse(content="""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medical Case Analysis System</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }
            .container {
                max-width: 800px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }
            h1 {
                text-align: center;
                color: #2c3e50;
                margin-bottom: 30px;
                font-size: 2.5em;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                font-weight: bold;
                color: #34495e;
            }
            input, select, textarea {
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 8px;
                font-size: 16px;
                transition: border-color 0.3s;
            }
            input:focus, select:focus, textarea:focus {
                outline: none;
                border-color: #3498db;
            }
            button {
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 8px;
                font-size: 18px;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s;
            }
            button:hover {
                transform: translateY(-2px);
            }
            .stats-button {
                background: linear-gradient(135deg, #27ae60, #229954);
                margin-top: 10px;
            }
            #loading {
                display: none;
                text-align: center;
                margin-top: 20px;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #3498db;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 2s linear infinite;
                margin: 0 auto;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏥 Medical Case Analysis System</h1>
            <p style="text-align: center; color: #7f8c8d; font-size: 1.2em; margin-bottom: 40px;">
                AI-Powered Similar Case Analysis & Visualization
            </p>
            
            <form id="caseForm" onsubmit="submitCase(event)">
                <div class="form-group">
                    <label for="case_id">Case ID:</label>
                    <input type="text" id="case_id" name="case_id" required placeholder="e.g., CASE_NEW_001">
                </div>
                
                <div class="form-group">
                    <label for="age">Age:</label>
                    <input type="number" id="age" name="age" min="1" max="120" required>
                </div>
                
                <div class="form-group">
                    <label for="gender">Gender:</label>
                    <select id="gender" name="gender" required>
                        <option value="">Select Gender</option>
                        <option value="Male">Male</option>
                        <option value="Female">Female</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="chief_complaint">Chief Complaint:</label>
                    <input type="text" id="chief_complaint" name="chief_complaint" required 
                           placeholder="e.g., Patient presents with chest pain">
                </div>
                
                <div class="form-group">
                    <label for="symptoms">Symptoms (comma-separated):</label>
                    <textarea id="symptoms" name="symptoms" rows="3" required 
                              placeholder="e.g., chest pain, shortness of breath, fatigue"></textarea>
                </div>
                
                <div class="form-group">
                    <label for="diagnosis">Diagnosis:</label>
                    <input type="text" id="diagnosis" name="diagnosis" required 
                           placeholder="e.g., Myocardial Infarction">
                </div>
                
                <div class="form-group">
                    <label for="treatment">Treatment:</label>
                    <input type="text" id="treatment" name="treatment" required 
                           placeholder="e.g., Antiplatelet therapy">
                </div>
                
                <div class="form-group">
                    <label for="outcome">Outcome:</label>
                    <select id="outcome" name="outcome" required>
                        <option value="">Select Outcome</option>
                        <option value="Full Recovery">Full Recovery</option>
                        <option value="Partial Recovery">Partial Recovery</option>
                        <option value="Stable">Stable</option>
                        <option value="Deteriorated">Deteriorated</option>
                        <option value="Deceased">Deceased</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="severity">Severity:</label>
                    <select id="severity" name="severity" required>
                        <option value="">Select Severity</option>
                        <option value="Mild">Mild</option>
                        <option value="Moderate">Moderate</option>
                        <option value="Severe">Severe</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="duration_days">Duration (days):</label>
                    <input type="number" id="duration_days" name="duration_days" min="1" required>
                </div>
                
                <div class="form-group">
                    <label for="cost">Cost ($):</label>
                    <input type="number" id="cost" name="cost" min="0" step="0.01" required>
                </div>
                
                <button type="submit">🔍 Analyze Case</button>
                <button type="button" class="stats-button" onclick="viewStats()">📊 View System Statistics</button>
            </form>
            
            <div id="loading">
                <div class="spinner"></div>
                <p>Analyzing case and generating report...</p>
            </div>
        </div>
        
        <script>
            async function submitCase(event) {
                event.preventDefault();
                
                const form = document.getElementById('caseForm');
                const loading = document.getElementById('loading');
                
                // Show loading
                form.style.display = 'none';
                loading.style.display = 'block';
                
                // Collect form data
                const formData = new FormData(form);
                const caseData = {
                    case_id: formData.get('case_id'),
                    age: parseInt(formData.get('age')),
                    gender: formData.get('gender'),
                    chief_complaint: formData.get('chief_complaint'),
                    symptoms: formData.get('symptoms').split(',').map(s => s.trim()),
                    diagnosis: formData.get('diagnosis'),
                    treatment: formData.get('treatment'),
                    outcome: formData.get('outcome'),
                    duration_days: parseInt(formData.get('duration_days')),
                    severity: formData.get('severity'),
                    comorbidities: [],
                    lab_values: {
                        hemoglobin: 12.5,
                        white_blood_cells: 7.0,
                        glucose: 100,
                        creatinine: 1.0,
                        sodium: 140
                    },
                    admission_date: new Date().toISOString().split('T')[0],
                    discharge_date: new Date(Date.now() + formData.get('duration_days') * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
                    cost: parseFloat(formData.get('cost'))
                };
                
                try {
                    const response = await fetch('/analyze', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(caseData)
                    });
                    
                    if (response.ok) {
                        const html = await response.text();
                        document.body.innerHTML = html;
                    } else {
                        alert('Error analyzing case. Please try again.');
                        form.style.display = 'block';
                        loading.style.display = 'none';
                    }
                } catch (error) {
                    alert('Error: ' + error.message);
                    form.style.display = 'block';
                    loading.style.display = 'none';
                }
            }
            
            async function viewStats() {
                window.location.href = '/stats';
            }
        </script>
    </body>
    </html>
    """)

@web_app.post("/analyze")
async def analyze_case(request: Request):
    """Analyze a new case and return HTML report"""
    case_data = await request.json()
    
    # Call the Modal function
    html_report = analyze_new_case.remote(case_data)
    
    return HTMLResponse(content=html_report)

@web_app.get("/stats")
async def get_stats():
    """Get system statistics"""
    stats = get_case_statistics.remote()
    
    return HTMLResponse(content=f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>System Statistics</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
            }}
            .container {{
                max-width: 1000px;
                margin: 0 auto;
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            }}
            h1 {{
                text-align: center;
                color: #2c3e50;
                margin-bottom: 30px;
            }}
            .stats-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 30px;
            }}
            .stat-card {{
                background: #f8f9fa;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                border: 1px solid #e9ecef;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                color: #3498db;
            }}
            .stat-label {{
                color: #6c757d;
                margin-top: 5px;
            }}
            .back-button {{
                background: linear-gradient(135deg, #3498db, #2980b9);
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 System Statistics</h1>
            
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value">{stats['total_cases']}</div>
                    <div class="stat-label">Total Cases</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{stats['age_stats']['avg']:.1f}</div>
                    <div class="stat-label">Average Age</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(stats['diagnoses'])}</div>
                    <div class="stat-label">Unique Diagnoses</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{len(stats['outcomes'])}</div>
                    <div class="stat-label">Outcome Types</div>
                </div>
            </div>
            
            <h3>Top Diagnoses</h3>
            <ul>
                {chr(10).join([f"<li>{diagnosis}: {count} cases</li>" for diagnosis, count in sorted(stats['diagnoses'].items(), key=lambda x: x[1], reverse=True)[:10]])}
            </ul>
            
            <h3>Outcome Distribution</h3>
            <ul>
                {chr(10).join([f"<li>{outcome}: {count} cases</li>" for outcome, count in stats['outcomes'].items()])}
            </ul>
            
            <a href="/" class="back-button">← Back to Analysis</a>
        </div>
    </body>
    </html>
    """)

@web_app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Serve the web app using Modal
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("nebius-api-key")]
)
@modal.asgi_app()
def fastapi_app():
    return web_app

# CLI functions for testing
@app.function(image=image, secrets=[modal.Secret.from_name("nebius-api-key")])
def test_system():
    """Test the system with a sample case"""
    
    # Initialize system
    result = initialize_system()
    print(result)
    
    # Create a test case
    test_case_data = {
        "case_id": "TEST_001",
        "age": 65,
        "gender": "Male",
        "chief_complaint": "Patient presents with chest pain",
        "symptoms": ["chest pain", "shortness of breath", "nausea"],
        "diagnosis": "Myocardial Infarction",
        "treatment": "Antiplatelet therapy",
        "outcome": "Full Recovery",
        "duration_days": 7,
        "severity": "Moderate",
        "comorbidities": ["Hypertension"],
        "lab_values": {
            "hemoglobin": 12.5,
            "white_blood_cells": 8.2,
            "glucose": 120,
            "creatinine": 1.1,
            "sodium": 140
        },
        "admission_date": "2024-01-15",
        "discharge_date": "2024-01-22",
        "cost": 25000.00
    }
    
    # Analyze the test case
    html_report = analyze_new_case(test_case_data)
    
    # Save report to file
    with open("/tmp/test_report.html", "w") as f:
        f.write(html_report)
    
    print("Test completed successfully!")
    print("Report saved to /tmp/test_report.html")
    
    return {"status": "Test completed", "report_length": len(html_report)}

if __name__ == "__main__":
    # For local development
    uvicorn.run(web_app, host="0.0.0.0", port=8000)