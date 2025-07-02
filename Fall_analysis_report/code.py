import os
import sys
from openai import OpenAI
from PIL import Image
import io
import json
from datetime import datetime
import hashlib
import base64
from jinja2 import Template

class FallPreventionAnalyzer:
    def __init__(self):
        # Initialize all required attributes
        self.report_data = {
            'analysis_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'system_info': self._get_system_info(),
            'image_metadata': {},
            'api_status': {},
            'results': {}
        }
        
        # Initialize OpenAI client for Nebius AI Studio
        self.client = OpenAI(
            base_url="https://api.studio.nebius.com/v1/",
            api_key=os.environ.get("NEBIUS_API_KEY")
        )
        
        if not self.client.api_key:
            print("Warning: NEBIUS_API_KEY environment variable not set")

    def _get_system_info(self):
        """Collect basic system information"""
        return {
            'python_version': sys.version,
            'platform': sys.platform,
            'working_directory': os.getcwd()
        }

    def _get_image_path(self):
        """Prompt user for local image file path with enhanced validation"""
        while True:
            file_path = input("Enter the full path to the image file: ").strip()
            
            if not os.path.exists(file_path):
                print("Error: File does not exist.")
                continue
                
            try:
                # Validate file size (max 10MB)
                file_size = os.path.getsize(file_path)
                if file_size > 10 * 1024 * 1024:
                    print("Error: Image file too large (max 10MB).")
                    continue
                    
                with open(file_path, 'rb') as f:
                    image_data = f.read()
                
                # Validate image format and content
                image = self._validate_image(image_data)
                if not image:
                    continue
                
                # Store comprehensive metadata
                self.report_data['image_metadata'] = {
                    'file_path': file_path,
                    'file_size': f"{file_size/1024:.2f} KB",
                    'dimensions': f"{image.width}x{image.height}",
                    'format': image.format,
                    'mode': image.mode,
                    'hash': hashlib.md5(image_data).hexdigest()
                }
                
                return image_data, image
                
            except Exception as e:
                print(f"Error loading image: {str(e)}")

    def _validate_image(self, image_data):
        """Thorough image validation with detailed error reporting"""
        try:
            # Verify image can be opened
            image = Image.open(io.BytesIO(image_data))
            
            # Check for common issues
            if image.format not in ['JPEG', 'PNG']:
                print(f"Unsupported format: {image.format}. Please use JPEG or PNG.")
                return None
                
            # Verify image content
            image.verify()  # Raises exception if invalid
            image = Image.open(io.BytesIO(image_data))  # Reopen for use
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            return image
            
        except Exception as e:
            print(f"Invalid image: {str(e)}")
            return None

    def _analyze_with_api(self, image_data):
        """Analyze the image using Nebius AI Studio API"""
        self.report_data['api_status'] = {
            'attempted': True,
            'success': False,
            'request_details': {},
            'response_details': {}
        }
        
        try:
            if not self.client.api_key:
                raise ValueError("API key not configured")
                
            # Prepare base64 encoded image
            base64_image = base64.b64encode(image_data).decode('utf-8')
            image_url = f"data:image/jpeg;base64,{base64_image}"
            
            # Make the API request
            response = self.client.chat.completions.create(
                model="Qwen/Qwen2-VL-72B-Instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": """Analyze this image for fall risk factors. Consider:
                            - Slippery surfaces (wet floors, spills)
                            - Uneven flooring or obstacles
                            - Poor lighting conditions
                            - Lack of handrails or support
                            - Other potential hazards
                            
                            Provide a detailed risk assessment with specific recommendations.
                            Rate the overall fall risk as high, moderate, or minimal."""},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.2
            )
            
            # Record API call details
            self.report_data['api_status']['request_details'] = {
                'model': "Qwen/Qwen2-VL-72B-Instruct",
                'max_tokens': 1000
            }
            
            self.report_data['api_status']['response_details'] = {
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            self.report_data['api_status']['success'] = True
            generated_text = response.choices[0].message.content
            
            return {
                'api_analysis': True,
                'model_response': generated_text,
                'risk_score': self._extract_risk_score(generated_text)
            }
            
        except Exception as e:
            error_info = {
                'type': type(e).__name__,
                'message': str(e)
            }
            self.report_data['api_status']['error'] = error_info
            return None

    def _extract_risk_score(self, text):
        """Extract a risk score from the model's text response"""
        text = text.lower()
        if 'high risk' in text:
            return 75
        elif 'moderate risk' in text or 'medium risk' in text:
            return 50
        elif 'minimal risk' in text:
            return 25
        return 50  # Default if no clear indication

    def _local_image_analysis(self, image):
        """Fallback local analysis when API fails"""
        try:
            # Convert to grayscale for analysis
            grayscale = image.convert('L')
            
            # Calculate image statistics
            histogram = grayscale.histogram()
            pixels = sum(histogram)
            brightness = sum(i * histogram[i] for i in range(256)) / (pixels * 255)
            
            # Calculate risk score
            risk_score = 0
            
            # Lighting conditions
            if brightness < 0.3:  # Dark image
                risk_score += 30
            elif brightness > 0.8:  # Overexposed
                risk_score += 10
                
            return {
                'local_analysis': True,
                'image_stats': {'brightness': brightness},
                'risk_score': min(100, risk_score)
            }
            
        except Exception as e:
            print(f"Local analysis error: {str(e)}")
            return {
                'local_analysis': True,
                'error': str(e),
                'risk_score': 0
            }

    def _parse_model_response(self, response_text):
        """Parse the model response into structured data"""
        categories = []
        overall_recommendations = []
        
        # Split into sections
        sections = [s.strip() for s in response_text.split('###') if s.strip()]
        
        for section in sections:
            if section.startswith('Overall Fall Risk'):
                # Parse overall risk
                lines = [l.strip() for l in section.split('\n') if l.strip()]
                if len(lines) > 1:
                    overall_recommendations.append(lines[-1])
            elif not section.startswith('Recommendations'):
                # Parse risk categories
                lines = [l.strip() for l in section.split('\n') if l.strip()]
                if len(lines) >= 3:
                    category_name = lines[0]
                    observation = lines[1].replace('- **Observation**:', '').strip()
                    risk = lines[2].replace('- **Risk**:', '').strip()
                    recommendations = []
                    
                    for line in lines[3:]:
                        if line.startswith('- **Recommendation**:'):
                            recommendations.append(line.replace('- **Recommendation**:', '').strip())
                    
                    # Determine risk level
                    risk_level = 'minimal'
                    if 'high' in risk.lower():
                        risk_level = 'high'
                    elif 'moderate' in risk.lower():
                        risk_level = 'moderate'
                    
                    categories.append({
                        'name': category_name,
                        'observation': observation,
                        'risk_assessment': risk,
                        'risk_level': risk_level,
                        'recommendations': recommendations
                    })
        
        # Fallback if parsing fails
        if not categories:
            categories = [{
                'name': 'General Analysis',
                'observation': 'Comprehensive analysis of the environment',
                'risk_assessment': 'See overall risk rating',
                'risk_level': 'moderate',
                'recommendations': overall_recommendations
            }]
        
        return {
            'categories': categories,
            'overall_recommendations': overall_recommendations
        }

    def _structure_local_findings(self, analysis_results):
        """Structure local analysis findings for the report"""
        categories = []
        
        if analysis_results.get('local_analysis'):
            # Lighting analysis
            brightness = analysis_results.get('image_stats', {}).get('brightness', 0)
            light_risk = 'high' if brightness < 0.3 else 'minimal'
            
            categories.append({
                'name': 'Lighting Conditions',
                'observation': f"Image brightness analysis ({brightness:.2f})",
                'risk_assessment': 'Poor lighting detected' if brightness < 0.3 else 'Adequate lighting',
                'risk_level': light_risk,
                'recommendations': [
                    'Improve lighting conditions' if brightness < 0.3 else 'Maintain current lighting levels',
                    'Ensure consistent lighting throughout the area'
                ]
            })
        
        overall_recommendations = []
        if analysis_results.get('risk_score', 0) > 70:
            overall_recommendations.append("Immediate professional assessment recommended due to high risk factors")
        elif analysis_results.get('risk_score', 0) > 40:
            overall_recommendations.append("Moderate risk detected - implement safety measures")
        else:
            overall_recommendations.append("Minimal risk environment - maintain standard precautions")
        
        return {
            'categories': categories,
            'overall_recommendations': overall_recommendations
        }

    def _generate_html_report(self, image_data, analysis_results):
        """Generate a professional HTML report with structured reasoning"""
        # Convert image to base64 for HTML embedding
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Process analysis results
        risk_score = analysis_results.get('risk_score', 0)
        risk_level = 'high' if risk_score > 70 else 'moderate' if risk_score > 40 else 'minimal'
        
        # Parse the model response into structured data if available
        if analysis_results.get('model_response'):
            structured_findings = self._parse_model_response(analysis_results['model_response'])
        else:
            structured_findings = self._structure_local_findings(analysis_results)
        
        # Determine risk color based on level
        if risk_level == 'high':
            risk_color = '#ff4d4d'
        elif risk_level == 'moderate':
            risk_color = '#ffa64d'
        else:
            risk_color = '#3a7bd5'  # Changed from green to accent color
        
        # Prepare the report data
        report_context = {
            'analysis_date': self.report_data['analysis_date'],
            'image_metadata': self.report_data['image_metadata'],
            'image_base64': base64_image,
            'analysis_method': 'API' if analysis_results.get('api_analysis') else 'Local',
            'risk_score': risk_score,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'structured_findings': structured_findings,
            'system_info': self.report_data['system_info'],
            'api_status': self.report_data['api_status'],
            'detailed_results': analysis_results
        }
        
        # HTML template with professional layout
        html_template = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Fall Risk Analysis Report</title>
            <style>
                :root {
                    --bg-dark: #121212;
                    --card-bg: #1e1e1e;
                    --text-primary: #e0e0e0;
                    --text-secondary: #a0a0a0;
                    --accent-color: #3a7bd5;
                    --border-color: #333333;
                    --warning-color: #FFC107;
                    --danger-color: #F44336;
                }
                
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background-color: var(--bg-dark);
                    color: var(--text-primary);
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                }
                
                .container {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                
                .header {
                    text-align: center;
                    margin-bottom: 30px;
                    padding-bottom: 20px;
                    border-bottom: 1px solid var(--border-color);
                }
                
                .header h1 {
                    color: var(--accent-color);
                    margin-bottom: 10px;
                    font-size: 2.2em;
                }
                
                .header .subtitle {
                    color: var(--text-secondary);
                    font-size: 1.1em;
                }
                
                .card {
                    background-color: var(--card-bg);
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                    padding: 25px;
                    margin-bottom: 30px;
                    border: 1px solid var(--border-color);
                }
                
                .card-title {
                    font-size: 1.4em;
                    margin-top: 0;
                    margin-bottom: 20px;
                    color: var(--accent-color);
                    border-bottom: 1px solid var(--border-color);
                    padding-bottom: 10px;
                }
                
                .image-container {
                    text-align: center;
                    margin: 25px 0;
                }
                
                .image-container img {
                    max-width: 100%;
                    max-height: 400px;
                    border-radius: 6px;
                    border: 1px solid var(--border-color);
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
                    object-fit: contain;
                }
                
                .risk-indicator {
                    display: flex;
                    align-items: center;
                    margin: 20px 0;
                    padding: 15px;
                    background-color: rgba(58, 123, 213, 0.1);
                    border-radius: 6px;
                    border-left: 4px solid {{ risk_color }};
                }
                
                .risk-value {
                    font-size: 1.8em;
                    font-weight: bold;
                    color: {{ risk_color }};
                    margin-right: 15px;
                    min-width: 80px;
                    text-align: center;
                }
                
                .risk-label {
                    flex-grow: 1;
                }
                
                .risk-label h3 {
                    margin: 0 0 5px 0;
                    color: {{ risk_color }};
                }
                
                .risk-label p {
                    margin: 0;
                    color: var(--text-secondary);
                }
                
                .findings-section {
                    margin-top: 30px;
                }
                
                .finding-card {
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 6px;
                    padding: 20px;
                    margin-bottom: 20px;
                    border-left: 4px solid var(--accent-color);
                }
                
                .finding-title {
                    font-size: 1.2em;
                    margin-top: 0;
                    margin-bottom: 15px;
                    color: var(--accent-color);
                }
                
                .finding-risk {
                    display: inline-block;
                    padding: 3px 10px;
                    border-radius: 4px;
                    font-size: 0.8em;
                    font-weight: bold;
                    margin-left: 10px;
                }
                
                .finding-risk.high {
                    background-color: var(--danger-color);
                    color: white;
                }
                
                .finding-risk.moderate {
                    background-color: var(--warning-color);
                    color: black;
                }
                
                .finding-details {
                    margin-bottom: 15px;
                }
                
                .finding-details p {
                    margin: 8px 0;
                }
                
                .recommendation-list {
                    margin-top: 15px;
                    padding-left: 20px;
                }
                
                .recommendation-list li {
                    margin-bottom: 8px;
                    position: relative;
                    list-style-type: none;
                    padding-left: 25px;
                }
                
                .recommendation-list li:before {
                    content: "•";
                    color: var(--accent-color);
                    font-size: 1.5em;
                    position: absolute;
                    left: 0;
                    top: -3px;
                }
                
                .two-column {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 30px;
                }
                
                .two-column > div {
                    flex: 1;
                    min-width: 300px;
                }
                
                .footer {
                    text-align: center;
                    margin-top: 40px;
                    color: var(--text-secondary);
                    font-size: 0.9em;
                    border-top: 1px solid var(--border-color);
                    padding-top: 20px;
                }
                
                .summary-stats {
                    display: flex;
                    gap: 20px;
                    margin: 20px 0;
                    flex-wrap: wrap;
                }
                
                .stat-box {
                    flex: 1;
                    min-width: 200px;
                    background-color: rgba(255, 255, 255, 0.05);
                    border-radius: 6px;
                    padding: 15px;
                    text-align: center;
                }
                
                .stat-value {
                    font-size: 1.8em;
                    font-weight: bold;
                    color: var(--accent-color);
                    margin: 5px 0;
                }
                
                .stat-label {
                    color: var(--text-secondary);
                    font-size: 0.9em;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Fall Risk Analysis Report</h1>
                    <div class="subtitle">Assessment of environmental fall hazards</div>
                </div>
                
                <div class="card">
                    <h2 class="card-title">Executive Summary</h2>
                    
                    <div class="two-column">
                        <div>
                            <div class="summary-stats">
                                <div class="stat-box">
                                    <div class="stat-value">{{ risk_score }}%</div>
                                    <div class="stat-label">Risk Probability</div>
                                </div>
                                <div class="stat-box">
                                    <div class="stat-value" style="color: {{ risk_color }}">{{ risk_level|upper }}</div>
                                    <div class="stat-label">Risk Level</div>
                                </div>
                            </div>
                            
                            <div class="risk-indicator">
                                <div class="risk-value">{{ risk_score }}%</div>
                                <div class="risk-label">
                                    <h3>{{ risk_level|upper }} RISK</h3>
                                    <p>
                                        {% if risk_level == 'high' %}
                                        Immediate attention required - significant fall hazards detected
                                        {% elif risk_level == 'moderate' %}
                                        Caution advised - some fall hazards present
                                        {% else %}
                                        Minimal risk environment - standard precautions recommended
                                        {% endif %}
                                    </p>
                                </div>
                            </div>
                            
                            <h3>Key Recommendations</h3>
                            <ol class="recommendation-list">
                                {% for rec in structured_findings.overall_recommendations %}
                                <li>{{ rec }}</li>
                                {% endfor %}
                            </ol>
                        </div>
                        
                        <div class="image-container">
                            <img src="data:image/{{ image_metadata.format|lower }};base64,{{ image_base64 }}" 
                                 alt="Analyzed Image" 
                                 title="{{ image_metadata.file_path }}">
                            <p style="color: var(--text-secondary); margin-top: 10px;">
                                {{ image_metadata.dimensions }} • {{ image_metadata.file_size }} • {{ image_metadata.format }}
                            </p>
                        </div>
                    </div>
                </div>
                
                <div class="card">
                    <h2 class="card-title">Detailed Risk Assessment</h2>
                    
                    <div class="findings-section">
                        {% for category in structured_findings.categories %}
                        <div class="finding-card">
                            <h3 class="finding-title">
                                {{ category.name }}
                                <span class="finding-risk {{ category.risk_level }}">
                                    {{ category.risk_level|upper }}
                                </span>
                            </h3>
                            
                            <div class="finding-details">
                                <p><strong>Observation:</strong> {{ category.observation }}</p>
                                <p><strong>Risk Assessment:</strong> {{ category.risk_assessment }}</p>
                            </div>
                            
                            <div class="recommendations">
                                <strong>Recommendations:</strong>
                                <ul class="recommendation-list">
                                    {% for rec in category.recommendations %}
                                    <li>{{ rec }}</li>
                                    {% endfor %}
                                </ul>
                            </div>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                
                <div class="footer">
                    <p>Fall Prevention Analysis Tool • Generated on {{ analysis_date }}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        # Create and render the template
        template = Template(html_template)
        rendered_html = template.render(**report_context)
        
        # Save the HTML report with UTF-8 encoding
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"fall_analysis_report_{timestamp}.html"
        
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(rendered_html)
            print(f"\nHTML report successfully generated: {report_filename}")
            return report_filename
        except Exception as e:
            print(f"\nError saving HTML report: {str(e)}")
            return None

    def run_analysis(self):
        """Main analysis workflow"""
        print("\nFall Prevention Analysis Tool")
        print("=" * 40)
        print("\nThis tool analyzes images for fall risk factors using Nebius AI Studio.")
        print("Please provide the path to a local image file (JPEG/PNG).\n")
        
        try:
            # Get and validate image
            image_data, image = self._get_image_path()
            
            # Attempt API analysis
            print("\nAttempting API analysis with Qwen2-VL-72B-Instruct...")
            api_results = self._analyze_with_api(image_data)
            
            if api_results:
                print("API analysis successful")
                analysis_results = api_results
            else:
                print("\nAPI analysis failed. Falling back to local analysis...")
                analysis_results = self._local_image_analysis(image)
            
            # Store results
            self.report_data['results'] = {
                'analysis_method': 'API' if analysis_results.get('api_analysis') else 'Local',
                'analysis_details': analysis_results
            }
            
            # Generate and display report
            print("\nGenerating analysis report...")
            report_file = self._generate_html_report(image_data, analysis_results)
            
            # Save detailed JSON report
            self._save_detailed_report()
            
        except Exception as e:
            print(f"\nFatal error during analysis: {str(e)}")
            print("\nPlease check the error details above and try again.")
            if hasattr(self, 'report_data'):
                self._save_detailed_report()

    def _save_detailed_report(self):
        """Save comprehensive report to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_filename = f"fall_analysis_report_{timestamp}.json"
        
        try:
            with open(report_filename, 'w', encoding='utf-8') as f:
                json.dump(self.report_data, f, indent=2, ensure_ascii=False)
            print(f"Detailed JSON report saved to: {report_filename}")
        except Exception as e:
            print(f"Failed to save JSON report: {str(e)}")

if __name__ == "__main__":
    analyzer = FallPreventionAnalyzer()
    analyzer.run_analysis()