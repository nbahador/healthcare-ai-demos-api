import os
import base64
import requests
from flask import Flask, request, render_template_string, send_from_directory
from openai import OpenAI
from datetime import datetime
import json

app = Flask(__name__)

# Initialize Nebius client
client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

# Directory to save reports
REPORTS_DIR = "saved_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def encode_image(image_file):
    """Encode image to base64"""
    return base64.b64encode(image_file.read()).decode('utf-8')

def identify_medication(image_base64):
    """Use Nebius vision model to identify medication"""
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2-VL-72B-Instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Identify the generic medication name in this image. Return only the single most likely generic medication name in lowercase, nothing else."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=50,
            temperature=0.1,
        )
        med_name = response.choices[0].message.content.strip().lower()
        return med_name.split()[0] if med_name else None
    except Exception as e:
        print(f"Error identifying medication: {str(e)}")
        return None

def get_medication_details(med_name):
    """Get comprehensive medication details using LLM"""
    if not med_name:
        return None
    
    try:
        prompt = f"""
        Gather comprehensive information about the medication {med_name} from reliable medical sources.
        Provide the information in JSON format with these fields:
        - form: The pharmaceutical form (tablet, capsule, etc.)
        - strength: Common available strengths
        - dosage: Standard dosage information
        - instructions: How to take the medication
        - therapeutic_class: Therapeutic class/category
        - primary_use: Primary indications/uses
        - storage: Storage instructions
        - interactions: List of significant drug interactions
        - overdose: Signs of overdose
        - side_effects: Common side effects
        - allergens: Potential allergens in the medication
        - generic_alternatives: Brand/generic alternatives
        - references: List of 3-5 reliable reference URLs from authoritative sources like:
          * Drugs.com
          * MedlinePlus
          * FDA
          * NIH
          * Mayo Clinic
          * WebMD
          * RxList
        
        Example references format:
        "references": [
            "https://www.drugs.com/{med_name}.html",
            "https://medlineplus.gov/druginfo/meds/a682878.html",
            "https://www.accessdata.fda.gov/drugsatfda_docs/label/2020/021923s020lbl.pdf"
        ]
        
        Only include information you can verify from reliable medical sources.
        """
        
        response = client.chat.completions.create(
            model="meta-llama/Meta-Llama-3.1-70B-Instruct",
            messages=[
                {"role": "system", "content": "You are a medical information assistant that provides accurate, verified information about medications."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1500
        )
        
        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            print("Failed to parse LLM response as JSON")
            return None
    except Exception as e:
        print(f"Error getting medication details from LLM: {str(e)}")
        return {
            "form": "Not specified",
            "strength": "Not specified",
            "dosage": "Not specified",
            "instructions": "Not specified",
            "therapeutic_class": "Not specified",
            "primary_use": "Not specified",
            "storage": "Not specified",
            "interactions": [],
            "overdose": "Not specified",
            "side_effects": [],
            "allergens": [],
            "generic_alternatives": [],
            "references": [
                f"https://www.drugs.com/search.php?searchterm={med_name}",
                f"https://medlineplus.gov/druginfo/meds/search.html?query={med_name}",
                f"https://www.fda.gov/drugs/drug-safety-and-availability/search-drugs"
            ]
        }

def check_allergies(med_details, user_allergies):
    """Check for potential allergies"""
    if not med_details or not user_allergies:
        return []
    
    allergens_in_med = med_details.get("allergens", [])
    return [allergen for allergen in allergens_in_med if allergen.lower() in [a.lower() for a in user_allergies]]

def save_report_to_file(html_content, med_name):
    """Save the HTML report to a file"""
    try:
        filename = f"{med_name.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        filepath = os.path.join(REPORTS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return filename
    except Exception as e:
        print(f"Error saving report: {str(e)}")
        return None

def generate_report(med_name, med_details, allergy_warnings, image_base64=None, save_to_file=False):
    """Generate HTML report with medication information"""
    # First generate the report HTML
    report_html = render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medication Report: {{ med_name }}</title>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary-color: #3498db;
                --secondary-color: #2ecc71;
                --danger-color: #e74c3c;
                --warning-color: #f39c12;
                --light-color: #ecf0f1;
                --dark-color: #2c3e50;
                --text-color: #333;
                --border-radius: 8px;
                --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            body {
                font-family: 'Roboto', sans-serif;
                line-height: 1.6;
                color: var(--text-color);
                background-color: #f5f7fa;
                margin: 0;
                padding: 20px;
            }
            
            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 20px;
            }
            
            .report-header {
                background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
                color: white;
                padding: 30px;
                border-radius: var(--border-radius);
                margin-bottom: 30px;
                text-align: center;
                box-shadow: var(--box-shadow);
            }
            
            .report-title {
                font-family: 'Montserrat', sans-serif;
                font-weight: 700;
                margin: 0;
                font-size: 2.2rem;
            }
            
            .report-date {
                opacity: 0.9;
                font-size: 0.9rem;
            }
            
            .medication-name {
                font-family: 'Montserrat', sans-serif;
                font-size: 1.8rem;
                color: var(--dark-color);
                margin-top: 10px;
            }
            
            .card {
                background: white;
                border-radius: var(--border-radius);
                padding: 25px;
                margin-bottom: 25px;
                box-shadow: var(--box-shadow);
                transition: transform 0.3s ease;
            }
            
            .card:hover {
                transform: translateY(-5px);
            }
            
            .card-title {
                font-family: 'Montserrat', sans-serif;
                font-weight: 600;
                color: var(--primary-color);
                margin-top: 0;
                margin-bottom: 15px;
                font-size: 1.3rem;
                border-bottom: 2px solid var(--light-color);
                padding-bottom: 10px;
            }
            
            .info-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            
            .info-item {
                margin-bottom: 15px;
            }
            
            .info-label {
                font-weight: 500;
                color: var(--dark-color);
                margin-bottom: 5px;
            }
            
            .info-value {
                background-color: var(--light-color);
                padding: 10px;
                border-radius: var(--border-radius);
            }
            
            .alert {
                padding: 15px;
                border-radius: var(--border-radius);
                margin-bottom: 20px;
            }
            
            .alert-warning {
                background-color: #fff3cd;
                color: #856404;
                border-left: 4px solid var(--warning-color);
            }
            
            .alert-danger {
                background-color: #f8d7da;
                color: #721c24;
                border-left: 4px solid var(--danger-color);
            }
            
            .reference-list {
                list-style-type: none;
                padding: 0;
            }
            
            .reference-item {
                margin-bottom: 8px;
            }
            
            .reference-link {
                color: var(--primary-color);
                text-decoration: none;
            }
            
            .reference-link:hover {
                text-decoration: underline;
            }
            
            .footer {
                text-align: center;
                margin-top: 40px;
                color: #7f8c8d;
                font-size: 0.9rem;
            }
            
            .download-btn {
                display: inline-block;
                background-color: var(--secondary-color);
                color: white;
                padding: 10px 15px;
                border-radius: var(--border-radius);
                text-decoration: none;
                margin-top: 20px;
            }
            
            .medication-image {
                max-width: 300px;
                max-height: 300px;
                border-radius: var(--border-radius);
                margin: 20px auto;
                display: block;
                box-shadow: var(--box-shadow);
            }
            
            @media (max-width: 768px) {
                .info-grid {
                    grid-template-columns: 1fr;
                }
                
                .medication-image {
                    max-width: 100%;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="report-header">
                <h1 class="report-title">Medication Safety Report</h1>
                <div class="report-date">{{ date }}</div>
            </div>
            
            {% if image_base64 %}
            <div class="card">
                <h3 class="card-title">Original Medication Image</h3>
                <img src="data:image/jpeg;base64,{{ image_base64 }}" alt="Uploaded medication image" class="medication-image">
            </div>
            {% endif %}
            
            {% if not med_details %}
            <div class="alert alert-warning">
                <h2>Medication: {{ med_name }}</h2>
                <p>We couldn't retrieve detailed information for this medication.</p>
                <p>Please try one of these reliable sources for more information:</p>
                <ul class="reference-list">
                    <li><a href="https://www.drugs.com/search.php?searchterm={{ med_name }}" target="_blank">Search on Drugs.com</a></li>
                    <li><a href="https://medlineplus.gov/druginfo/meds/search.html?query={{ med_name }}" target="_blank">Search on MedlinePlus</a></li>
                    <li><a href="https://www.fda.gov/drugs/drug-safety-and-availability/search-drugs" target="_blank">FDA Drug Information</a></li>
                </ul>
            </div>
            {% else %}
            <div class="card">
                <h2 class="medication-name">{{ med_name.capitalize() }}</h2>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Form</div>
                        <div class="info-value">{{ med_details.get('form', 'Not specified') }}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Strength</div>
                        <div class="info-value">{{ med_details.get('strength', 'Not specified') }}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Therapeutic Class</div>
                        <div class="info-value">{{ med_details.get('therapeutic_class', 'Not specified') }}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Primary Use</div>
                        <div class="info-value">{{ med_details.get('primary_use', 'Not specified') }}</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3 class="card-title">Dosage Information</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Recommended Dosage</div>
                        <div class="info-value">{{ med_details.get('dosage', 'Not specified') }}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Instructions for Use</div>
                        <div class="info-value">{{ med_details.get('instructions', 'Not specified') }}</div>
                    </div>
                </div>
            </div>
            
            <div class="card">
                <h3 class="card-title">Safety Information</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Storage Instructions</div>
                        <div class="info-value">{{ med_details.get('storage', 'Not specified') }}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Signs of Overdose</div>
                        <div class="info-value">{{ med_details.get('overdose', 'Not specified') }}</div>
                    </div>
                </div>
            </div>
            
            {% if allergy_warnings %}
            <div class="card alert-danger">
                <h3 class="card-title">⚠️ Allergy Warning</h3>
                <p>This medication contains substances you may be allergic to:</p>
                <ul>
                    {% for allergen in allergy_warnings %}
                    <li>{{ allergen }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            <div class="card">
                <h3 class="card-title">Potential Side Effects</h3>
                <ul>
                    {% for effect in med_details.get('side_effects', ['No side effects information available']) %}
                    <li>{{ effect }}</li>
                    {% endfor %}
                </ul>
            </div>
            
            <div class="card">
                <h3 class="card-title">Drug Interactions</h3>
                {% if med_details.get('interactions') %}
                <p>This medication may interact with:</p>
                <ul>
                    {% for interaction in med_details.get('interactions', []) %}
                    <li>{{ interaction }}</li>
                    {% endfor %}
                </ul>
                {% else %}
                <p>No significant drug interactions reported.</p>
                {% endif %}
            </div>
            
            {% if med_details.get('generic_alternatives') %}
            <div class="card">
                <h3 class="card-title">Generic Alternatives</h3>
                <ul>
                    {% for alt in med_details.get('generic_alternatives', []) %}
                    <li>{{ alt }}</li>
                    {% endfor %}
                </ul>
            </div>
            {% endif %}
            
            <div class="card">
                <h3 class="card-title">References</h3>
                <p>For more information, please consult these verified sources:</p>
                <ul class="reference-list">
                    {% for ref in med_details.get('references', [
                        'https://www.drugs.com/search.php?searchterm=' + med_name,
                        'https://medlineplus.gov/druginfo/meds/search.html?query=' + med_name,
                        'https://www.fda.gov/drugs/drug-safety-and-availability/search-drugs'
                    ]) %}
                    <li class="reference-item">
                        <a href="{{ ref }}" class="reference-link" target="_blank">
                            {% if 'drugs.com' in ref %}Drugs.com
                            {% elif 'medlineplus' in ref %}MedlinePlus
                            {% elif 'fda.gov' in ref %}FDA
                            {% elif 'nih.gov' in ref %}NIH
                            {% elif 'mayoclinic' in ref %}Mayo Clinic
                            {% elif 'webmd' in ref %}WebMD
                            {% elif 'rxlist' in ref %}RxList
                            {% else %}{{ ref }}{% endif %}
                        </a>
                    </li>
                    {% endfor %}
                </ul>
            </div>
            
            {% if save_to_file and filename %}
            <div class="text-center">
                <a href="/download/{{ filename }}" class="download-btn">Download This Report</a>
            </div>
            {% endif %}
            {% endif %}
            
            <div class="footer">
                <p>Report generated on {{ date }} | Medication Wallet App</p>
                <p>This information is for educational purposes only and does not constitute medical advice.</p>
            </div>
        </div>
    </body>
    </html>
    """, 
    med_name=med_name, 
    med_details=med_details, 
    allergy_warnings=allergy_warnings, 
    image_base64=image_base64,
    date=datetime.now().strftime("%B %d, %Y %H:%M"),
    save_to_file=save_to_file,
    filename=None)  # We'll handle the filename separately

    # Save the report to file if requested
    filename = None
    if save_to_file:
        filename = save_report_to_file(report_html, med_name)
        # If we need to include the download link, we need to re-render with the filename
        if filename:
            report_html = render_template_string(report_html, 
                med_name=med_name, 
                med_details=med_details, 
                allergy_warnings=allergy_warnings, 
                image_base64=image_base64,
                date=datetime.now().strftime("%B %d, %Y %H:%M"),
                save_to_file=save_to_file,
                filename=filename)

    return report_html

@app.route('/download/<filename>')
def download_report(filename):
    """Endpoint to download saved reports"""
    try:
        return send_from_directory(REPORTS_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        return "Report not found", 404

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        user_allergies = [a.strip().lower() for a in request.form.get('allergies', '').split(',') if a.strip()]
        
        if 'med_image' not in request.files and 'image_url' not in request.form:
            return "No image uploaded or URL provided", 400
            
        try:
            image_base64 = ""
            if 'med_image' in request.files and request.files['med_image'].filename != '':
                image_file = request.files['med_image']
                image_base64 = encode_image(image_file)
            elif 'image_url' in request.form and request.form['image_url']:
                response = requests.get(request.form['image_url'])
                if response.status_code == 200:
                    image_base64 = base64.b64encode(response.content).decode('utf-8')
                else:
                    return "Failed to download image from URL", 400
            
            med_name = identify_medication(image_base64)
            if not med_name:
                return "Could not identify medication from image", 400
            
            med_details = get_medication_details(med_name)
            allergy_warnings = check_allergies(med_details, user_allergies)
            
            # Generate and return report, saving a copy to file
            return generate_report(med_name, med_details, allergy_warnings, image_base64, save_to_file=True)
            
        except Exception as e:
            return f"Error processing medication: {str(e)}", 500
    
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Medication Wallet</title>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400;500;600;700&family=Roboto:wght@300;400;500&display=swap" rel="stylesheet">
        <style>
            :root {
                --primary-color: #3498db;
                --secondary-color: #2ecc71;
                --light-color: #ecf0f1;
                --dark-color: #2c3e50;
                --border-radius: 8px;
                --box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            
            body {
                font-family: 'Roboto', sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f5f7fa;
                margin: 0;
                padding: 0;
            }
            
            .container {
                max-width: 600px;
                margin: 40px auto;
                padding: 20px;
            }
            
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            
            .app-title {
                font-family: 'Montserrat', sans-serif;
                font-weight: 700;
                color: var(--dark-color);
                font-size: 2.5rem;
                margin-bottom: 10px;
            }
            
            .app-subtitle {
                color: #7f8c8d;
                font-size: 1.1rem;
            }
            
            .tab-container {
                margin-bottom: 30px;
            }
            
            .tab-nav {
                display: flex;
                border-bottom: 1px solid #ddd;
                margin-bottom: 20px;
            }
            
            .tab-button {
                padding: 12px 20px;
                background: none;
                border: none;
                cursor: pointer;
                font-family: 'Montserrat', sans-serif;
                font-weight: 500;
                color: #7f8c8d;
                transition: all 0.3s;
                position: relative;
            }
            
            .tab-button.active {
                color: var(--primary-color);
            }
            
            .tab-button.active::after {
                content: '';
                position: absolute;
                bottom: -1px;
                left: 0;
                right: 0;
                height: 3px;
                background-color: var(--primary-color);
            }
            
            .tab-content {
                display: none;
                animation: fadeIn 0.5s;
            }
            
            .tab-content.active {
                display: block;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 500;
                color: var(--dark-color);
            }
            
            input[type="text"], input[type="file"], textarea {
                width: 100%;
                padding: 12px;
                border: 1px solid #ddd;
                border-radius: var(--border-radius);
                font-family: 'Roboto', sans-serif;
                font-size: 1rem;
                transition: border 0.3s;
            }
            
            input[type="text"]:focus, input[type="file"]:focus, textarea:focus {
                outline: none;
                border-color: var(--primary-color);
                box-shadow: 0 0 0 2px rgba(52, 152, 219, 0.2);
            }
            
            .submit-btn {
                background-color: var(--primary-color);
                color: white;
                border: none;
                padding: 14px 20px;
                border-radius: var(--border-radius);
                font-family: 'Montserrat', sans-serif;
                font-weight: 500;
                font-size: 1rem;
                cursor: pointer;
                width: 100%;
                transition: background-color 0.3s;
            }
            
            .submit-btn:hover {
                background-color: #2980b9;
            }
            
            .footer {
                text-align: center;
                margin-top: 40px;
                color: #7f8c8d;
                font-size: 0.9rem;
            }
            
            @keyframes fadeIn {
                from { opacity: 0; }
                to { opacity: 1; }
            }
            
            @media (max-width: 768px) {
                .container {
                    padding: 15px;
                    margin: 20px auto;
                }
                
                .app-title {
                    font-size: 2rem;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1 class="app-title">Medication Wallet</h1>
                <p class="app-subtitle">Your personal medication safety assistant</p>
            </div>
            
            <div class="tab-container">
                <div class="tab-nav">
                    <button class="tab-button active" onclick="openTab(event, 'upload-tab')">Upload Image</button>
                    <button class="tab-button" onclick="openTab(event, 'url-tab')">Enter URL</button>
                </div>
                
                <div id="upload-tab" class="tab-content active">
                    <form method="post" enctype="multipart/form-data">
                        <div class="form-group">
                            <label for="allergies">Known Allergies (comma separated):</label>
                            <input type="text" id="allergies" name="allergies" placeholder="e.g., Penicillin, NSAIDs">
                        </div>
                        <div class="form-group">
                            <label for="med_image">Upload Medication Image:</label>
                            <input type="file" id="med_image" name="med_image" accept="image/*" required>
                        </div>
                        <button type="submit" class="submit-btn">Analyze Medication</button>
                    </form>
                </div>
                
                <div id="url-tab" class="tab-content">
                    <form method="post">
                        <div class="form-group">
                            <label for="allergies_url">Known Allergies (comma separated):</label>
                            <input type="text" id="allergies_url" name="allergies" placeholder="e.g., Penicillin, NSAIDs">
                        </div>
                        <div class="form-group">
                            <label for="image_url">Medication Image URL:</label>
                            <input type="text" id="image_url" name="image_url" placeholder="https://example.com/medication.jpg" required>
                        </div>
                        <button type="submit" class="submit-btn">Analyze Medication</button>
                    </form>
                </div>
            </div>
            
            <div class="footer">
                <p>Always consult with your healthcare provider about medications</p>
            </div>
        </div>
        
        <script>
            function openTab(evt, tabName) {
                var i, tabcontent, tabbuttons;
                
                tabcontent = document.getElementsByClassName("tab-content");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].classList.remove("active");
                }
                
                tabbuttons = document.getElementsByClassName("tab-button");
                for (i = 0; i < tabbuttons.length; i++) {
                    tabbuttons[i].classList.remove("active");
                }
                
                document.getElementById(tabName).classList.add("active");
                evt.currentTarget.classList.add("active");
            }
        </script>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)