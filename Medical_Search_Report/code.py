import os
import re
import requests
from bs4 import BeautifulSoup
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import webbrowser
from typing import List, Dict, Any
from urllib.parse import urljoin
import json
import random

class MedicalSearchSystem:
    def __init__(self, api_key: str):
        """
        Initialize the medical search system with Nebius AI Studio API
        """
        self.api_key = api_key
        self.base_url = "https://api.studio.nebius.com/v1/"
        self.embedding_model = "BAAI/bge-en-icl"
        self.llm_model = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def search_medical_info(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for medical information based on user query
        """
        # Extract key medical terms from query
        medical_terms = self._extract_medical_terms(query)
        
        # Fetch data from medical sources
        results = []
        results.extend(self._search_pubmed(medical_terms))
        results.extend(self._search_clinical_trials(medical_terms))
        results.extend(self._search_cdc(medical_terms))
        
        print(f"Found {len(results)} relevant medical documents")
        return results

    def _extract_medical_terms(self, query: str) -> List[str]:
        """
        Use LLM to extract key medical terms from query
        """
        prompt = f"""
        Extract the key medical terms from this query that would be useful for searching medical literature.
        Return only a comma-separated list of terms, no additional text.
        
        Query: {query}
        """
        
        try:
            response = requests.post(
                f"{self.base_url}completions",
                headers=self.headers,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "max_tokens": 50,
                    "temperature": 0.3
                },
                timeout=30
            )
            
            if response.status_code == 200:
                terms = response.json()['choices'][0]['text'].strip()
                return [term.strip() for term in terms.split(',') if term.strip()]
        except Exception as e:
            print(f"Error extracting medical terms: {str(e)}")
        
        # Fallback to simple keyword extraction if LLM fails
        return list(set(re.findall(r'\b[a-z]+\b', query.lower())))

    def _search_pubmed(self, terms: List[str]) -> List[Dict[str, Any]]:
        """
        Search PubMed for relevant articles
        """
        base_url = "https://pubmed.ncbi.nlm.nih.gov/"
        results = []
        
        for term in terms[:3]:  # Limit to top 3 terms
            try:
                search_url = f"{base_url}?term={term.replace(' ', '+')}"
                response = requests.get(search_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                articles = soup.find_all('article', class_='full-docsum')[:5]  # Limit to 5 results
                
                for article in articles:
                    try:
                        title = article.find('a', class_='docsum-title').text.strip()
                        abstract_url = urljoin(base_url, article.find('a', class_='docsum-title')['href'])
                        
                        # Fetch abstract
                        abstract_response = requests.get(abstract_url, timeout=10)
                        abstract_soup = BeautifulSoup(abstract_response.text, 'html.parser')
                        abstract = abstract_soup.find('div', class_='abstract-content').text.strip() if abstract_soup.find('div', class_='abstract-content') else ""
                        
                        results.append({
                            'source': 'PubMed',
                            'title': title,
                            'content': abstract,
                            'url': abstract_url
                        })
                    except Exception as e:
                        print(f"Error processing PubMed article: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error searching PubMed for {term}: {str(e)}")
                continue
                
        return results

    def _search_clinical_trials(self, terms: List[str]) -> List[Dict[str, Any]]:
        """
        Search ClinicalTrials.gov for relevant studies
        """
        base_url = "https://clinicaltrials.gov/"
        results = []
        
        for term in terms[:2]:  # Limit to top 2 terms
            try:
                search_url = f"{base_url}ct2/results?cond={term.replace(' ', '+')}"
                response = requests.get(search_url, timeout=15)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                studies = soup.find_all('div', class_='study-info')[:3]  # Limit to 3 results
                
                for study in studies:
                    try:
                        title = study.find('a', class_='study-link').text.strip()
                        study_url = urljoin(base_url, study.find('a', class_='study-link')['href'])
                        
                        # Fetch study details
                        study_response = requests.get(study_url, timeout=15)
                        study_soup = BeautifulSoup(study_response.text, 'html.parser')
                        
                        conditions = study_soup.find('div', id='conditions').text.strip() if study_soup.find('div', id='conditions') else ""
                        criteria = study_soup.find('div', id='eligibility').text.strip() if study_soup.find('div', id='eligibility') else ""
                        interventions = study_soup.find('div', id='interventions').text.strip() if study_soup.find('div', id='interventions') else ""
                        
                        content = f"Conditions: {conditions}\nEligibility: {criteria}\nInterventions: {interventions}"
                        
                        results.append({
                            'source': 'ClinicalTrials.gov',
                            'title': title,
                            'content': content,
                            'url': study_url
                        })
                    except Exception as e:
                        print(f"Error processing clinical trial: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error searching ClinicalTrials.gov for {term}: {str(e)}")
                continue
                
        return results

    def _search_cdc(self, terms: List[str]) -> List[Dict[str, Any]]:
        """
        Search CDC health topics
        """
        base_url = "https://www.cdc.gov/"
        results = []
        
        for term in terms[:1]:  # Limit to top term
            try:
                search_url = f"{base_url}search/index.html?query={term.replace(' ', '+')}"
                response = requests.get(search_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Find top 2 results
                items = soup.select('.search-results .item')[:2]
                
                for item in items:
                    try:
                        title = item.find('h3').text.strip()
                        page_url = urljoin(base_url, item.find('a')['href'])
                        
                        # Fetch page content
                        page_response = requests.get(page_url, timeout=10)
                        page_soup = BeautifulSoup(page_response.text, 'html.parser')
                        
                        # Extract main content
                        main_content = (page_soup.find('main') or 
                                       page_soup.find('div', id='content') or 
                                       page_soup.find('article'))
                        
                        if not main_content:
                            continue
                            
                        # Remove unwanted elements
                        for element in main_content.find_all(['nav', 'footer', 'aside', 'script', 'style']):
                            element.decompose()
                            
                        # Extract meaningful content
                        content_parts = []
                        for p in main_content.find_all(['p', 'h1', 'h2', 'h3']):
                            if p.text.strip():
                                content_parts.append(p.text.strip())
                        
                        content = ' '.join(content_parts[:500])  # Limit content length
                        
                        results.append({
                            'source': 'CDC',
                            'title': title,
                            'content': content,
                            'url': page_url
                        })
                    except Exception as e:
                        print(f"Error processing CDC page: {str(e)}")
                        continue
            except Exception as e:
                print(f"Error searching CDC for {term}: {str(e)}")
                continue
                
        return results

    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings using Nebius AI Studio API
        """
        embeddings = []
        
        for text in texts:
            try:
                if not text.strip():
                    embeddings.append(None)
                    continue
                    
                response = requests.post(
                    f"{self.base_url}embeddings",
                    headers=self.headers,
                    json={
                        "model": self.embedding_model,
                        "input": text,
                        "encoding_format": "float"
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    embedding_data = response.json()
                    embeddings.append(embedding_data['data'][0]['embedding'])
                else:
                    print(f"Error creating embedding: {response.status_code} - {response.text}")
                    embeddings.append(None)
            except Exception as e:
                print(f"Exception creating embedding: {str(e)}")
                embeddings.append(None)
                
        return embeddings

    def rank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank results by relevance to query using embeddings
        """
        if not results:
            return []
            
        # Create embeddings for query and all results
        query_embedding = self.create_embeddings([query])[0]
        if not query_embedding:
            return results
            
        result_texts = [f"{r['title']}. {r['content'][:500]}" for r in results]
        result_embeddings = self.create_embeddings(result_texts)
        
        # Calculate similarities
        ranked_results = []
        for i, result in enumerate(results):
            if result_embeddings[i] is None:
                continue
                
            similarity = cosine_similarity(
                np.array(query_embedding).reshape(1, -1),
                np.array(result_embeddings[i]).reshape(1, -1)
            )[0][0]
            
            ranked_results.append({
                **result,
                'similarity_score': similarity
            })
        
        # Sort by similarity
        ranked_results.sort(key=lambda x: x['similarity_score'], reverse=True)
        return ranked_results

    def generate_medical_analysis(self, query: str, results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate comprehensive medical analysis using LLM
        """
        # Prepare context from search results
        context = "\n\n".join([f"Source: {r['source']}\nTitle: {r['title']}\nContent: {r['content'][:1000]}" for r in results[:5]])
        
        prompt = f"""
        Based on the following medical search results, provide a comprehensive analysis of:
        
        1. Potential causes for the patient's symptoms
        2. Common diagnostic approaches
        3. Typical treatment protocols
        4. Medications commonly prescribed
        5. Prognosis and outcomes
        6. Similar case profiles
        
        Patient Query: {query}
        
        Search Results:
        {context}
        
        Structure your response with these sections:
        
        ### Potential Causes
        - Bullet points of possible causes
        
        ### Diagnostic Approaches
        - Common tests and procedures
        
        ### Treatment Options
        - Medications
        - Therapies
        - Lifestyle changes
        
        ### Similar Cases
        - Typical patient profiles
        - Common symptom patterns
        
        ### Clinical Recommendations
        - When to seek immediate care
        - Specialist referrals
        
        Use professional medical language but keep it clear and concise.
        """
        
        try:
            response = requests.post(
                f"{self.base_url}completions",
                headers=self.headers,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "max_tokens": 2000,
                    "temperature": 0.3
                },
                timeout=60
            )
            
            if response.status_code == 200:
                return {
                    'analysis': response.json()['choices'][0]['text'].strip(),
                    'status': 'success'
                }
        except Exception as e:
            print(f"Error generating medical analysis: {str(e)}")
        
        return {
            'analysis': "Could not generate medical analysis due to technical issues.",
            'status': 'error'
        }

    def generate_report(self, query: str, results: List[Dict[str, Any]], filename: str = "medical_report.html") -> str:
        """
        Generate professional HTML report with medical analysis
        """
        # Generate medical analysis
        analysis = self.generate_medical_analysis(query, results)
        
        # Prepare data for LLM
        report_data = {
            "query": query,
            "results": results,
            "analysis": analysis['analysis'],
            "date": datetime.now().strftime("%B %d, %Y %H:%M:%S")
        }
        
        # Generate HTML with LLM
        prompt = f"""
        Create a professional HTML medical report with these sections:
        
        1. Header with title and date
        2. Search query section
        3. Medical Analysis (comprehensive insights from expert analysis)
        4. Summary statistics (number of results, average similarity)
        5. Detailed results section with cards for each result showing:
           - Title
           - Source
           - Similarity score (0-1)
           - Content snippet
           - Link to full content
        6. Results grouped by source
        7. Footer
        
        Medical Analysis Content:
        {report_data['analysis']}
        
        Search Results Data:
        {json.dumps({k: v for k, v in report_data.items() if k != 'analysis'})}
        
        Include CSS styling for a professional, responsive design with:
        - Clean typography
        - Card-based layout
        - Color coding by source
        - Highlighted medical insights section
        - Mobile responsiveness
        
        Return ONLY the complete HTML document, no additional text or markdown.
        """
        
        try:
            response = requests.post(
                f"{self.base_url}completions",
                headers=self.headers,
                json={
                    "model": self.llm_model,
                    "prompt": prompt,
                    "max_tokens": 4000,
                    "temperature": 0.3
                },
                timeout=60
            )
            
            if response.status_code == 200:
                html_content = response.json()['choices'][0]['text'].strip()
                
                # Save HTML file
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                
                print(f"HTML report generated: {filename}")
                return filename
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            return self._generate_elegant_html(query, results, analysis['analysis'], filename)

    def _generate_elegant_html(self, query: str, results: List[Dict[str, Any]], analysis: str, filename: str) -> str:
        """
        Generate an extremely elegant HTML report with stunning design
        """
        current_date = datetime.now().strftime("%B %d, %Y %H:%M:%S")
        avg_similarity = sum(r.get('similarity_score', 0) for r in results) / len(results) if results else 0
        
        # Professional color scheme (dark blue, slate, with teal accent)
        colors = {
            'primary': '#2c3e50',
            'secondary': '#34495e',
            'accent': '#1abc9c',
            'light': '#f8f9fa',
            'dark': '#2c3e50',
            'text': '#333333',
            'light_text': '#7f8c8d'
        }
        
        # Generate source icons
        source_icons = {
            'PubMed': '🔬',
            'ClinicalTrials.gov': '📊',
            'CDC': '🏥'
        }
        
        # Parse analysis into sections
        analysis_sections = {}
        current_section = None
        for line in analysis.split('\n'):
            if line.startswith('### '):
                current_section = line[4:].strip()
                analysis_sections[current_section] = []
            elif current_section and line.strip():
                analysis_sections[current_section].append(line.strip())
        
        # Generate analysis cards
        analysis_cards = ""
        section_icons = {
            'Potential Causes': '🩺',
            'Diagnostic Approaches': '🔍',
            'Treatment Options': '💊',
            'Similar Cases': '👥',
            'Clinical Recommendations': '⚠️'
        }
        
        for section, content in analysis_sections.items():
            icon = section_icons.get(section, '📌')
            bullet_points = "".join([f"<li>{point[2:] if point.startswith('- ') else point}</li>" for point in content if point])
            analysis_cards += f"""
            <div class="analysis-card">
                <div class="analysis-card-header">
                    <span class="analysis-card-icon">{icon}</span>
                    <h3>{section}</h3>
                </div>
                <div class="analysis-card-content">
                    <ul>{bullet_points}</ul>
                </div>
            </div>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Medical Search Report | {query[:30]}...</title>
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&family=Merriweather:wght@400;700&display=swap" rel="stylesheet">
            <style>
                :root {{
                    --primary: {colors['primary']};
                    --secondary: {colors['secondary']};
                    --accent: {colors['accent']};
                    --light: {colors['light']};
                    --dark: {colors['dark']};
                    --text: {colors['text']};
                    --light-text: {colors['light_text']};
                }}
                
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Roboto', sans-serif;
                    line-height: 1.6;
                    color: var(--text);
                    background-color: #ffffff;
                    padding: 0;
                    margin: 0;
                }}
                
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 2rem;
                }}
                
                header {{
                    background: linear-gradient(135deg, var(--primary), var(--secondary));
                    color: white;
                    padding: 4rem 2rem;
                    text-align: center;
                    margin-bottom: 3rem;
                    position: relative;
                    overflow: hidden;
                }}
                
                header:after {{
                    content: '';
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    right: 0;
                    height: 4px;
                    background: var(--accent);
                }}
                
                h1, h2, h3 {{
                    font-family: 'Merriweather', serif;
                    font-weight: 700;
                    color: var(--dark);
                }}
                
                h1 {{
                    font-size: 2.5rem;
                    margin-bottom: 1rem;
                }}
                
                h2 {{
                    font-size: 1.8rem;
                    margin: 2.5rem 0 1.5rem;
                    position: relative;
                    padding-bottom: 0.5rem;
                }}
                
                h2:after {{
                    content: '';
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    width: 50px;
                    height: 3px;
                    background: var(--accent);
                }}
                
                h3 {{
                    font-size: 1.3rem;
                    margin: 1.2rem 0;
                }}
                
                .query-section {{
                    background: white;
                    padding: 2rem;
                    border-radius: 8px;
                    box-shadow: 0 2px 15px rgba(0,0,0,0.05);
                    margin-bottom: 2.5rem;
                }}
                
                .query-text {{
                    font-size: 1.1rem;
                    font-weight: 500;
                    color: var(--dark);
                    padding: 1.2rem;
                    background: var(--light);
                    border-left: 4px solid var(--accent);
                    border-radius: 0 4px 4px 0;
                }}
                
                .stats {{
                    display: flex;
                    gap: 1.2rem;
                    margin: 1.5rem 0;
                }}
                
                .stat-box {{
                    flex: 1;
                    background: white;
                    padding: 1.5rem;
                    border-radius: 6px;
                    box-shadow: 0 2px 8px rgba(0,0,0,0.04);
                    text-align: center;
                    border-top: 3px solid var(--accent);
                }}
                
                .stat-value {{
                    font-size: 1.8rem;
                    font-weight: 700;
                    color: var(--primary);
                    margin-bottom: 0.3rem;
                }}
                
                .stat-label {{
                    font-size: 0.85rem;
                    color: var(--light-text);
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                
                .analysis-section {{
                    margin-bottom: 2.5rem;
                }}
                
                .analysis-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                    gap: 1.5rem;
                    margin-top: 1.5rem;
                }}
                
                .analysis-card {{
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.06);
                    overflow: hidden;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                
                .analysis-card:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                
                .analysis-card-header {{
                    padding: 1.2rem 1.5rem;
                    background: var(--light);
                    border-bottom: 1px solid rgba(0,0,0,0.05);
                    display: flex;
                    align-items: center;
                    gap: 0.8rem;
                }}
                
                .analysis-card-icon {{
                    font-size: 1.3rem;
                    color: var(--accent);
                }}
                
                .analysis-card-content {{
                    padding: 1.5rem;
                }}
                
                .analysis-card-content ul {{
                    list-style-type: none;
                }}
                
                .analysis-card-content li {{
                    margin-bottom: 0.6rem;
                    padding-left: 1.2rem;
                    position: relative;
                    color: var(--text);
                }}
                
                .analysis-card-content li:before {{
                    content: '•';
                    position: absolute;
                    left: 0;
                    color: var(--accent);
                    font-weight: bold;
                }}
                
                .results-section {{
                    margin-bottom: 3rem;
                }}
                
                .result-card {{
                    background: white;
                    border-radius: 8px;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.06);
                    margin-bottom: 1.5rem;
                    overflow: hidden;
                    transition: transform 0.3s ease, box-shadow 0.3s ease;
                }}
                
                .result-card:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                
                .result-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 1.2rem 1.5rem;
                    background: var(--light);
                    border-bottom: 1px solid rgba(0,0,0,0.05);
                }}
                
                .result-source {{
                    display: flex;
                    align-items: center;
                    gap: 0.6rem;
                    font-weight: 500;
                    color: var(--light-text);
                    font-size: 0.9rem;
                }}
                
                .result-score {{
                    background: var(--accent);
                    color: white;
                    padding: 0.3rem 0.8rem;
                    border-radius: 20px;
                    font-weight: 600;
                    font-size: 0.85rem;
                }}
                
                .result-body {{
                    padding: 1.5rem;
                }}
                
                .result-title {{
                    font-size: 1.2rem;
                    margin-bottom: 1rem;
                    color: var(--dark);
                }}
                
                .result-content {{
                    margin-bottom: 1.5rem;
                    color: var(--text);
                    line-height: 1.6;
                }}
                
                .result-link {{
                    display: inline-flex;
                    align-items: center;
                    gap: 0.5rem;
                    padding: 0.6rem 1.2rem;
                    background: var(--accent);
                    color: white;
                    text-decoration: none;
                    border-radius: 4px;
                    font-weight: 500;
                    font-size: 0.9rem;
                    transition: background 0.3s ease;
                }}
                
                .result-link:hover {{
                    background: var(--primary);
                }}
                
                footer {{
                    text-align: center;
                    padding: 2rem;
                    color: var(--light-text);
                    font-size: 0.85rem;
                    border-top: 1px solid rgba(0,0,0,0.08);
                }}
                
                @media (max-width: 768px) {{
                    .container {{
                        padding: 1.2rem;
                    }}
                    
                    h1 {{
                        font-size: 2rem;
                    }}
                    
                    h2 {{
                        font-size: 1.5rem;
                    }}
                    
                    .stats {{
                        flex-direction: column;
                    }}
                    
                    .analysis-grid {{
                        grid-template-columns: 1fr;
                    }}
                }}
                
                /* Animation effects */
                @keyframes fadeIn {{
                    from {{ opacity: 0; transform: translateY(15px); }}
                    to {{ opacity: 1; transform: translateY(0); }}
                }}
                
                .query-section, .analysis-card, .result-card {{
                    animation: fadeIn 0.5s ease-out forwards;
                }}
                
                .analysis-card:nth-child(1) {{ animation-delay: 0.1s; }}
                .analysis-card:nth-child(2) {{ animation-delay: 0.2s; }}
                .analysis-card:nth-child(3) {{ animation-delay: 0.3s; }}
                .analysis-card:nth-child(4) {{ animation-delay: 0.4s; }}
                .analysis-card:nth-child(5) {{ animation-delay: 0.5s; }}
                
                .result-card:nth-child(1) {{ animation-delay: 0.1s; }}
                .result-card:nth-child(2) {{ animation-delay: 0.2s; }}
                .result-card:nth-child(3) {{ animation-delay: 0.3s; }}
                .result-card:nth-child(4) {{ animation-delay: 0.4s; }}
                .result-card:nth-child(5) {{ animation-delay: 0.5s; }}
            </style>
        </head>
        <body>
            <header>
                <div class="container">
                    <h1>Medical Insights Report</h1>
                    <p>Comprehensive clinical analysis based on latest research</p>
                </div>
            </header>
            
            <main class="container">
                <section class="query-section">
                    <h2>Patient Query</h2>
                    <div class="query-text">{query}</div>
                    
                    <div class="stats">
                        <div class="stat-box">
                            <div class="stat-value">{len(results)}</div>
                            <div class="stat-label">Relevant Findings</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{avg_similarity:.2f}</div>
                            <div class="stat-label">Average Relevance</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{current_date.split()[0]}</div>
                            <div class="stat-label">Report Date</div>
                        </div>
                    </div>
                </section>
                
                <section class="analysis-section">
                    <h2>Clinical Analysis</h2>
                    <div class="analysis-grid">
                        {analysis_cards}
                    </div>
                </section>
                
                <section class="results-section">
                    <h2>Research Findings</h2>
                    {"".join([f"""
                    <article class="result-card">
                        <div class="result-header">
                            <div class="result-source">
                                {source_icons.get(r['source'], '📄')} {r['source']}
                            </div>
                            <div class="result-score">Relevance: {r.get('similarity_score', 0):.2f}</div>
                        </div>
                        <div class="result-body">
                            <h3 class="result-title">{r['title']}</h3>
                            <div class="result-content">
                                {r['content'][:400]}...
                            </div>
                            <a href="{r['url']}" class="result-link" target="_blank">
                                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                                    <path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"></path>
                                    <polyline points="15 3 21 3 21 9"></polyline>
                                    <line x1="10" y1="14" x2="21" y2="3"></line>
                                </svg>
                                View Full Content
                            </a>
                        </div>
                    </article>
                    """ for r in results])}
                </section>
            </main>
            
            <footer>
                <div class="container">
                    <p>Report generated on {current_date} by Medical Search System</p>
                    <p>This clinical analysis is for informational purposes only and should not replace professional medical advice.</p>
                </div>
            </footer>
            
            <script>
                document.addEventListener('DOMContentLoaded', function() {{
                    // Animate elements when they come into view
                    const animateOnScroll = function() {{
                        const elements = document.querySelectorAll('.analysis-card, .result-card');
                        elements.forEach(element => {{
                            const elementPosition = element.getBoundingClientRect().top;
                            const screenPosition = window.innerHeight / 1.2;
                            
                            if (elementPosition < screenPosition) {{
                                element.style.opacity = '1';
                                element.style.transform = 'translateY(0)';
                            }}
                        }});
                    }};
                    
                    window.addEventListener('scroll', animateOnScroll);
                    animateOnScroll();
                }});
            </script>
        </body>
        </html>
        """
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Elegant HTML report generated: {filename}")
        return filename


def main():
    # Initialize with your Nebius API key
    api_key = os.getenv("NEBIUS_API_KEY")
    if not api_key:
        raise ValueError("Please set the NEBIUS_API_KEY environment variable")
    
    # Get user query
    query = input("Enter your medical query (e.g., '35-year-old with chest pain and shortness of breath'): ")
    if not query.strip():
        print("No query provided")
        return
    
    print("\nInitializing Medical Search System...")
    med_search = MedicalSearchSystem(api_key)
    
    # Step 1: Search for medical information
    print("\nSearching for relevant medical information...")
    results = med_search.search_medical_info(query)
    
    if not results:
        print("No relevant medical information found")
        return
    
    # Step 2: Rank results by relevance
    print("\nRanking results by relevance...")
    ranked_results = med_search.rank_results(query, results)
    
    # Step 3: Generate and display report
    print("\nGenerating report...")
    report_filename = med_search.generate_report(query, ranked_results[:5])  # Top 5 results
    
    # Open the report in default browser
    try:
        webbrowser.open(f'file://{os.path.abspath(report_filename)}')
    except Exception as e:
        print(f"Could not open report automatically: {str(e)}")
        print(f"Please open the file manually: {os.path.abspath(report_filename)}")

if __name__ == "__main__":
    main()