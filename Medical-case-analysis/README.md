# README.md
# Medical Case Analysis System

## Overview

An AI-powered medical case analysis system that generates synthetic patient case data, finds similar cases using semantic embeddings, and creates comprehensive visualized HTML reports. The system is deployed on Modal for scalable cloud execution.

## Features

### 🏥 Comprehensive Case Analysis
- **Synthetic Data Generation**: Creates realistic patient cases with demographics, symptoms, diagnoses, treatments, and outcomes
- **Semantic Similarity Search**: Uses advanced embeddings to find similar historical cases
- **Intelligent Recommendations**: Provides treatment suggestions based on similar case outcomes

### 📊 Rich Visualizations
- **Interactive Charts**: Similarity scores, outcome distributions, treatment effectiveness
- **Demographics Analysis**: Age and gender distributions of similar cases
- **Success Rate Metrics**: Treatment success rates based on historical data

### 🤖 AI-Powered Insights
- **Outcome Prediction**: Predicts likely outcomes based on similar cases
- **Cost Estimation**: Estimates treatment costs using historical patterns
- **Risk Assessment**: Identifies potential complications based on patient profile

### 🌐 Web Interface
- **User-Friendly Forms**: Easy case input with validation
- **Real-time Analysis**: Instant report generation
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Interface │    │  Modal Functions │    │  Nebius API     │
│                 │    │                 │    │                 │
│ • FastAPI App   │◄──►│ • Case Analysis │◄──►│ • Embeddings    │
│ • HTML Forms    │    │ • Data Generation│    │ • Similarity    │
│ • Visualizations│    │ • Report Creation│    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Reports  │    │  Synthetic Data │    │   Vector Store  │
│                 │    │                 │    │                 │
│ • HTML Reports  │    │ • Patient Cases │    │ • Case Embeddings│
│ • Visualizations│    │ • Demographics  │    │ • Similarity Index│
│ • Recommendations│   │ • Lab Values    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Setup and Deployment

### Prerequisites
- Python 3.11+
- Modal account and CLI
- Nebius API key for embeddings

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd medical-case-analysis
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env and add your NEBIUS_API_KEY
```

4. **Install Modal CLI**
```bash
pip install modal
```

5. **Authenticate with Modal**
```bash
modal token new
```

### Deployment

1. **Run the deployment script**
```bash
chmod +x deploy.sh
./deploy.sh
```

2. **Or deploy manually**
```bash
# Create secret
modal secret create nebius-api-key NEBIUS_API_KEY=your_api_key

# Deploy application
modal deploy main.py

# Test deployment
modal run main.py::test_system
```

### Local Development

```bash
# Run locally for development
python main.py
# Visit http://localhost:8000
```

## Usage

### Web Interface

1. **Access the application** at the URL provided by Modal after deployment
2. **Fill in patient case details** in the web form:
   - Basic demographics (age, gender)
   - Clinical information (symptoms, diagnosis, treatment)
   - Case details (severity, duration, cost)
3. **Submit for analysis** and receive a comprehensive HTML report

### API Endpoints

- `GET /` - Main web interface
- `POST /analyze` - Analyze a new case
- `GET /stats` - System statistics
- `GET /health` - Health check

### Modal Functions

```python
# Initialize the system
modal run main.py::initialize_system

# Analyze a specific case
modal run main.py::analyze_new_case --case-data '{"case_id": "TEST_001", ...}'

# Get system statistics
modal run main.py::get_case_statistics

# Test the system
modal run main.py::test_system
```

## Technical Components

### Data Generation
- **SyntheticDataGenerator**: Creates realistic patient cases with proper medical correlations
- **Disease-Symptom Mapping**: Maintains accurate relationships between conditions and presentations
- **Demographic Modeling**: Age-appropriate comorbidities and complications

### Embedding & Similarity
- **EmbeddingService**: Interfaces with Nebius API for high-quality embeddings
- **CaseSimilarityAnalyzer**: Finds semantically similar cases using cosine similarity
- **Vector Storage**: Efficient storage and retrieval of case embeddings

### Visualization
- **VisualizationGenerator**: Creates interactive Plotly charts
- **ReportGenerator**: Generates comprehensive HTML reports
- **AI Reasoning**: Provides intelligent insights and recommendations

### Web Application
- **FastAPI**: Modern, fast web framework
- **Responsive Design**: Mobile-friendly interface
- **Real-time Processing**: Instant analysis and report generation

## Configuration

### Environment Variables
```bash
NEBIUS_API_KEY=your_nebius_api_key_here
```

### Modal Secrets
```bash
modal secret create nebius-api-key NEBIUS_API_KEY=your_key
```

## Sample Output

The system generates comprehensive HTML reports including:

- **Case Summary**: Patient demographics and clinical details
- **Key Insights**: Statistics from similar cases
- **AI Reasoning**: Intelligent analysis and recommendations
- **Visualizations**: 
  - Case similarity scores
  - Outcome distributions
  - Treatment effectiveness
  - Patient demographics
- **Similar Cases Table**: Top matching cases with details

## Performance

- **Scalability**: Modal provides automatic scaling based on demand
- **Speed**: Optimized embedding generation and similarity search
- **Reliability**: Built-in error handling and fallback mechanisms
- **Cost-Effective**: Pay-per-use model with Modal's pricing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the Modal documentation for deployment issues
- Verify your Nebius API key configuration

## Roadmap

- [ ] Add more sophisticated ML models for outcome prediction
- [ ] Implement real-time case streaming
- [ ] Add user authentication and case history
- [ ] Integrate with electronic health records (EHR) systems
- [ ] Add multi-language support
- [ ] Implement advanced analytics dashboard