# Documentation for the Multi-Modal AI Agent

"""
# Multi-Modal AI Agent

A comprehensive multi-modal reasoning AI agent that can understand images and text to generate detailed reports. Built for deployment on Modal with MCP (Model Context Protocol) server integration.

## Features

- **Multi-Modal Processing**: Handles both text and image inputs
- **Multiple Report Types**: Comprehensive, technical, executive, and research reports
- **Web Interface**: Streamlit-based UI for easy interaction
- **API Endpoints**: REST API for programmatic access
- **MCP Server**: Model Context Protocol server for advanced integrations
- **Modal Deployment**: Serverless deployment with automatic scaling

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │    │   FastAPI REST  │    │   MCP Server   │
│   (Port 8000)   │    │   (Port 8080)   │    │   (Port 8001)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Core AI Agent  │
                    │ (Qwen2.5-VL-72B)│
                    └─────────────────┘
```

## Quick Start

### 1. Setup

```bash
# Clone the repository
git clone <repository-url>
cd multi-modal-ai-agent

# Install Modal CLI
pip install modal

# Set up your Nebius API key
modal secret create nebius-api-key NEBIUS_API_KEY=your_api_key_here
```

### 2. Deploy to Modal

```bash
# Deploy all services
modal deploy app.py

# Or deploy individual components
modal deploy app.py::serve_ui      # Web UI
modal deploy app.py::serve_api     # REST API
modal deploy app.py::serve_mcp     # MCP Server
```

### 3. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variable
export NEBIUS_API_KEY=your_api_key_here

# Run locally (for testing)
modal run app.py main --prompt "Analyze this image" --image-paths "path/to/image.jpg"
```

## Usage

### Web Interface

Access the Streamlit web interface at your Modal app URL (port 8000).

1. Enter your prompt or question
2. Upload images (optional)
3. Select report type
4. Click "Generate Report"

### REST API

```python
import requests

# Analyze with text only
response = requests.post("https://your-modal-app.modal.run/analyze", 
    data={"prompt": "Explain machine learning", "report_type": "comprehensive"})

# Analyze with images
files = {"images": open("image.jpg", "rb")}
data = {"prompt": "What's in this image?", "report_type": "technical"}
response = requests.post("https://your-modal-app.modal.run/analyze", 
    data=data, files=files)
```

### MCP Server

Connect to the WebSocket MCP server:

```python
import asyncio
import websockets
import json

async def connect_mcp():
    uri = "ws://your-modal-app.modal.run:8001"
    async with websockets.connect(uri) as websocket:
        # Send analysis request
        request = {
            "id": 1,
            "method": "analyze",
            "params": {
                "prompt": "Analyze this data",
                "report_type": "research"
            }
        }
        await websocket.send(json.dumps(request))
        response = await websocket.recv()
        print(json.loads(response))

asyncio.run(connect_mcp())
```

### CLI Usage

```bash
# Basic text analysis
modal run app.py main --prompt "Explain quantum computing" --report-type "technical"

# Image analysis
modal run app.py main --prompt "Describe these images" --image-paths "img1.jpg,img2.png" --report-type "comprehensive"
```

## Report Types

### Comprehensive
- Executive summary
- Detailed analysis
- Visual analysis (for images)
- Key insights
- Recommendations
- Technical details
- Conclusion

### Technical
- Technical overview
- System/process analysis
- Performance metrics
- Technical specifications
- Implementation details
- Issues and solutions

### Executive
- Executive summary
- Business impact
- Strategic recommendations
- Resource requirements
- Timeline and next steps
- Risk assessment

### Research
- Research question/hypothesis
- Methodology
- Findings and evidence
- Data analysis
- Literature context
- Limitations
- Future research directions

## Configuration

### Environment Variables

- `NEBIUS_API_KEY`: Your Nebius AI API key (required)
- `MODAL_ENVIRONMENT`: Set to "true" when running in Modal

### Model Settings

The agent uses Qwen2.5-VL-72B-Instruct model with these default parameters:
- Temperature: 0.7
- Top-p: 0.9
- Max tokens: 4000-6000 (depending on report type)

## API Reference

### Agent Methods

```python
from agent import MultiModalAgent

agent = MultiModalAgent(api_key="your_key")

# Process request
result = agent.process_request(
    prompt="Your question",
    images=[image_bytes],  # Optional
    report_type="comprehensive"
)

# Analyze single image
result = agent.analyze_image(image_bytes, "Describe this image")

# Batch processing
requests = [{"prompt": "Q1", "report_type": "technical"}]
results = agent.batch_analyze(requests)
```

### Utility Functions

```python
from utils import ImageProcessor, ReportFormatter, RequestValidator

# Image processing
ImageProcessor.resize_image(image_bytes, max_size=(800, 600))
ImageProcessor.get_image_info(image_bytes)

# Report formatting
ReportFormatter.to_markdown(report_data)
ReportFormatter.to_html(report_data)

# Validation
RequestValidator.validate_prompt(prompt)
RequestValidator.validate_images(image_list)
```

## Limitations

- Maximum 10 images per request
- Total image size limit: 50MB
- Prompt length limit: 10,000 characters
- Request timeout: 5 minutes

## Error Handling

The agent includes comprehensive error handling:

- Invalid image formats
- API timeouts
- Network errors
- Rate limiting
- Invalid prompts

All errors are returned in a structured format with error codes and messages.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Troubleshooting

### Common Issues

1. **API Key Issues**
   ```bash
   # Check if secret is set
   modal secret list
   
   # Update secret
   modal secret create nebius-api-key NEBIUS_API_KEY=new_key
   ```

2. **Image Upload Errors**
   - Ensure images are in supported formats (PNG, JPG, JPEG, GIF, BMP)
   - Check file sizes (max 50MB total)
   - Verify image files are not corrupted

3. **Deployment Issues**
   ```bash
   # Check Modal app status
   modal app list
   
   # View logs
   modal logs your-app-name
   ```

4. **Network Timeouts**
   - Large images may take longer to process
   - Complex prompts require more processing time
   - Check your internet connection

### Support

For issues and questions:
1. Check the troubleshooting section
2. Review Modal documentation
3. Check Nebius API status
4. Open an issue in the repository

## Performance Tips

1. **Image Optimization**
   - Resize large images before upload
   - Use JPEG format for photographs
   - Compress images when possible

2. **Prompt Engineering**
   - Be specific and clear in prompts
   - Use appropriate report types
   - Break complex requests into smaller parts

3. **Batch Processing**
   - Use batch_analyze for multiple requests
   - Cache results for repeated queries
   - Process images in parallel when possible

## Changelog

### v1.0.0
- Initial release
- Multi-modal processing
- Web UI and API endpoints
- MCP server integration
- Modal deployment support
"""
