import modal
import os
import sys
import base64
import requests
from typing import List, Dict, Any, Optional
from pathlib import Path
import json
import logging
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Modal app
app = modal.App("multimodal-ai-agent")

# Docker image with corrected dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "fastapi>=0.104.0",
        "python-multipart>=0.0.6",
        "pillow>=10.0.0",
        "requests>=2.31.0",
        "semanticscholar>=0.10.0",  # Fixed version constraint
        "arxiv>=2.1.0",
        "crossrefapi>=1.0.0"
    ])
)

class MultiModalAgent:
    """Complete MultiModalAgent implementation with all template methods"""
    def __init__(self, api_key: str, base_url: str = "https://api.studio.nebius.ai/v1"):
        self.api_key = api_key
        self.base_url = base_url
        self.model = "Qwen/Qwen2.5-VL-72B-Instruct"
        self.report_templates = {
            "comprehensive": self._comprehensive_template,
            "technical": self._technical_template,
            "executive": self._executive_template,
            "research": self._research_template
        }

    def _encode_image(self, image_data: bytes) -> str:
        """Encode image to base64"""
        return base64.b64encode(image_data).decode('utf-8')

    def _prepare_messages(self, prompt: str, images: Optional[List[bytes]] = None) -> List[Dict]:
        """Prepare messages for the API call"""
        content = [{"type": "text", "text": prompt}]
        if images:
            for img_data in images:
                img_b64 = self._encode_image(img_data)
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}
                })
        return [{"role": "user", "content": content}]

    def _call_nebius_api(self, messages: List[Dict], max_tokens: int = 4000) -> str:
        """Make API call to Nebius/Qwen model"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=120
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logger.error(f"API call failed: {e}")
            raise Exception(f"Failed to get response from AI model: {e}")

    def _comprehensive_template(self, base_prompt: str) -> str:
        return f"""
        Analyze the provided content comprehensively and generate a detailed report.

        Original Request: {base_prompt}

        Please provide a comprehensive analysis following this structure:

        # Executive Summary
        [Brief overview of key findings]

        # Detailed Analysis
        [Thorough examination of all aspects]

        # Visual Analysis (if images provided)
        [Describe what you see in images, key elements, patterns, etc.]

        # Key Insights
        [Important discoveries and observations]

        # Recommendations
        [Actionable suggestions based on analysis]

        # Technical Details
        [Relevant technical information]

        # Conclusion
        [Summary and final thoughts]

        Provide specific, actionable insights with clear reasoning.
        """

    def _technical_template(self, base_prompt: str) -> str:
        return f"""
        Provide a technical analysis of the content with focus on:

        Request: {base_prompt}

        # Technical Overview
        # System/Process Analysis
        # Performance Metrics (if applicable)
        # Technical Specifications
        # Implementation Details
        # Potential Issues and Solutions
        # Technical Recommendations

        Focus on technical accuracy and practical implementation details.
        """

    def _executive_template(self, base_prompt: str) -> str:
        return f"""
        Create an executive-level report for: {base_prompt}

        # Executive Summary (Key points in 3-5 bullets)
        # Business Impact
        # Strategic Recommendations
        # Resource Requirements
        # Timeline and Next Steps
        # Risk Assessment

        Keep language business-focused and strategic.
        """

    def _research_template(self, base_prompt: str) -> str:
        return f"""
        Conduct a research-style analysis of: {base_prompt}

        # Research Question/Hypothesis
        # Methodology
        # Findings and Evidence
        # Data Analysis
        # Literature Context (if applicable)
        # Limitations
        # Future Research Directions
        # Conclusions

        Use rigorous analytical approach with evidence-based conclusions.
        """

    def process_request(
        self,
        prompt: str,
        images: Optional[List[bytes]] = None,
        report_type: str = "comprehensive",
        **kwargs
    ) -> Dict[str, Any]:
        """Process a multi-modal request and generate a report"""
        try:
            # Apply report template
            if report_type in self.report_templates:
                enhanced_prompt = self.report_templates[report_type](prompt)
            else:
                enhanced_prompt = prompt

            # Prepare messages
            messages = self._prepare_messages(enhanced_prompt, images)

            # Get AI response
            logger.info(f"Processing {report_type} report request")
            response = self._call_nebius_api(messages, max_tokens=6000)

            # Prepare result
            result = {
                "success": True,
                "report": response,
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "report_type": report_type,
                    "model": self.model,
                    "has_images": bool(images),
                    "num_images": len(images) if images else 0,
                    "original_prompt": prompt
                }
            }

            logger.info("Request processed successfully")
            return result

        except Exception as e:
            logger.error(f"Error processing request: {e}")
            return {
                "success": False,
                "error": str(e),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "report_type": report_type,
                    "original_prompt": prompt
                }
            }

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("nebius-api-key")],
    timeout=300,
    keep_warm=1
)
def run_agent(
    prompt: str,
    images: Optional[List[bytes]] = None,
    report_type: str = "comprehensive",
    **kwargs
) -> Dict[str, Any]:
    """Core agent processing function"""
    try:
        api_key = os.getenv("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("Nebius API key not found")
        
        logger.info(f"Processing request - Type: {report_type}")
        
        agent = MultiModalAgent(api_key=api_key)
        return agent.process_request(
            prompt=prompt,
            images=images,
            report_type=report_type,
            **kwargs
        )
    except Exception as e:
        logger.error(f"Error in run_agent: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "report_type": report_type,
                "original_prompt": prompt
            }
        }

@app.function(
    image=image,
    secrets=[modal.Secret.from_name("nebius-api-key")],
    keep_warm=1
)
@modal.asgi_app()
def serve_api():
    """FastAPI endpoint with proper Modal integration"""
    api = FastAPI(
        title="Multi-Modal AI Agent API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url=None
    )

    # Add CORS middleware
    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    valid_report_types = ["comprehensive", "technical", "executive", "research"]

    @api.post("/analyze")
    async def analyze(
        prompt: str = Form(...),
        report_type: str = Form("comprehensive"),
        images: List[UploadFile] = File(None)
    ):
        try:
            # Validate inputs
            if not prompt.strip():
                raise HTTPException(status_code=400, detail="Prompt cannot be empty")
            
            if report_type not in valid_report_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid report type. Must be one of: {valid_report_types}"
                )

            image_data = []
            if images:
                if len(images) > 10:
                    raise HTTPException(status_code=400, detail="Maximum 10 images allowed")

                total_size = 0
                for img in images:
                    content = await img.read()
                    total_size += len(content)
                    if len(content) > 10 * 1024 * 1024:  # 10MB per image
                        raise HTTPException(
                            status_code=400,
                            detail=f"Image {img.filename} exceeds 10MB size limit"
                        )
                    if total_size > 50 * 1024 * 1024:  # 50MB total
                        raise HTTPException(
                            status_code=400,
                            detail="Total images size exceeds 50MB limit"
                        )
                    image_data.append(content)

            # Call the agent function
            result = run_agent.remote(
                prompt=prompt,
                images=image_data if image_data else None,
                report_type=report_type
            )
            
            # Ensure proper response format
            if not isinstance(result, dict):
                raise HTTPException(
                    status_code=500,
                    detail="Invalid response format from agent"
                )
                
            if not result.get("success"):
                raise HTTPException(
                    status_code=500,
                    detail=result.get("error", "Analysis failed")
                )
                
            return JSONResponse(result)

        except HTTPException as he:
            raise he
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error: {str(e)}"
            )

    @api.get("/")
    async def health():
        return {"status": "healthy", "service": "Multi-Modal AI Agent"}

    @api.get("/config")
    async def get_config():
        """Endpoint for retrieving current configuration"""
        return {
            "max_images": 10,
            "max_image_size": "10MB",
            "max_total_size": "50MB",
            "supported_report_types": valid_report_types
        }

    return api

if __name__ == "__main__":
    # For local development testing
    import uvicorn
    print("Starting local development server...")
    uvicorn.run("app:serve_api", host="0.0.0.0", port=8000, reload=True)