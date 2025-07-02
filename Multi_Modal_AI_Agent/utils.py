import base64
import io
from PIL import Image
import hashlib
import json
from typing import Dict, Any, List, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Utility class for image processing operations"""
    
    @staticmethod
    def validate_image(image_data: bytes) -> bool:
        """Validate if the data is a valid image"""
        try:
            img = Image.open(io.BytesIO(image_data))
            img.verify()
            return True
        except Exception:
            return False
    
    @staticmethod
    def resize_image(image_data: bytes, max_size: tuple = (1024, 1024)) -> bytes:
        """Resize image while maintaining aspect ratio"""
        try:
            img = Image.open(io.BytesIO(image_data))
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            
            output = io.BytesIO()
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            img.save(output, format='JPEG', quality=85)
            return output.getvalue()
        except Exception as e:
            logger.error(f"Error resizing image: {e}")
            return image_data
    
    @staticmethod
    def get_image_info(image_data: bytes) -> Dict[str, Any]:
        """Get basic information about an image"""
        try:
            img = Image.open(io.BytesIO(image_data))
            return {
                "format": img.format,
                "mode": img.mode,
                "size": img.size,
                "width": img.width,
                "height": img.height,
                "has_transparency": img.mode in ('RGBA', 'LA') or 'transparency' in img.info
            }
        except Exception as e:
            logger.error(f"Error getting image info: {e}")
            return {}

class ReportFormatter:
    """Utility class for formatting reports"""
    
    @staticmethod
    def to_markdown(report_data: Dict[str, Any]) -> str:
        """Convert report data to markdown format"""
        md_content = []
        
        # Header
        md_content.append(f"# AI Analysis Report")
        md_content.append(f"*Generated: {report_data.get('timestamp', datetime.now().isoformat())}*\n")
        
        # Metadata
        if 'metadata' in report_data:
            meta = report_data['metadata']
            md_content.append("## Report Information")
            md_content.append(f"- **Type**: {meta.get('report_type', 'N/A')}")
            md_content.append(f"- **Model**: {meta.get('model', 'N/A')}")
            md_content.append(f"- **Images**: {meta.get('num_images', 0)}")
            md_content.append("")
        
        # Original prompt
        if 'original_prompt' in report_data.get('metadata', {}):
            md_content.append("## Original Request")
            md_content.append(report_data['metadata']['original_prompt'])
            md_content.append("")
        
        # Report content
        md_content.append("## Analysis")
        md_content.append(report_data.get('report', 'No report content available'))
        md_content.append("")
        
        return "\n".join(md_content)
    
    @staticmethod
    def to_html(report_data: Dict[str, Any]) -> str:
        """Convert report data to HTML format"""
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { color: #1f77b4; border-bottom: 2px solid #1f77b4; }
                .metadata { background: #f8f9fa; padding: 15px; border-radius: 5px; }
                .content { margin-top: 20px; }
                .timestamp { color: #6c757d; font-style: italic; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Analysis Report</h1>
                <p class="timestamp">Generated: {timestamp}</p>
            </div>
            
            <div class="metadata">
                <h2>Report Information</h2>
                <ul>
                    <li><strong>Model:</strong> {model}</li>
                    <li><strong>Images:</strong> {num_images}</li>
                </ul>
            </div>
            
            {original_prompt_section}
            
            <div class="content">
                <h2>Analysis</h2>
                <div>{report_content}</div>
            </div>
        </body>
        </html>
        """
        
        # Prepare data
        timestamp = report_data.get('timestamp', datetime.now().isoformat())
        metadata = report_data.get('metadata', {})
        
        original_prompt_section = ""
        if metadata.get('original_prompt'):
            original_prompt_section = f"""
            <div class="content">
                <h2>Original Request</h2>
                <p>{metadata['original_prompt']}</p>
            </div>
            """
        
        return html_template.format(
            timestamp=timestamp,
            model=metadata.get('model', 'N/A'),
            num_images=metadata.get('num_images', 0),
            original_prompt_section=original_prompt_section,
            report_content=report_data.get('report', 'No report content available').replace('\n', '<br>')
        )

class RequestValidator:
    """Utility class for validating requests"""
    
    @staticmethod
    def validate_prompt(prompt: str) -> Dict[str, Any]:
        """Validate prompt input"""
        if not prompt or not prompt.strip():
            return {"valid": False, "error": "Prompt cannot be empty"}
        
        if len(prompt) > 10000:
            return {"valid": False, "error": "Prompt too long (max 10,000 characters)"}
        
        return {"valid": True}
    
    @staticmethod
    def validate_images(images: List[bytes]) -> Dict[str, Any]:
        """Validate image inputs"""
        if not images:
            return {"valid": True}
        
        if len(images) > 10:
            return {"valid": False, "error": "Too many images (max 10)"}
        
        total_size = sum(len(img) for img in images)
        if total_size > 50 * 1024 * 1024:  # 50MB total
            return {"valid": False, "error": "Total image size too large (max 50MB)"}
        
        for i, img_data in enumerate(images):
            if not ImageProcessor.validate_image(img_data):
                return {"valid": False, "error": f"Invalid image format at index {i}"}
        
        return {"valid": True}

class CacheManager:
    """Simple in-memory cache for responses"""
    
    def __init__(self, max_size: int = 100):
        self.cache = {}
        self.max_size = max_size
        self.access_order = []
    
    def _generate_key(self, prompt: str, images: Optional[List[bytes]] = None) -> str:
        """Generate cache key"""
        key_data = prompt
        if images:
            # Add image hashes to key
            img_hashes = [hashlib.md5(img).hexdigest()[:8] for img in images]
            key_data += "_" + "_".join(img_hashes)
        
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get(self, prompt: str, images: Optional[List[bytes]] = None) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        key = self._generate_key(prompt, images)
        if key in self.cache:
            # Update access order
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def set(self, prompt: str, response: Dict[str, Any], images: Optional[List[bytes]] = None):
        """Cache response"""
        key = self._generate_key(prompt, images)
        
        # Remove oldest if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        self.cache[key] = response
        if key not in self.access_order:
            self.access_order.append(key)