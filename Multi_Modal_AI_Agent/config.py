import os
from typing import Dict, Any
from pathlib import Path

class Config:
    """Configuration management for the multi-modal AI agent"""
    
    # API Configuration
    NEBIUS_API_KEY = os.getenv("NEBIUS_API_KEY")
    NEBIUS_BASE_URL = os.getenv("NEBIUS_BASE_URL", "https://api.studio.nebius.ai/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-72B-Instruct")
    
    # Model Parameters
    DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
    DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "4000"))
    
    # Processing Limits
    MAX_IMAGES = int(os.getenv("MAX_IMAGES", "10"))
    MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "50")) * 1024 * 1024  # 50MB in bytes
    MAX_PROMPT_LENGTH = int(os.getenv("MAX_PROMPT_LENGTH", "10000"))
    REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "300"))  # 5 minutes
    
    # Cache Configuration
    ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "100"))
    
    # UI Configuration
    UI_PORT = int(os.getenv("UI_PORT", "8000"))
    API_PORT = int(os.getenv("API_PORT", "8080"))
    MCP_PORT = int(os.getenv("MCP_PORT", "8001"))
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    @classmethod
    def validate(cls) -> Dict[str, Any]:
        """Validate configuration"""
        issues = []
        
        if not cls.NEBIUS_API_KEY:
            issues.append("NEBIUS_API_KEY is required")
        
        if cls.MAX_IMAGES <= 0:
            issues.append("MAX_IMAGES must be positive")
        
        if cls.MAX_PROMPT_LENGTH <= 0:
            issues.append("MAX_PROMPT_LENGTH must be positive")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "api": {
                "base_url": cls.NEBIUS_BASE_URL,
                "model": cls.MODEL_NAME,
                "temperature": cls.DEFAULT_TEMPERATURE,
                "top_p": cls.DEFAULT_TOP_P,
                "max_tokens": cls.DEFAULT_MAX_TOKENS,
            },
            "limits": {
                "max_images": cls.MAX_IMAGES,
                "max_image_size": cls.MAX_IMAGE_SIZE,
                "max_prompt_length": cls.MAX_PROMPT_LENGTH,
                "request_timeout": cls.REQUEST_TIMEOUT,
            },
            "cache": {
                "enabled": cls.ENABLE_CACHE,
                "size": cls.CACHE_SIZE,
            },
            "ports": {
                "ui": cls.UI_PORT,
                "api": cls.API_PORT,
                "mcp": cls.MCP_PORT,
            }
        }