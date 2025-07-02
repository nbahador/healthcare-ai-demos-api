import unittest
from unittest.mock import Mock, patch
import json
import io
from PIL import Image

class TestMultiModalAgent(unittest.TestCase):
    """Test cases for the MultiModalAgent class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.api_key = "test_api_key"
        # We'll mock the agent to avoid actual API calls
    
    @patch('agent.requests.post')
    def test_agent_initialization(self, mock_post):
        """Test agent initialization"""
        from agent import MultiModalAgent
        
        agent = MultiModalAgent(self.api_key)
        self.assertEqual(agent.api_key, self.api_key)
        self.assertEqual(agent.model, "Qwen/Qwen2.5-VL-72B-Instruct")
    
    def test_image_encoding(self):
        """Test image encoding functionality"""
        from agent import MultiModalAgent
        
        agent = MultiModalAgent(self.api_key)
        
        # Create a simple test image
        img = Image.new('RGB', (100, 100), color='red')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_data = img_bytes.getvalue()
        
        encoded = agent._encode_image(img_data)
        self.assertIsInstance(encoded, str)
        self.assertTrue(len(encoded) > 0)
    
    def test_message_preparation(self):
        """Test message preparation for API calls"""
        from agent import MultiModalAgent
        
        agent = MultiModalAgent(self.api_key)
        prompt = "Test prompt"
        
        # Test text-only message
        messages = agent._prepare_messages(prompt)
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(len(messages[0]["content"]), 1)
        self.assertEqual(messages[0]["content"][0]["type"], "text")
        
        # Test with images
        img = Image.new('RGB', (50, 50), color='blue')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='JPEG')
        img_data = img_bytes.getvalue()
        
        messages = agent._prepare_messages(prompt, [img_data])
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(messages[0]["content"]), 2)  # text + image

class TestUtils(unittest.TestCase):
    """Test cases for utility functions"""
    
    def test_image_validation(self):
        """Test image validation"""
        from utils import ImageProcessor
        
        # Create valid image
        img = Image.new('RGB', (100, 100), color='green')
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        valid_img_data = img_bytes.getvalue()
        
        self.assertTrue(ImageProcessor.validate_image(valid_img_data))
        
        # Test invalid image data
        invalid_data = b"not an image"
        self.assertFalse(ImageProcessor.validate_image(invalid_data))
    
    def test_image_info(self):
        """Test image info extraction"""
        from utils import ImageProcessor
        
        img = Image.new('RGBA', (200, 150), color=(255, 0, 0, 128))
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_data = img_bytes.getvalue()
        
        info = ImageProcessor.get_image_info(img_data)
        self.assertEqual(info['width'], 200)
        self.assertEqual(info['height'], 150)
        self.assertEqual(info['format'], 'PNG')
        self.assertTrue(info['has_transparency'])
    
    def test_request_validation(self):
        """Test request validation"""
        from utils import RequestValidator
        
        # Valid prompt
        result = RequestValidator.validate_prompt("Valid prompt")
        self.assertTrue(result['valid'])
        
        # Empty prompt
        result = RequestValidator.validate_prompt("")
        self.assertFalse(result['valid'])
        
        # Too long prompt
        long_prompt = "x" * 20000
        result = RequestValidator.validate_prompt(long_prompt)
        self.assertFalse(result['valid'])
    
    def test_report_formatting(self):
        """Test report formatting"""
        from utils import ReportFormatter
        
        report_data = {
            'report': 'Test report content',
            'timestamp': '2024-01-01T12:00:00',
            'metadata': {
                'report_type': 'comprehensive',
                'model': 'test-model',
                'num_images': 2,
                'original_prompt': 'Test prompt'
            }
        }
        
        # Test markdown formatting
        md_content = ReportFormatter.to_markdown(report_data)
        self.assertIn('# AI Analysis Report', md_content)
        self.assertIn('Test report content', md_content)
        self.assertIn('comprehensive', md_content)
        
        # Test HTML formatting
        html_content = ReportFormatter.to_html(report_data)
        self.assertIn('<html>', html_content)
        self.assertIn('Test report content', html_content)

class TestConfig(unittest.TestCase):
    """Test configuration management"""
    
    def test_config_validation(self):
        """Test configuration validation"""
        from config import Config
        
        # This would test actual config validation
        # In a real scenario, you'd mock environment variables
        validation = Config.validate()
        self.assertIn('valid', validation)
        self.assertIn('issues', validation)
    
    def test_config_to_dict(self):
        """Test configuration dictionary conversion"""
        from config import Config
        
        config_dict = Config.to_dict()
        self.assertIn('api', config_dict)
        self.assertIn('limits', config_dict)
        self.assertIn('cache', config_dict)
        self.assertIn('ports', config_dict)

if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)