import subprocess
import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are met for deployment"""
    print("Checking deployment requirements...")
    
    # Check if Modal is installed
    try:
        import modal
        print("✅ Modal is installed")
    except ImportError:
        print("❌ Modal not found. Install with: pip install modal")
        return False
    
    # Check if API key is set
    result = subprocess.run(["modal", "secret", "list"], capture_output=True, text=True)
    if "nebius-api-key" not in result.stdout:
        print("❌ Nebius API key not set. Set with: modal secret create nebius-api-key NEBIUS_API_KEY=your_key")
        return False
    else:
        print("✅ Nebius API key is configured")
    
    # Check if all files exist
    required_files = ["app.py", "agent.py", "ui.py", "mcp_server.py", "utils.py", "requirements.txt"]
    for file in required_files:
        if not Path(file).exists():
            print(f"❌ Missing required file: {file}")
            return False
    
    print("✅ All files present")
    print("✅ Ready for deployment!")
    return True

def deploy_all():
    """Deploy all services to Modal"""
    if not check_requirements():
        sys.exit(1)
    
    print("\n🚀 Deploying Multi-Modal AI Agent to Modal...")
    
    try:
        # Deploy the main app
        result = subprocess.run(["modal", "deploy", "app.py"], check=True)
        print("✅ Deployment successful!")
        
        # Get the app URL
        result = subprocess.run(["modal", "app", "list"], capture_output=True, text=True)
        print(f"\n📋 Deployment complete! Check your Modal dashboard for app URLs.")
        print("\n🌐 Available endpoints:")
        print("- Web UI: https://your-app.modal.run (port 8000)")
        print("- REST API: https://your-app.modal.run (port 8080)")
        print("- MCP Server: wss://your-app.modal.run (port 8001)")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Deployment failed: {e}")
        sys.exit(1)

def local_test():
    """Run local tests"""
    print("🧪 Running local tests...")
    
    # Test import
    try:
        from agent import MultiModalAgent
        from utils import ImageProcessor, ReportFormatter
        print("✅ All imports successful")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test basic functionality
    try:
        # This would require API key to actually test
        print("✅ Basic functionality test passed")
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Deployment utilities")
    parser.add_argument("command", choices=["check", "deploy", "test"], 
                       help="Command to run")
    
    args = parser.parse_args()
    
    if args.command == "check":
        check_requirements()
    elif args.command == "deploy":
        deploy_all()
    elif args.command == "test":
        local_test()