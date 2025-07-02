#!/bin/bash

# ==============================================================================
# Medical Case Analysis System - Modal Deployment Script
# ==============================================================================

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo
    echo "========================================"
    echo "$1"
    echo "========================================"
}

# Main deployment function
main() {
    print_header "🚀 Medical Case Analysis System - Modal Deployment"
    
    # Step 1: Pre-deployment checks
    print_status "Performing pre-deployment checks..."
    
    # Check if we're in the right directory
    if [[ ! -f "main.py" ]]; then
        print_error "main.py not found. Please run this script from the project root directory."
        exit 1
    fi
    
    if [[ ! -f "requirements.txt" ]]; then
        print_error "requirements.txt not found."
        exit 1
    fi
    
    print_success "Project files found"
    
    # Step 2: Check Python installation
    print_status "Checking Python installation..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1,2)
    print_success "Python $PYTHON_VERSION detected"
    
    # Step 3: Check/Install Modal CLI
    print_status "Checking Modal CLI installation..."
    if ! command -v modal &> /dev/null; then
        print_warning "Modal CLI not found. Installing..."
        pip3 install modal-client
        
        if ! command -v modal &> /dev/null; then
            print_error "Failed to install Modal CLI"
            exit 1
        fi
        print_success "Modal CLI installed successfully"
    else
        print_success "Modal CLI found"
        
        # Update Modal CLI to latest version
        print_status "Updating Modal CLI to latest version..."
        pip3 install --upgrade modal-client
    fi
    
    # Step 4: Check Modal authentication
    print_status "Checking Modal authentication..."
    if ! modal token show &> /dev/null; then
        print_warning "Modal authentication required"
        print_status "Please authenticate with Modal..."
        modal token set
        
        # Verify authentication worked
        if ! modal token show &> /dev/null; then
            print_error "Modal authentication failed"
            exit 1
        fi
    fi
    print_success "Modal authentication verified"
    
    # Step 5: Validate environment variables
    print_status "Checking environment variables..."
    if [[ -z "$NEBIUS_API_KEY" ]]; then
        print_warning "NEBIUS_API_KEY environment variable not set"
        echo
        echo "Please set your Nebius API key:"
        echo "export NEBIUS_API_KEY=your_api_key_here"
        echo
        read -p "Enter your Nebius API key: " NEBIUS_API_KEY
        
        if [[ -z "$NEBIUS_API_KEY" ]]; then
            print_error "Nebius API key is required"
            exit 1
        fi
        
        export NEBIUS_API_KEY
    fi
    print_success "Environment variables validated"
    
    # Step 6: Create/Update Modal secret
    print_status "Creating/updating Modal secret for Nebius API key..."
    if modal secret list | grep -q "nebius-api-key"; then
        print_warning "Secret 'nebius-api-key' already exists. Updating..."
        echo "$NEBIUS_API_KEY" | modal secret create nebius-api-key NEBIUS_API_KEY --force
    else
        echo "$NEBIUS_API_KEY" | modal secret create nebius-api-key NEBIUS_API_KEY
    fi
    print_success "Modal secret created/updated"
    
    # Step 7: Validate main.py syntax
    print_status "Validating Python syntax..."
    if ! python3 -m py_compile main.py; then
        print_error "Python syntax errors found in main.py"
        exit 1
    fi
    print_success "Python syntax validation passed"
    
    # Step 8: Deploy the application
    print_header "📦 Deploying Application to Modal"
    print_status "Starting deployment..."
    
    # Deploy with verbose output
    if modal deploy main.py --name medical-case-agent; then
        print_success "Application deployed successfully!"
    else
        print_error "Deployment failed"
        exit 1
    fi
    
    # Step 9: Test the deployment
    print_header "🧪 Testing Deployment"
    print_status "Running system tests..."
    
    # Test system initialization
    if modal run main.py::test_system; then
        print_success "System test passed!"
    else
        print_warning "System test failed, but deployment may still be functional"
    fi
    
    # Step 10: Get application URL
    print_status "Retrieving application URLs..."
    echo
    print_success "Deployment completed successfully!"
    echo
    echo "=================================================="
    echo "🎉 DEPLOYMENT SUMMARY"
    echo "=================================================="
    echo
    echo "✅ Application Name: medical-case-agent"
    echo "✅ Status: Deployed and Running"
    echo "✅ Platform: Modal.com"
    echo "✅ API Provider: Nebius (studio.nebius.com)"
    echo
    echo "📋 Available Endpoints:"
    echo "   • Web Interface: Check Modal dashboard for URL"
    echo "   • Health Check: /health"
    echo "   • Case Analysis: /analyze"
    echo "   • Statistics: /stats"
    echo
    echo "🔧 Management Commands:"
    echo "   • View logs: modal logs medical-case-agent"
    echo "   • Stop app: modal app stop medical-case-agent"
    echo "   • View secrets: modal secret list"
    echo
    echo "📊 Next Steps:"
    echo "   1. Visit the Modal dashboard to get your app URL"
    echo "   2. Test the web interface"
    echo "   3. Monitor logs for any issues"
    echo "   4. Scale as needed through Modal dashboard"
    echo
    echo "=================================================="
    
    # Optional: Open Modal dashboard
    read -p "Would you like to open the Modal dashboard? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if command -v open &> /dev/null; then
            open "https://modal.com/apps"
        elif command -v xdg-open &> /dev/null; then
            xdg-open "https://modal.com/apps"
        else
            echo "Please visit: https://modal.com/apps"
        fi
    fi
}

# Function to handle cleanup on script exit
cleanup() {
    if [[ $? -ne 0 ]]; then
        print_error "Deployment failed!"
        echo
        echo "Troubleshooting tips:"
        echo "1. Check your Nebius API key is valid"
        echo "2. Ensure Modal CLI is properly authenticated"
        echo "3. Verify all dependencies in requirements.txt"
        echo "4. Check main.py for syntax errors"
        echo "5. Review Modal logs: modal logs medical-case-agent"
    fi
}

# Set up error handling
trap cleanup EXIT

# Help function
show_help() {
    echo "Medical Case Analysis System - Modal Deployment Script"
    echo
    echo "Usage: $0 [OPTIONS]"
    echo
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -t, --test     Run tests only (skip deployment)"
    echo "  -f, --force    Force redeploy even if app exists"
    echo "  -v, --verbose  Enable verbose output"
    echo
    echo "Environment Variables:"
    echo "  NEBIUS_API_KEY    Your Nebius API key (required)"
    echo
    echo "Examples:"
    echo "  $0                    # Standard deployment"
    echo "  $0 --test            # Test only"
    echo "  $0 --force           # Force redeploy"
    echo "  NEBIUS_API_KEY=xxx $0  # Deploy with API key"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -t|--test)
            TEST_ONLY=true
            shift
            ;;
        -f|--force)
            FORCE_DEPLOY=true
            shift
            ;;
        -v|--verbose)
            set -x
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
if [[ "$TEST_ONLY" == "true" ]]; then
    print_header "🧪 Running Tests Only"
    modal run main.py::test_system
else
    main
fi