"""
Setup script for Metacognitive Therapy Chatbot

This script handles initial setup, dependency installation, and system configuration.
"""

import subprocess
import sys
import os
from pathlib import Path
import requests

def run_command(command, description):
    """Run a shell command with proper error handling"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    commands = [
        "pip install --upgrade pip",
        "pip install -r requirements.txt",
        "python -m spacy download en_core_web_sm"
    ]
    
    for cmd in commands:
        if not run_command(cmd, f"Running: {cmd}"):
            return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    directories = [
        "data",
        "index",
        "index/chroma_db", 
        "models",
        "notebooks",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def create_env_file():
    """Create environment configuration file"""
    print("⚙️  Creating environment configuration...")
    
    env_content = """# Metacognitive Therapy Chatbot Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
FRONTEND_PORT=8501

# Model Configuration  
DEFAULT_MODEL=GPT4ALL_MISTRAL
MAX_TOKENS=512
TEMPERATURE=0.7

# Database Configuration
DB_PATH=./index/chroma_db
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Security (generate secure keys for production)
SECRET_KEY=your-secret-key-here
SESSION_TIMEOUT=3600

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log

# Rate Limiting
RATE_LIMIT_REQUESTS=20
RATE_LIMIT_WINDOW=60

# Development
DEBUG=True
RELOAD=True
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env configuration file")
    return True

def download_sample_data():
    """Download sample therapy data if needed"""
    print("📚 Checking for sample data...")
    
    data_file = Path("data/Raw data.pdf")
    if data_file.exists():
        print("✅ Sample data already exists")
        return True
    
    print("ℹ️  No sample data found. You'll need to add your therapy literature to data/Raw data.pdf")
    
    # Create a placeholder file with instructions
    instructions = """# Therapy Literature Instructions

To complete the setup of your metacognitive therapy chatbot:

1. Obtain a PDF book or document on metacognitive therapy
   - Recommended: Adrian Wells' "Metacognitive Therapy for Anxiety and Depression"
   - Or any evidence-based therapy literature in PDF format

2. Rename your PDF file to "Raw data.pdf"

3. Place it in this data/ directory

4. Run the setup command:
   python main.py --setup "data/Raw data.pdf"

This will process the document and create your knowledge base for the chatbot.

Note: Ensure you have proper licensing/permission to use the therapeutic content.
"""
    
    with open("data/README.txt", "w") as f:
        f.write(instructions)
    
    print("✅ Created data directory with instructions")
    return True

def verify_installation():
    """Verify that installation was successful"""
    print("🔍 Verifying installation...")
    
    # Check if key dependencies can be imported
    key_imports = [
        "gpt4all",
        "langchain", 
        "chromadb",
        "sentence_transformers",
        "fastapi",
        "streamlit",
        "pypdf"
    ]
    
    failed_imports = []
    for module in key_imports:
        try:
            __import__(module.replace("-", "_"))
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"❌ Installation verification failed. Missing modules: {failed_imports}")
        return False
    
    print("✅ All key dependencies verified successfully")
    return True

def main():
    """Main setup function"""
    print("🧠 Metacognitive Therapy Chatbot Setup")
    print("=" * 50)
    
    setup_steps = [
        ("Python Version Check", check_python_version),
        ("Directory Creation", create_directories),
        ("Environment Configuration", create_env_file),
        ("Dependency Installation", install_dependencies),
        ("Sample Data Setup", download_sample_data),
        ("Installation Verification", verify_installation)
    ]
    
    failed_steps = []
    
    for step_name, step_function in setup_steps:
        print(f"\n📋 Step: {step_name}")
        print("-" * 30)
        
        if not step_function():
            failed_steps.append(step_name)
            print(f"❌ Step '{step_name}' failed")
        else:
            print(f"✅ Step '{step_name}' completed")
    
    print("\n" + "=" * 50)
    
    if failed_steps:
        print(f"❌ Setup completed with {len(failed_steps)} failed steps:")
        for step in failed_steps:
            print(f"   • {step}")
        print("\nPlease resolve the issues above and run setup again.")
        return False
    else:
        print("🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Add your therapy literature PDF to data/Raw data.pdf")
        print("2. Run: python main.py --setup 'data/Raw data.pdf'")
        print("3. Run: python main.py --check (to verify everything works)")
        print("4. Run: python run_system.py (to start the application)")
        print("\nFor help: python main.py --help")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)