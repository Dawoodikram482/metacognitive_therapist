#!/usr/bin/env python3
"""
Convenient system runner for Metacognitive Therapy Chatbot

This script provides easy startup options for different system components.
"""

import subprocess
import sys
import time
import threading
from pathlib import Path

def run_backend():
    """Run FastAPI backend server"""
    print("🚀 Starting FastAPI Backend...")
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "src.api:app", 
        "--reload", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ])

def run_frontend():
    """Run Streamlit frontend"""
    print("🎨 Starting Streamlit Frontend...")
    time.sleep(3)  # Wait for backend to start
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "src/streamlit_app.py", 
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def main():
    """Main runner function"""
    print("🧠 Metacognitive Therapy Chatbot System")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    print("Starting system components...")
    
    try:
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        # Start frontend (this will block)
        run_frontend()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down system...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()