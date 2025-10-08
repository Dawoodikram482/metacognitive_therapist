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
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "src.api:app", 
            "--reload", 
            "--host", "localhost", 
            "--port", "8000",
            "--timeout-keep-alive", "120",  # Keep connections alive longer
            "--timeout-graceful-shutdown", "30"  # Graceful shutdown timeout
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Backend failed to start: {e}")
        print("Check if port 8000 is already in use")
    except Exception as e:
        print(f"❌ Unexpected backend error: {e}")

def run_frontend():
    """Run Streamlit frontend"""
    print("🎨 Starting Streamlit Frontend...")
    
    # Wait for backend to be ready
    print("⏳ Waiting for backend to be ready...")
    time.sleep(5)  # Give backend more time to start
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/streamlit_app.py", 
            "--server.port", "8501",
            "--server.address", "localhost"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Frontend failed to start: {e}")
        print("Check if port 8501 is already in use")
    except Exception as e:
        print(f"❌ Unexpected frontend error: {e}")

def check_prerequisites():
    """Check if system is ready to run"""
    print("🔍 Checking prerequisites...")
    
    # Check required files
    required_files = ["src/api.py", "src/streamlit_app.py", "src/vector_database.py"]
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Missing required file: {file_path}")
            return False
    
    # Check if vector database is populated
    try:
        from src.vector_database import AdvancedVectorDatabase
        db = AdvancedVectorDatabase()
        if not db.is_populated():
            print("❌ Vector database not populated.")
            print("   Run first: python main.py --setup 'data/Raw data.pdf'")
            return False
        
        stats = db.get_collection_stats()
        total_docs = sum(collection_stats.get("document_count", 0) for collection_stats in stats.values())
        print(f"✅ Vector database ready ({total_docs} documents)")
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return False
    
    print("✅ All prerequisites met")
    return True

def main():
    """Main runner function"""
    print("🧠 Metacognitive Therapy Chatbot System")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    # Check prerequisites
    if not check_prerequisites():
        print("\n🔧 Setup required before running the web system")
        return
    
    print("\nStarting system components...")
    print("📍 Backend will be available at: http://localhost:8000")
    print("🌐 Frontend will be available at: http://localhost:8501")
    print("📚 API Documentation: http://localhost:8000/docs")
    print("⏳ Please wait for both services to start...")
    print("🛑 Press Ctrl+C to stop both services")
    
    try:
        # Start backend in a separate thread
        backend_thread = threading.Thread(target=run_backend, daemon=True)
        backend_thread.start()
        
        print("✅ Backend starting...")
        print("🎨 Starting frontend (this may take a moment)...")
        
        # Start frontend (this will block)
        run_frontend()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down system...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you're in the project root directory")
        print("2. Check that ports 8000 and 8501 are not in use")
        print("3. Verify the vector database is set up")
        sys.exit(1)
if __name__ == "__main__":
    main()