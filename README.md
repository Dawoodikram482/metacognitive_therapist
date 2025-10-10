# 🧠 Metacognitive Therapy Chatbot

**An Advanced AI-Powered Therapy Assistant using RAG, Local LLMs, and Metacognitive Therapy Principles**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

This project implements a sophisticated **Retrieval-Augmented Generation (RAG)** chatbot specifically designed for **metacognitive therapy** guidance. Built entirely with **free, open-source technologies**, it demonstrates advanced NLP capabilities, vector databases, local LLM inference, and modern web development practices.


## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
│  • Streamlit Web UI with Professional Styling               │
│  • Real-time Chat Interface                                 │
│  • Analytics Dashboard & Progress Tracking                  │
│  • Crisis Resource Integration                              │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   API Gateway Layer                         │
│  • FastAPI Backend with OpenAPI Documentation               │
│  • Session Management & Authentication                      │
│  • Rate Limiting & Security Middleware                      │
│  • Comprehensive Error Handling                             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              RAG Orchestration Layer                        │
│  • Advanced Query Classification                            │
│  • Multi-Strategy Document Retrieval                        │
│  • Context Compression & Optimization                       │
│  • Conversation Memory Management                           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                Vector Database Layer                        │
│  • ChromaDB with 5 Specialized Collections                  │
│  • Semantic Similarity Search                               │
│  • Advanced Metadata Filtering                              │
│  • Hybrid Search Capabilities                               │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│               Local LLM & Embedding Layer                   │
│  • GPT4All Orca Mini 3B                                     │
│  • SentenceTransformers all-MiniLM-L6-v2 Embeddings         │
│  • Unlimited Generation Time (No Artificial Timeouts)       │
│  • Enhanced Response Quality & Completeness                 │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

### 🤖 **Advanced AI Capabilities**
- **Retrieval-Augmented Generation (RAG)** with sophisticated context management and fixed similarity scoring
- **Local LLM Inference** - GPT4All Orca Mini 3B model, completely free, no API costs
- **Multi-Collection Vector Database** (6 specialized collections) for categorized therapy content
- **Intelligent Query Classification** and response routing with metadata filtering
- **Context-Aware Conversations** with session management and conversation memory
- **Unlimited Generation Time** - no artificial timeouts, complete responses guaranteed



### 🧠 **Therapy-Specific Features**
- **Metacognitive Therapy Techniques** integration with evidence-based responses
- **Document Processing** with advanced PDF parsing and semantic chunking
- **Therapeutic Content Management** across 6 specialized collections
- **Professional Disclaimers** and safety considerations
- **Evidence-Based Content** from validated therapy literature

### 💰 **100% Free Technologies**
- **No API costs** - everything runs locally with optimized models
- **Open-source models** (GPT4All Orca Mini 3B, SentenceTransformers)
- **Free hosting options** (Streamlit Cloud, Render, Railway)
- **No paid subscriptions** or external service dependencies

## 📁 Project Structure

```
metacognitive_therapist/
├── 📂 src/                          # Core application code
│   ├── 🐍 document_processor.py     # Advanced PDF processing & chunking
│   ├── 🗄️ vector_database.py        # ChromaDB management & search
│   ├── 🤖 rag_system.py             # RAG orchestration & context mgmt
│   ├── 🧠 llm_manager.py            # Local LLM management & safety
│   ├── 🌐 api.py                    # FastAPI backend server
│   └── 🎨 streamlit_app.py          # Frontend user interface
├── 📂 data/                         # Therapy literature storage
│   └── 📄 Raw data.pdf              # Your therapy book/literature
├── 📂 notebooks/                    # Jupyter notebooks for experimentation
├── 🐍 main.py                       # System orchestrator & demos
├── 📋 requirements.txt              # Python dependencies
├── 🚀 run_system.py                 # Easy startup script
├── 🔧 setup.py                      # Easy setup script
└── 📖 README.md                     # This documentation
```

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.8+** 
- **8GB+ RAM** (for local LLM inference)
- **5GB+ Storage** (for models and vector database)

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/Dawoodikram482/metacognitive_therapist.git
cd metacognitive_therapist

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Additional Models
```bash
# Download spaCy English model
python -m spacy download en_core_web_sm

# The LLM models will download automatically on first use
```

### 3. Setup Knowledge Base
```bash
# Place your therapy literature PDF in the data/ folder
# Then run the setup
python main.py --setup "data/Raw data.pdf"
```

### 4. System Check
```bash
# Verify everything is working
python main.py --check
```

## 🚀 Running the Application

### Option 1: Quick Start (Recommended)
```bash
# Use the convenient runner script - starts both backend and frontend
python run_system.py
```

### Option 2: Manual Setup (Advanced)
```bash
# Terminal 1: Start the FastAPI backend with extended timeouts
cd src
uvicorn api:app --reload --host localhost --port 8000 --timeout-keep-alive 120

# Terminal 2: Start the Streamlit frontend
streamlit run src/streamlit_app.py --server.port 8501
```

### Option 3: CLI Demo
```bash
# Run interactive CLI demo for testing
python main.py --demo
```

**⚠️ Important:** The system now allows unlimited generation time for complete responses. Initial responses may take longer but will be more comprehensive and complete.

## 🌐 Accessing the Application

Once running, access these URLs:

- **🎨 Main Chat Interface**: http://localhost:8501 
- **📚 API Documentation**: http://localhost:8000/docs 
- **🔧 API Health Check**: http://localhost:8000/health
- **📊 System Statistics**: http://localhost:8000/admin/stats

## 🧪 Testing & Validation

### System Check
```bash
python main.py --check
```

### Interactive Testing
```bash
python main.py --demo
```

### API Testing
```bash
# Test health endpoint
curl http://localhost:8000/health

# Test chat endpoint
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"message": "I feel anxious about work"}'
```

### **Architecture Decisions:**
- **Why Local LLMs?** Cost-effective, privacy-preserving, no API dependencies
- **Why ChromaDB?** Open-source, lightweight, perfect for RAG applications
- **Why FastAPI?** Modern, fast, automatic documentation, great for ML APIs
- **Why Streamlit?** Rapid prototyping, built for data science, easy deployment


## 🔒 Privacy & Ethics

### Privacy Features:
- **Local processing** - no data sent to external APIs
- **Session-based storage** - no permanent user data
- **Anonymized analytics** - no personal information stored
- **Secure communication** - HTTPS ready

### Ethical Considerations:
- **Clear disclaimers** about AI limitations
- **Crisis resource integration** for emergencies  
- **Professional help recommendations** when appropriate
- **Evidence-based content** from validated therapy literature
- **Safety filtering** to prevent harmful advice

### Performance Tips
- **First Run**: Model download may take time, subsequent runs are faster
- **Memory**: Ensure 8GB+ RAM for optimal LLM performance
- **Storage**: Keep 5GB+ free space for models and vector database
- **Patience**: First response may take longer as the model loads, subsequent responses are faster

## �📚 Learning Resources

### Metacognitive Therapy:
- Adrian Wells' MCT books and research
- MCT Institute resources
- Clinical psychology journals

### Technical Resources:
- LangChain documentation
- ChromaDB guides
- GPT4All model hub
- FastAPI tutorials
- Streamlit gallery

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Adrian Wells** for Metacognitive Therapy development
- **LangChain team** for the excellent RAG framework
- **GPT4All project** for free local LLM access
- **ChromaDB** for the vector database solution
- **Streamlit** for the amazing frontend framework


---

**⚠️ Important Disclaimer**: This AI assistant is for educational and supportive purposes only. It does not replace professional mental health care, diagnosis, or treatment. If you're experiencing a mental health crisis, please contact emergency services or a mental health professional immediately.