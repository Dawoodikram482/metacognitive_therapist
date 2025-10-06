# 🧠 Metacognitive Therapy Chatbot

**An Advanced AI-Powered Therapy Assistant using RAG, Local LLMs, and Metacognitive Therapy Principles**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🌟 Overview

This project implements a sophisticated **Retrieval-Augmented Generation (RAG)** chatbot specifically designed for **metacognitive therapy** guidance. Built entirely with **free, open-source technologies**, it demonstrates advanced NLP capabilities, vector databases, local LLM inference, and modern web development practices.

### 🎯 Perfect for CV/Portfolio because it showcases:
- **Advanced RAG Implementation** with multi-collection vector databases
- **Local LLM Integration** (GPT4All, Ollama) - completely free
- **Professional FastAPI Backend** with authentication, rate limiting, analytics
- **Modern Frontend** with Streamlit and real-time chat
- **Sophisticated NLP Pipeline** with semantic chunking and metadata extraction
- **Production-Ready Architecture** with proper error handling, logging, monitoring
- **Domain Expertise** in mental health/therapy applications

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                           │
│  • Streamlit Web UI with Professional Styling              │
│  • Real-time Chat Interface                                │
│  • Analytics Dashboard & Progress Tracking                 │
│  • Crisis Resource Integration                             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   API Gateway Layer                        │
│  • FastAPI Backend with OpenAPI Documentation              │
│  • Session Management & Authentication                     │
│  • Rate Limiting & Security Middleware                     │
│  • Comprehensive Error Handling                            │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│              RAG Orchestration Layer                       │
│  • Advanced Query Classification                           │
│  • Multi-Strategy Document Retrieval                      │
│  • Context Compression & Optimization                     │
│  • Conversation Memory Management                         │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                Vector Database Layer                       │
│  • ChromaDB with 5 Specialized Collections                │
│  • Semantic Similarity Search                             │
│  • Advanced Metadata Filtering                            │
│  • Hybrid Search Capabilities                             │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│               Local LLM & Embedding Layer                 │
│  • GPT4All (Mistral, Llama2, Orca) - FREE                │
│  • SentenceTransformers Embeddings                       │
│  • Safety Filtering & Therapeutic Optimization           │
│  • Fallback Model Support                                │
└─────────────────────────────────────────────────────────────┘
```

## 🚀 Key Features

### 🤖 **Advanced AI Capabilities**
- **Retrieval-Augmented Generation (RAG)** with sophisticated context management
- **Local LLM Inference** - completely free, no API costs
- **Multi-Collection Vector Database** for specialized therapy content
- **Intelligent Query Classification** and response routing
- **Context-Aware Conversations** with memory management

### 🔧 **Technical Excellence**
- **Professional FastAPI Backend** with OpenAPI documentation
- **Modern Streamlit Frontend** with real-time features
- **Comprehensive Error Handling** and logging
- **Session Management** and user tracking
- **Rate Limiting** and security features
- **Analytics & Monitoring** dashboard

### 🧠 **Therapy-Specific Features**
- **Metacognitive Therapy Techniques** integration
- **Crisis Detection** and resource recommendations
- **Therapeutic Response Optimization** with safety filtering
- **Progress Tracking** and conversation insights
- **Evidence-Based Content** from therapy literature

### 💰 **100% Free Technologies**
- **No API costs** - everything runs locally
- **Open-source models** (GPT4All, SentenceTransformers)
- **Free hosting options** (Streamlit Cloud, Render, Railway)
- **No paid subscriptions** required

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
├── 📂 index/                        # Vector database storage
│   └── 🗃️ chroma_db/               # ChromaDB persistent storage
├── 📂 models/                       # Downloaded LLM models
├── 📂 notebooks/                    # Jupyter notebooks for experimentation
├── 🐍 main.py                       # System orchestrator & demos
├── 📋 requirements.txt              # Python dependencies
├── 🚀 run_system.py                 # Easy startup script
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
git clone https://github.com/yourusername/metacognitive_therapist.git
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

### Option 1: Complete System (Recommended)
```bash
# Terminal 1: Start the FastAPI backend
cd src
uvicorn api:app --reload --host 0.0.0.0 --port 8000

# Terminal 2: Start the Streamlit frontend
streamlit run src/streamlit_app.py --server.port 8501
```

### Option 2: Quick Demo
```bash
# Run interactive CLI demo
python main.py --demo
```

### Option 3: Development Mode
```bash
# Use the convenient runner script
python run_system.py
```

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

## 📈 Deployment Options (All Free!)

### 1. Streamlit Cloud (Easiest)
- Push to GitHub
- Connect at [share.streamlit.io](https://share.streamlit.io)
- Deploy with one click
- **Cost: FREE**

### 2. Render (Full Stack)
- Deploy FastAPI backend on Render
- Deploy Streamlit frontend separately
- **Cost: FREE tier available**

### 3. Railway (Modern)
- Deploy both services with railway.app
- Automatic deployments from GitHub
- **Cost: FREE tier available**

### 4. Self-Hosted
- Use Docker containers
- Deploy on your own VPS
- **Cost: Just server costs**

## 🎯 CV/Portfolio Highlights

### **Technical Skills Demonstrated:**

**🤖 AI/ML:**
- Retrieval-Augmented Generation (RAG) implementation
- Vector database design and optimization
- Local LLM integration and management
- Natural Language Processing pipelines
- Semantic search and similarity matching

**🌐 Web Development:**
- FastAPI backend development
- RESTful API design
- Modern frontend with Streamlit
- Real-time chat applications
- Session management and authentication

**🗄️ Data Engineering:**
- Vector database management (ChromaDB)
- Document processing and chunking
- Metadata extraction and indexing
- Advanced search algorithms
- Data pipeline optimization

**🔧 Software Engineering:**
- Clean, modular architecture
- Comprehensive error handling
- Logging and monitoring
- Testing and validation
- Professional documentation

**🧠 Domain Expertise:**
- Mental health application development
- Therapy technique integration
- Safety filtering and crisis detection
- Evidence-based content management
- User experience for sensitive applications

### **Architecture Decisions:**
- **Why Local LLMs?** Cost-effective, privacy-preserving, no API dependencies
- **Why ChromaDB?** Open-source, lightweight, perfect for RAG applications
- **Why FastAPI?** Modern, fast, automatic documentation, great for ML APIs
- **Why Streamlit?** Rapid prototyping, built for data science, easy deployment

## 🔍 System Components Deep Dive

### Document Processor (`document_processor.py`)
- **Advanced PDF parsing** with formatting preservation
- **Semantic chunking** using multiple strategies
- **Metadata extraction** for therapy-specific concepts
- **Quality filtering** and optimization
- **spaCy integration** for NLP analysis

### Vector Database (`vector_database.py`)
- **5 specialized collections** for different content types
- **Hybrid search** with semantic and keyword matching
- **Advanced ranking** algorithms
- **Metadata filtering** and boosting
- **Backup and recovery** functionality

### RAG System (`rag_system.py`)
- **Query classification** for appropriate response routing
- **Multi-step retrieval** with context compression
- **Conversation memory** management
- **Context optimization** for token efficiency
- **Response quality** scoring and validation

### LLM Manager (`llm_manager.py`)
- **Multiple model support** (GPT4All, Ollama)
- **Safety filtering** for therapeutic appropriateness
- **Crisis detection** and response protocols
- **Performance monitoring** and fallback handling
- **Therapeutic optimization** for response quality

## 📊 Performance Metrics

The system tracks comprehensive metrics:
- **Response time** and latency monitoring
- **Model performance** and success rates
- **User engagement** and session analytics
- **System health** and resource usage
- **Content effectiveness** and retrieval quality

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

## 🤝 Contributing

This project welcomes contributions! Areas for enhancement:

- **Additional therapy modalities** (CBT, DBT, etc.)
- **Multi-language support** 
- **Voice interface** integration
- **Mobile app** development
- **Additional LLM models** support
- **Advanced analytics** features

## 📚 Learning Resources

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

## 📞 Support & Contact

For questions, issues, or collaboration:
- **GitHub Issues**: [Project Issues](https://github.com/yourusername/metacognitive_therapist/issues)
- **Email**: your.email@example.com
- **LinkedIn**: [Your LinkedIn Profile]

---

**⚠️ Important Disclaimer**: This AI assistant is for educational and supportive purposes only. It does not replace professional mental health care, diagnosis, or treatment. If you're experiencing a mental health crisis, please contact emergency services or a mental health professional immediately.