"""
Main initialization and setup script for Metacognitive Therapy Chatbot

This script provides easy setup and initialization of the complete system.
"""

import logging
import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.document_processor import AdvancedDocumentProcessor, ProcessingConfig
from src.vector_database import AdvancedVectorDatabase
from src.rag_system import AdvancedRAGSystem, RAGConfig
from src.llm_manager import LLMManager, ModelType, GenerationConfig

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetacognitiveTherapistSystem:
    """Main system orchestrator for the Metacognitive Therapy Chatbot"""
    
    def __init__(self):
        self.document_processor = None
        self.vector_db = None
        self.llm_manager = None
        self.rag_system = None
        self.is_initialized = False
    
    async def initialize(self, model_type: ModelType = ModelType.GPT4ALL_MISTRAL):
        """Initialize all system components"""
        logger.info("Initializing Metacognitive Therapy Chatbot System...")
        
        try:
            # Initialize document processor
            logger.info("Initializing document processor...")
            processing_config = ProcessingConfig(
                chunk_size=1000,
                chunk_overlap=200,
                enable_semantic_chunking=True,
                enable_metadata_extraction=True
            )
            self.document_processor = AdvancedDocumentProcessor(processing_config)
            
            # Initialize vector database
            logger.info("Initializing vector database...")
            self.vector_db = AdvancedVectorDatabase()
            
            # Initialize LLM manager
            logger.info("Initializing LLM manager...")
            generation_config = GenerationConfig(
                max_tokens=512,
                temperature=0.7,
                enable_safety_filter=True,
                enable_therapeutic_optimization=True
            )
            self.llm_manager = LLMManager(model_type, generation_config)
            
            # Initialize RAG system
            logger.info("Initializing RAG system...")
            rag_config = RAGConfig(
                max_context_length=4000,
                max_retrieved_docs=8,
                similarity_threshold=0.7,
                enable_query_expansion=True,
                enable_context_compression=True
            )
            self.rag_system = AdvancedRAGSystem(self.vector_db, self.llm_manager, rag_config)
            
            self.is_initialized = True
            logger.info("✅ System initialization completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ System initialization failed: {e}")
            return False
    
    async def setup_knowledge_base(self, pdf_path: str):
        """Setup knowledge base from therapy literature"""
        if not self.is_initialized:
            logger.error("System not initialized. Call initialize() first.")
            return False
        
        try:
            logger.info(f"Processing therapy literature: {pdf_path}")
            
            # Process the PDF document
            pdf_file = Path(pdf_path)
            if not pdf_file.exists():
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            documents = self.document_processor.process_document(pdf_file)
            
            if not documents:
                logger.error("No documents were processed from the PDF")
                return False
            
            logger.info(f"Processed {len(documents)} document chunks")
            
            # Categorize and add documents to appropriate collections
            await self._categorize_and_store_documents(documents)
            
            logger.info("✅ Knowledge base setup completed!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Knowledge base setup failed: {e}")
            return False
    
    async def _categorize_and_store_documents(self, documents):
        """Categorize documents and store in appropriate collections"""
        
        # Simple categorization based on content
        collections_map = {
            "therapy_techniques": [],
            "theoretical_concepts": [],
            "case_studies": [],
            "therapeutic_exercises": [],
            "assessment_tools": []
        }
        
        for doc in documents:
            content_lower = doc.page_content.lower()
            
            # Categorize based on keywords
            if any(keyword in content_lower for keyword in ["technique", "method", "intervention", "att", "detached mindfulness"]):
                collections_map["therapy_techniques"].append(doc)
            elif any(keyword in content_lower for keyword in ["theory", "concept", "model", "framework", "research"]):
                collections_map["theoretical_concepts"].append(doc)
            elif any(keyword in content_lower for keyword in ["case", "patient", "client", "example", "clinical"]):
                collections_map["case_studies"].append(doc)
            elif any(keyword in content_lower for keyword in ["exercise", "practice", "homework", "activity", "training"]):
                collections_map["therapeutic_exercises"].append(doc)
            elif any(keyword in content_lower for keyword in ["assessment", "questionnaire", "scale", "measure", "test"]):
                collections_map["assessment_tools"].append(doc)
            else:
                # Default to therapy techniques
                collections_map["therapy_techniques"].append(doc)
        
        # Add documents to collections
        for collection_name, docs in collections_map.items():
            if docs:
                success = self.vector_db.add_documents(docs, collection_name)
                if success:
                    logger.info(f"Added {len(docs)} documents to {collection_name}")
                else:
                    logger.warning(f"Failed to add documents to {collection_name}")
    
    def chat(self, message: str, conversation_history: list = None) -> dict:
        """Process a chat message and return response"""
        if not self.is_initialized:
            return {"error": "System not initialized"}
        
        try:
            # Convert conversation history if provided
            history_messages = None
            if conversation_history:
                from langchain.schema import HumanMessage, AIMessage
                history_messages = []
                for msg in conversation_history:
                    if msg.get("role") == "user":
                        history_messages.append(HumanMessage(content=msg["content"]))
                    elif msg.get("role") == "assistant":
                        history_messages.append(AIMessage(content=msg["content"]))
            
            # Generate response using RAG system
            response = self.rag_system.generate_rag_response(message, history_messages)
            return response
            
        except Exception as e:
            logger.error(f"Chat processing error: {e}")
            return {"error": str(e)}
    
    def get_system_status(self) -> dict:
        """Get system status and statistics"""
        status = {
            "initialized": self.is_initialized,
            "components": {
                "document_processor": self.document_processor is not None,
                "vector_database": self.vector_db is not None,
                "llm_manager": self.llm_manager is not None,
                "rag_system": self.rag_system is not None
            }
        }
        
        if self.is_initialized:
            # Add detailed statistics
            if self.vector_db:
                status["database_stats"] = self.vector_db.get_collection_stats()
            
            if self.llm_manager:
                status["model_info"] = self.llm_manager.get_model_info()
        
        return status

# Demo and testing functions
async def run_demo():
    """Run a simple demo of the system"""
    print("🧠 Metacognitive Therapy Chatbot Demo")
    print("=" * 50)
    
    # Initialize system
    system = MetacognitiveTherapistSystem()
    success = await system.initialize()
    
    if not success:
        print("❌ Failed to initialize system")
        return
    
    # Check if we have therapy data
    data_path = Path("data/Raw data.pdf")
    if data_path.exists():
        print("📚 Setting up knowledge base...")
        await system.setup_knowledge_base(str(data_path))
    else:
        print("⚠️  No therapy literature found. Place your PDF in data/Raw data.pdf")
    
    # Interactive chat demo
    print("\n💬 Chat Demo (type 'quit' to exit)")
    print("-" * 30)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ['quit', 'exit', 'bye']:
                break
            
            if not user_input:
                continue
            
            print("🤔 AI Therapist is thinking...")
            response = system.chat(user_input, conversation_history)
            
            if "error" in response:
                print(f"❌ Error: {response['error']}")
            else:
                print(f"\n🤖 AI Therapist: {response['response']}")
                
                # Update conversation history
                conversation_history.append({"role": "user", "content": user_input})
                conversation_history.append({"role": "assistant", "content": response["response"]})
                
                # Keep only last 10 messages
                if len(conversation_history) > 10:
                    conversation_history = conversation_history[-10:]
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print("\n👋 Thank you for using the Metacognitive Therapy Chatbot!")
    print("Remember: This AI assistant is for educational purposes and doesn't replace professional mental health care.")

# System status check
async def check_system():
    """Check system components and dependencies"""
    print("🔧 System Check")
    print("=" * 30)
    
    # Check dependencies
    dependencies = [
        "gpt4all", "langchain", "chromadb", "sentence-transformers",
        "fastapi", "uvicorn", "streamlit", "pypdf", "python-dotenv"
    ]
    
    print("📦 Checking dependencies...")
    for dep in dependencies:
        try:
            __import__(dep.replace("-", "_"))
            print(f"✅ {dep}")
        except ImportError:
            print(f"❌ {dep} - Not installed")
    
    # Check system initialization
    print("\n🔧 Testing system initialization...")
    system = MetacognitiveTherapistSystem()
    success = await system.initialize()
    
    if success:
        print("✅ System initialization successful")
        status = system.get_system_status()
        
        print("\n📊 System Status:")
        for component, available in status["components"].items():
            status_symbol = "✅" if available else "❌"
            print(f"{status_symbol} {component}")
        
        # Test model
        if system.llm_manager:
            print("\n🧠 Testing LLM...")
            test_result = system.llm_manager.test_model()
            if test_result["status"] == "success":
                print(f"✅ LLM test successful (Response time: {test_result['response_time']:.2f}s)")
            else:
                print(f"❌ LLM test failed: {test_result.get('error', 'Unknown error')}")
    
    else:
        print("❌ System initialization failed")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Metacognitive Therapy Chatbot")
    parser.add_argument("--demo", action="store_true", help="Run interactive demo")
    parser.add_argument("--check", action="store_true", help="Check system status")
    parser.add_argument("--setup", type=str, help="Setup knowledge base from PDF file")
    
    args = parser.parse_args()
    
    if args.demo:
        asyncio.run(run_demo())
    elif args.check:
        asyncio.run(check_system())
    elif args.setup:
        async def setup_kb():
            system = MetacognitiveTherapistSystem()
            await system.initialize()
            await system.setup_knowledge_base(args.setup)
        asyncio.run(setup_kb())
    else:
        print("Metacognitive Therapy Chatbot")
        print("Use --demo for interactive demo")
        print("Use --check for system check")
        print("Use --setup <pdf_path> to setup knowledge base")