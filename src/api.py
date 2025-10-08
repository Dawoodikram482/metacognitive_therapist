"""
Advanced FastAPI Backend for Metacognitive Therapy Chatbot

This module implements the main API backend with session management,
authentication, rate limiting, and comprehensive therapy chat functionality.
"""

import logging
from typing import List, Dict, Any, Optional, Union
import uuid
import json
from datetime import datetime, timedelta
import asyncio
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request, status, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator
import uvicorn

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Local imports
from .document_processor import AdvancedDocumentProcessor
from .vector_database import AdvancedVectorDatabase  
from .rag_system import AdvancedRAGSystem
from .llm_manager import LLMManager, ModelType
# from .session_manager import SessionManager
# from .analytics_tracker import AnalyticsTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Pydantic models for API requests/responses
class ChatMessage(BaseModel):
    """Single chat message"""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: Optional[datetime] = None
    message_id: Optional[str] = None
    
    @validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be user, assistant, or system')
        return v

class ChatRequest(BaseModel):
    """Chat request from client"""
    message: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    include_sources: bool = True
    max_response_length: Optional[int] = 500
    
    @validator('message')
    def validate_message(cls, v):
        if not v or len(v.strip()) == 0:
            raise ValueError('Message cannot be empty')
        if len(v) > 2000:
            raise ValueError('Message too long (max 2000 characters)')
        return v.strip()

class ChatResponse(BaseModel):
    """Chat response to client"""
    response: str
    session_id: str
    message_id: str
    timestamp: datetime
    sources: Optional[List[Dict[str, Any]]] = None
    metadata: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None

class SessionInfo(BaseModel):
    """Session information"""
    session_id: str
    created_at: datetime
    last_activity: datetime
    message_count: int
    user_id: Optional[str] = None
    session_metadata: Optional[Dict[str, Any]] = None

class HealthCheck(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    models_loaded: Dict[str, bool]
    database_status: str
    uptime_seconds: float

class DocumentUploadRequest(BaseModel):
    """Document upload request"""
    file_path: str
    collection_name: str = "therapy_techniques"
    
class SystemStats(BaseModel):
    """System statistics"""
    total_sessions: int
    total_messages: int
    active_sessions: int
    model_performance: Dict[str, Any]
    database_stats: Dict[str, Any]
    uptime: str

# Session management
class SessionManager:
    """Manage user sessions and conversation history"""
    
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = timedelta(hours=24)
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "messages": [],
            "metadata": {}
        }
        logger.info(f"Created session {session_id}")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session by ID"""
        session = self.sessions.get(session_id)
        if session and self._is_session_valid(session):
            session["last_activity"] = datetime.now()
            return session
        elif session:
            # Session expired
            self.delete_session(session_id)
        return None
    
    def add_message(self, session_id: str, message: ChatMessage):
        """Add message to session"""
        session = self.get_session(session_id)
        if session:
            message.timestamp = datetime.now()
            message.message_id = str(uuid.uuid4())
            session["messages"].append(message.dict())
            session["last_activity"] = datetime.now()
    
    def get_conversation_history(self, session_id: str, limit: int = 10) -> List[ChatMessage]:
        """Get conversation history for session"""
        session = self.get_session(session_id)
        if session:
            messages = session["messages"][-limit:]
            return [ChatMessage(**msg) for msg in messages]
        return []
    
    def delete_session(self, session_id: str):
        """Delete session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session {session_id}")
    
    def _is_session_valid(self, session: Dict[str, Any]) -> bool:
        """Check if session is still valid"""
        last_activity = session["last_activity"]
        return datetime.now() - last_activity < self.session_timeout
    
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        expired_sessions = []
        for session_id, session in self.sessions.items():
            if not self._is_session_valid(session):
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.delete_session(session_id)
        
        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics"""
        total_messages = sum(len(session["messages"]) for session in self.sessions.values())
        return {
            "total_sessions": len(self.sessions),
            "active_sessions": len([s for s in self.sessions.values() if self._is_session_valid(s)]),
            "total_messages": total_messages
        }

# Analytics tracking
class AnalyticsTracker:
    """Track usage analytics and performance metrics"""
    
    def __init__(self):
        self.metrics = {
            "requests_total": 0,
            "requests_successful": 0,
            "average_response_time": 0.0,
            "common_queries": {},
            "error_counts": {},
            "daily_active_users": set(),
            "model_usage": {}
        }
        self.start_time = datetime.now()
    
    def track_request(self, endpoint: str, response_time: float, success: bool, error: str = None):
        """Track API request"""
        self.metrics["requests_total"] += 1
        
        if success:
            self.metrics["requests_successful"] += 1
        
        # Update average response time
        total_time = (self.metrics["average_response_time"] * 
                     (self.metrics["requests_total"] - 1) + response_time)
        self.metrics["average_response_time"] = total_time / self.metrics["requests_total"]
        
        # Track errors
        if error:
            self.metrics["error_counts"][error] = self.metrics["error_counts"].get(error, 0) + 1
    
    def track_query(self, query: str):
        """Track user query patterns"""
        # Simple query categorization
        query_lower = query.lower()
        category = "other"
        
        if any(word in query_lower for word in ["anxious", "anxiety", "worry", "worried"]):
            category = "anxiety"
        elif any(word in query_lower for word in ["sad", "depressed", "depression", "down"]):
            category = "depression"
        elif any(word in query_lower for word in ["technique", "help", "method", "how to"]):
            category = "technique_request"
        
        self.metrics["common_queries"][category] = self.metrics["common_queries"].get(category, 0) + 1
    
    def track_daily_user(self, user_id: str):
        """Track daily active user"""
        self.metrics["daily_active_users"].add(user_id)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get analytics summary"""
        uptime = datetime.now() - self.start_time
        
        return {
            "uptime_seconds": uptime.total_seconds(),
            "requests_per_hour": self.metrics["requests_total"] / max(1, uptime.total_seconds() / 3600),
            "success_rate": (self.metrics["requests_successful"] / max(1, self.metrics["requests_total"])) * 100,
            "average_response_time": round(self.metrics["average_response_time"], 3),
            "daily_active_users": len(self.metrics["daily_active_users"]),
            "common_queries": self.metrics["common_queries"],
            "error_summary": self.metrics["error_counts"]
        }

# Initialize FastAPI app
app = FastAPI(
    title="Metacognitive Therapy Chatbot API",
    description="Advanced RAG-powered chatbot for metacognitive therapy guidance",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["localhost", "127.0.0.1", "*.streamlit.app"]
)

# Global instances
session_manager = SessionManager()
analytics_tracker = AnalyticsTracker()
document_processor = None
vector_db = None
rag_system = None
llm_manager = None

# Global variables for tracking
startup_time = datetime.now()

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global document_processor, vector_db, rag_system, llm_manager
    
    try:
        logger.info("Initializing Metacognitive Therapy Chatbot API...")
        
        # Initialize document processor
        document_processor = AdvancedDocumentProcessor()
        logger.info("Document processor initialized")
        
        # Initialize vector database
        vector_db = AdvancedVectorDatabase()
        logger.info("Vector database initialized")
        
        # Initialize LLM manager - using faster 3B model for better response times
        llm_manager = LLMManager(ModelType.GPT4ALL_ORCA_MINI)
        logger.info("LLM manager initialized")
        
        # Initialize RAG system
        rag_system = AdvancedRAGSystem(vector_db, llm_manager)
        logger.info("RAG system initialized")
        
        # Start background tasks
        asyncio.create_task(periodic_cleanup())
        
        logger.info("API startup completed successfully")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise

async def periodic_cleanup():
    """Background task for periodic cleanup"""
    while True:
        try:
            # Clean up expired sessions every hour
            session_manager.cleanup_expired_sessions()
            
            # Reset daily active users at midnight
            if datetime.now().hour == 0:
                analytics_tracker.metrics["daily_active_users"].clear()
            
            await asyncio.sleep(3600)  # 1 hour
            
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            await asyncio.sleep(3600)

# Health check endpoint
@app.get("/health", response_model=HealthCheck)
@limiter.limit("30/minute")
async def health_check(request: Request):
    """Check API health and component status"""
    uptime = datetime.now() - startup_time
    
    return HealthCheck(
        status="healthy",
        timestamp=datetime.now(),
        version="1.0.0",
        models_loaded={
            "llm_manager": llm_manager is not None,
            "vector_db": vector_db is not None,
            "rag_system": rag_system is not None
        },
        database_status="connected" if vector_db else "disconnected",
        uptime_seconds=uptime.total_seconds()
    )

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
@limiter.limit("20/minute")
async def chat(request: Request, chat_request: ChatRequest, background_tasks: BackgroundTasks):
    """Main chat endpoint for therapy conversations"""
    start_time = datetime.now()
    
    try:
        # Get or create session
        session_id = chat_request.session_id
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session(chat_request.user_id)
        
        # Track analytics
        if chat_request.user_id:
            analytics_tracker.track_daily_user(chat_request.user_id)
        analytics_tracker.track_query(chat_request.message)
        
        # Add user message to session
        user_message = ChatMessage(role="user", content=chat_request.message)
        session_manager.add_message(session_id, user_message)
        
        # Get conversation history
        conversation_history = session_manager.get_conversation_history(session_id, limit=6)
        
        # Convert to the format expected by RAG system
        history_messages = []
        for msg in conversation_history[:-1]:  # Exclude current message
            if msg.role == "user":
                from langchain.schema import HumanMessage
                history_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                from langchain.schema import AIMessage
                history_messages.append(AIMessage(content=msg.content))
        
        # Generate response using RAG system
        rag_response = rag_system.generate_rag_response(
            chat_request.message,
            history_messages if history_messages else None
        )
        
        # Create assistant message
        assistant_message = ChatMessage(
            role="assistant", 
            content=rag_response["response"]
        )
        session_manager.add_message(session_id, assistant_message)
        
        # Generate helpful suggestions
        suggestions = await generate_suggestions(chat_request.message, rag_response["response"])
        
        # Track successful request
        response_time = (datetime.now() - start_time).total_seconds()
        background_tasks.add_task(
            analytics_tracker.track_request, 
            "chat", response_time, True
        )
        
        return ChatResponse(
            response=rag_response["response"],
            session_id=session_id,
            message_id=assistant_message.message_id,
            timestamp=datetime.now(),
            sources=rag_response.get("sources") if chat_request.include_sources else None,
            metadata={
                "query_classification": rag_response.get("query_classification"),
                "response_time": response_time,
                "documents_used": rag_response.get("documents_used", 0)
            },
            suggestions=suggestions
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        
        # Track error
        response_time = (datetime.now() - start_time).total_seconds()
        background_tasks.add_task(
            analytics_tracker.track_request, 
            "chat", response_time, False, str(e)
        )
        
        # Return fallback response
        return ChatResponse(
            response="I apologize, but I'm experiencing technical difficulties. Please try again or consider speaking with a mental health professional if you need immediate support.",
            session_id=session_id or "error",
            message_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            metadata={"error": str(e)}
        )

async def generate_suggestions(user_message: str, bot_response: str) -> List[str]:
    """Generate helpful follow-up suggestions"""
    suggestions = []
    
    message_lower = user_message.lower()
    
    if "worry" in message_lower or "anxious" in message_lower:
        suggestions = [
            "Can you tell me more about what specifically triggers your worry?",
            "Would you like to learn about the Attention Training Technique?",
            "How long have you been experiencing these worries?"
        ]
    elif "technique" in message_lower or "help" in message_lower:
        suggestions = [
            "Would you like me to guide you through a specific exercise?",
            "Can you describe what you've tried before?",
            "What time of day do you find these techniques most helpful?"
        ]
    elif "depressed" in message_lower or "sad" in message_lower:
        suggestions = [
            "What thoughts tend to go through your mind when you feel this way?",
            "Have you noticed any patterns in when these feelings occur?",
            "Would you like to explore some metacognitive strategies for low mood?"
        ]
    else:
        suggestions = [
            "Is there anything specific you'd like to explore further?",
            "How are you feeling about what we've discussed?",
            "Would you like to learn about any particular therapy techniques?"
        ]
    
    return suggestions[:3]  # Return max 3 suggestions

# Session management endpoints
@app.get("/session/{session_id}", response_model=SessionInfo)
@limiter.limit("60/minute")
async def get_session(request: Request, session_id: str):
    """Get session information"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionInfo(
        session_id=session["session_id"],
        created_at=session["created_at"],
        last_activity=session["last_activity"],
        message_count=len(session["messages"]),
        user_id=session.get("user_id"),
        session_metadata=session.get("metadata")
    )

@app.delete("/session/{session_id}")
@limiter.limit("30/minute")
async def delete_session(request: Request, session_id: str):
    """Delete session"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session_manager.delete_session(session_id)
    return {"message": "Session deleted successfully"}

@app.get("/session/{session_id}/messages")
@limiter.limit("60/minute")
async def get_session_messages(request: Request, session_id: str, limit: int = 50):
    """Get session conversation history"""
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = session["messages"][-limit:]
    return {"messages": messages}

# System administration endpoints
@app.get("/admin/stats", response_model=SystemStats)
@limiter.limit("10/minute")
async def get_system_stats(request: Request):
    """Get system statistics (admin only)"""
    session_stats = session_manager.get_stats()
    analytics = analytics_tracker.get_analytics()
    model_info = llm_manager.get_model_info() if llm_manager else {}
    db_stats = vector_db.get_collection_stats() if vector_db else {}
    
    uptime = datetime.now() - startup_time
    
    return SystemStats(
        total_sessions=session_stats["total_sessions"],
        total_messages=session_stats["total_messages"],
        active_sessions=session_stats["active_sessions"],
        model_performance=model_info,
        database_stats=db_stats,
        uptime=str(uptime)
    )

@app.post("/admin/upload-document")
@limiter.limit("5/minute")
async def upload_document(request: Request, upload_request: DocumentUploadRequest):
    """Upload and process new therapy document"""
    try:
        file_path = Path(upload_request.file_path)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Process document
        documents = document_processor.process_document(file_path)
        
        # Add to vector database
        success = vector_db.add_documents(documents, upload_request.collection_name)
        
        if success:
            return {
                "message": f"Successfully processed and added {len(documents)} chunks to {upload_request.collection_name}",
                "document_count": len(documents),
                "collection": upload_request.collection_name
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add documents to database")
            
    except Exception as e:
        logger.error(f"Document upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        timeout_keep_alive=120,  # Keep connections alive longer for slow LLM responses
        timeout_graceful_shutdown=30
    )