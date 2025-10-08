"""
Advanced RAG (Retrieval-Augmented Generation) System for Metacognitive Therapy

This module implements a sophisticated RAG system with context management,
query enhancement, and multi-step reasoning for therapy conversations.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import re
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory, ConversationSummaryBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
from sentence_transformers import SentenceTransformer
import numpy as np

from .vector_database import AdvancedVectorDatabase
from .llm_manager import LLMManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Enum for different types of user queries"""
    SYMPTOM_INQUIRY = "symptom_inquiry"
    TECHNIQUE_REQUEST = "technique_request"
    GENERAL_THERAPY = "general_therapy"
    CRISIS_SUPPORT = "crisis_support"
    PROGRESS_CHECK = "progress_check"
    EDUCATIONAL = "educational"

@dataclass
class RAGConfig:
    """Configuration for RAG system"""
    max_context_length: int = 4000
    max_retrieved_docs: int = 8
    similarity_threshold: float = 0.3  # Adjusted for new relevance scoring (1/(1+distance))
    context_window_size: int = 10
    enable_query_expansion: bool = True
    enable_context_compression: bool = True
    enable_multi_step_reasoning: bool = True

class AdvancedRAGSystem:
    """
    Sophisticated RAG system implementing advanced retrieval strategies,
    context management, and multi-step reasoning for therapy conversations.
    """
    
    def __init__(
        self, 
        vector_db: AdvancedVectorDatabase,
        llm_manager: 'LLMManager',
        config: RAGConfig = None
    ):
        self.vector_db = vector_db
        self.llm_manager = llm_manager
        self.config = config or RAGConfig()
        
        # Initialize components
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.conversation_memory = ConversationBufferWindowMemory(
            k=self.config.context_window_size,
            return_messages=True
        )
        
        # Therapy-specific query patterns
        self.query_patterns = self._initialize_query_patterns()
        
        # Context compression templates
        self.compression_templates = self._initialize_compression_templates()
        
        logger.info("Advanced RAG System initialized")
    
    def _initialize_query_patterns(self) -> Dict[QueryType, List[str]]:
        """Initialize regex patterns for query classification"""
        return {
            QueryType.SYMPTOM_INQUIRY: [
                r'(?i)(feel|feeling|symptoms?|experiencing|having trouble)',
                r'(?i)(anxious|anxiety|worried|worry|depressed|sad|panic)',
                r'(?i)(can\'t stop|keeps happening|overwhelming)'
            ],
            QueryType.TECHNIQUE_REQUEST: [
                r'(?i)(how to|help me|teach me|show me|technique|method)',
                r'(?i)(cope|deal with|manage|handle|overcome)',
                r'(?i)(exercise|practice|skill|strategy)'
            ],
            QueryType.CRISIS_SUPPORT: [
                r'(?i)(crisis|emergency|urgent|immediate|desperate)',
                r'(?i)(hurt myself|suicide|end it all|can\'t go on)',
                r'(?i)(emergency|need help now)'
            ],
            QueryType.EDUCATIONAL: [
                r'(?i)(what is|explain|tell me about|understand)',
                r'(?i)(definition|meaning|concept|theory)',
                r'(?i)(learn|study|research|information)'
            ]
        }
    
    def _initialize_compression_templates(self) -> Dict[str, str]:
        """Initialize templates for context compression"""
        return {
            "summary": """
            Summarize the following therapy content concisely while preserving key therapeutic concepts:
            {content}
            
            Summary:
            """,
            "key_points": """
            Extract the most important therapeutic points from:
            {content}
            
            Key Points:
            """,
            "technique_extraction": """
            Identify specific therapeutic techniques mentioned in:
            {content}
            
            Techniques:
            """
        }
    
    def classify_query(self, query: str) -> QueryType:
        """
        Classify user query to determine appropriate retrieval strategy
        """
        query_lower = query.lower()
        
        # Check for crisis indicators first
        if any(re.search(pattern, query) for pattern in self.query_patterns[QueryType.CRISIS_SUPPORT]):
            return QueryType.CRISIS_SUPPORT
        
        # Check other patterns
        for query_type, patterns in self.query_patterns.items():
            if query_type == QueryType.CRISIS_SUPPORT:
                continue
            
            if any(re.search(pattern, query) for pattern in patterns):
                return query_type
        
        return QueryType.GENERAL_THERAPY
    
    def expand_query(self, query: str, query_type: QueryType) -> List[str]:
        """
        Expand query with synonyms and related terms for better retrieval
        """
        if not self.config.enable_query_expansion:
            return [query]
        
        expanded_queries = [query]
        
        # Metacognitive therapy specific expansions
        expansion_map = {
            "worry": ["rumination", "perseverative thinking", "negative thoughts"],
            "anxiety": ["anxious thoughts", "fear", "panic", "stress"],
            "attention": ["focus", "concentration", "mindfulness", "awareness"],
            "control": ["manage", "regulate", "cope", "handle"],
            "technique": ["method", "strategy", "approach", "intervention"],
            "metacognitive": ["meta-cognitive", "thinking about thinking", "awareness of thoughts"]
        }
        
        query_words = query.lower().split()
        for word in query_words:
            if word in expansion_map:
                for expansion in expansion_map[word]:
                    expanded_query = query.replace(word, expansion)
                    expanded_queries.append(expanded_query)
        
        return expanded_queries[:3]  # Limit to avoid too many queries
    
    def retrieve_relevant_context(
        self, 
        query: str, 
        conversation_history: Optional[List[BaseMessage]] = None
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        Retrieve relevant context using multi-strategy approach
        """
        # Classify query
        query_type = self.classify_query(query)
        
        # Expand query
        expanded_queries = self.expand_query(query, query_type)
        
        # Determine collections to search based on query type
        collection_strategy = self._get_collection_strategy(query_type)
        
        # Retrieve documents for each expanded query
        all_documents = []
        retrieval_metadata = {
            "query_type": query_type.value,
            "expanded_queries": expanded_queries,
            "collections_searched": collection_strategy["collections"],
            "retrieval_timestamp": datetime.now().isoformat()
        }
        
        for expanded_query in expanded_queries:
            # Hybrid search across relevant collections
            results = self.vector_db.hybrid_search(
                query=expanded_query,
                collection_names=collection_strategy["collections"],
                n_results=collection_strategy["max_results"],
                similarity_threshold=self.config.similarity_threshold
            )
            
            logger.info(f"Search for '{expanded_query[:50]}...' found {len(results)} results")
            
            # Convert results to Document objects
            for result in results:
                doc = Document(
                    page_content=result["document"],
                    metadata={
                        **result["metadata"],
                        "retrieval_score": result["final_score"],
                        "query_used": expanded_query,
                        "collection_source": result["collection"]
                    }
                )
                all_documents.append(doc)
        
        # Remove duplicates and rank
        unique_documents = self._deduplicate_documents(all_documents)
        ranked_documents = self._rank_documents(unique_documents, query, query_type)
        
        # Apply conversation context if available
        if conversation_history:
            contextualized_documents = self._apply_conversation_context(
                ranked_documents, conversation_history
            )
        else:
            contextualized_documents = ranked_documents
        
        # Limit to max retrieved docs
        final_documents = contextualized_documents[:self.config.max_retrieved_docs]
        
        retrieval_metadata["final_document_count"] = len(final_documents)
        retrieval_metadata["collections_used"] = list(set([
            doc.metadata.get("collection_source", "unknown") for doc in final_documents
        ]))
        
        return final_documents, retrieval_metadata
    
    def _get_collection_strategy(self, query_type: QueryType) -> Dict[str, Any]:
        """Get retrieval strategy based on query type"""
        strategies = {
            QueryType.SYMPTOM_INQUIRY: {
                "collections": ["case_studies", "theoretical_concepts", "therapy_techniques"],
                "max_results": 6
            },
            QueryType.TECHNIQUE_REQUEST: {
                "collections": ["therapy_techniques", "therapeutic_exercises"],
                "max_results": 8
            },
            QueryType.CRISIS_SUPPORT: {
                "collections": ["therapy_techniques", "case_studies"],
                "max_results": 4
            },
            QueryType.EDUCATIONAL: {
                "collections": ["theoretical_concepts", "assessment_tools"],
                "max_results": 6
            },
            QueryType.GENERAL_THERAPY: {
                "collections": ["therapy_techniques", "case_studies", "theoretical_concepts"],
                "max_results": 6
            }
        }
        
        return strategies.get(query_type, strategies[QueryType.GENERAL_THERAPY])
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity"""
        if not documents:
            return []
        
        unique_docs = []
        embeddings = []
        
        for doc in documents:
            embedding = self.embedding_model.encode(doc.page_content)
            
            # Check similarity with existing documents
            is_duplicate = False
            for existing_embedding in embeddings:
                similarity = np.dot(embedding, existing_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(existing_embedding)
                )
                if similarity > 0.95:  # High similarity threshold for duplicates
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_docs.append(doc)
                embeddings.append(embedding)
        
        return unique_docs
    
    def _rank_documents(
        self, 
        documents: List[Document], 
        query: str, 
        query_type: QueryType
    ) -> List[Document]:
        """Advanced document ranking considering multiple factors"""
        if not documents:
            return []
        
        # Calculate ranking scores
        for doc in documents:
            base_score = doc.metadata.get("retrieval_score", 0.5)
            
            # Query type specific boosting
            type_boost = self._calculate_type_boost(doc, query_type)
            
            # Content quality scoring
            quality_score = self._calculate_content_quality(doc.page_content)
            
            # Recency scoring
            recency_score = self._calculate_recency_score(doc.metadata)
            
            # Final ranking score
            doc.metadata["final_ranking_score"] = (
                base_score * 0.4 + 
                type_boost * 0.3 + 
                quality_score * 0.2 + 
                recency_score * 0.1
            )
        
        # Sort by final ranking score
        ranked_docs = sorted(
            documents, 
            key=lambda x: x.metadata.get("final_ranking_score", 0), 
            reverse=True
        )
        
        return ranked_docs
    
    def _calculate_type_boost(self, doc: Document, query_type: QueryType) -> float:
        """Calculate boost based on query type and document collection"""
        collection = doc.metadata.get("collection_source", "")
        
        type_collection_weights = {
            QueryType.TECHNIQUE_REQUEST: {
                "therapy_techniques": 1.0,
                "therapeutic_exercises": 0.9,
                "case_studies": 0.7
            },
            QueryType.SYMPTOM_INQUIRY: {
                "case_studies": 1.0,
                "theoretical_concepts": 0.8,
                "therapy_techniques": 0.9
            },
            QueryType.EDUCATIONAL: {
                "theoretical_concepts": 1.0,
                "assessment_tools": 0.8,
                "therapy_techniques": 0.6
            }
        }
        
        weights = type_collection_weights.get(query_type, {})
        return weights.get(collection, 0.5)
    
    def _calculate_content_quality(self, content: str) -> float:
        """Calculate content quality score"""
        # Basic quality metrics
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        
        # Ideal length scoring
        if 100 <= word_count <= 500:
            length_score = 1.0
        elif word_count < 100:
            length_score = word_count / 100
        else:
            length_score = max(0.5, 500 / word_count)
        
        # Therapy-specific term density
        therapy_terms = [
            "metacognitive", "attention", "worry", "rumination", "technique",
            "therapy", "cognitive", "behavioral", "intervention", "treatment"
        ]
        
        term_count = sum(1 for term in therapy_terms if term in content.lower())
        term_density = min(1.0, term_count / 10)
        
        return (length_score * 0.6 + term_density * 0.4)
    
    def _calculate_recency_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate recency score for document"""
        try:
            if "added_timestamp" in metadata:
                added_date = datetime.fromisoformat(metadata["added_timestamp"])
                days_old = (datetime.now() - added_date).days
                # Recent content gets higher score
                return max(0.1, 1.0 - (days_old / 365))
        except:
            pass
        
        return 0.5  # Default score
    
    def _apply_conversation_context(
        self, 
        documents: List[Document], 
        conversation_history: List[BaseMessage]
    ) -> List[Document]:
        """Apply conversation context to re-rank documents"""
        if not conversation_history:
            return documents
        
        # Extract recent conversation themes
        recent_messages = conversation_history[-6:]  # Last 6 messages
        conversation_text = " ".join([msg.content for msg in recent_messages])
        
        # Re-rank based on conversation relevance
        conversation_embedding = self.embedding_model.encode(conversation_text)
        
        for doc in documents:
            doc_embedding = self.embedding_model.encode(doc.page_content)
            
            # Calculate conversation similarity
            conversation_similarity = np.dot(conversation_embedding, doc_embedding) / (
                np.linalg.norm(conversation_embedding) * np.linalg.norm(doc_embedding)
            )
            
            # Adjust ranking score
            current_score = doc.metadata.get("final_ranking_score", 0.5)
            doc.metadata["final_ranking_score"] = (
                current_score * 0.7 + conversation_similarity * 0.3
            )
        
        # Re-sort by updated scores
        return sorted(
            documents, 
            key=lambda x: x.metadata.get("final_ranking_score", 0), 
            reverse=True
        )
    
    def compress_context(self, documents: List[Document], max_length: int = None) -> str:
        """
        Compress retrieved context to fit within token limits while preserving key information
        """
        if not documents:
            return ""
        
        max_length = max_length or self.config.max_context_length
        
        if not self.config.enable_context_compression:
            # Simple concatenation
            full_context = "\n\n".join([doc.page_content for doc in documents])
            return full_context[:max_length]
        
        # Advanced compression
        compressed_sections = []
        current_length = 0
        
        for i, doc in enumerate(documents):
            content = doc.page_content
            
            # If adding this document would exceed limit, compress it
            if current_length + len(content) > max_length:
                remaining_space = max_length - current_length
                if remaining_space > 100:  # Minimum useful space
                    # Compress using LLM
                    compressed_content = self._compress_with_llm(content, remaining_space)
                    compressed_sections.append(f"[Source {i+1}]: {compressed_content}")
                break
            else:
                compressed_sections.append(f"[Source {i+1}]: {content}")
                current_length += len(content) + 20  # Account for source labeling
        
        return "\n\n".join(compressed_sections)
    
    def _compress_with_llm(self, content: str, max_length: int) -> str:
        """Use LLM to compress content while preserving key information"""
        try:
            compression_prompt = f"""
            Compress the following therapeutic content to approximately {max_length} characters 
            while preserving all key therapeutic concepts, techniques, and important details:
            
            {content}
            
            Compressed version:
            """
            
            compressed = self.llm_manager.generate_response(
                compression_prompt, 
                max_tokens=max_length // 4  # Rough token estimate
            )
            
            return compressed.strip()
            
        except Exception as e:
            logger.warning(f"LLM compression failed: {e}")
            # Fallback to simple truncation
            return content[:max_length] + "..."
    
    def generate_rag_response(
        self, 
        query: str, 
        conversation_history: Optional[List[BaseMessage]] = None
    ) -> Dict[str, Any]:
        """
        Generate complete RAG response with context retrieval and answer generation
        """
        logger.info(f"Processing RAG query: {query[:100]}...")
        
        # Retrieve relevant context
        retrieved_docs, retrieval_metadata = self.retrieve_relevant_context(
            query, conversation_history
        )
        
        if not retrieved_docs:
            return {
                "response": "I apologize, but I couldn't find relevant information to answer your question. Could you please rephrase or provide more context?",
                "sources": [],
                "retrieval_metadata": retrieval_metadata,
                "error": "No relevant documents found"
            }
        
        # Compress context
        compressed_context = self.compress_context(retrieved_docs)
        
        # Generate response using LLM
        response_data = self._generate_contextual_response(
            query, compressed_context, retrieved_docs, conversation_history
        )
        
        # Add retrieval metadata
        response_data["retrieval_metadata"] = retrieval_metadata
        response_data["context_length"] = len(compressed_context)
        response_data["documents_used"] = len(retrieved_docs)
        
        # Update conversation memory
        if conversation_history is not None:
            self.conversation_memory.chat_memory.add_user_message(query)
            self.conversation_memory.chat_memory.add_ai_message(response_data["response"])
        
        return response_data
    
    def _generate_contextual_response(
        self,
        query: str,
        context: str,
        source_docs: List[Document],
        conversation_history: Optional[List[BaseMessage]] = None
    ) -> Dict[str, Any]:
        """Generate contextual response using retrieved information"""
        
        # Build conversation context
        conversation_context = ""
        if conversation_history and len(conversation_history) > 0:
            recent_messages = conversation_history[-4:]  # Last 4 messages for context
            context_parts = []
            for msg in recent_messages:
                role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                context_parts.append(f"{role}: {msg.content}")
            conversation_context = "\n".join(context_parts)
        
        # Create comprehensive prompt
        system_prompt = """You are an expert metacognitive therapist AI assistant. You provide evidence-based guidance using metacognitive therapy principles and techniques.

INSTRUCTIONS:
1. Base your response primarily on the provided context from therapy literature
2. Maintain a warm, empathetic, and professional therapeutic tone
3. Provide specific, actionable guidance when appropriate
4. Always prioritize client safety - refer to professional help for crisis situations  
5. Use metacognitive therapy concepts and terminology appropriately
6. Be concise but thorough in your explanations

CONTEXT FROM THERAPY LITERATURE:
{context}

{conversation_section}

Please respond to the client's question/concern with therapeutic guidance based on the provided context."""
        
        conversation_section = f"\nRECENT CONVERSATION:\n{conversation_context}\n" if conversation_context else ""
        
        formatted_prompt = system_prompt.format(
            context=context,
            conversation_section=conversation_section
        )
        
        user_prompt = f"Client: {query}"
        
        try:
            # Generate response
            response = self.llm_manager.generate_therapeutic_response(
                formatted_prompt, 
                user_prompt,
                max_tokens=500
            )
            
            # Extract sources information
            sources = []
            for doc in source_docs:
                source_info = {
                    "content_preview": doc.page_content[:200] + "...",
                    "collection": doc.metadata.get("collection_source", "unknown"),
                    "relevance_score": doc.metadata.get("final_ranking_score", 0),
                    "source_type": doc.metadata.get("document_type", "therapy_literature")
                }
                sources.append(source_info)
            
            return {
                "response": response,
                "sources": sources,
                "query_classification": self.classify_query(query).value,
                "processing_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "response": "I apologize, but I'm having difficulty processing your request right now. Please try again or rephrase your question.",
                "sources": [],
                "error": str(e),
                "query_classification": self.classify_query(query).value,
                "processing_timestamp": datetime.now().isoformat()
            }

if __name__ == "__main__":
    # This would be initialized with actual vector_db and llm_manager instances
    pass