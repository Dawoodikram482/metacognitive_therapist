"""
Advanced Vector Database Management for Metacognitive Therapy Chatbot

This module implements sophisticated vector database operations with ChromaDB,
including multiple collections, advanced querying, and semantic search optimization.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import uuid
import json
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedVectorDatabase:
    """
    Sophisticated vector database manager implementing multiple collections,
    hybrid search, and advanced retrieval strategies for therapy content.
    """
    
    def __init__(self, db_path: str = "./index/chroma_db", embedding_model: str = "all-MiniLM-L6-v2"):
        self.db_path = Path(db_path)
        self.embedding_model_name = embedding_model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Define collection schemas
        self.collections = {}
        self._initialize_collections()
        
        logger.info(f"Initialized Vector Database at {self.db_path}")
    
    def _initialize_collections(self):
        """Initialize specialized collections for different types of therapy content"""
        
        collection_configs = {
            "therapy_techniques": {
                "description": "Metacognitive therapy techniques and interventions",
                "metadata_schema": ["technique_type", "difficulty_level", "session_phase", "target_symptoms"]
            },
            "case_studies": {
                "description": "Clinical case studies and examples",
                "metadata_schema": ["case_type", "patient_profile", "outcome", "duration"]
            },
            "theoretical_concepts": {
                "description": "Theoretical foundations and concepts",
                "metadata_schema": ["concept_category", "complexity_level", "related_theories", "evidence_base"]
            },
            "therapeutic_exercises": {
                "description": "Practical exercises and homework assignments",
                "metadata_schema": ["exercise_type", "time_required", "materials_needed", "target_skills"]
            },
            "assessment_tools": {
                "description": "Assessment instruments and questionnaires",
                "metadata_schema": ["tool_type", "validated", "target_population", "psychometric_properties"]
            }
        }
        
        # Create embedding function
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.embedding_model_name
        )
        
        for collection_name, config in collection_configs.items():
            try:
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=embedding_function,
                    metadata={"description": config["description"]}
                )
                self.collections[collection_name] = collection
                logger.info(f"Initialized collection: {collection_name}")
            except Exception as e:
                logger.error(f"Error creating collection {collection_name}: {e}")
    
    def add_documents(self, documents: List[Document], collection_name: str = "therapy_techniques") -> bool:
        """
        Add documents to specified collection with advanced metadata processing
        """
        try:
            if collection_name not in self.collections:
                logger.error(f"Collection {collection_name} not found")
                return False
            
            collection = self.collections[collection_name]
            
            # Prepare data for ChromaDB
            ids = []
            documents_text = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                # Generate unique ID
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                
                # Extract text content
                documents_text.append(doc.page_content)
                
                # Process metadata
                metadata = self._process_metadata(doc.metadata, collection_name)
                metadatas.append(metadata)
            
            # Add to collection
            collection.add(
                documents=documents_text,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding documents to {collection_name}: {e}")
            return False
    
    def _process_metadata(self, metadata: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
        """Process and validate metadata for specific collection"""
        processed_metadata = {}
        
        # Convert all values to strings (ChromaDB requirement)
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)
        
        # Add collection-specific metadata
        processed_metadata.update({
            "collection_type": collection_name,
            "added_timestamp": datetime.now().isoformat(),
            "embedding_model": self.embedding_model_name
        })
        
        return processed_metadata
    
    def hybrid_search(
        self, 
        query: str, 
        collection_names: Optional[List[str]] = None,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search across multiple collections with advanced ranking
        """
        if collection_names is None:
            collection_names = list(self.collections.keys())
        
        all_results = []
        
        for collection_name in collection_names:
            if collection_name not in self.collections:
                continue
            
            try:
                collection = self.collections[collection_name]
                
                # Perform semantic search
                results = collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=filter_metadata
                )
                
                # Process results
                for i in range(len(results['ids'][0])):
                    result = {
                        'id': results['ids'][0][i],
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i],
                        'collection': collection_name,
                        'relevance_score': 1 - results['distances'][0][i]  # Convert distance to similarity
                    }
                    
                    # Apply similarity threshold
                    if result['relevance_score'] >= similarity_threshold:
                        all_results.append(result)
                
            except Exception as e:
                logger.error(f"Error searching collection {collection_name}: {e}")
        
        # Rank results by relevance and diversity
        ranked_results = self._rank_hybrid_results(all_results, query)
        
        return ranked_results[:n_results]
    
    def _rank_hybrid_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """
        Advanced ranking algorithm considering relevance, diversity, and collection importance
        """
        if not results:
            return []
        
        # Collection importance weights
        collection_weights = {
            "therapy_techniques": 1.2,
            "case_studies": 1.0,
            "theoretical_concepts": 0.9,
            "therapeutic_exercises": 1.1,
            "assessment_tools": 0.8
        }
        
        # Calculate enhanced scores
        for result in results:
            base_score = result['relevance_score']
            collection_weight = collection_weights.get(result['collection'], 1.0)
            
            # Metadata-based boosting
            metadata_boost = self._calculate_metadata_boost(result['metadata'], query)
            
            # Final score calculation
            result['final_score'] = base_score * collection_weight * metadata_boost
        
        # Sort by final score
        ranked_results = sorted(results, key=lambda x: x['final_score'], reverse=True)
        
        # Apply diversity filtering to avoid too many results from same collection
        diverse_results = self._apply_diversity_filter(ranked_results)
        
        return diverse_results
    
    def _calculate_metadata_boost(self, metadata: Dict[str, Any], query: str) -> float:
        """Calculate boost score based on metadata relevance"""
        boost = 1.0
        query_lower = query.lower()
        
        # Boost for therapy-specific terms in metadata
        therapy_terms = ["anxiety", "depression", "worry", "rumination", "metacognitive", "attention"]
        
        for term in therapy_terms:
            if any(term in str(value).lower() for value in metadata.values()):
                boost += 0.1
        
        # Boost for recent content
        if "added_timestamp" in metadata:
            try:
                added_date = datetime.fromisoformat(metadata["added_timestamp"])
                days_old = (datetime.now() - added_date).days
                if days_old < 30:  # Recent content gets boost
                    boost += 0.05
            except:
                pass
        
        return min(boost, 2.0)  # Cap the boost
    
    def _apply_diversity_filter(self, results: List[Dict[str, Any]], max_per_collection: int = 3) -> List[Dict[str, Any]]:
        """Apply diversity filtering to ensure balanced results across collections"""
        collection_counts = {}
        diverse_results = []
        
        for result in results:
            collection = result['collection']
            current_count = collection_counts.get(collection, 0)
            
            if current_count < max_per_collection:
                diverse_results.append(result)
                collection_counts[collection] = current_count + 1
        
        return diverse_results
    
    def get_collection_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all collections"""
        stats = {}
        
        for collection_name, collection in self.collections.items():
            try:
                count = collection.count()
                stats[collection_name] = {
                    "document_count": count,
                    "description": collection.metadata.get("description", ""),
                    "status": "active"
                }
            except Exception as e:
                stats[collection_name] = {
                    "document_count": 0,
                    "description": "",
                    "status": f"error: {e}"
                }
        
        return stats
    
    def semantic_similarity_search(
        self, 
        query: str, 
        collection_name: str,
        n_results: int = 10,
        threshold: float = 0.7
    ) -> List[Document]:
        """
        Perform semantic similarity search with threshold filtering
        """
        try:
            if collection_name not in self.collections:
                logger.error(f"Collection {collection_name} not found")
                return []
            
            collection = self.collections[collection_name]
            
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            documents = []
            for i in range(len(results['ids'][0])):
                similarity_score = 1 - results['distances'][0][i]
                
                if similarity_score >= threshold:
                    doc = Document(
                        page_content=results['documents'][0][i],
                        metadata={
                            **results['metadatas'][0][i],
                            'similarity_score': similarity_score,
                            'search_query': query
                        }
                    )
                    documents.append(doc)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error in semantic similarity search: {e}")
            return []
    
    def update_document(self, document_id: str, new_content: str, collection_name: str) -> bool:
        """Update existing document in collection"""
        try:
            collection = self.collections[collection_name]
            
            # ChromaDB doesn't support direct updates, so we delete and re-add
            collection.delete(ids=[document_id])
            
            collection.add(
                documents=[new_content],
                ids=[document_id],
                metadatas=[{"updated_timestamp": datetime.now().isoformat()}]
            )
            
            logger.info(f"Updated document {document_id} in {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating document: {e}")
            return False
    
    def delete_document(self, document_id: str, collection_name: str) -> bool:
        """Delete document from collection"""
        try:
            collection = self.collections[collection_name]
            collection.delete(ids=[document_id])
            
            logger.info(f"Deleted document {document_id} from {collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {e}")
            return False
    
    def reset_collection(self, collection_name: str) -> bool:
        """Reset (clear) a specific collection"""
        try:
            if collection_name in self.collections:
                self.client.delete_collection(collection_name)
                # Recreate the collection
                self._initialize_collections()
                logger.info(f"Reset collection: {collection_name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    def backup_database(self, backup_path: str) -> bool:
        """Create backup of the entire database"""
        try:
            import shutil
            backup_path = Path(backup_path)
            backup_path.mkdir(parents=True, exist_ok=True)
            
            # Copy database files
            shutil.copytree(self.db_path, backup_path / f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            logger.info(f"Database backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize vector database
    vector_db = AdvancedVectorDatabase()
    
    # Get collection statistics
    stats = vector_db.get_collection_stats()
    print("Collection Statistics:")
    for collection, stat in stats.items():
        print(f"  {collection}: {stat['document_count']} documents")