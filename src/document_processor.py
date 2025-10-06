"""
Advanced Document Processing Pipeline for Metacognitive Therapy Chatbot

This module handles the sophisticated processing of therapy literature,
implementing multiple chunking strategies and metadata extraction.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import re
from dataclasses import dataclass

import pypdf
import spacy
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SpacyTextSplitter,
    TokenTextSplitter
)
from langchain.docstore.document import Document
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingConfig:
    """Configuration for document processing pipeline"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    similarity_threshold: float = 0.7
    enable_semantic_chunking: bool = True
    enable_metadata_extraction: bool = True

class AdvancedDocumentProcessor:
    """
    Sophisticated document processor that implements multiple chunking strategies,
    semantic analysis, and metadata extraction for therapy literature.
    """
    
    def __init__(self, config: ProcessingConfig = None):
        self.config = config or ProcessingConfig()
        self.nlp = None
        self.embedding_model = None
        self.setup_models()
        
    def setup_models(self):
        """Initialize NLP models and embeddings"""
        try:
            # Load spaCy model for advanced text processing
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy English model")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
            
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Loaded SentenceTransformer embedding model")
    
    def extract_pdf_content(self, pdf_path: Path) -> str:
        """
        Extract text content from PDF with enhanced formatting preservation
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                text_content = []
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    
                    # Clean and format text
                    cleaned_text = self._clean_pdf_text(page_text)
                    if cleaned_text.strip():
                        text_content.append(f"[Page {page_num + 1}]\n{cleaned_text}")
                
                full_text = "\n\n".join(text_content)
                logger.info(f"Extracted {len(full_text)} characters from {pdf_path}")
                return full_text
                
        except Exception as e:
            logger.error(f"Error extracting PDF content: {e}")
            return ""
    
    def _clean_pdf_text(self, text: str) -> str:
        """Clean and normalize PDF text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common PDF extraction issues
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)  # Fix hyphenated words
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)      # Normalize paragraph breaks
        
        # Remove headers/footers (basic heuristic)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            # Skip likely headers/footers
            if len(line) < 10 or re.match(r'^\d+$', line) or 'copyright' in line.lower():
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def create_semantic_chunks(self, text: str) -> List[Document]:
        """
        Create semantically coherent chunks using multiple strategies
        """
        chunks = []
        
        # Strategy 1: Recursive Character Text Splitter (Primary)
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        primary_chunks = recursive_splitter.split_text(text)
        
        # Strategy 2: Semantic Coherence Enhancement
        if self.config.enable_semantic_chunking and self.nlp:
            enhanced_chunks = self._enhance_semantic_coherence(primary_chunks)
            chunks.extend(enhanced_chunks)
        else:
            chunks.extend([Document(page_content=chunk) for chunk in primary_chunks])
        
        # Strategy 3: Add metadata and filter
        processed_chunks = self._add_chunk_metadata(chunks)
        filtered_chunks = self._filter_chunks(processed_chunks)
        
        logger.info(f"Created {len(filtered_chunks)} semantic chunks")
        return filtered_chunks
    
    def _enhance_semantic_coherence(self, chunks: List[str]) -> List[Document]:
        """
        Enhance chunks for semantic coherence using NLP analysis
        """
        enhanced_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Analyze chunk with spaCy
            doc = self.nlp(chunk)
            
            # Extract key information
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            sentences = [sent.text.strip() for sent in doc.sents]
            
            # Identify therapy-specific concepts
            therapy_concepts = self._extract_therapy_concepts(chunk)
            
            # Create document with enhanced metadata
            document = Document(
                page_content=chunk,
                metadata={
                    "chunk_id": i,
                    "entities": entities,
                    "sentence_count": len(sentences),
                    "therapy_concepts": therapy_concepts,
                    "word_count": len(chunk.split()),
                    "character_count": len(chunk)
                }
            )
            enhanced_chunks.append(document)
        
        return enhanced_chunks
    
    def _extract_therapy_concepts(self, text: str) -> List[str]:
        """
        Extract metacognitive therapy specific concepts and techniques
        """
        # Metacognitive therapy keywords and concepts
        mct_keywords = [
            "metacognitive", "worry", "rumination", "attention training",
            "detached mindfulness", "cognitive attentional syndrome",
            "meta-worry", "thought suppression", "cognitive control",
            "executive attention", "monitoring", "beliefs about thinking",
            "thought control strategies", "threat monitoring",
            "perseverative thinking", "attentional bias"
        ]
        
        therapy_techniques = [
            "attention training technique", "ATT", "detached mindfulness",
            "postponement", "exposure", "behavioral experiments",
            "situational attentional refocusing", "SAR"
        ]
        
        found_concepts = []
        text_lower = text.lower()
        
        for concept in mct_keywords + therapy_techniques:
            if concept.lower() in text_lower:
                found_concepts.append(concept)
        
        return found_concepts
    
    def _add_chunk_metadata(self, chunks: List[Document]) -> List[Document]:
        """Add comprehensive metadata to chunks"""
        if not self.config.enable_metadata_extraction:
            return chunks
        
        for i, chunk in enumerate(chunks):
            # Calculate embedding for semantic search
            embedding = self.embedding_model.encode(chunk.page_content)
            
            # Enhance existing metadata
            chunk.metadata.update({
                "chunk_index": i,
                "embedding_model": "all-MiniLM-L6-v2",
                "processing_timestamp": str(pd.Timestamp.now()),
                "semantic_density": self._calculate_semantic_density(chunk.page_content)
            })
        
        return chunks
    
    def _calculate_semantic_density(self, text: str) -> float:
        """Calculate semantic density score for content quality assessment"""
        if not self.nlp:
            return 0.5  # Default score if spaCy not available
        
        doc = self.nlp(text)
        
        # Calculate various density metrics
        entity_density = len(doc.ents) / len(doc) if len(doc) > 0 else 0
        noun_density = len([token for token in doc if token.pos_ == "NOUN"]) / len(doc) if len(doc) > 0 else 0
        unique_token_ratio = len(set([token.lemma_ for token in doc])) / len(doc) if len(doc) > 0 else 0
        
        # Combine metrics for overall semantic density
        semantic_density = (entity_density + noun_density + unique_token_ratio) / 3
        return min(semantic_density, 1.0)
    
    def _filter_chunks(self, chunks: List[Document]) -> List[Document]:
        """Filter chunks based on quality metrics"""
        filtered_chunks = []
        
        for chunk in chunks:
            # Quality filters
            if len(chunk.page_content.strip()) < self.config.min_chunk_size:
                continue
            
            if len(chunk.page_content) > self.config.max_chunk_size:
                # Split large chunks further
                large_chunk_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.config.max_chunk_size,
                    chunk_overlap=self.config.chunk_overlap
                )
                sub_chunks = large_chunk_splitter.split_text(chunk.page_content)
                for sub_chunk in sub_chunks:
                    filtered_chunks.append(Document(
                        page_content=sub_chunk,
                        metadata={**chunk.metadata, "is_sub_chunk": True}
                    ))
            else:
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def process_document(self, pdf_path: Path) -> List[Document]:
        """
        Complete document processing pipeline
        """
        logger.info(f"Starting document processing for: {pdf_path}")
        
        # Extract text content
        text_content = self.extract_pdf_content(pdf_path)
        if not text_content:
            logger.error("No content extracted from document")
            return []
        
        # Create semantic chunks
        chunks = self.create_semantic_chunks(text_content)
        
        # Add document-level metadata
        for chunk in chunks:
            chunk.metadata.update({
                "source_document": str(pdf_path),
                "document_type": "therapy_literature",
                "processing_version": "1.0"
            })
        
        logger.info(f"Successfully processed document into {len(chunks)} chunks")
        return chunks

# Example usage and testing
if __name__ == "__main__":
    processor = AdvancedDocumentProcessor()
    # This would be used to process your therapy book
    # chunks = processor.process_document(Path("data/Raw data.pdf"))