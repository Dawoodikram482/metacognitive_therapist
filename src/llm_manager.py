"""
Advanced Local LLM Management System for Metacognitive Therapy Chatbot

This module manages multiple local LLMs (GPT4All, Ollama) with sophisticated
prompt engineering, safety checks, and therapeutic response optimization.
"""

import logging
from typing import List, Dict, Any, Optional, Union, Callable, Tuple
import json
import re
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import threading
import time

# GPT4All for local LLM inference
from gpt4all import GPT4All

# Alternative: Ollama integration (if available)
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported local LLM types"""
    GPT4ALL_MISTRAL = "mistral-7b-openorca.Q4_0.gguf"
    GPT4ALL_LLAMA2 = "llama-2-7b-chat.q4_0.gguf"
    GPT4ALL_ORCA_MINI = "orca-mini-3b-gguf2-q4_0.gguf"
    OLLAMA_LLAMA2 = "llama2:7b"
    OLLAMA_MISTRAL = "mistral:7b"

@dataclass
class GenerationConfig:
    """Configuration for text generation - BALANCED FOR COMPLETE RESPONSES"""
    max_tokens: int = 300       # Allow for complete responses (~200-250 words)
    temperature: float = 0.3    # Balanced temp for quality responses
    top_p: float = 0.8          # Reasonable search space
    top_k: int = 20            # Balanced search space
    repeat_penalty: float = 1.05  # Lower penalty for speed
    n_batch: int = 32          # Larger batch for efficiency
    n_threads: int = 8         # More threads for speed
    enable_safety_filter: bool = False    # Disable for speed
    enable_therapeutic_optimization: bool = False  # Disable for speed  
    
    @classmethod
    def fast_config(cls):
        """Create a configuration optimized for speed but with complete responses"""
        return cls(
            max_tokens=250,  # Allow for complete responses
            temperature=0.3,
            top_p=0.8,
            top_k=20,
            n_batch=32,
            n_threads=8,
            enable_safety_filter=False,  
            enable_therapeutic_optimization=False  
        )

class SafetyFilter:
    """Safety filter for therapeutic responses"""
    
    def __init__(self):
        # Crisis indicators that require professional intervention
        self.crisis_patterns = [
            r'(?i)(kill|suicide|hurt|harm)\s+(myself|me)',
            r'(?i)(end\s+it\s+all|don\'t\s+want\s+to\s+live)',
            r'(?i)(going\s+to\s+hurt|plan\s+to\s+hurt)',
            r'(?i)(worthless|hopeless|no\s+point)'
        ]
        
        # Inappropriate response patterns to filter
        self.inappropriate_patterns = [
            r'(?i)(diagnose|diagnosis|prescribe|medication)',
            r'(?i)(you\s+should\s+take|you\s+need\s+to\s+take)',
            r'(?i)(emergency|call\s+(911|112)|seek\s+immediate)',
            r'(?i)(replace\s+professional|instead\s+of\s+therapy)'
        ]
    
    def check_crisis_indicators(self, text: str) -> bool:
        """Check if text contains crisis indicators"""
        for pattern in self.crisis_patterns:
            if re.search(pattern, text):
                return True
        return False
    
    def filter_response(self, response: str) -> Tuple[str, bool]:
        """Filter response for therapeutic appropriateness"""
        filtered_response = response
        needs_modification = False
        
        # Check for inappropriate content
        for pattern in self.inappropriate_patterns:
            if re.search(pattern, response):
                needs_modification = True
                break
        
        if needs_modification:
            # Replace with more appropriate language
            filtered_response = self._make_response_appropriate(response)
        
        return filtered_response, needs_modification
    
    def _make_response_appropriate(self, response: str) -> str:
        """Make response more therapeutically appropriate"""
        # Replace diagnostic language
        response = re.sub(
            r'(?i)(diagnose|diagnosis)', 
            'understand your experience', 
            response
        )
        
        # Replace prescriptive language
        response = re.sub(
            r'(?i)(you should|you must|you need to)', 
            'you might consider|it could be helpful to', 
            response
        )
        
        return response

class TherapeuticPromptEngineer:
    """Advanced prompt engineering for therapeutic responses"""
    
    def __init__(self):
        self.base_system_prompt = """You are a knowledgeable and empathetic AI assistant trained in metacognitive therapy principles. You provide supportive, evidence-based guidance while maintaining appropriate therapeutic boundaries.

CORE PRINCIPLES:
- Use warm, empathetic, and non-judgmental language
- Focus on metacognitive therapy techniques and concepts
- Encourage self-awareness and self-reflection
- Never diagnose or prescribe medication
- Always prioritize client safety and well-being
- Refer to professional help when appropriate

THERAPEUTIC TECHNIQUES TO REFERENCE:
- Attention Training Technique (ATT)
- Detached Mindfulness
- Worry Postponement
- Metacognitive Awareness
- Challenging Meta-worries
- Situational Attentional Refocusing (SAR)

RESPONSE STYLE:
- Be supportive and validating
- Ask thoughtful follow-up questions when appropriate
- Provide concrete, actionable suggestions
- Use person-first language
- Maintain professional therapeutic boundaries"""

    def create_therapeutic_prompt(
        self, 
        context: str, 
        user_query: str, 
        conversation_history: Optional[str] = None
    ) -> str:
        """Create optimized prompt for therapeutic response"""
        
        prompt_parts = [self.base_system_prompt]
        
        if context:
            prompt_parts.append(f"\nRELEVANT THERAPY CONTEXT:\n{context}")
        
        if conversation_history:
            prompt_parts.append(f"\nCONVERSATION HISTORY:\n{conversation_history}")
        
        prompt_parts.append(f"\nCLIENT MESSAGE: {user_query}")
        prompt_parts.append("\nTHERAPEUTIC RESPONSE:")
        
        return "\n".join(prompt_parts)
    
    def create_crisis_response_prompt(self, user_query: str) -> str:
        """Create specialized prompt for crisis situations"""
        return f"""You are responding to someone who may be in emotional distress. Your response must be:

1. Immediately supportive and validating
2. Emphasize that their life has value
3. Encourage professional help (therapist, counselor, crisis hotline)
4. Provide crisis resources if appropriate
5. Be warm but direct about the importance of professional support

CRISIS MESSAGE: {user_query}

SUPPORTIVE RESPONSE:"""

class LLMManager:
    """
    Advanced manager for local LLMs with therapeutic optimization
    """
    
    def __init__(self, model_type: ModelType = ModelType.GPT4ALL_MISTRAL, config: GenerationConfig = None):
        self.model_type = model_type
        self.config = config or GenerationConfig()
        
        # Initialize components
        self.safety_filter = SafetyFilter()
        self.prompt_engineer = TherapeuticPromptEngineer()
        
        # Model instances
        self.gpt4all_model = None
        self.ollama_client = None
        
        # Performance tracking
        self.generation_stats = {
            "total_generations": 0,
            "successful_generations": 0,
            "average_response_time": 0.0,
            "safety_filtered_responses": 0
        }
        
        # Initialize model
        self._initialize_model()
        
        logger.info(f"LLM Manager initialized with {self.model_type.value}")
    
    def _initialize_model(self):
        """Initialize the specified local LLM"""
        try:
            if self.model_type.value.endswith('.gguf'):
                # GPT4All model
                self._initialize_gpt4all()
            elif OLLAMA_AVAILABLE and self.model_type.value.startswith('ollama'):
                # Ollama model
                self._initialize_ollama()
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            # Fallback to a different model if available
            self._initialize_fallback_model()
    
    def _initialize_gpt4all(self):
        """Initialize GPT4All model"""
        try:
            model_name = self.model_type.value
            logger.info(f"Loading GPT4All model: {model_name}")
            
            self.gpt4all_model = GPT4All(
                model_name=model_name,
                model_path="./models/",  # Will download if not exists
                allow_download=True,
                device='cpu'  # Use CPU for compatibility
            )
            
            logger.info("GPT4All model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading GPT4All model: {e}")
            raise
    
    def _initialize_ollama(self):
        """Initialize Ollama client"""
        if not OLLAMA_AVAILABLE:
            raise ImportError("Ollama not available")
        
        try:
            self.ollama_client = ollama.Client()
            model_name = self.model_type.value.replace('ollama_', '').replace('_', ':')
            
            # Test connection
            response = self.ollama_client.generate(
                model=model_name,
                prompt="Test connection",
                stream=False
            )
            
            logger.info(f"Ollama model {model_name} initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            raise
    
    def _initialize_fallback_model(self):
        """Initialize fallback model if primary fails"""
        try:
            # Try GPT4All Orca Mini as lightweight fallback
            self.model_type = ModelType.GPT4ALL_ORCA_MINI
            self._initialize_gpt4all()
            logger.info("Fallback model initialized")
        except Exception as e:
            logger.error(f"Fallback model initialization failed: {e}")
            raise RuntimeError("No local LLM could be initialized")
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generate response using the configured local LLM - NO TIMEOUT, LET IT RUN"""
        start_time = time.time()
        
        try:
            # Use configured parameters for complete responses
            max_tokens = max_tokens or self.config.max_tokens
            temperature = temperature or self.config.temperature
            
            # Generate response directly without timeout restrictions
            if self.gpt4all_model:
                response = self._generate_with_gpt4all(prompt, max_tokens, temperature)
            elif self.ollama_client:
                response = self._generate_with_ollama(prompt, max_tokens, temperature)
            else:
                raise RuntimeError("No LLM model available")
            
            # Log generation time and update statistics
            generation_time = time.time() - start_time
            logger.info(f"Generation completed in {generation_time:.1f}s")
            
            # Update statistics for successful generation
            self._update_stats(generation_time, True)
            
            # Return the response
            return response.strip() if response else ""
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            self._update_stats(time.time() - start_time, False)
            raise  # Re-raise the exception instead of returning fallback
    
    def _generate_with_gpt4all(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using GPT4All - ULTRA SPEED OPTIMIZED"""
        try:
            # Use balanced settings for complete responses
            with self.gpt4all_model.chat_session():
                response = self.gpt4all_model.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,            # Use full requested token count
                    temp=temperature,                 # Use requested temperature
                    top_p=self.config.top_p,          # Use config values
                    top_k=self.config.top_k,          # Use config values
                    repeat_penalty=self.config.repeat_penalty,  # Use config values
                    n_batch=64,                       # Large batch for speed
                    streaming=False
                )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"GPT4All generation error: {e}")
            raise
    
    def _generate_with_ollama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        """Generate response using Ollama"""
        try:
            model_name = self.model_type.value.replace('ollama_', '').replace('_', ':')
            
            response = self.ollama_client.generate(
                model=model_name,
                prompt=prompt,
                options={
                    'num_predict': max_tokens,
                    'temperature': temperature,
                    'top_p': self.config.top_p,
                    'top_k': self.config.top_k,
                    'repeat_penalty': self.config.repeat_penalty
                },
                stream=False
            )
            
            return response['response'].strip()
            
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise
    
    def generate_therapeutic_response(
        self,
        system_context: str,
        user_query: str,
        conversation_history: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate therapeutic response with safety filtering and optimization
        """
        # Check for crisis indicators
        if self.safety_filter.check_crisis_indicators(user_query):
            return self._generate_crisis_response(user_query)
        
        # Create therapeutic prompt
        if self.config.enable_therapeutic_optimization:
            therapeutic_prompt = self.prompt_engineer.create_therapeutic_prompt(
                system_context, user_query, conversation_history
            )
        else:
            therapeutic_prompt = f"{system_context}\n\nUser: {user_query}\nAssistant:"
        
        # Generate response
        raw_response = self.generate_response(
            therapeutic_prompt, 
            max_tokens=max_tokens
        )
        
        # Apply safety filtering
        if self.config.enable_safety_filter:
            filtered_response, was_filtered = self.safety_filter.filter_response(raw_response)
            if was_filtered:
                self.generation_stats["safety_filtered_responses"] += 1
            return filtered_response
        
        return raw_response
    
    def _generate_crisis_response(self, user_query: str) -> str:
        """Generate specialized crisis response"""
        crisis_prompt = self.prompt_engineer.create_crisis_response_prompt(user_query)
        
        # Use more conservative generation parameters for crisis responses
        crisis_config = GenerationConfig(
            max_tokens=300,
            temperature=0.5,  # Lower temperature for more consistent responses
            top_p=0.8
        )
        
        return self.generate_response(crisis_prompt, 
                                    max_tokens=crisis_config.max_tokens,
                                    temperature=crisis_config.temperature)
    
    def _get_fallback_response(self) -> str:
        """Get fallback response when generation fails"""
        fallback_responses = [
            "I understand you're reaching out, and I want to help. Could you please rephrase your question or share more about what you're experiencing?",
            "I'm having difficulty processing your request right now. Would you like to try asking in a different way?",
            "I apologize for the technical difficulty. Your wellbeing is important - please feel free to rephrase your question or consider speaking with a mental health professional if you need immediate support."
        ]
        
        import random
        return random.choice(fallback_responses)
    
    def _get_fast_fallback_response(self) -> str:
        """Get instant fallback response when model is too slow"""
        return "I'm processing your message. While I prepare a response, remember: you're taking a positive step by reaching out. How are you feeling right now?"
    
    def _update_stats(self, response_time: float, success: bool):
        """Update generation statistics"""
        self.generation_stats["total_generations"] += 1
        
        if success:
            self.generation_stats["successful_generations"] += 1
        
        # Update average response time
        total_time = (self.generation_stats["average_response_time"] * 
                     (self.generation_stats["total_generations"] - 1) + response_time)
        self.generation_stats["average_response_time"] = total_time / self.generation_stats["total_generations"]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "model_type": self.model_type.value,
            "model_loaded": self.gpt4all_model is not None or self.ollama_client is not None,
            "generation_stats": self.generation_stats,
            "config": {
                "max_tokens": self.config.max_tokens,
                "temperature": self.config.temperature,
                "safety_filtering": self.config.enable_safety_filter,
                "therapeutic_optimization": self.config.enable_therapeutic_optimization
            }
        }
    
    def switch_model(self, new_model_type: ModelType) -> bool:
        """Switch to a different model"""
        try:
            old_model_type = self.model_type
            self.model_type = new_model_type
            
            # Clean up old model
            self.gpt4all_model = None
            self.ollama_client = None
            
            # Initialize new model
            self._initialize_model()
            
            logger.info(f"Switched from {old_model_type.value} to {new_model_type.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to switch models: {e}")
            # Revert to old model
            self.model_type = old_model_type
            self._initialize_model()
            return False
    
    def test_model(self) -> Dict[str, Any]:
        """Test model functionality and performance"""
        test_prompt = "Hello, how can I help you today?"
        
        start_time = time.time()
        try:
            response = self.generate_response(test_prompt, max_tokens=50)
            response_time = time.time() - start_time
            
            return {
                "status": "success",
                "response_time": response_time,
                "response_preview": response[:100],
                "model_info": self.get_model_info()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "response_time": time.time() - start_time,
                "model_info": self.get_model_info()
            }

if __name__ == "__main__":
    # Initialize LLM manager
    llm_manager = LLMManager(ModelType.GPT4ALL_MISTRAL)
    
    # Test the model
    test_result = llm_manager.test_model()
    print("Model Test Result:", json.dumps(test_result, indent=2))
    
    # Generate a therapeutic response
    test_query = "I've been feeling very anxious lately and can't stop worrying about everything."
    test_context = "Metacognitive therapy focuses on changing the way people think about their thoughts, particularly worry and rumination."
    
    response = llm_manager.generate_therapeutic_response(
        test_context, 
        test_query
    )
    
    print(f"\nTest Response: {response}")