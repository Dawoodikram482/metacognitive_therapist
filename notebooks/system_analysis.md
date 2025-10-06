# Example Jupyter Notebook for Metacognitive Therapy Chatbot Analysis

This notebook demonstrates various aspects of the system for analysis and experimentation.

## Setup

```python
import sys
sys.path.append('../src')

from document_processor import AdvancedDocumentProcessor
from vector_database import AdvancedVectorDatabase
from rag_system import AdvancedRAGSystem
from llm_manager import LLMManager, ModelType

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
```

## Document Processing Analysis

```python
# Initialize document processor
processor = AdvancedDocumentProcessor()

# Process a sample document
documents = processor.process_document("../data/Raw data.pdf")

# Analyze document chunks
chunk_lengths = [len(doc.page_content) for doc in documents]
chunk_metadata = [doc.metadata for doc in documents]

# Visualize chunk distribution
fig = px.histogram(x=chunk_lengths, title="Document Chunk Length Distribution")
fig.show()
```

## Vector Database Analysis

```python
# Initialize vector database
vector_db = AdvancedVectorDatabase()

# Get collection statistics
stats = vector_db.get_collection_stats()
print("Collection Statistics:")
for collection, stat in stats.items():
    print(f"  {collection}: {stat['document_count']} documents")

# Analyze search performance
test_queries = [
    "anxiety management techniques",
    "worry and rumination",
    "attention training therapy",
    "metacognitive strategies"
]

search_results = {}
for query in test_queries:
    results = vector_db.hybrid_search(query, n_results=5)
    search_results[query] = results
    print(f"\nQuery: {query}")
    print(f"Results: {len(results)}")
```

## RAG System Performance

```python
# Initialize complete system
llm_manager = LLMManager(ModelType.GPT4ALL_MISTRAL)
rag_system = AdvancedRAGSystem(vector_db, llm_manager)

# Test response generation
test_conversations = [
    "I've been feeling very anxious lately",
    "Can you teach me some coping techniques?",
    "What is metacognitive therapy?",
    "I can't stop worrying about work"
]

for query in test_conversations:
    response = rag_system.generate_rag_response(query)
    print(f"\nQuery: {query}")
    print(f"Response: {response['response'][:200]}...")
    print(f"Sources used: {response.get('documents_used', 0)}")
```

## System Analytics

```python
# Create analytics dashboard
import time
import numpy as np

# Simulate performance metrics
response_times = np.random.gamma(2, 2, 100)  # Simulated response times
success_rates = np.random.beta(9, 1, 100)    # Simulated success rates

# Create performance visualization
fig = go.Figure()
fig.add_trace(go.Scatter(
    y=response_times,
    mode='lines+markers',
    name='Response Time (s)',
    yaxis='y'
))

fig.add_trace(go.Scatter(
    y=success_rates * 100,
    mode='lines+markers',
    name='Success Rate (%)',
    yaxis='y2'
))

fig.update_layout(
    title='System Performance Metrics',
    xaxis_title='Request Number',
    yaxis=dict(title='Response Time (s)', side='left'),
    yaxis2=dict(title='Success Rate (%)', side='right', overlaying='y')
)

fig.show()
```

## Model Comparison

```python
# Compare different models (if available)
models_to_test = [ModelType.GPT4ALL_MISTRAL, ModelType.GPT4ALL_ORCA_MINI]
model_performance = {}

for model_type in models_to_test:
    try:
        llm = LLMManager(model_type)
        test_result = llm.test_model()
        model_performance[model_type.value] = test_result
    except Exception as e:
        print(f"Could not test {model_type.value}: {e}")

# Create comparison chart
if model_performance:
    model_names = list(model_performance.keys())
    response_times = [model_performance[name]['response_time'] for name in model_names]
    
    fig = px.bar(x=model_names, y=response_times, title="Model Response Time Comparison")
    fig.update_yaxis(title="Response Time (seconds)")
    fig.show()
```

## Therapy Content Analysis

```python
# Analyze therapy concepts in the knowledge base
therapy_concepts = [
    "worry", "anxiety", "rumination", "metacognitive", 
    "attention", "mindfulness", "cognitive", "behavioral"
]

concept_frequency = {}
for concept in therapy_concepts:
    results = vector_db.hybrid_search(concept, n_results=10)
    concept_frequency[concept] = len(results)

# Visualize concept coverage
fig = px.bar(
    x=list(concept_frequency.keys()), 
    y=list(concept_frequency.values()),
    title="Therapy Concept Coverage in Knowledge Base"
)
fig.update_yaxis(title="Number of Relevant Documents")
fig.show()
```

## User Interaction Patterns

```python
# Simulate user interaction data
import datetime
import random

# Generate sample interaction data
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
daily_users = [random.randint(10, 100) for _ in dates]
session_lengths = [random.gamma(2, 10) for _ in dates]

interaction_df = pd.DataFrame({
    'date': dates,
    'daily_users': daily_users,
    'avg_session_length': session_lengths
})

# Create usage trends
fig = px.line(interaction_df, x='date', y='daily_users', 
              title='Daily Active Users Over Time')
fig.show()

fig2 = px.scatter(interaction_df, x='daily_users', y='avg_session_length',
                  title='Session Length vs Daily Users')
fig2.show()
```

This notebook provides a comprehensive analysis framework for understanding and optimizing the metacognitive therapy chatbot system.