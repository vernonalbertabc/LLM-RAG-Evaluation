# LLM RAG Evaluation - Bank Debtor Study

Case Study for Data Scientist / Machine Learning Engineer with LangChain RAG pipeline for evaluating debtor analysis in banking context.

## Overview

This project provides a comprehensive evaluation framework for Large Language Model (LLM) Retrieval-Augmented Generation (RAG) systems specifically designed for banking debtor studies. The system evaluates how effectively RAG pipelines can retrieve and analyze relevant information from financial documents and transcripts to support debtor assessment and risk analysis.

## Features

- RAG pipeline evaluation with multiple metrics
- Chunk visualization and analysis
- Query engine for document retrieval
- Support for various LLM providers (OpenAI, Ollama)
- MLflow integration for experiment tracking
- Comprehensive evaluation metrics including BERT-score, ROUGE, and RAGAS

## Installation

```bash
# Install dependencies using Poetry
poetry install

## Troubleshooting

### Common Issues

1. **Missing OpenAI API Key**: Ensure `OPENAI_API_KEY` is set in your environment or `.env` file
2. **MLflow Connection**: If MLflow UI isn't accessible, check if the server is running on port 5000
3. **Memory Issues**: For large documents, consider reducing chunk_size or top_k parameters
4. **Package Installation**: If installation fails, try using Python 3.10-3.12 and updating pip

### MLflow Setup Details

- **Local Storage**: By default, MLflow stores experiments in `./mlruns/`
- **Remote Tracking**: Set `MLFLOW_TRACKING_URI` to use remote MLflow server
- **Experiment Artifacts**: Includes generated visualizations and model outputs

## Quick Start

### Step 1: Install Dependencies

```bash
# Using Poetry 
poetry install
```

### Step 2: Set Up Environment Variables

Create a `.env` file in the project root:

```bash
# For OpenAI API (required)
OPENAI_API_KEY=your_openai_api_key_here

# For MLflow tracking (optional - defaults to local)
MLFLOW_TRACKING_URI=http://localhost:5000
```

### Step 3: Start MLflow Server (Optional)

To track experiments with MLflow UI:

```bash
# Start MLflow server in the background
mlflow server --host 0.0.0.0 --port 5000 &

# Or run in foreground
mlflow server --host 0.0.0.0 --port 5000
```

Access MLflow UI at: http://localhost:5000

### Step 4: Run the RAG Evaluation Pipeline

The main pipeline supports two types of experiments:

#### Option A: Chunk Size Experiments

Tests different chunk sizes (500, 1000, 1500 tokens) to find optimal chunking strategy:

```bash
python main.py chunk_size
```

#### Option B: Contextualization Experiments

Compares performance with and without chunk contextualization and title inclusion:

```bash
python main.py contextualization
```

#### Option C: Default Run

If no experiment type is specified, defaults to chunk_size experiments:

```bash
python main.py
```

### Step 5: View Results

- **Console Output**: Real-time progress and metrics during execution
- **MLflow UI**: Navigate to http://localhost:5000 to view:
  - Experiment runs and comparisons
  - Evaluation metrics (BERT-score, ROUGE, RAGAS)
  - Parameters and artifacts
  - Chunk visualizations

## Pipeline Details

### What the Pipeline Does

1. **Document Processing**: Loads financial documents and transcripts from the `data/` directory
2. **Chunk Creation**: Splits documents using various chunking strategies
3. **Vector Store**: Creates embeddings and stores in ChromaDB
4. **RAG Pipeline**: Sets up LangChain retrieval system with configurable parameters
5. **Evaluation**: Runs comprehensive evaluation using multiple metrics:
   - **BERT-score**: Semantic similarity between generated and reference answers
   - **ROUGE**: N-gram overlap scores
   - **RAGAS**: RAG-specific metrics (faithfulness, relevance, etc.)
6. **Visualization**: Generates chunk analysis visualizations
7. **MLflow Logging**: Tracks all experiments, parameters, and results

### Experiment Configurations

#### Chunk Size Experiments
- **Run 1**: 500 tokens (overlap: 100)
- **Run 2**: 1000 tokens (overlap: 200) 
- **Run 3**: 1500 tokens (overlap: 300)

#### Contextualization Experiments
- **Run 1**: Baseline (no contextualization)
- **Run 2**: With chunk recontextualization
- **Run 3**: With recontextualization + title inclusion

### Manual Usage (Advanced)

```python
from llm_evaluation.langchain_rag_pipeline import LangChainRAGPipeline
from llm_evaluation.eval_retrieval import RetrievalEvaluator

# Initialize RAG pipeline with custom parameters
pipeline = LangChainRAGPipeline(
    chunk_size=1000,
    chunk_overlap=200,
    top_k=10,
    recontextualize=False
)

# Set up and run evaluation
evaluator = RetrievalEvaluator(pipeline)
results = evaluator.evaluate_all()
```

## Project Structure

- `src/llm_evaluation/` - Main package
  - `langchain_rag_pipeline.py` - Complete RAG pipeline implementation
  - `eval_retrieval.py` - Evaluation metrics and retrieval testing
  - `chunk_visualization.py` - Visualization utilities
  - `run_experiments.py` - Experiment configuration and execution
  - `evaluation_questions.json` - Test questions and ground truth data
  - `utilities/` - Helper modules for metrics calculation
- `data/` - Financial documents and transcripts
- `chunk_visualizations/` - Generated visualization outputs
- `main.py` - Main entry point for running experiments

## Dependencies

- Python >=3.10, <3.13
- LangChain >=0.2.0
- ChromaDB ^0.4.24
- OpenAI ^1.14.3
- Sentence Transformers ^2.6.1
- MLflow ^3.1.4
- RAGAS ^0.3.0
- And more (see pyproject.toml)

## Development

```bash
# Install development dependencies
poetry install --with dev

# Run tests
pytest

# Lint code
ruff check src/
```

