#!/usr/bin/env python3
"""
Script to run RAG evaluation experiments with different parameters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import eval_retrieval
from eval_retrieval import main as run_evaluation

# Chunk size experiments
CHUNK_SIZE_CONFIGS = [
    {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 10,
        "recontextualize": False,
        "include_title_in_content": False
    },
    {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 10,
        "recontextualize": False,
        "include_title_in_content": False
    },
    {
        "chunk_size": 1500,
        "chunk_overlap": 300,
        "top_k": 10,
        "recontextualize": False,
        "include_title_in_content": False
    }
]

# Contextualization experiments
CONTEXTUALIZATION_CONFIGS = [
    {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 5,
        "recontextualize": False,
        "include_title_in_content": False
    },
    {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 5,
        "recontextualize": True,
        "include_title_in_content": False
    },
    {
        "chunk_size": 500,
        "chunk_overlap": 100,
        "top_k": 5,
        "recontextualize": True,
        "include_title_in_content": True
    }
]

def get_experiment_configs(experiment_type: str):
    """Get experiment configurations based on type."""
    if experiment_type == "chunk_size":
        return CHUNK_SIZE_CONFIGS
    elif experiment_type == "contextualization":
        return CONTEXTUALIZATION_CONFIGS
    else:
        raise ValueError(f"Unknown experiment type: {experiment_type}")

def get_available_experiment_types():
    """Get list of available experiment types."""
    return ["chunk_size", "contextualization"]

def main():
    """Main function to run different types of experiments."""
    if len(sys.argv) < 2:
        print("Usage: python run_experiments.py [chunk_size|contextualization|top_k]")
        print("\nOptions:")
        print("  chunk_size       - Run experiments with different chunk sizes (500, 1000, 1500)")
        print("  contextualization - Run experiments comparing contextualized vs non-contextualized chunks")
        return
    
    experiment_type = sys.argv[1].lower()
    
    if experiment_type not in get_available_experiment_types():
        print(f"Unknown experiment type: {experiment_type}")
        print(f"Available types: {', '.join(get_available_experiment_types())}")
        return
    
    print(f"Setting up {experiment_type} experiments...")
    configs = get_experiment_configs(experiment_type)
    print(f"\n{experiment_type.upper()} experiments configured:")
    for config in configs:
        recontext_status = "with recontextualization" if config['recontextualize'] else "without recontextualization"
        title_status = "with title in content" if config.get('include_title_in_content', False) else "without title in content"
        print(f"- chunk_size={config['chunk_size']}, top_k={config['top_k']}, chunk_overlap={config['chunk_overlap']} ({recontext_status}, {title_status})")
    
    
    eval_retrieval.get_experiment_configs = lambda: configs
    run_evaluation()

if __name__ == "__main__":
    main() 
    