#!/usr/bin/env python3
"""
Chunk Visualization Script using Chonkie
Creates HTML visualizations of document chunking with the same parameters as LangChain pipeline.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))

from chonkie import SentenceChunker, Visualizer
import tiktoken

from langchain_rag_pipeline import LangChainRAGPipeline


class ChunkVisualizer:
    """
    Visualizes document chunking using Chonkie with LangChain pipeline parameters.
    """
    
    def __init__(
        self,
        data_dir: str = "../../data",
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        output_dir: str = "./chunk_visualizations"
    ):
        """
        Initialize the chunk visualizer.
        
        Args:
            data_dir: Directory containing text files
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            output_dir: Directory to save HTML visualizations
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize Chonkie SentenceChunker with same parameters as LangChain pipeline
        tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        self.chunker = SentenceChunker(
            tokenizer_or_token_counter=tokenizer,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            min_sentences_per_chunk=1, 
            delim=["\n\n", "\n", " ", ""]
        )
        
        # Initialize Chonkie visualizer
        self.visualizer = Visualizer()
        
        print(f"Initialized ChunkVisualizer with:")
        print(f"  Chunk size: {self.chunk_size}")
        print(f"  Chunk overlap: {self.chunk_overlap}")
        print(f"  Output dir: {self.output_dir}")
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load documents from the data directory.
        
        Returns:
            List of document dictionaries with content and metadata
        """
        documents = []
        
        for file_path in self.data_dir.glob("*.txt"):
            print(f"Loading document: {file_path.name}")
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # No title modification needed for this configuration
                
                documents.append({
                    'content': content,
                    'source': file_path.name,
                    'file_path': str(file_path),
                    'file_size': len(content),
                    'name_of_the_file': file_path.name
                })
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
        
        print(f"Loaded {len(documents)} documents")
        return documents
    
    def create_chunk_visualization(self, document: Dict[str, Any]) -> str:
        """
        Create chunk visualization for a single document.
        
        Args:
            document: Document dictionary with content and metadata
            
        Returns:
            Path to the generated HTML file
        """
        content = document['content']
        source = document['source']
        
        print(f"Creating visualization for: {source}")
        
        # Chunk the document using Chonkie
        chunks = self.chunker.chunk(content)
        
        print(f"Created {len(chunks)} chunks")
        
        # Save HTML file using Chonkie Visualizer
        safe_filename = source.replace('.txt', '').replace(' ', '_').replace('/', '_')
        config_suffix = f"_chunk{self.chunk_size}_overlap{self.chunk_overlap}"
        
        html_filename = f"{safe_filename}{config_suffix}.html"
        html_path = self.output_dir / html_filename
        
        # Use Chonkie Visualizer to save the HTML
        self.visualizer.save(str(html_path), chunks)
        
        print(f"Saved visualization to: {html_path}")
        return str(html_path)
    
    def create_all_visualizations(self) -> List[str]:
        """
        Create chunk visualizations for all documents.
        
        Returns:
            List of paths to generated HTML files
        """
        documents = self.load_documents()
        html_files = []
        
        for doc in documents:
            html_file = self.create_chunk_visualization(doc)
            html_files.append(html_file)
        
        return html_files
    
    def print_chunking_stats(self, documents: List[Dict[str, Any]]) -> None:
        """
        Print chunking statistics for all documents.
        
        Args:
            documents: List of document dictionaries
        """
        print("\n" + "="*50)
        print("CHUNKING STATISTICS")
        print("="*50)
        
        total_chunks = 0
        
        for doc in documents:
            content = doc['content']
            chunks = self.chunker.chunk(content)
            
            avg_chunk_length = sum(len(chunk.text) for chunk in chunks) / len(chunks) if chunks else 0
            
            print(f"\nDocument: {doc['source']}")
            print(f"  Original Length: {len(content):,} characters")
            print(f"  Number of Chunks: {len(chunks)}")
            print(f"  Average Chunk Length: {avg_chunk_length:.1f} characters")
            
            total_chunks += len(chunks)
        
        print(f"\nTOTAL:")
        print(f"  Documents: {len(documents)}")
        print(f"  Total Chunks: {total_chunks}")
        print(f"  Configuration: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")
        print("="*50)


def create_visualizations_for_pipeline_config(pipeline: LangChainRAGPipeline) -> List[str]:
    """
    Create visualizations using the same configuration as a LangChain pipeline.
    
    Args:
        pipeline: LangChainRAGPipeline instance
        
    Returns:
        List of paths to generated HTML files
    """
    visualizer = ChunkVisualizer(
        data_dir=str(pipeline.data_dir),
        chunk_size=pipeline.chunk_size,
        chunk_overlap=pipeline.chunk_overlap
    )
    
    # Create individual visualizations
    html_files = visualizer.create_all_visualizations()
    
    # Print statistics
    documents = visualizer.load_documents()
    visualizer.print_chunking_stats(documents)
    
    return html_files


def main():
    """Main function to create chunk visualizations."""
    load_dotenv()
    
    print("=" * 60)
    print("CHUNK VISUALIZATION WITH CHONKIE")
    print("=" * 60)
    
    # Use the specified configuration: chunk_size=500, chunk_overlap=100
    visualizer = ChunkVisualizer(
        data_dir="./data",  # Correct path from project root
        chunk_size=500,
        chunk_overlap=100
    )
    
    # Create visualizations
    html_files = visualizer.create_all_visualizations()
    
    # Print statistics
    documents = visualizer.load_documents()
    visualizer.print_chunking_stats(documents)
    
    print(f"\n{'='*60}")
    print("VISUALIZATION COMPLETED")
    print(f"{'='*60}")
    print(f"HTML files created: {len(html_files)}")
    print("\nGenerated files:")
    for html_file in html_files:
        print(f"  - {html_file}")
    
    print(f"\nOpen any of these HTML files in your browser to view the chunk visualizations!")


if __name__ == "__main__":
    main()