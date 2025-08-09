
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import pandas as pd

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI


class LangChainRAGPipeline:
    """
    A complete RAG pipeline using LangChain for processing transcript data.
    
    This class handles document loading, chunking, vector storage, and querying
    with a simple RAG chain implementation using LangChain.
    """
    
    def __init__(
        self,
        data_dir: str = "data",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        persist_dir: str = "./vector_store",
        top_k: int = 5,
        recontextualize: bool = False,
        include_title_in_content: bool = False
    ):
        """
        Initialize the LangChain RAG pipeline.
        
        Args:
            data_dir: Directory containing transcript files
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between consecutive chunks
            persist_dir: Directory to persist vector store
            top_k: Number of top chunks to retrieve
            recontextualize: Whether to recontextualize chunks with ChatGPT
            include_title_in_content: Whether to prepend file title to document content
        """
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.persist_dir = Path(persist_dir)
        self.top_k = top_k
        self.recontextualize = recontextualize
        self.include_title_in_content = include_title_in_content
        
        # Initialize components
        self._setup_models()
        self._setup_storage()
        
        # Initialize DataFrame to store vector database results with configuration-based naming
        self._generate_config_name()
        self.vector_db_df = pd.DataFrame()
        self.chunk_data_df = pd.DataFrame()
        
    def _generate_config_name(self):
        """Generate a configuration name based on parameters."""
        recontext_str = "_recontext" if self.recontextualize else ""
        title_str = "_title" if self.include_title_in_content else ""
        self.config_name = f"chunk{self.chunk_size}_overlap{self.chunk_overlap}_topk{self.top_k}{recontext_str}{title_str}"
        
    def _setup_models(self):
        """Setup embedding and LLM models."""
        # Always use OpenAI embeddings
        self.embeddings = OpenAIEmbeddings()     
        self.llm = ChatOpenAI(model="gpt-4o-mini")

        
    def _setup_storage(self):
        """Setup vector storage directory."""
        self.persist_dir.mkdir(exist_ok=True)       

        
    def load_documents(self) -> List[Document]:
        """
        Load transcript documents from the data directory.
        
        Returns:
            List of Document objects
        """
        documents = []
        
        for file_path in self.data_dir.glob("*.txt"):
            print(f"Loading document: {file_path.name}")
            
            try:
                # Use LangChain's TextLoader
                loader = TextLoader(str(file_path), encoding='utf-8')
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    # Add title to content if enabled
                    if self.include_title_in_content:
                        title_prefix = f"title = {file_path.stem}\n\n"
                        doc.page_content = title_prefix + doc.page_content
                    
                    doc.metadata.update({
                        "source": file_path.name,
                        "file_path": str(file_path),
                        "file_size": len(doc.page_content),
                        "name_of_the_file": file_path.name,
                    })
                
                documents.extend(docs)
                
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue
            
        print(f"Loaded {len(documents)} documents")
        return documents
        
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Chunk documents into smaller pieces for better retrieval.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        # Create text splitter with specified parameters
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunked_docs = []
        
        for doc in documents:
            # Split the document into chunks
            chunks = text_splitter.split_text(doc.page_content)
            
            for i, chunk in enumerate(chunks):
                # Create new document for each chunk
                chunk_doc = Document(
                    page_content=chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_id": i,
                        "total_chunks": len(chunks),
                    }
                )
                chunked_docs.append(chunk_doc)
                
        print(f"Created {len(chunked_docs)} chunks from {len(documents)} documents")
        
        # Apply recontextualization if enabled
        if self.recontextualize:
            print("Applying recontextualization with ChatGPT...")
            chunked_docs = self._recontextualize_chunks(chunked_docs, documents)
        
        # Remove duplicate chunks
        seen_contents = set()
        unique_chunked_docs = []
        
        for doc in chunked_docs:
            if doc.page_content not in seen_contents:
                seen_contents.add(doc.page_content)
                unique_chunked_docs.append(doc)
        
        print(f"Removed {len(chunked_docs) - len(unique_chunked_docs)} duplicate chunks")
        print(f"Final unique chunks: {len(unique_chunked_docs)}")
        
        # Store chunk data in DataFrame
        self._store_chunk_data(unique_chunked_docs)
            
        return unique_chunked_docs
    
    def _store_chunk_data(self, chunked_docs: List[Document]) -> None:
        """
        Store chunk data in a pandas DataFrame.
        
        Args:
            chunked_docs: List of chunked documents
        """
        chunk_data = []
        
        for i, doc in enumerate(chunked_docs):
            chunk_data.append({
                'chunk_id': i,
                'content': doc.page_content,
                'source': doc.metadata.get('source', ''),
                'file_path': doc.metadata.get('file_path', ''),
                'file_size': doc.metadata.get('file_size', 0),
                'name_of_the_file': doc.metadata.get('name_of_the_file', ''),
                'chunk_id_in_doc': doc.metadata.get('chunk_id', 0),
                'total_chunks_in_doc': doc.metadata.get('total_chunks', 0),
                'recontextualized': doc.metadata.get('recontextualized', False),
                'original_chunk': doc.metadata.get('original_chunk', ''),
                'recontextualized_answer': doc.metadata.get('recontextualized_answer', ''),
                'num_preceding': doc.metadata.get('num_preceding', 0),
                'num_next': doc.metadata.get('num_next', 0),
                'content_length': len(doc.page_content),
                'chunk_size': self.chunk_size,
                'chunk_overlap': self.chunk_overlap,
                'recontextualize': self.recontextualize,
                'include_title_in_content': self.include_title_in_content
            })
        
        self.chunk_data_df = pd.DataFrame(chunk_data)
        print(f"Stored {len(self.chunk_data_df)} chunks in DataFrame for configuration: {self.config_name}")
    
    def _recontextualize_chunks(self, chunked_docs: List[Document], original_docs: List[Document]) -> List[Document]:
        """
        Recontextualize chunks using ChatGPT by providing the preceding and next chunks as context.
        
        Args:
            chunked_docs: List of chunked documents
            original_docs: List of original documents
            
        Returns:
            List of recontextualized chunked documents
        """
        # Group chunks by source document
        chunks_by_source = {}
        for chunk_doc in chunked_docs:
            source_file = chunk_doc.metadata.get("source", "")
            if source_file not in chunks_by_source:
                chunks_by_source[source_file] = []
            chunks_by_source[source_file].append(chunk_doc)
        
        # Sort chunks by chunk_id within each source
        for source_file in chunks_by_source:
            chunks_by_source[source_file].sort(key=lambda x: x.metadata.get("chunk_id", 0))
        
        recontextualized_docs = []
        
        for source_file, chunks in chunks_by_source.items():
            for i, chunk_doc in enumerate(chunks):
                original_content = chunk_doc.page_content
                
                # Get 1 preceding and 1 next chunk
                preceding_chunks = []
                next_chunks = []
                
                # Get 1 preceding chunk
                if i >= 1:  # There is at least 1 preceding chunk
                    preceding_chunks = [chunks[i-1].page_content]
                
                # Get 1 next chunk
                if i <= len(chunks) - 2:  # There is at least 1 next chunk
                    next_chunks = [chunks[i+1].page_content]
                                   
                # Create context from preceding and next chunks, maintaining original order
                preceding_context = ""
                following_context = ""
                
                if preceding_chunks:
                    preceding_context = f"Preceding chunk: {preceding_chunks[0]}"
                
                if next_chunks:
                    following_context = f"Following chunk: {next_chunks[0]}"
                
                # Create prompt for recontextualization
                recontextualization_prompt = f"""Given the following original chunk and its surrounding context, please provide a recontextualized version that enhances the original chunk with broader context from the neighboring chunks.
                The recontextualized version should maintain the core information from the original chunk while incorporating relevant context from the surrounding chunks, preserving the natural flow and order of information.
                
                Context that comes BEFORE the original chunk:
                {preceding_context if preceding_context else "No preceding context available."}
                
                Original Chunk:
                {original_content}
                
                Context that comes AFTER the original chunk:
                {following_context if following_context else "No following context available."}
                
                Please provide the recontextualized version of the original chunk enhanced with surrounding context:"""
                
                try:
                    # Get recontextualized version from ChatGPT
                    recontextualized_content = self.llm.invoke(recontextualization_prompt).content
                    
                    # Check if the original content had a title prefix
                    title_prefix = ""
                    if self.include_title_in_content:
                        source_file = chunk_doc.metadata.get("source", "")
                        if source_file:
                            title_prefix = f"title = {Path(source_file).stem}\n\n"
                    
                    # Store both recontextualized answer and actual chunk as the main content
                    # Keep the recontextualized answer in metadata as well
                    combined_content = f"{title_prefix} recontextualized_answer: {recontextualized_content}\nactual_chunk: {original_content}"
                    
                    recontextualized_doc = Document(
                        page_content=combined_content,  # Use both recontextualized and original content
                        metadata={
                            **chunk_doc.metadata,
                            "original_chunk": original_content,
                            "recontextualized_answer": recontextualized_content,
                            "recontextualized": True,
                            "num_preceding": len(preceding_chunks),
                            "num_next": len(next_chunks)
                        }
                    )
                    recontextualized_docs.append(recontextualized_doc)
                    
                except Exception as e:
                    print(f"Error recontextualizing chunk {chunk_doc.metadata.get('chunk_id', 'unknown')}: {e}")
                    # Fall back to original chunk if recontextualization fails
                    recontextualized_docs.append(chunk_doc)
        
        print(f"Recontextualized {len(recontextualized_docs)} chunks using preceding/next chunk context")
        return recontextualized_docs
        
    def build_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Build vector store from documents.
        
        Args:
            documents: List of documents to index
            
        Returns:
            Chroma vector store
        """
        print("Building vector store...")
        
        # Create vector store
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir)
        )
        
        # Persist the vector store
        vectorstore.persist()
        
        print("Vector store built and persisted successfully")
        return vectorstore
    
    def clear_vectorstore(self):
        """Clear the vector store to prevent duplicates."""
        import shutil
        if self.persist_dir.exists():
            shutil.rmtree(self.persist_dir)
            print(f"Cleared vector store at {self.persist_dir}")
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
    def load_vectorstore(self) -> Optional[Chroma]:
        """
        Load existing vector store from storage.
        
        Returns:
            Chroma vector store if exists, None otherwise
        """
        try:
            vectorstore = Chroma(
                persist_directory=str(self.persist_dir),
                embedding_function=self.embeddings
            )
            
            # Check if the collection has documents
            if vectorstore._collection.count() > 0:
                print("Loaded existing vector store from storage")
                return vectorstore
            else:
                print("Vector store exists but is empty")
                return None
                
        except Exception as e:
            print(f"Could not load existing vector store: {e}")
            return None
            
    def create_rag_chain(self, vectorstore: Chroma):
        """
        Create a RAG chain with retrieval capabilities using retriever | prompt | llm format.
        
        Args:
            vectorstore: Chroma vector store to query
            
        Returns:
            Chain composed of retriever | prompt | llm
        """
        # Create retriever
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.top_k}
        )
        
        def get_context_from_question(input_dict):
            question = input_dict["question"]
            docs = retriever.get_relevant_documents(question)
            return "\n\n".join(doc.page_content for doc in docs)
        
        def create_prompt(input_dict):
            context = input_dict["context"]
            question = input_dict["question"]
            
            prompt_text = f"""You are a financial risk department of a bank, we need to monitor the financial health of our debtors. 
            Use the following context to answer the question at the end. Please provide a detailed answer 100 words or more.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            
            Context: {context}
            
            Question: {question}
            
            Answer:"""
            
            return prompt_text
        
        rag_chain = (
            {"context": RunnableLambda(get_context_from_question), "question": RunnablePassthrough()}
            | RunnableLambda(create_prompt)
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
        
    def setup_pipeline(self, force_rebuild: bool = False):
        """
        Setup the complete RAG pipeline.
        
        Args:
            force_rebuild: Whether to force rebuild the vector store
            
        Returns:
            RAG chain ready for querying
        """

        vectorstore = self.load_vectorstore()
        
        if force_rebuild or vectorstore is None:
            print("Building new vector store...")
            documents = self.load_documents()
            chunked_docs = self.chunk_documents(documents)
            vectorstore = self.build_vectorstore(chunked_docs)
        else:
            print("Using existing vector store...")
        
        return self.create_rag_chain(vectorstore)
        
    def query(self, rag_chain, question: str) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Args:
            rag_chain: The RAG chain to use
            question: The question to ask
            
        Returns:
            Dictionary containing answer
        """
        print(f"Querying: {question}")
        
        result = rag_chain.invoke({"question": question})
        
        print(f"Answer: {result}")
        return result
        
    def get_source_chunks(self, vectorstore: Chroma, question: str) -> List[Dict[str, Any]]:
        """
        Get source chunks for a query without generating a response.
        
        Args:
            vectorstore: The vector store to query
            question: The question to ask
            
        Returns:
            List of source chunks with metadata
        """
        # Search for similar documents
        docs = vectorstore.similarity_search(question, k=self.top_k)
        
        chunks = []
        seen_texts = set()
        
        for i, doc in enumerate(docs):
            # Only add if we haven't seen this text before
            if doc.page_content not in seen_texts:
                seen_texts.add(doc.page_content)
                chunks.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                    "rank": len(chunks) + 1,  # Adjust rank for unique chunks
                })
            
        return chunks
    
    def get_vectorstore_info(self) -> Dict[str, Any]:
        """Get information about the vector store contents."""
        vectorstore = self.load_vectorstore()
        if vectorstore is None:
            return {"status": "not_found", "count": 0}
        
        count = vectorstore._collection.count()
        return {
            "status": "loaded",
            "count": count,
            "persist_directory": str(self.persist_dir)
        }
    
    def get_chunk_dataframe(self) -> pd.DataFrame:
        """
        Get the chunk data DataFrame.
        
        Returns:
            DataFrame containing chunk information
        """
        return self.chunk_data_df
    
    def save_chunk_dataframe(self, filepath: str = None) -> None:
        """
        Save the chunk data DataFrame to a CSV file.
        
        Args:
            filepath: Path to save the CSV file (if None, uses configuration-based naming)
        """
        if not self.chunk_data_df.empty:
            if filepath is None:
                filepath = f"chunk_data_{self.config_name}.csv"
            
            self.chunk_data_df.to_csv(filepath, index=False)
            print(f"Saved chunk data to {filepath}")
        else:
            print("No chunk data to save")
    
    def get_vector_db_results(self, question: str = None) -> pd.DataFrame:
        """
        Get vector database results including chunk information and retrieval results.
        
        Args:
            question: Optional question to include retrieval results
            
        Returns:
            DataFrame with vector database results
        """
        results_data = []
        
        # Add chunk data
        for _, chunk_row in self.chunk_data_df.iterrows():
            result_row = {
                'chunk_id': chunk_row['chunk_id'],
                'content': chunk_row['content'],
                'source': chunk_row['source'],
                'file_path': chunk_row['file_path'],
                'file_size': chunk_row['file_size'],
                'name_of_the_file': chunk_row['name_of_the_file'],
                'chunk_id_in_doc': chunk_row['chunk_id_in_doc'],
                'total_chunks_in_doc': chunk_row['total_chunks_in_doc'],
                'recontextualized': chunk_row['recontextualized'],
                'original_chunk': chunk_row['original_chunk'],
                'recontextualized_answer': chunk_row['recontextualized_answer'],
                'num_preceding': chunk_row['num_preceding'],
                'num_next': chunk_row['num_next'],
                'content_length': chunk_row['content_length'],
                'chunk_size': chunk_row['chunk_size'],
                'chunk_overlap': chunk_row['chunk_overlap'],
                'recontextualize': chunk_row['recontextualize'],
                'include_title_in_content': chunk_row['include_title_in_content'],
                'question': question if question else None,
                'retrieved': False,
                'retrieval_rank': None,
                'retrieval_score': None
            }
            results_data.append(result_row)
        
        # Add retrieval results if question is provided
        if question:
            vectorstore = self.load_vectorstore()
            if vectorstore is not None:
                retrieved_chunks = self.get_source_chunks(vectorstore, question)
                
                # Mark retrieved chunks
                for chunk in retrieved_chunks:
                    chunk_text = chunk['text']
                    # Find matching chunk in results_data
                    for result_row in results_data:
                        if result_row['content'] == chunk_text:
                            result_row['retrieved'] = True
                            result_row['retrieval_rank'] = chunk['rank']
                            # Note: Chroma doesn't provide similarity scores by default
                            result_row['retrieval_score'] = None
                            break
        
        self.vector_db_df = pd.DataFrame(results_data)
        return self.vector_db_df
    
    def save_vector_db_results(self, filepath: str = None, question: str = None) -> None:
        """
        Save vector database results to a CSV file.
        
        Args:
            filepath: Path to save the CSV file (if None, uses configuration-based naming)
            question: Optional question to include retrieval results
        """
        results_df = self.get_vector_db_results(question)
        if not results_df.empty:
            if filepath is None:
                question_suffix = "_with_retrieval" if question else ""
                filepath = f"vector_db_results_{self.config_name}{question_suffix}.csv"
            
            results_df.to_csv(filepath, index=False)
            print(f"Saved vector database results to {filepath}")
        else:
            print("No vector database results to save")
    
    def get_configuration_name(self) -> str:
        """
        Get the configuration name.
        
        Returns:
            Configuration name string
        """
        return self.config_name


def main():
    """Example usage of the LangChain RAG pipeline."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Initialize pipeline with OpenAI API
    pipeline = LangChainRAGPipeline(
        chunk_size=1000,
        chunk_overlap=200,
        top_k=5,
    )
    
    # Setup pipeline
    rag_chain = pipeline.setup_pipeline()
    
    # Example queries for credit risk assessment
    example_questions = [
        "What were OMV's Q1 2025 financial results and key performance indicators?",
        "What is OMV's current debt level and financial leverage position in 2024?",
        "What are OMV's major capital expenditure projects and their funding sources in 2024?",
        "What is OMV's cash flow generation and liquidity position in 2024?",
        "What are the key risks and uncertainties affecting OMV's financial performance in 2024?",
        "What is OMV's dividend policy and payout ratio in 2024?",
        "What are OMV's strategic partnerships and their financial implication in 2025 and 2024?",
        "What is OMV's exposure to commodity price volatility and hedging strategies through time?",
    ]
    
    print("\n" + "="*50)
    print("LangChain RAG Pipeline Demo")
    print("="*50)
    
    for question in example_questions:
        print(f"\nQuestion: {question}")
        print("-" * 30)
        
        # Get answer
        result = pipeline.query(rag_chain, question)
        print(f"Answer: {result}")
    
    # Save vector database results
    pipeline.save_chunk_dataframe()
    pipeline.save_vector_db_results(example_questions[0])


if __name__ == "__main__":
    main() 