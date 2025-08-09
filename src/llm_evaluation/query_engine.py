from llama_index.core import VectorStoreIndex, Document
from llama_index.core.embeddings.utils import EmbedType
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.llms.llm import LLM
from llama_index.core.node_parser import SentenceWindowNodeParser
from llama_index.core.postprocessor import MetadataReplacementPostProcessor, LLMRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.query_engine import BaseQueryEngine

from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Configuration model for the query engine.

    Attributes:
        window_size (int): Window size for text preprocessing. Default is 3.
        window_metadata_key (str): Metadata key indicating the location of the text window.
        top_k (int): Number of top results to return for similarity search. Default is 10.
        use_rerank (bool): Flag indicating whether to use LLM reranking. Default is False.
        rerank_top_n (int): Number of top results to consider for reranking. Default is 3.
    """

    window_size: int = Field(default=3)
    window_metadata_key: str = Field(default="content")
    top_k: int = Field(default=10)
    use_rerank: bool = Field(default=False)
    rerank_top_n: int = Field(default=3)


def generate_pipeline(config: Config) -> IngestionPipeline:
    """
    Creates and returns an IngestionPipeline configured with a SentenceWindowNodeParser
    transformation. The parser splits input documents into overlapping windows of sentences,
    where the window size and metadata key are determined by the provided configuration.

    Args:
        config (Config): Configuration object specifying window size and metadata key.

    Returns:
        IngestionPipeline: An ingestion pipeline that processes documents into sentence windows
        for downstream indexing and retrieval.
    """
    transformations = [
        SentenceWindowNodeParser(
            window_size=config.window_size,
            window_metadata_key=config.window_metadata_key,
        )
    ]
    return IngestionPipeline(transformations=transformations)  # type: ignore


def generate_index(
    documents: list[Document], pipeline: IngestionPipeline, embed_model: EmbedType
) -> VectorStoreIndex:
    """
    Generates a VectorStoreIndex from a list of documents using the specified ingestion pipeline and
    embedding model.

    Args:
        documents (list[Document]): A list of Document objects to be indexed.
        pipeline (IngestionPipeline): The pipeline used to transform the documents into nodes.
        embed_model (EmbedType): The embedding model to generate vector representations of the nodes.

    Returns:
        VectorStoreIndex: An index containing the vectorized representations of the processed nodes.
    """
    nodes = pipeline.run(documents=documents)
    index = VectorStoreIndex(nodes=nodes, embed_model=embed_model)
    return index


def generate_query_engine(index: VectorStoreIndex, llm: LLM, config: Config) -> BaseQueryEngine:
    """
    Generates a query engine from the provided index, LLM, and configuration.

    Args:
        index (VectorStoreIndex): The vector store index to query.
        llm (LLM): The language model used for reranking (if enabled).
        config (Config): Configuration object specifying query parameters.

    Returns:
        BaseQueryEngine: A query engine configured with the specified index, LLM, and postprocessors.
    """
    node_postprocessors: list[BaseNodePostprocessor] = [
        MetadataReplacementPostProcessor(target_metadata_key="content")
    ]

    if config.use_rerank:
        node_postprocessors.append(LLMRerank(llm=llm, top_n=config.rerank_top_n))

    query_engine = index.as_query_engine(
        similarity_top_k=config.top_k,
        node_postprocessors=node_postprocessors,
    )
    return query_engine
