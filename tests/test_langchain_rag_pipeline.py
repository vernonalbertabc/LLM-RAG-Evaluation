import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
from langchain_core.documents import Document

from llm_evaluation.langchain_rag_pipeline import LangChainRAGPipeline


def make_pipeline(data_dir, persist_dir, **kwargs):
    """Instantiate the pipeline with LLM/embedding calls mocked."""
    with (
        patch("llm_evaluation.langchain_rag_pipeline.OpenAIEmbeddings", return_value=MagicMock()),
        patch("llm_evaluation.langchain_rag_pipeline.ChatOpenAI", return_value=MagicMock()),
    ):
        return LangChainRAGPipeline(
            data_dir=str(data_dir),
            persist_dir=str(persist_dir),
            **kwargs,
        )


def make_doc(content, source="test.txt"):
    return Document(
        page_content=content,
        metadata={
            "source": source,
            "file_path": f"/tmp/{source}",
            "file_size": len(content),
            "name_of_the_file": source,
        },
    )


class TestConfigName(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.pipeline = make_pipeline(self.tmp, self.tmp + "/vs", chunk_size=500, chunk_overlap=50, top_k=10)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_default_format(self):
        self.assertEqual(self.pipeline.config_name, "chunk500_overlap50_topk10")

    def test_recontext_suffix(self):
        p = make_pipeline(self.tmp, self.tmp + "/vs", recontextualize=True)
        self.assertIn("_recontext", p.config_name)

    def test_title_suffix(self):
        p = make_pipeline(self.tmp, self.tmp + "/vs", include_title_in_content=True)
        self.assertIn("_title", p.config_name)


class TestLoadDocuments(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pipeline = make_pipeline(self.tmp, self.tmp / "vs")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loads_txt_files(self):
        (self.tmp / "a.txt").write_text("Hello")
        (self.tmp / "b.txt").write_text("World")
        docs = self.pipeline.load_documents()
        self.assertEqual(len(docs), 2)

    def test_ignores_non_txt(self):
        (self.tmp / "notes.md").write_text("markdown")
        docs = self.pipeline.load_documents()
        self.assertEqual(docs, [])

    def test_empty_dir_returns_empty(self):
        self.assertEqual(self.pipeline.load_documents(), [])

    def test_metadata_keys_present(self):
        (self.tmp / "doc.txt").write_text("content")
        docs = self.pipeline.load_documents()
        for key in ("source", "file_path", "file_size", "name_of_the_file"):
            self.assertIn(key, docs[0].metadata)

    def test_title_prepended_when_enabled(self):
        (self.tmp / "report.txt").write_text("some content")
        pipeline = make_pipeline(self.tmp, self.tmp / "vs", include_title_in_content=True)
        docs = pipeline.load_documents()
        self.assertTrue(docs[0].page_content.startswith("title = report\n\n"))


class TestChunkDocuments(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pipeline = make_pipeline(self.tmp, self.tmp / "vs", chunk_size=100, chunk_overlap=20)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_long_doc_produces_multiple_chunks(self):
        docs = [make_doc("word " * 200)]
        chunks = self.pipeline.chunk_documents(docs)
        self.assertGreater(len(chunks), 1)

    def test_chunk_has_metadata_fields(self):
        chunks = self.pipeline.chunk_documents([make_doc("word " * 200)])
        for chunk in chunks:
            self.assertIn("chunk_id", chunk.metadata)
            self.assertIn("total_chunks", chunk.metadata)

    def test_duplicate_content_removed(self):
        text = "word " * 200
        docs = [make_doc(text, "a.txt"), make_doc(text, "b.txt")]
        chunks = self.pipeline.chunk_documents(docs)
        contents = [c.page_content for c in chunks]
        self.assertEqual(len(contents), len(set(contents)))

    def test_chunk_dataframe_populated(self):
        self.pipeline.chunk_documents([make_doc("word " * 50)])
        self.assertFalse(self.pipeline.chunk_data_df.empty)
        self.assertIn("content", self.pipeline.chunk_data_df.columns)


class TestRecontextualizeChunks(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pipeline = make_pipeline(self.tmp, self.tmp / "vs")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _chunk(self, content, chunk_id):
        return Document(page_content=content, metadata={"source": "doc.txt", "chunk_id": chunk_id})

    def test_llm_called_once_per_chunk(self):
        mock_resp = MagicMock()
        mock_resp.content = "recontextualized"
        self.pipeline.llm.invoke.return_value = mock_resp

        chunks = [self._chunk(f"chunk {i}", i) for i in range(3)]
        self.pipeline._recontextualize_chunks(chunks, [])

        self.assertEqual(self.pipeline.llm.invoke.call_count, 3)

    def test_recontextualized_flag_set_in_metadata(self):
        mock_resp = MagicMock()
        mock_resp.content = "better text"
        self.pipeline.llm.invoke.return_value = mock_resp

        result = self.pipeline._recontextualize_chunks([self._chunk("original", 0)], [])

        self.assertTrue(result[0].metadata["recontextualized"])
        self.assertEqual(result[0].metadata["original_chunk"], "original")

    def test_falls_back_to_original_on_llm_error(self):
        self.pipeline.llm.invoke.side_effect = Exception("API error")
        chunks = [self._chunk("original content", 0)]
        result = self.pipeline._recontextualize_chunks(chunks, [])
        self.assertEqual(result[0].page_content, "original content")


class TestQuery(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pipeline = make_pipeline(self.tmp, self.tmp / "vs")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_invokes_chain_and_returns_result(self):
        mock_chain = MagicMock()
        mock_chain.invoke.return_value = "The answer."
        result = self.pipeline.query(mock_chain, "What is revenue?")
        mock_chain.invoke.assert_called_once_with({"question": "What is revenue?"})
        self.assertEqual(result, "The answer.")


class TestGetSourceChunks(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pipeline = make_pipeline(self.tmp, self.tmp / "vs", top_k=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _mock_vectorstore(self, docs):
        vs = MagicMock()
        vs.similarity_search.return_value = docs
        return vs

    def test_deduplicates_results(self):
        docs = [
            Document(page_content="chunk A", metadata={}),
            Document(page_content="chunk B", metadata={}),
            Document(page_content="chunk A", metadata={}),  # duplicate
        ]
        chunks = self.pipeline.get_source_chunks(self._mock_vectorstore(docs), "question")
        texts = [c["text"] for c in chunks]
        self.assertEqual(len(texts), len(set(texts)))

    def test_ranks_are_sequential(self):
        docs = [Document(page_content=f"chunk {i}", metadata={}) for i in range(3)]
        chunks = self.pipeline.get_source_chunks(self._mock_vectorstore(docs), "q")
        self.assertEqual([c["rank"] for c in chunks], [1, 2, 3])


class TestChunkDataframe(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.pipeline = make_pipeline(self.tmp, self.tmp / "vs", chunk_size=100, chunk_overlap=10)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_get_chunk_dataframe_returns_dataframe(self):
        self.pipeline.chunk_documents([make_doc("word " * 20)])
        self.assertIsInstance(self.pipeline.get_chunk_dataframe(), pd.DataFrame)

    def test_save_writes_csv(self):
        self.pipeline.chunk_documents([make_doc("word " * 20)])
        out = str(self.tmp / "out.csv")
        self.pipeline.save_chunk_dataframe(out)
        self.assertTrue(Path(out).exists())

    def test_save_with_no_data_prints_message(self):
        from io import StringIO
        import sys
        buf = StringIO()
        sys.stdout = buf
        try:
            self.pipeline.save_chunk_dataframe()
        finally:
            sys.stdout = sys.__stdout__
        self.assertIn("No chunk data", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
