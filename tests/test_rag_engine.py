from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from core.rag_engine import RAGEngine


@patch("core.rag_engine.Chroma")
@patch("core.rag_engine.OllamaEmbeddings")
def test_get_vectorstore_existing_and_cached(mock_emb, mock_chroma, tmp_path):
    persist_dir = tmp_path / "chroma"
    persist_dir.mkdir()
    (persist_dir / "chroma.sqlite3").write_text("dummy", encoding="utf-8")

    mock_vs = MagicMock()
    mock_chroma.return_value = mock_vs

    engine = RAGEngine(persist_dir=str(persist_dir), core_kb_dir=str(tmp_path / "kb"))
    vs1 = engine.get_vectorstore()
    vs2 = engine.get_vectorstore()

    assert vs1 is mock_vs
    assert vs2 is mock_vs
    mock_chroma.assert_called_once()


@patch("core.rag_engine.DirectoryLoader")
@patch("core.rag_engine.Chroma")
@patch("core.rag_engine.OllamaEmbeddings")
def test_get_vectorstore_initializes_from_core_kb(
    mock_emb, mock_chroma, mock_loader_cls, tmp_path
):
    persist_dir = tmp_path / "empty_chroma"
    kb_dir = tmp_path / "kb"
    kb_dir.mkdir()
    (kb_dir / "guideline.txt").write_text(
        "Synthetic security guideline.", encoding="utf-8"
    )

    mock_vs = MagicMock()
    mock_chroma.return_value = mock_vs

    mock_loader = MagicMock()
    mock_loader.load.return_value = [
        Document(page_content="Synthetic security guideline.")
    ]
    mock_loader_cls.return_value = mock_loader

    engine = RAGEngine(persist_dir=str(persist_dir), core_kb_dir=str(kb_dir))
    vs = engine.get_vectorstore()

    assert vs is mock_vs
    mock_vs.add_documents.assert_called_once()


@patch("core.rag_engine.Chroma")
@patch("core.rag_engine.OllamaEmbeddings")
def test_add_documents_from_texts_and_retrieve_context(mock_emb, mock_chroma, tmp_path):
    mock_vs = MagicMock()
    mock_chroma.return_value = mock_vs

    engine = RAGEngine(
        persist_dir=str(tmp_path / "chroma"),
        core_kb_dir=str(tmp_path / "missing_kb"),
    )

    assert engine.add_documents_from_texts([]) == 0
    assert (
        engine.add_documents_from_texts([{"content": "", "source_name": "a.txt"}]) == 0
    )

    count = engine.add_documents_from_texts(
        [{"content": "Policy rules for gateway.", "source_name": "policy.txt"}]
    )
    assert count >= 1
    mock_vs.add_documents.assert_called_once()

    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [
        Document(page_content="Chunk 1"),
        Document(page_content="Chunk 2"),
    ]
    mock_vs.as_retriever.return_value = mock_retriever
    assert engine.retrieve_context("gateway", k=2) == "Chunk 1\nChunk 2"

    mock_retriever.invoke.return_value = []
    assert engine.retrieve_context("unknown") == "No additional context available."
