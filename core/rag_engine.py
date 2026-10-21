import os
from typing import List, Optional, Any
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import BASE_URL, EMBEDDING_MODEL, CORE_KB_DIR, CHROMA_PERSIST_DIR


class RAGEngine:
    """
    Gestore del Vector Store per la Knowledge Base e per i documenti caricati dinamicamente.
    Completamente slegato da Streamlit o qualsiasi altro framework UI.
    """

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        core_kb_dir: str = CORE_KB_DIR,
        embedding_model: str = EMBEDDING_MODEL,
        base_url: str = BASE_URL,
    ):
        self.persist_dir = persist_dir
        self.core_kb_dir = core_kb_dir
        self.embeddings = OllamaEmbeddings(model=embedding_model, base_url=base_url)
        self._vectorstore: Optional[Chroma] = None

    def get_vectorstore(self) -> Chroma:
        """Carica o inizializza il VectorStore di Chroma."""
        if self._vectorstore is not None:
            return self._vectorstore

        if os.path.exists(self.persist_dir) and os.listdir(self.persist_dir):
            self._vectorstore = Chroma(
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )
            return self._vectorstore

        self._vectorstore = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )

        # Se la knowledge base di base esiste ed contiene file, popoliamo il vectorstore iniziale
        if os.path.exists(self.core_kb_dir) and os.listdir(self.core_kb_dir):
            loader = DirectoryLoader(
                self.core_kb_dir, glob="**/*.txt", loader_cls=TextLoader
            )
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = text_splitter.split_documents(docs)
            if splits:
                self._vectorstore.add_documents(documents=splits)

        return self._vectorstore

    def add_documents_from_texts(
        self, text_sources: List[dict]
    ) -> int:
        """
        Aggiunge al vectorstore una lista di testi forniti come dizionari.
        Formato di ciascun elemento: {"content": str, "source_name": str}
        Rende la funzione del tutto generica rispetto alla sorgente (file caricati via API, CLI o UI).
        """
        if not text_sources:
            return 0

        vectorstore = self.get_vectorstore()
        new_docs = []

        for src in text_sources:
            content = src.get("content", "")
            source_name = src.get("source_name", "unknown")
            if content:
                doc = Document(page_content=content, metadata={"source": source_name})
                new_docs.append(doc)

        if not new_docs:
            return 0

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(new_docs)

        if splits:
            vectorstore.add_documents(documents=splits)

        return len(splits)

    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Esegue la ricerca di similarità sul vectorstore e formatta il testo estratto."""
        vectorstore = self.get_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": k})
        retrieved_docs = retriever.invoke(query)

        if retrieved_docs:
            return "\n".join([doc.page_content for doc in retrieved_docs])
        return "No additional context available."