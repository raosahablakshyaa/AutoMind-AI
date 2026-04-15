import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

_CACHED_RETRIEVER = None
_CACHED_RETRIEVER_KEY = None


class GuidelineRetriever:
    def __init__(self, db_dir: str = "chroma_db", source_file: str = "data/maintenance_guidelines.txt"):
        self.db_dir = db_dir
        self.source_file = source_file
        self.retriever = self._build_or_load_retriever()
        self.fallback_guidelines = [
            "If brake wear exceeds 80%, schedule immediate brake pad and rotor inspection within 24 hours.",
            "For engine temperature consistently above 105C, stop non-essential operation and perform cooling system diagnostics.",
            "If oil quality drops below 35%, perform oil and filter replacement at the earliest maintenance window.",
            "Vehicles overdue service by more than 180 days should be prioritized for preventive maintenance.",
        ]

    def _build_or_load_retriever(self):
        global _CACHED_RETRIEVER, _CACHED_RETRIEVER_KEY

        cache_key = f"{self.db_dir}|{self.source_file}"
        if _CACHED_RETRIEVER_KEY == cache_key and _CACHED_RETRIEVER is not None:
            return _CACHED_RETRIEVER

        source_path = Path(self.source_file)
        source_path.parent.mkdir(parents=True, exist_ok=True)

        if not source_path.exists():
            source_path.write_text("General maintenance guideline: Inspect critical systems monthly.")

        loader = TextLoader(str(source_path))
        documents = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = splitter.split_documents(documents)

        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=self.db_dir)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            _CACHED_RETRIEVER = retriever
            _CACHED_RETRIEVER_KEY = cache_key
            return retriever
        except Exception:
            # Keep the app usable even when optional embedding dependencies are missing.
            return None

    def retrieve(self, query: str) -> List[str]:
        if self.retriever is None:
            return self.fallback_guidelines
        try:
            docs = self.retriever.invoke(query)
            return [d.page_content for d in docs]
        except Exception:
            return self.fallback_guidelines
