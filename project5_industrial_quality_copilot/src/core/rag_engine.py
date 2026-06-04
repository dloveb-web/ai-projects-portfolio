"""
RAG (Retrieval-Augmented Generation) engine for industrial defect knowledge.

Uses ChromaDB for vector storage and sentence-transformers for embeddings.
Supports adding, searching, and finding similar defect cases.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..database.settings import settings

logger = logging.getLogger(__name__)


class RAGEngine:
    """Vector-store backed knowledge base for defect cases."""

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        self.persist_directory = persist_directory or settings.chroma_persist_directory
        self.embedding_model_name = embedding_model or settings.embedding_model
        self.embedding_model = None
        self.client = None
        self.collection = None
        self._initialize()

    # -- initialization ------------------------------------------------------

    def _initialize(self) -> None:
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)

        # Load embedding model
        try:
            from sentence_transformers import SentenceTransformer
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("Loaded embedding model: %s", self.embedding_model_name)
        except Exception as exc:
            logger.warning("Failed to load embedding model: %s", exc)
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                logger.info("Loaded fallback embedding model")
            except Exception:
                self.embedding_model = None

        # Initialize ChromaDB — use PersistentClient (new API)
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name="defect_knowledge",
                metadata={"description": "Industrial defect knowledge base"},
            )
            logger.info("Initialized ChromaDB collection (persistent)")
        except (TypeError, ImportError):
            # Fallback for older chromadb versions or missing dependency
            try:
                import chromadb
                from chromadb.config import Settings as ChromaSettings
                self.client = chromadb.Client(ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=self.persist_directory,
                    anonymized_telemetry=False,
                ))
                self.collection = self.client.get_or_create_collection(
                    name="defect_knowledge",
                    metadata={"description": "Industrial defect knowledge base"},
                )
                logger.info("Initialized ChromaDB (legacy API fallback)")
            except (ImportError, Exception) as exc:
                logger.warning("Failed to initialize ChromaDB: %s", exc)
                self.client = None

    # -- embedding -----------------------------------------------------------

    def _generate_embedding(self, text: str) -> List[float]:
        if self.embedding_model is None:
            return [0.0] * 384  # all-MiniLM-L6-v2 dimension
        embedding = self.embedding_model.encode(text)
        return embedding.tolist()

    # -- CRUD ----------------------------------------------------------------

    def add_document(
        self,
        document_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Add a single document to the knowledge base."""
        if self.collection is None:
            logger.warning("ChromaDB collection not available")
            return False

        try:
            embedding = self._generate_embedding(content)
            self.collection.add(
                ids=[document_id],
                embeddings=[embedding],
                documents=[content],
                metadatas=[metadata or {}],
            )
            logger.info("Added document %s", document_id)
            return True
        except Exception as exc:
            logger.error("Failed to add document %s: %s", document_id, exc)
            return False

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """Batch-add documents. Returns count of successfully added docs."""
        if self.collection is None:
            return 0

        success_count = 0
        for doc in documents:
            if self.add_document(
                doc["id"],
                doc["content"],
                doc.get("metadata"),
            ):
                success_count += 1
        return success_count

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Semantic search over the knowledge base."""
        if self.collection is None:
            return []

        try:
            query_embedding = self._generate_embedding(query)

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata,
            )

            documents: List[Dict[str, Any]] = []
            if results and results.get("documents") and results["documents"][0]:
                ids_list = results.get("ids", [[]])[0]
                docs_list = results["documents"][0]
                mds_list = results.get("metadatas", [[{}]])[0]
                dists_list = results.get("distances", [[0.0]])[0]

                for i, doc in enumerate(docs_list):
                    documents.append({
                        "id": ids_list[i] if i < len(ids_list) else "",
                        "content": doc,
                        "metadata": mds_list[i] if i < len(mds_list) else {},
                        "distance": dists_list[i] if i < len(dists_list) else 0.0,
                    })

            return documents

        except Exception as exc:
            logger.error("Search error: %s", exc)
            return []

    def search_by_defect_type(
        self, defect_type: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for cases matching a specific defect type."""
        return self.search(
            query=f"defect type: {defect_type}",
            top_k=top_k,
            filter_metadata={"category": "defect_case"},
        )

    def find_similar_cases(
        self,
        defect_description: str,
        defect_type: str,
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Find historically similar defect cases."""
        query = f"{defect_type} defect: {defect_description}"
        return self.search(query, top_k=top_k)

    def delete_document(self, document_id: str) -> bool:
        """Remove a document by ID."""
        if self.collection is None:
            return False
        try:
            self.collection.delete(ids=[document_id])
            return True
        except Exception as exc:
            logger.error("Failed to delete document %s: %s", document_id, exc)
            return False

    def reset(self) -> None:
        """Reset the entire knowledge base."""
        if self.client is not None:
            try:
                self.client.reset()
            except AttributeError:
                # PersistentClient doesn't have reset(); delete & recreate
                try:
                    self.client.delete_collection("defect_knowledge")
                except Exception:
                    pass
            self.collection = self.client.get_or_create_collection(
                name="defect_knowledge",
                metadata={"description": "Industrial defect knowledge base"},
            )

    def get_collection_count(self) -> int:
        """Return the number of documents in the collection."""
        if self.collection is None:
            return 0
        return self.collection.count()


# Module-level singleton (lazy-initialized to avoid import-time failures)
_rag_engine: Optional[RAGEngine] = None


def _get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        _rag_engine = RAGEngine()
    return _rag_engine


# Access via _get_rag_engine() for lazy initialization
