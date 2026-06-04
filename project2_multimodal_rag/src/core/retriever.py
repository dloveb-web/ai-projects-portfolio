"""
Hybrid retriever combining BM25 and vector search for RAG.
"""
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from src.config import (
    settings,
    SearchConfig,
    ModelConfig,
    PromptTemplates
)

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# Optional imports with graceful fallback
try:
    import faiss
except ImportError:
    faiss = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

try:
    from dashscope import Generation
except ImportError:
    Generation = None


class HybridRetriever:
    """
    Hybrid retriever combining BM25 keyword search and FAISS vector search.
    Implements soft deletion for efficient document removal.
    """
    
    def __init__(self, use_vector: bool = settings.USE_VECTOR_SEARCH):
        """
        Initialize the hybrid retriever.
        
        Args:
            use_vector: Whether to use vector search
        """
        self.embedding_model: Optional[Any] = None
        self.index: Optional[Any] = None
        self.documents: List[Dict[str, Any]] = []
        self.bm25: Optional[Any] = None
        self.bm25_corpus: List[str] = []
        self.use_vector_search: bool = False
        self.model_loaded: bool = False
        self.use_vector: bool = use_vector
        self.embedding_dimension: Optional[int] = None
    
    def load_embedding_model(self) -> None:
        """
        Load the embedding model from Hugging Face or use cached version.
        """
        if self.model_loaded:
            return
        
        if not self.use_vector:
            print("⚠️ 跳过Embedding模型加载，直接使用BM25检索")
            self.use_vector_search = False
            self.model_loaded = True
            return
        
        if SentenceTransformer:
            try:
                print("⏳ 正在加载Embedding模型...")
                os.environ["HF_HUB_TIMEOUT"] = str(settings.HF_HUB_TIMEOUT)
                os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = str(settings.HF_HUB_DOWNLOAD_TIMEOUT)
                
                import signal
                
                def timeout_handler(signum: int, frame: Any) -> None:
                    raise TimeoutError("模型加载超时")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(180)
                
                self.embedding_model = SentenceTransformer(ModelConfig.EMBEDDING_MODEL)
                self.use_vector_search = True
                self.model_loaded = True
                
                # Auto-detect embedding dimension
                if self.embedding_model:
                    test_embedding = self.embedding_model.encode(["test"])
                    self.embedding_dimension = test_embedding.shape[1]
                    ModelConfig.EMBEDDING_DIMENSION = self.embedding_dimension
                    print(f"✅ Embedding模型加载成功，维度: {self.embedding_dimension}")
                
                signal.alarm(0)
            except Exception as e:
                print(f"⚠️ 初始化Embedding模型失败，将使用纯BM25检索：{e}")
                self.use_vector_search = False
                self.model_loaded = True
        else:
            print("⚠️ 跳过Embedding模型加载，直接使用BM25检索")
            self.use_vector_search = False
            self.model_loaded = True
    
    def init_faiss(self, dimension: Optional[int] = None) -> None:
        """
        Initialize FAISS index with specified or auto-detected dimension.
        
        Args:
            dimension: Embedding dimension (auto-detected if None
        """
        if not faiss:
            raise ImportError("FAISS未安装，请安装：pip install faiss-cpu")
        
        if dimension is None:
            if self.embedding_dimension is None:
                raise ValueError("Embedding dimension not available and not provided")
            dimension = self.embedding_dimension
        
        self.index = faiss.IndexFlatL2(dimension)
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the retriever.
        
        Args:
            documents: List of document dictionaries with "content" and metadata
        """
        # Ensure documents have is_active flag
        for doc in documents:
            if "is_active" not in doc:
                doc["is_active"] = True
        
        self.documents.extend(documents)
        self._update_bm25()
        self._update_vector_index(documents)
    
    def _update_bm25(self) -> None:
        """Update BM25 index with active documents only."""
        active_docs = [doc for doc in self.documents if doc.get("is_active", True)]
        self.bm25_corpus = [doc["content"] for doc in active_docs]
        
        if BM25Okapi and self.bm25_corpus:
            try:
                tokenized_corpus: List[List[str]] = []
                for doc in self.bm25_corpus:
                    if doc and doc.strip():
                        tokenized_corpus.append(doc.split())
                    else:
                        tokenized_corpus.append(["empty"])
                
                if tokenized_corpus:
                    self.bm25 = BM25Okapi(tokenized_corpus)
            except Exception as e:
                print(f"BM25索引更新失败：{e}")
                self.bm25 = None
    
    def _update_vector_index(self, new_documents: List[Dict[str, Any]]) -> None:
        """
        Update vector index with new documents.
        
        Args:
            new_documents: New documents to add to index
        """
        self.load_embedding_model()
        if self.embedding_model and faiss and np:
            try:
                active_new_docs = [doc for doc in new_documents if doc.get("is_active", True)]
                if active_new_docs:
                    embeddings = self.embedding_model.encode([doc["content"] for doc in active_new_docs])
                    if self.index is None:
                        self.init_faiss(embeddings.shape[1])
                    self.index.add(np.array(embeddings).astype(np.float32))
            except Exception as e:
                print(f"向量索引更新失败：{e}")
    
    def _get_active_documents(self) -> List[Dict[str, Any]]:
        """Get list of active documents (not soft-deleted)."""
        return [doc for doc in self.documents if doc.get("is_active", True)]
    
    def _get_active_indices(self) -> List[int]:
        """Get indices of active documents."""
        return [idx for idx, doc in enumerate(self.documents) if doc.get("is_active", True)]
    
    def search_bm25(
        self,
        query: str,
        top_k: int = SearchConfig.TOP_K,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using BM25 keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filters: Optional metadata filters
            
        Returns:
            List of search results with scores
        """
        active_docs = self._get_active_documents()
        if not active_docs:
            return []
        
        # Apply metadata filters if provided
        if metadata_filters:
            filtered_docs: List[Dict[str, Any]] = []
            for doc in active_docs:
                match = True
                for key, value in metadata_filters.items():
                    if doc.get(key) != value:
                        match = False
                        break
                if match:
                    filtered_docs.append(doc)
            active_docs = filtered_docs
        
        if not active_docs:
            return []
        
        if self.bm25:
            try:
                tokens = query.split()
                scores = self.bm25.get_scores(tokens)
                
                if np:
                    top_indices = np.argsort(scores)[::-1][:top_k]
                else:
                    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
                
                results: List[Dict[str, Any]] = []
                for idx in top_indices[:top_k]:
                    try:
                        if np and scores[idx] > 0:
                            results.append({
                                "document": active_docs[idx],
                                "score": float(scores[idx]),
                                "type": "bm25"
                            })
                        elif idx < len(active_docs):
                            results.append({
                                "document": active_docs[idx],
                                "score": 1.0,
                                "type": "simple"
                            })
                    except Exception:
                        pass
                
                return results
            except Exception as e:
                print(f"BM25搜索失败：{e}")
        
        # Fallback to simple keyword search
        results: List[Dict[str, Any]] = []
        query_lower = query.lower()
        for doc in active_docs:
            content = doc["content"].lower()
            if any(keyword in content for keyword in query_lower.split()):
                results.append({
                    "document": doc,
                    "score": 1.0,
                    "type": "simple"
                })
        
        return results[:top_k]
    
    def search_vector(
        self,
        query: str,
        top_k: int = SearchConfig.TOP_K,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using vector similarity.
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filters: Optional metadata filters
            
        Returns:
            List of search results with scores
        """
        self.load_embedding_model()
        if not self.embedding_model or not self.index or not np:
            return []
        
        try:
            query_embedding = self.embedding_model.encode([query])
            distances, indices = self.index.search(np.array(query_embedding).astype(np.float32), top_k)
            
            results: List[Dict[str, Any]] = []
            active_indices = self._get_active_indices()
            
            for i, idx in enumerate(indices[0]):
                if idx != -1 and idx < len(self.documents):
                    doc = self.documents[idx]
                    
                    # Skip soft-deleted check
                    if not doc.get("is_active", True):
                        continue
                    
                    # Apply metadata filters
                    if metadata_filters:
                        match = True
                        for key, value in metadata_filters.items():
                            if doc.get(key) != value:
                                match = False
                                break
                        if not match:
                            continue
                    
                    results.append({
                        "document": doc,
                        "score": float(1 / (1 + distances[0][i])),
                        "type": "vector"
                    })
            
            return results[:top_k]
        except Exception as e:
            print(f"向量搜索失败：{e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = SearchConfig.TOP_K,
        bm25_weight: float = SearchConfig.BM25_WEIGHT,
        vector_weight: float = SearchConfig.VECTOR_WEIGHT,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining BM25 and vector search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            bm25_weight: Weight for BM25 scores
            vector_weight: Weight for vector scores
            metadata_filters: Optional metadata filters
            
        Returns:
            List of combined search results
        """
        bm25_results = self.search_bm25(query, top_k=top_k, metadata_filters=metadata_filters)
        
        if not self.use_vector_search:
            return bm25_results
        
        vector_results = self.search_vector(query, top_k=top_k, metadata_filters=metadata_filters)
        
        if not bm25_results and not vector_results:
            return []
        
        if not bm25_results:
            return vector_results[:top_k]
        if not vector_results:
            return bm25_results[:top_k]
        
        combined: Dict[int, Dict[str, Any]] = {}
        for result in bm25_results:
            doc_id = id(result["document"])
            combined[doc_id] = {
                "document": result["document"],
                "bm25_score": result["score"],
                "vector_score": 0.0,
                "combined_score": result["score"] * bm25_weight
            }
        
        for result in vector_results:
            doc_id = id(result["document"])
            if doc_id in combined:
                combined[doc_id]["vector_score"] = result["score"]
                combined[doc_id]["combined_score"] += result["score"] * vector_weight
            else:
                combined[doc_id] = {
                    "document": result["document"],
                    "bm25_score": 0.0,
                    "vector_score": result["score"],
                    "combined_score": result["score"] * vector_weight
                }
        
        sorted_results = sorted(combined.values(), key=lambda x: x["combined_score"], reverse=True)
        return sorted_results[:top_k]
    
    def generate_hyde(self, query: str) -> str:
        """
        Generate hypothetical document embedding using LLM.
        
        Args:
            query: Original search query
            
        Returns:
            Hypothetical document text
        """
        if not Generation:
            print("⚠️ DashScope未安装，HyDE不可用")
            return query
        
        prompt = PromptTemplates.HYDE_PROMPT.format(query=query)
        
        try:
            response = Generation.call(
                model=ModelConfig.LLM_MODEL,
                prompt=prompt,
                api_key=settings.DASHSCOPE_API_KEY
            )
            return response.output.text.strip()
        except Exception as e:
            print(f"HyDE生成失败：{e}")
            return query
    
    def search_with_hyde(
        self,
        query: str,
        top_k: int = SearchConfig.TOP_K,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search using HyDE (Hypothetical Document Embedding).
        
        Args:
            query: Search query
            top_k: Number of results to return
            metadata_filters: Optional metadata filters
            
        Returns:
            List of search results
        """
        hyde_query = self.generate_hyde(query)
        return self.hybrid_search(hyde_query, top_k=top_k, metadata_filters=metadata_filters)
    
    def save_index(self, path: str) -> None:
        """
        Save the index to disk.
        
        Args:
            path: Path to save index
        """
        if not faiss:
            print("FAISS未安装，无法保存索引")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if self.index:
            faiss.write_index(self.index, path)
        
        with open(path + ".json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
    
    def remove_documents_by_source(self, source_id: str) -> None:
        """
        Soft delete documents by source ID (soft deletion for efficiency.
        
        Args:
            source_id: Document source ID to remove
        """
        for doc in self.documents:
            if doc.get("source") == source_id:
                doc["is_active"] = False
        
        # Update BM25 index (rebuilds with active docs only)
        self._update_bm25()
        
        # For vector index, we keep soft delete - still there but filtered out in searches
        # For full rebuild, we'd do it less frequently
    
    def permanently_remove_deleted(self) -> None:
        """
        Permanently remove soft-deleted documents and rebuild indexes.
        Use this periodically to clean up.
        """
        active_docs = [doc for doc in self.documents if doc.get("is_active", True)]
        self.documents = active_docs
        
        # Rebuild indexes from scratch
        self._update_bm25()
        
        if self.embedding_model and faiss and np and self.documents:
            try:
                embeddings = self.embedding_model.encode([doc["content"] for doc in self.documents])
                self.init_faiss(embeddings.shape[1])
                self.index.add(np.array(embeddings).astype(np.float32))
            except Exception as e:
                print(f"永久删除后重建索引失败：{e}")
    
    def load_index(self, path: str) -> None:
        """
        Load the index from disk.
        
        Args:
            path: Path to load index from
        """
        if faiss and os.path.exists(path):
            self.index = faiss.read_index(path)
        
        json_path = path + ".json"
        if os.path.exists(json_path):
            with open(json_path, "r", encoding="utf-8") as f:
                self.documents = json.load(f)
            self._update_bm25()
