"""
API routes for the multimodal RAG document center.
"""
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
import os

from src.models.document import Document, get_db, DocumentChunk
from src.api.models import DocumentResponse, QueryRequest, QueryResponse, VersionResponse, CompareResponse
from src.core.parser import DocumentParser
from src.core.retriever import HybridRetriever
from src.core.rag_chain import RAGChain
from src.core.version import VersionManager
from src.core.chunker import create_chunker
from src.utils.helpers import generate_unique_id, generate_file_hash, ensure_dir_exists
from src.config import (
    settings,
    FileConfig,
    ChunkingConfig,
    SearchConfig
)

router = APIRouter()

# Initialize core components
try:
    retriever = HybridRetriever(use_vector=settings.USE_VECTOR_SEARCH)
    rag_chain = RAGChain(retriever)
except Exception as e:
    print(f"初始化检索器失败：{e}")
    retriever = None
    rag_chain = None

version_manager = VersionManager()
document_parser = DocumentParser(init_layoutlm=settings.INIT_LAYOUTLM)
chunker = create_chunker(strategy="recursive")

DOCUMENT_STORAGE_PATH = settings.DOCUMENT_STORAGE_PATH
ensure_dir_exists(DOCUMENT_STORAGE_PATH)


def load_existing_documents_to_retriever():
    """Load existing documents from database into retriever on startup."""
    if not retriever:
        return
    
    try:
        db_gen = get_db()
        db = next(db_gen)
        
        active_docs = db.query(Document).filter(Document.is_active == True).all()
        if not active_docs:
            print("ℹ️ 数据库中没有活跃的文档")
            db.close()
            return
        
        print(f"⏳ 正在从数据库加载 {len(active_docs)} 个文档到检索系统...")
        loaded_count = 0
        
        for doc in active_docs:
            try:
                chunks = db.query(DocumentChunk).filter(DocumentChunk.document_id == doc.id).all()
                if chunks:
                    doc_chunks = []
                    for chunk in chunks:
                        doc_chunks.append({
                            "content": chunk.content,
                            "source": doc.id,
                            "page_number": chunk.page_number,
                            "chunk_index": chunk.chunk_index
                        })
                    if doc_chunks:
                        retriever.add_documents(doc_chunks)
                        loaded_count += 1
                else:
                    # If no chunks in DB, try to re-parse the file
                    file_path = os.path.join(DOCUMENT_STORAGE_PATH, doc.filename)
                    if os.path.exists(file_path):
                        try:
                            parsed = document_parser.parse(file_path)
                            all_content = []
                            for page in parsed.get("pages", []):
                                content = page.get("text", "")
                                if content.strip():
                                    all_content.append(content)
                            
                            if all_content:
                                full_content = "\n".join(all_content)
                                chunks = chunker.split_text(full_content)
                                
                                if chunks:
                                    chunk_docs = []
                                    for idx, chunk_content in enumerate(chunks):
                                        chunk_docs.append({
                                            "content": chunk_content,
                                            "source": doc.id,
                                            "page_number": 1,
                                            "chunk_index": idx
                                        })
                                    retriever.add_documents(chunk_docs)
                                    loaded_count += 1
                        except Exception as e:
                            print(f"⚠️ 重新解析文档 {doc.filename} 失败: {e}")
            except Exception as e:
                print(f"⚠️ 加载文档 {doc.filename} 失败: {e}")
        
        db.close()
        print(f"✅ 已加载 {loaded_count} 个文档到检索系统")
        
    except Exception as e:
        print(f"❌ 从数据库加载文档失败：{e}")


# Load existing documents on startup
load_existing_documents_to_retriever()


class QueryWithFilters(QueryRequest):
    """Extended query request with metadata filters."""
    metadata_filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters for search (e.g. {'source': 'doc1'})"
    )


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Upload and process a document.
    
    Args:
        file: Upload file
        db: Database session
        
    Returns:
        Upload result with document ID
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in FileConfig.SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的文件格式: {file_ext}")
    
    safe_filename = os.path.basename(file.filename)
    file_path = os.path.join(DOCUMENT_STORAGE_PATH, safe_filename)
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")
    
    file_hash = generate_file_hash(file_path)
    
    existing_doc = db.query(Document).filter(Document.file_hash == file_hash).first()
    if existing_doc:
        return {"message": "文档已存在", "document_id": existing_doc.id}
    
    doc_id = generate_unique_id()
    
    new_doc = Document(
        id=doc_id,
        title=safe_filename,
        filename=safe_filename,
        file_hash=file_hash,
        file_size=os.path.getsize(file_path),
        content_type=file.content_type
    )
    
    db.add(new_doc)
    db.commit()
    db.refresh(new_doc)
    
    try:
        parsed = document_parser.parse(file_path)
        
        all_content: List[str] = []
        for page in parsed.get("pages", []):
            content = page.get("text", "")
            if content.strip():
                all_content.append(content)
                chunk = DocumentChunk(
                    id=generate_unique_id(),
                    document_id=doc_id,
                    content=content,
                    content_type="text",
                    page_number=page.get("page_number", 1),
                    chunk_index=0,
                    chunk_metadata=str(page.get("images", []))
                )
                db.add(chunk)
        
        if retriever and all_content:
            full_content = "\n".join(all_content)
            # Use chunker for semantic chunking
            chunks = chunker.split_text(full_content)
            
            if chunks:
                # Add each chunk as a separate document for retrieval
                chunk_docs: List[Dict[str, Any]] = []
                for idx, chunk_content in enumerate(chunks):
                    chunk_docs.append({
                        "content": chunk_content,
                        "source": doc_id,
                        "page_number": 1,
                        "chunk_index": idx
                    })
                retriever.add_documents(chunk_docs)
            else:
                # Fallback to single document
                retriever.add_documents([{
                    "content": full_content,
                    "source": doc_id,
                    "page_number": 1
                }])
    except Exception as e:
        return {"message": f"文档上传成功，但解析失败：{str(e)}", "document_id": doc_id}
    
    db.commit()
    
    version_manager.create_version(doc_id, file_path, description="初始版本", db=db)
    
    return {"message": "文档上传成功", "document_id": doc_id}


@router.get("/documents")
def list_documents(db: Session = Depends(get_db)) -> List[DocumentResponse]:
    """
    List all active documents.
    
    Args:
        db: Database session
        
    Returns:
        List of document responses
    """
    docs = db.query(Document).filter(Document.is_active == True).all()
    return [DocumentResponse.model_validate(doc) for doc in docs]


@router.get("/documents/{document_id}")
def get_document(document_id: str, db: Session = Depends(get_db)) -> DocumentResponse:
    """
    Get a single document by ID.
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        Document response
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    return DocumentResponse.model_validate(doc)


@router.delete("/documents/{document_id}")
def delete_document(document_id: str, db: Session = Depends(get_db)) -> Dict[str, str]:
    """
    Delete a document (soft deletion).
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        Deletion result
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="文档不存在")
    
    doc.is_active = False
    db.commit()
    
    if retriever:
        retriever.remove_documents_by_source(document_id)
    
    return {"message": "文档已删除"}


@router.post("/query", response_model=QueryResponse)
def query(
    request: QueryRequest,
    metadata_filters: Optional[Dict[str, Any]] = None
) -> QueryResponse:
    """
    Query the RAG system.
    
    Args:
        request: Query request
        metadata_filters: Optional metadata filters
        
    Returns:
        Query response with answer and sources
    """
    if not rag_chain:
        return QueryResponse(
            answer="检索功能未初始化，请确保已安装所有依赖",
            sources=[],
            contexts=[]
        )
    try:
        result = rag_chain.answer(
            request.query,
            use_hyde=request.use_hyde,
            metadata_filters=metadata_filters
        )
        return QueryResponse(**result)
    except Exception as e:
        return QueryResponse(
            answer=f"查询失败：{str(e)}",
            sources=[],
            contexts=[]
        )


@router.post("/query-with-filter", response_model=QueryResponse)
def query_with_filter(
    request: QueryWithFilters
) -> QueryResponse:
    """
    Query the RAG system with optional metadata filters.
    
    Args:
        request: Query request with optional filters
        
    Returns:
        Query response with answer and sources
    """
    if not rag_chain:
        return QueryResponse(
            answer="检索功能未初始化，请确保已安装所有依赖",
            sources=[],
            contexts=[]
        )
    try:
        result = rag_chain.answer(
            request.query,
            use_hyde=request.use_hyde,
            metadata_filters=request.metadata_filters
        )
        return QueryResponse(**result)
    except Exception as e:
        return QueryResponse(
            answer=f"查询失败：{str(e)}",
            sources=[],
            contexts=[]
        )


@router.get("/documents/{document_id}/versions", response_model=List[VersionResponse])
def get_document_versions(document_id: str, db: Session = Depends(get_db)) -> List[VersionResponse]:
    """
    Get document versions.
    
    Args:
        document_id: Document ID
        db: Database session
        
    Returns:
        List of version responses
    """
    versions = version_manager.get_versions(document_id, db=db)
    return versions


@router.post("/documents/{document_id}/versions")
async def add_version(
    document_id: str,
    file: UploadFile = File(...),
    description: str = "",
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Add a new document version.
    
    Args:
        document_id: Document ID
        file: New file version
        description: Version description
        db: Database session
        
    Returns:
        Creation result
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    
    file_path = os.path.join(DOCUMENT_STORAGE_PATH, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    
    version = version_manager.create_version(document_id, file_path, description, db=db)
    
    return {"message": "版本创建成功", "version_number": version.version_number}


@router.get("/documents/{document_id}/versions/compare")
def compare_versions(
    document_id: str,
    v1: int,
    v2: int,
    db: Session = Depends(get_db)
) -> CompareResponse:
    """
    Compare two document versions.
    
    Args:
        document_id: Document ID
        v1: First version number
        v2: Second version number
        db: Database session
        
    Returns:
        Comparison result
    """
    result = version_manager.compare_versions(document_id, v1, v2, db=db)
    if not result:
        raise HTTPException(status_code=404, detail="版本不存在")
    return result


@router.delete("/documents/{document_id}/versions/{version_number}")
def delete_version(
    document_id: str,
    version_number: int,
    db: Session = Depends(get_db)
) -> Dict[str, str]:
    """
    Delete a document version.
    
    Args:
        document_id: Document ID
        version_number: Version number
        db: Database session
        
    Returns:
        Deletion result
    """
    success = version_manager.delete_version(document_id, version_number, db=db)
    if not success:
        raise HTTPException(status_code=404, detail="版本不存在")
    return {"message": "版本已删除"}
