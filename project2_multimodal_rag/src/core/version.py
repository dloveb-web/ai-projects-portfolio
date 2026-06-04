"""
Version management for documents.
"""
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from src.models.document import Document, DocumentVersion, get_db
from src.utils.helpers import generate_unique_id, generate_file_hash
from datetime import datetime
import os


class VersionManager:
    """Manager for document versioning operations."""
    
    def __init__(self):
        pass
    
    def create_version(
        self,
        document_id: str,
        file_path: str,
        description: str = "",
        db: Optional[Session] = None
    ) -> DocumentVersion:
        """
        Create a new document version.
        
        Args:
            document_id: Document ID
            file_path: Path to the new file
            description: Version description
            db: Optional database session (if not provided, creates a new one)
            
        Returns:
            New DocumentVersion object
        """
        if db is None:
            db_gen = get_db()
            db = next(db_gen)
            try:
                return self.create_version(document_id, file_path, description, db)
            finally:
                try:
                    db.close()
                except:
                    pass
        
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            raise ValueError("文档不存在")
        
        file_hash = generate_file_hash(file_path)
        
        existing_version = db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id,
            DocumentVersion.file_hash == file_hash
        ).first()
        
        if existing_version:
            return existing_version
        
        version_number = db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id
        ).count() + 1
        
        new_version = DocumentVersion(
            id=generate_unique_id(),
            document_id=document_id,
            version_number=version_number,
            file_hash=file_hash,
            description=description
        )
        
        db.add(new_version)
        db.commit()
        db.refresh(new_version)
        
        return new_version
    
    def get_versions(
        self,
        document_id: str,
        db: Optional[Session] = None
    ) -> List[Dict[str, Any]]:
        """
        Get all versions for a document.
        
        Args:
            document_id: Document ID
            db: Optional database session
            
        Returns:
            List of version dictionaries
        """
        if db is None:
            db_gen = get_db()
            db = next(db_gen)
            try:
                return self.get_versions(document_id, db)
            finally:
                try:
                    db.close()
                except:
                    pass
        
        versions = db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id
        ).order_by(DocumentVersion.version_number).all()
        
        return [{
            "id": v.id,
            "version_number": v.version_number,
            "created_at": v.created_at.isoformat(),
            "description": v.description
        } for v in versions]
    
    def get_version(
        self,
        document_id: str,
        version_number: int,
        db: Optional[Session] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get a specific document version.
        
        Args:
            document_id: Document ID
            version_number: Version number
            db: Optional database session
            
        Returns:
            Version dictionary or None
        """
        if db is None:
            db_gen = get_db()
            db = next(db_gen)
            try:
                return self.get_version(document_id, version_number, db)
            finally:
                try:
                    db.close()
                except:
                    pass
        
        version = db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id,
            DocumentVersion.version_number == version_number
        ).first()
        
        if not version:
            return None
        
        return {
            "id": version.id,
            "version_number": version.version_number,
            "created_at": version.created_at.isoformat(),
            "description": version.description
        }
    
    def compare_versions(
        self,
        document_id: str,
        version1: int,
        version2: int,
        db: Optional[Session] = None
    ) -> Dict[str, Any]:
        """
        Compare two document versions.
        
        Args:
            document_id: Document ID
            version1: First version number
            version2: Second version number
            db: Optional database session
            
        Returns:
            Comparison dictionary
        """
        if db is None:
            db_gen = get_db()
            db = next(db_gen)
            try:
                return self.compare_versions(document_id, version1, version2, db)
            finally:
                try:
                    db.close()
                except:
                    pass
        
        v1 = db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id,
            DocumentVersion.version_number == version1
        ).first()
        
        v2 = db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id,
            DocumentVersion.version_number == version2
        ).first()
        
        if not v1 or not v2:
            return {}
        
        return {
            "version1": {
                "number": v1.version_number,
                "created_at": v1.created_at.isoformat(),
                "hash": v1.file_hash
            },
            "version2": {
                "number": v2.version_number,
                "created_at": v2.created_at.isoformat(),
                "hash": v2.file_hash
            },
            "same_content": v1.file_hash == v2.file_hash
        }
    
    def delete_version(
        self,
        document_id: str,
        version_number: int,
        db: Optional[Session] = None
    ) -> bool:
        """
        Delete a document version and re-number remaining versions.
        
        Args:
            document_id: Document ID
            version_number: Version number to delete
            db: Optional database session
            
        Returns:
            True if successful, False otherwise
        """
        if db is None:
            db_gen = get_db()
            db = next(db_gen)
            try:
                return self.delete_version(document_id, version_number, db)
            finally:
                try:
                    db.close()
                except:
                    pass
        
        version = db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id,
            DocumentVersion.version_number == version_number
        ).first()
        
        if not version:
            return False
        
        db.delete(version)
        db.commit()
        
        remaining_versions = db.query(DocumentVersion).filter(
            DocumentVersion.document_id == document_id
        ).order_by(DocumentVersion.version_number).all()
        
        for i, v in enumerate(remaining_versions):
            v.version_number = i + 1
        
        db.commit()
        return True
