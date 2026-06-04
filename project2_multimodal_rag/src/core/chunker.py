"""
Text chunking functionality for the RAG system.
"""
from typing import List, Optional
from src.config import ChunkingConfig


class RecursiveCharacterTextSplitter:
    """
    Recursively splits text into chunks based on separators.
    Implements a simple version of LangChain's RecursiveCharacterTextSplitter.
    """
    
    def __init__(
        self,
        chunk_size: int = ChunkingConfig.CHUNK_SIZE,
        chunk_overlap: int = ChunkingConfig.CHUNK_OVERLAP,
        separators: Optional[List[str]] = None,
        min_chunk_size: int = ChunkingConfig.MIN_CHUNK_SIZE
    ):
        """
        Initialize the text splitter.
        
        Args:
            chunk_size: Target size of each chunk in characters
            chunk_overlap: Overlap between consecutive chunks
            separators: List of separators to use for splitting (in priority order)
            min_chunk_size: Minimum chunk size to keep
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ChunkingConfig.SEPARATORS
        self.min_chunk_size = min_chunk_size
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text into chunks.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            return []
        
        chunks = self._split_text_recursive(text, self.separators)
        chunks = self._merge_small_chunks(chunks)
        
        return chunks
    
    def _split_text_recursive(self, text: str, separators: List[str]) -> List[str]:
        """
        Recursively split text using separators.
        
        Args:
            text: Text to split
            separators: List of separators to try
            
        Returns:
            List of text chunks
        """
        separator = separators[0]
        new_separators = separators[1:] if len(separators) > 1 else []
        
        splits = text.split(separator)
        good_splits: List[str] = []
        
        for split in splits:
            if len(split) <= self.chunk_size:
                good_splits.append(split)
            else:
                if new_separators:
                    good_splits.extend(self._split_text_recursive(split, new_separators))
                else:
                    # Fallback: force split into fixed size
                    good_splits.extend(self._split_fixed_size(split))
        
        return good_splits
    
    def _split_fixed_size(self, text: str) -> List[str]:
        """
        Split text into fixed-size chunks.
        
        Args:
            text: Text to split
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_length = len(text)
        
        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.chunk_overlap
        
        return chunks
    
    def _merge_small_chunks(self, chunks: List[str]) -> List[str]:
        """
        Merge small chunks to meet minimum size requirement.
        
        Args:
            chunks: List of text chunks
            
        Returns:
            List of merged chunks
        """
        if not chunks:
            return []
        
        merged_chunks: List[str] = []
        current_chunk = ""
        
        for chunk in chunks:
            if len(current_chunk) + len(chunk) <= self.chunk_size:
                if current_chunk:
                    current_chunk += " " + chunk
                else:
                    current_chunk = chunk
            else:
                if len(current_chunk) >= self.min_chunk_size:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk
        
        if len(current_chunk) >= self.min_chunk_size:
            merged_chunks.append(current_chunk)
        elif merged_chunks and len(current_chunk) > 0:
            # Merge last small chunk with previous
            merged_chunks[-1] += " " + current_chunk
        
        return merged_chunks


class SimpleParagraphSplitter:
    """
    Simple text splitter that splits by paragraphs.
    """
    
    def __init__(
        self,
        chunk_size: int = ChunkingConfig.CHUNK_SIZE,
        chunk_overlap: int = ChunkingConfig.CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """
        Split text by paragraphs and merge as needed.
        
        Args:
            text: Input text to split
            
        Returns:
            List of text chunks
        """
        if not text or len(text.strip()) == 0:
            return []
        
        paragraphs = text.split("\n\n")
        chunks: List[str] = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
            
            if len(current_chunk) + len(paragraph) + 2 <= self.chunk_size:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks


def create_chunker(
    strategy: str = "recursive",
    chunk_size: int = ChunkingConfig.CHUNK_SIZE,
    chunk_overlap: int = ChunkingConfig.CHUNK_OVERLAP
):
    """
    Factory function to create a text chunker.
    
    Args:
        strategy: Chunking strategy ("recursive" or "paragraph")
        chunk_size: Target chunk size
        chunk_overlap: Chunk overlap
        
    Returns:
        Chunker instance
    """
    if strategy == "paragraph":
        return SimpleParagraphSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    else:
        return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
