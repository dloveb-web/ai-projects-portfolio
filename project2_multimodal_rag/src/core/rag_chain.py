"""
RAG (Retrieval-Augmented Generation) chain for question answering.
"""
import os
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv
import base64

from src.config import (
    settings,
    SearchConfig,
    ModelConfig,
    PromptTemplates
)

# Optional imports with graceful fallback
try:
    from dashscope import Generation
except ImportError:
    Generation = None

try:
    from dashscope import MultiModalConversation
except ImportError:
    MultiModalConversation = None

load_dotenv()


class RAGChain:
    """
    RAG chain that combines retrieval with LLM generation.
    """
    
    def __init__(self, retriever: Any):
        """
        Initialize the RAG chain.
        
        Args:
            retriever: Hybrid retriever instance
        """
        self.retriever = retriever
    
    def build_prompt(
        self,
        query: str,
        contexts: List[Dict[str, Any]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Build the prompt for the LLM.
        
        Args:
            query: User question
            contexts: Retrieved contexts
            
        Returns:
            Tuple of (prompt text, sources list)
        """
        context_text = ""
        sources: List[Dict[str, Any]] = []
        
        for i, ctx in enumerate(contexts):
            doc = ctx["document"]
            context_text += f"【来源{i+1}】\n{doc.get('content', '')}\n\n"
            sources.append({
                "source": doc.get("source", "unknown"),
                "page": doc.get("page_number", "unknown"),
                "score": ctx.get("combined_score", 0)
            })
        
        prompt = PromptTemplates.RAG_PROMPT.format(
            context=context_text,
            query=query
        )
        
        return prompt, sources
    
    def answer(
        self,
        query: str,
        use_hyde: bool = True,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.
        
        Args:
            query: User question
            use_hyde: Whether to use HyDE for enhanced retrieval
            metadata_filters: Optional metadata filters for retrieval
            
        Returns:
            Answer dict with answer, sources, and contexts
        """
        if not self.retriever:
            return {
                "answer": "检索系统未初始化",
                "sources": [],
                "contexts": []
            }
        
        try:
            # Retrieve contexts
            if use_hyde:
                contexts = self.retriever.search_with_hyde(
                    query,
                    top_k=SearchConfig.TOP_K,
                    metadata_filters=metadata_filters
                )
            else:
                contexts = self.retriever.hybrid_search(
                    query,
                    top_k=SearchConfig.TOP_K,
                    metadata_filters=metadata_filters
                )
            
            if not contexts:
                return {
                    "answer": "未找到相关文档",
                    "sources": [],
                    "contexts": []
                }
            
            prompt, sources = self.build_prompt(query, contexts)
            
            if Generation:
                try:
                    response = Generation.call(
                        model=ModelConfig.LLM_MODEL,
                        prompt=prompt,
                        api_key=settings.DASHSCOPE_API_KEY,
                        temperature=settings.TEMPERATURE
                    )
                    
                    answer = response.output.text.strip()
                    
                    for i, source in enumerate(sources):
                        answer += f"\n\n引用来源{i+1}: 文档-{source['source']}, 页码-{source['page']}"
                except Exception as e:
                    answer = f"生成回答时出错: {str(e)}\n\n找到的相关内容:\n"
                    for i, ctx in enumerate(contexts):
                        answer += f"\n{i+1}. {ctx['document']['content'][:200]}..."
            else:
                answer = "请安装 dashscope 依赖以使用完整的问答功能。\n\n找到的相关内容:\n"
                for i, ctx in enumerate(contexts):
                    answer += f"\n{i+1}. {ctx['document']['content'][:200]}..."
            
            return {
                "answer": answer,
                "sources": sources,
                "contexts": contexts
            }
        except Exception as e:
            return {
                "answer": f"回答生成失败: {str(e)}",
                "sources": [],
                "contexts": []
            }
    
    def multimodal_answer(
        self,
        query: str,
        image_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Answer a multimodal question (text + image).
        
        Args:
            query: User question
            image_bytes: Optional image bytes
            
        Returns:
            Answer dict
        """
        messages: List[Dict[str, Any]] = []
        
        if image_bytes:
            base64_image = base64.b64encode(image_bytes).decode('utf-8')
            messages.append({
                "role": "user",
                "content": [
                    {"image": f"data:image/png;base64,{base64_image}"},
                    {"text": query}
                ]
            })
        else:
            if not self.retriever:
                return {
                    "answer": "检索系统未初始化",
                    "sources": [],
                    "contexts": []
                }
            contexts = self.retriever.search_with_hyde(query, top_k=SearchConfig.TOP_K)
            context_text = "\n\n".join([ctx["document"].get("content", "") for ctx in contexts])
            
            messages.append({
                "role": "user",
                "content": [
                    {"text": f"参考文档:\n{context_text}\n\n用户问题: {query}"}
                ]
            })
        
        if MultiModalConversation:
            try:
                response = MultiModalConversation.call(
                    model="qwen-vl-plus",
                    messages=messages,
                    api_key=settings.DASHSCOPE_API_KEY
                )
                
                return {
                    "answer": response.output.choices[0].message.content[0]["text"],
                    "sources": [],
                    "contexts": []
                }
            except Exception as e:
                return {
                    "answer": f"多模态回答生成失败: {str(e)}",
                    "sources": [],
                    "contexts": []
                }
        else:
            return {
                "answer": "多模态功能需要 dashscope 依赖",
                "sources": [],
                "contexts": []
            }
