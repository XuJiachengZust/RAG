"""高级检索模块

基于LangChain原生能力的高级检索技术实现，包括：
- ChromaDB混合搜索适配器
- 智能查询重写
- 多视角检索
- 长上下文重排序
"""

from .chroma_hybrid_retriever import ChromaHybridRetriever
from .advanced_retrieval_manager import AdvancedRetrievalManager, get_advanced_retrieval_manager

__all__ = [
    'ChromaHybridRetriever',
    'AdvancedRetrievalManager',
    'get_advanced_retrieval_manager'
]