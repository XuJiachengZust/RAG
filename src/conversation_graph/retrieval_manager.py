"""检索管理器模块

负责管理文档检索逻辑，与初始化逻辑分离。
实现单一职责原则：只负责检索相关功能。
"""

import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever

from .document_initializer import get_document_initializer, DocumentInitializer

logger = logging.getLogger(__name__)


def calculate_retrieval_quality(query: str, documents: List[Document]) -> float:
    """计算检索质量分数
    
    基于文档相关性、多样性和完整性评估检索质量。
    
    Args:
        query: 查询文本
        documents: 检索到的文档列表
        
    Returns:
        质量分数 (0.0-1.0)
    """
    if not documents:
        return 0.0
    
    try:
        # 基础分数：基于文档数量
        base_score = min(len(documents) / 5.0, 1.0)  # 假设5个文档为理想数量
        
        # 内容质量分数：基于文档内容长度和多样性
        content_lengths = [len(doc.page_content) for doc in documents]
        avg_length = sum(content_lengths) / len(content_lengths)
        
        # 长度分数：理想长度为200-1000字符
        length_score = 1.0
        if avg_length < 100:
            length_score = avg_length / 100.0
        elif avg_length > 2000:
            length_score = max(0.5, 2000.0 / avg_length)
        
        # 多样性分数：基于文档来源的多样性
        sources = set()
        for doc in documents:
            source = doc.metadata.get('source', 'unknown')
            sources.add(source)
        
        diversity_score = min(len(sources) / len(documents), 1.0)
        
        # 综合分数
        quality_score = (base_score * 0.3 + length_score * 0.4 + diversity_score * 0.3)
        
        return min(quality_score, 1.0)
        
    except Exception as e:
        logger.warning(f"计算检索质量时出错: {str(e)}")
        return 0.5  # 返回中等分数作为默认值


class RetrievalManager:
    """检索管理器
    
    职责：
    1. 管理已初始化的向量库实例
    2. 提供不同类型的检索方法（语义检索、混合检索等）
    3. 缓存检索器实例，避免重复创建
    4. 提供检索质量评估和优化
    5. 支持多种检索策略和参数配置
    
    设计原则：
    - 单一职责：只负责检索逻辑
    - 依赖注入：接收已初始化的向量库实例
    - 缓存优化：避免重复创建检索器
    - 配置灵活：支持不同的检索策略和参数
    """
    
    def __init__(self, document_initializer: Optional[DocumentInitializer] = None):
        """初始化检索管理器
        
        Args:
            document_initializer: 已初始化的文档初始化器实例
        """
        self.document_initializer = document_initializer
        self._retriever_cache: Dict[str, BaseRetriever] = {}
        self._vector_store_cache: Dict[str, VectorStore] = {}
        self.stats = {
            'total_retrievals': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'average_quality_score': 0.0
        }
        
    def get_retriever(self, 
                     k: int = 5, 
                     search_type: str = "similarity",
                     search_kwargs: Optional[Dict[str, Any]] = None) -> BaseRetriever:
        """获取检索器实例
        
        Args:
            k: 返回的文档数量
            search_type: 检索类型 (similarity, mmr, similarity_score_threshold)
            search_kwargs: 额外的检索参数
            
        Returns:
            配置好的检索器实例
            
        Raises:
            RuntimeError: 当向量库未初始化时
        """
        # 生成缓存键
        cache_key = self._generate_cache_key(k, search_type, search_kwargs)
        
        # 检查缓存
        if cache_key in self._retriever_cache:
            self.stats['cache_hits'] += 1
            logger.debug(f"从缓存获取检索器: {cache_key}")
            return self._retriever_cache[cache_key]
        
        # 缓存未命中，创建新的检索器
        self.stats['cache_misses'] += 1
        
        # 确保文档初始化器可用
        if not self.document_initializer:
            raise RuntimeError("文档初始化器未设置，无法创建检索器")
        
        # 获取向量库
        vector_store = self.document_initializer.vector_store
        if not vector_store:
            raise RuntimeError("向量库未初始化，请先调用文档初始化器的 initialize() 方法")
        
        # 准备检索参数
        final_search_kwargs = {"k": k}
        if search_kwargs:
            final_search_kwargs.update(search_kwargs)
        
        # 创建检索器
        retriever = vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=final_search_kwargs
        )
        
        # 缓存检索器
        self._retriever_cache[cache_key] = retriever
        logger.debug(f"创建并缓存新检索器: {cache_key}")
        
        return retriever
    
    def retrieve_documents(self, 
                          query: str, 
                          k: int = 5,
                          search_type: str = "similarity",
                          search_kwargs: Optional[Dict[str, Any]] = None,
                          quality_threshold: float = 0.0) -> Dict[str, Any]:
        """检索文档并返回详细结果
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            search_type: 检索类型
            search_kwargs: 额外的检索参数
            quality_threshold: 质量阈值，低于此值的结果将被过滤
            
        Returns:
            包含文档、质量分数和元数据的字典
        """
        try:
            # 获取检索器
            retriever = self.get_retriever(k, search_type, search_kwargs)
            
            # 执行检索
            documents = retriever.invoke(query)
            
            # 计算检索质量
            quality_score = calculate_retrieval_quality(query, documents)
            
            # 过滤低质量结果
            if quality_threshold > 0.0:
                if quality_score < quality_threshold:
                    logger.warning(f"检索质量分数 {quality_score:.3f} 低于阈值 {quality_threshold}")
            
            # 更新统计信息
            self.stats['total_retrievals'] += 1
            self._update_average_quality(quality_score)
            
            result = {
                'documents': documents,
                'quality_score': quality_score,
                'query': query,
                'retrieval_params': {
                    'k': k,
                    'search_type': search_type,
                    'search_kwargs': search_kwargs
                },
                'metadata': {
                    'document_count': len(documents),
                    'meets_threshold': quality_score >= quality_threshold,
                    'retrieval_id': self.stats['total_retrievals']
                }
            }
            
            logger.debug(f"检索完成: 查询='{query}', 文档数={len(documents)}, 质量分数={quality_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"文档检索失败: {str(e)}")
            raise
    
    def semantic_search(self, 
                       query: str, 
                       k: int = 5,
                       score_threshold: Optional[float] = None) -> List[Document]:
        """语义相似度检索
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            score_threshold: 相似度分数阈值
            
        Returns:
            检索到的文档列表
        """
        search_kwargs = {}
        if score_threshold is not None:
            search_kwargs['score_threshold'] = score_threshold
            
        result = self.retrieve_documents(
            query=query,
            k=k,
            search_type="similarity_score_threshold" if score_threshold else "similarity",
            search_kwargs=search_kwargs
        )
        return result['documents']
    
    def mmr_search(self, 
                   query: str, 
                   k: int = 5,
                   fetch_k: int = 20,
                   lambda_mult: float = 0.5) -> List[Document]:
        """最大边际相关性检索
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            fetch_k: 初始获取的文档数量
            lambda_mult: 多样性参数 (0-1)
            
        Returns:
            检索到的文档列表
        """
        search_kwargs = {
            'fetch_k': fetch_k,
            'lambda_mult': lambda_mult
        }
        
        result = self.retrieve_documents(
            query=query,
            k=k,
            search_type="mmr",
            search_kwargs=search_kwargs
        )
        return result['documents']
    
    def hybrid_search(self, 
                     query: str, 
                     k: int = 5,
                     keywords: Optional[List[str]] = None) -> List[Document]:
        """混合检索：结合语义检索和关键词检索
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            keywords: 额外的关键词列表
            
        Returns:
            检索到的文档列表
        """
        # 首先进行语义检索
        semantic_docs = self.semantic_search(query, k)
        
        # 如果提供了关键词，进行关键词检索并合并结果
        if keywords:
            keyword_query = " ".join(keywords)
            keyword_docs = self.semantic_search(keyword_query, k//2)
            
            # 合并文档，去重
            seen_content = {doc.page_content for doc in semantic_docs}
            for doc in keyword_docs:
                if doc.page_content not in seen_content:
                    semantic_docs.append(doc)
                    seen_content.add(doc.page_content)
        
        return semantic_docs[:k]
    
    def clear_cache(self):
        """清空检索器缓存"""
        self._retriever_cache.clear()
        self._vector_store_cache.clear()
        logger.info("检索器缓存已清空")
    
    def get_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        return {
            **self.stats,
            'cache_size': len(self._retriever_cache),
            'cache_hit_rate': self.stats['cache_hits'] / max(1, self.stats['cache_hits'] + self.stats['cache_misses'])
        }
    
    def set_document_initializer(self, document_initializer: DocumentInitializer):
        """设置文档初始化器
        
        Args:
            document_initializer: 文档初始化器实例
        """
        self.document_initializer = document_initializer
        # 清空缓存，因为向量库可能已更改
        self.clear_cache()
        logger.info("文档初始化器已更新，缓存已清空")
    
    def _generate_cache_key(self, 
                           k: int, 
                           search_type: str, 
                           search_kwargs: Optional[Dict[str, Any]]) -> str:
        """生成缓存键"""
        kwargs_str = ""
        if search_kwargs:
            # 将字典转换为排序后的字符串
            sorted_items = sorted(search_kwargs.items())
            kwargs_str = str(sorted_items)
        
        return f"{search_type}_{k}_{kwargs_str}"
    
    def _update_average_quality(self, quality_score: float):
        """更新平均质量分数"""
        total = self.stats['total_retrievals']
        current_avg = self.stats['average_quality_score']
        
        # 计算新的平均值
        new_avg = ((current_avg * (total - 1)) + quality_score) / total
        self.stats['average_quality_score'] = new_avg


# 全局检索管理器实例
_global_retrieval_manager: Optional[RetrievalManager] = None


def get_retrieval_manager(document_initializer: Optional[DocumentInitializer] = None) -> RetrievalManager:
    """获取全局检索管理器实例
    
    Args:
        document_initializer: 文档初始化器实例
        
    Returns:
        检索管理器实例
    """
    global _global_retrieval_manager
    
    if _global_retrieval_manager is None:
        # 如果没有提供初始化器，尝试获取默认的
        if document_initializer is None:
            document_initializer = get_document_initializer()
        
        _global_retrieval_manager = RetrievalManager(document_initializer)
        logger.info("创建全局检索管理器实例")
    else:
        # 如果提供了新的初始化器，更新它
        if document_initializer is not None:
            _global_retrieval_manager.set_document_initializer(document_initializer)
    
    return _global_retrieval_manager


def reset_retrieval_manager():
    """重置全局检索管理器实例"""
    global _global_retrieval_manager
    _global_retrieval_manager = None
    logger.info("全局检索管理器实例已重置")