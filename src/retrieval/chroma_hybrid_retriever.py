"""ChromaDB混合搜索适配器

基于LangChain原生能力的ChromaDB混合检索实现，结合向量搜索和关键词匹配。
使用LangChain原生的similarity_search_with_score方法获取标准评分。
"""

import logging

from typing import List, Dict, Any, Optional, Tuple, Union
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)


class ChromaHybridRetriever(BaseRetriever):
    """ChromaDB混合检索器
    
    结合ChromaDB向量搜索和jieba关键词匹配，使用LangChain原生评分机制。
    
    特性：
    - 使用LangChain原生的similarity_search_with_score获取标准评分
    - 结合向量相似度和关键词匹配评分
    - 支持多种检索策略（向量搜索、混合搜索、关键词搜索）
    - 自动结果合并和去重
    - 兼容现有ChromaDB配置
    """
    
    # Pydantic字段定义
    vector_store: VectorStore
    k: int = 5
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    min_score_threshold: float = 0.0
    enable_keyword_boost: bool = True
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        """
        初始化ChromaDB混合检索器
        """
        super().__init__(**kwargs)
        
        # 验证权重
        if abs(self.vector_weight + self.keyword_weight - 1.0) > 0.01:
            logger.warning(f"权重之和不等于1.0: vector_weight={self.vector_weight}, keyword_weight={self.keyword_weight}")
        
        # 确保是ChromaDB实例
        if not isinstance(self.vector_store, Chroma):
            logger.warning("向量存储不是ChromaDB实例，某些功能可能不可用")
    
    def _get_relevant_documents(self, query: str) -> List[Document]:
        """获取相关文档（BaseRetriever接口实现）"""
        return self.hybrid_search(query, self.k)
    
    def vector_search(self, query: str, k: int = None) -> List[Tuple[Document, float]]:
        """纯向量搜索，使用LangChain原生评分
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            
        Returns:
            (文档, 分数)元组列表，分数为LangChain原生相似度分数
        """
        k = k or self.k
        
        try:
            # 使用LangChain原生的similarity_search_with_score方法
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            # 过滤低分结果
            filtered_results = [
                (doc, score) for doc, score in results 
                if score >= self.min_score_threshold
            ]
            
            logger.debug(f"向量搜索完成: 查询='{query}', 原始结果={len(results)}, 过滤后={len(filtered_results)}")
            return filtered_results
            
        except Exception as e:
            logger.error(f"向量搜索失败: {str(e)}")
            return []
    
    def keyword_search(self, query: str, k: int = None, keywords: Optional[List[str]] = None) -> List[Tuple[Document, float]]:
        """关键词搜索，基于Chroma全文检索
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            keywords: 额外的关键词列表
            
        Returns:
            (文档, 关键词匹配分数)元组列表
        """
        k = k or self.k
        
        try:
            # 提取查询关键词
            if keywords:
                query_keywords = list(keywords)
            else:
                # 直接按空格分割查询文本作为关键词
                query_keywords = query.split()
            
            # 去重并过滤停用词
            query_keywords = [kw.strip() for kw in set(query_keywords) if len(kw.strip()) > 1]
            
            if len(query_keywords) < 2:
                logger.warning(f"关键词数量不足2个: {query_keywords}，返回空结果")
                return []
            
            # 使用Chroma的where_document进行全文检索
            # 构建至少匹配两个关键词的查询条件
            where_conditions = []
            for keyword in query_keywords:
                if keyword.strip() and len(keyword.strip()) > 1:  # 忽略空关键词和单字符关键词
                    where_conditions.append({"$contains": keyword.strip()})
            
            logger.debug(f"提取的关键词: {[kw for kw in query_keywords if kw.strip() and len(kw.strip()) > 1]}")
            
            if len(where_conditions) < 2:
                # 如果关键词少于2个，尝试放宽条件：只要求匹配一个关键词
                if len(where_conditions) == 1:
                    where_document = where_conditions[0]
                    logger.debug(f"关键词数量为1，使用单关键词搜索: {where_document}")
                else:
                    logger.debug(f"关键词数量不足({len(where_conditions)})，回退到原有方法")
                    return self._fallback_keyword_search(query, k, query_keywords)
            elif len(where_conditions) == 2:
                # 两个关键词，要求都匹配
                where_document = {"$and": where_conditions}
                logger.debug(f"使用AND条件: {where_document}")
            else:
                # 多个关键词，先尝试要求至少匹配两个关键词
                from itertools import combinations
                pair_conditions = []
                for pair in combinations(where_conditions, 2):
                    pair_conditions.append({"$and": list(pair)})
                where_document = {"$or": pair_conditions}
                logger.debug(f"使用OR条件(至少匹配2个): {len(pair_conditions)} 个组合")
            
            logger.debug(f"Chroma全文检索条件: {where_document}")
            
            # 执行Chroma全文检索
            try:
                # 使用Chroma的get方法进行全文检索（不需要embedding）
                if hasattr(self.vector_store, '_collection') and self.vector_store._collection is not None:
                    # 直接使用Chroma的collection进行查询
                    collection = self.vector_store._collection
                    
                    # 使用Chroma的get方法进行全文检索（不需要query_texts）
                    results = collection.get(
                        limit=min(k * 3, 100),  # 获取更多结果以便后续过滤，但限制最大数量
                        where_document=where_document,
                        include=["documents", "metadatas"]
                    )
                    
                    logger.debug(f"Chroma全文检索结果: {len(results.get('documents', []))} 个文档")
                    
                    # 转换为Document格式
                    scored_docs = []
                    if results and 'documents' in results and results['documents']:
                        documents = results['documents']
                        metadatas = results.get('metadatas', []) or [{}] * len(documents)
                        
                        for i, doc_text in enumerate(documents):
                            if doc_text:  # 确保文档内容不为空
                                metadata = metadatas[i] if i < len(metadatas) else {}
                                # 计算关键词匹配分数
                                score = self._calculate_keyword_score_with_minimum_match(doc_text, query_keywords)
                                if score > 0:
                                    # 添加检索相关的元数据
                                    if metadata is None:
                                        metadata = {}
                                    metadata['keyword_score'] = score
                                    
                                    doc = Document(page_content=doc_text, metadata=metadata)
                                    scored_docs.append((doc, score))
                    else:
                        logger.debug("Chroma全文检索未返回结果")
                else:
                    # 回退到原有方法
                    logger.warning("无法访问Chroma collection，回退到原有方法")
                    return self._fallback_keyword_search(query, k, query_keywords)
                
            except Exception as chroma_error:
                logger.warning(f"Chroma全文检索失败: {chroma_error}，回退到原有方法")
                return self._fallback_keyword_search(query, k, query_keywords)
            
            # 按分数排序并返回前k个
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            result = scored_docs[:k]
            
            logger.debug(f"Chroma全文检索完成: 关键词={query_keywords}, 匹配文档={len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"关键词搜索失败: {str(e)}")
            return []
    
    def hybrid_search(self, query: str, k: int = None, keywords: Optional[List[str]] = None) -> List[Document]:
        """混合搜索：结合向量搜索和关键词搜索
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            keywords: 额外的关键词列表
            
        Returns:
            合并后的文档列表
        """
        k = k or self.k
        
        # 执行向量搜索
        vector_results = self.vector_search(query, k)
        
        # 执行关键词搜索（如果启用）
        keyword_results = []
        if self.enable_keyword_boost:
            keyword_results = self.keyword_search(query, k, keywords)
        
        # 合并结果
        merged_results = self._merge_results_with_native_scores(
            vector_results, keyword_results, k
        )
        
        # 提取文档
        documents = [doc for doc, _ in merged_results]
        
        logger.info(f"混合搜索完成: 向量结果={len(vector_results)}, 关键词结果={len(keyword_results)}, 最终结果={len(documents)}")
        return documents
    
    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """计算关键词匹配分数
        
        Args:
            content: 文档内容
            keywords: 关键词列表
            
        Returns:
            关键词匹配分数 (0-1)
        """
        if not keywords or not content:
            return 0.0
        
        content_lower = content.lower()
        total_score = 0.0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # 精确匹配得分更高
            if keyword_lower in content_lower:
                # 计算关键词在内容中的出现次数
                count = content_lower.count(keyword_lower)
                # 基于出现次数和关键词长度计算分数
                score = min(count * len(keyword) / len(content), 1.0)
                total_score += score
        
        # 归一化分数
        normalized_score = min(total_score / len(keywords), 1.0)
        return normalized_score
    
    def _calculate_keyword_score_with_minimum_match(self, content: str, keywords: List[str]) -> float:
        """计算关键词匹配分数，要求至少匹配两个关键词
        
        Args:
            content: 文档内容
            keywords: 关键词列表
            
        Returns:
            关键词匹配分数 (0-1)，如果匹配关键词少于2个则返回0
        """
        if not keywords or not content or len(keywords) < 2:
            return 0.0
        
        content_lower = content.lower()
        matched_keywords = 0
        total_score = 0.0
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # 精确匹配得分更高
            if keyword_lower in content_lower:
                matched_keywords += 1
                # 计算关键词在内容中的出现次数
                count = content_lower.count(keyword_lower)
                # 基于出现次数和关键词长度计算分数
                score = min(count * len(keyword) / len(content), 1.0)
                total_score += score
        
        # 如果匹配的关键词少于2个，返回0
        if matched_keywords < 2:
            return 0.0
        
        # 归一化分数，并给予匹配更多关键词的文档更高分数
        normalized_score = min(total_score / len(keywords), 1.0)
        # 额外奖励匹配更多关键词的文档
        bonus = min(matched_keywords / len(keywords), 1.0) * 0.2
        
        return min(normalized_score + bonus, 1.0)
    
    def _fallback_keyword_search(self, query: str, k: int, query_keywords: List[str]) -> List[Tuple[Document, float]]:
        """回退的关键词搜索方法，基于jieba分词和文本匹配
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            query_keywords: 查询关键词列表
            
        Returns:
            (文档, 关键词匹配分数)元组列表
        """
        try:
            # 获取所有文档进行关键词匹配
            # 注意：这里使用大的k值获取更多候选文档
            all_docs = self.vector_store.similarity_search(query, k=k*3)
            
            # 计算关键词匹配分数，要求至少匹配两个关键词
            scored_docs = []
            for doc in all_docs:
                score = self._calculate_keyword_score_with_minimum_match(doc.page_content, query_keywords)
                if score > 0:
                    scored_docs.append((doc, score))
            
            # 按分数排序并返回前k个
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            result = scored_docs[:k]
            
            logger.debug(f"回退关键词搜索完成: 关键词={query_keywords}, 匹配文档={len(result)}")
            return result
            
        except Exception as e:
            logger.error(f"回退关键词搜索失败: {str(e)}")
            return []
    
    def _merge_results_with_native_scores(self, 
                                        vector_results: List[Tuple[Document, float]], 
                                        keyword_results: List[Tuple[Document, float]], 
                                        k: int) -> List[Tuple[Document, float]]:
        """合并向量搜索和关键词搜索结果，保留LangChain原生评分
        
        Args:
            vector_results: 向量搜索结果
            keyword_results: 关键词搜索结果
            k: 返回的文档数量
            
        Returns:
            合并后的(文档, 综合分数)列表
        """
        # 使用文档内容作为去重键
        doc_scores = {}
        
        # 处理向量搜索结果
        for doc, vector_score in vector_results:
            content_key = doc.page_content
            if content_key not in doc_scores:
                doc_scores[content_key] = {
                    'document': doc,
                    'vector_score': vector_score,
                    'keyword_score': 0.0
                }
        
        # 处理关键词搜索结果
        for doc, keyword_score in keyword_results:
            content_key = doc.page_content
            if content_key in doc_scores:
                # 更新已存在文档的关键词分数
                doc_scores[content_key]['keyword_score'] = keyword_score
            else:
                # 添加新文档（仅关键词匹配）
                doc_scores[content_key] = {
                    'document': doc,
                    'vector_score': 0.0,
                    'keyword_score': keyword_score
                }
        
        # 计算综合分数并排序
        final_results = []
        for content_key, scores in doc_scores.items():
            # 综合分数 = 向量分数 * 向量权重 + 关键词分数 * 关键词权重
            combined_score = (
                scores['vector_score'] * self.vector_weight + 
                scores['keyword_score'] * self.keyword_weight
            )
            
            # 添加原生分数到文档元数据
            doc = scores['document']
            if 'retrieval_scores' not in doc.metadata:
                doc.metadata['retrieval_scores'] = {}
            
            doc.metadata['retrieval_scores'].update({
                'vector_score': scores['vector_score'],
                'keyword_score': scores['keyword_score'],
                'combined_score': combined_score,
                'vector_weight': self.vector_weight,
                'keyword_weight': self.keyword_weight
            })
            
            final_results.append((doc, combined_score))
        
        # 按综合分数排序
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前k个结果
        return final_results[:k]
    
    def update_weights(self, vector_weight: float, keyword_weight: float):
        """更新搜索权重
        
        Args:
            vector_weight: 向量搜索权重
            keyword_weight: 关键词搜索权重
        """
        if abs(vector_weight + keyword_weight - 1.0) > 0.01:
            raise ValueError(f"权重之和必须等于1.0: {vector_weight} + {keyword_weight} = {vector_weight + keyword_weight}")
        
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
        logger.info(f"搜索权重已更新: vector={vector_weight}, keyword={keyword_weight}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return {
            'k': self.k,
            'vector_weight': self.vector_weight,
            'keyword_weight': self.keyword_weight,
            'min_score_threshold': self.min_score_threshold,
            'enable_keyword_boost': self.enable_keyword_boost,
            'vector_store_type': type(self.vector_store).__name__
        }