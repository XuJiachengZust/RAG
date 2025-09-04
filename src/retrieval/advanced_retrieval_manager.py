"""高级检索管理器

基于LangChain原生能力的高级检索管理器，集成多种检索策略和智能评分机制。
"""

import logging
import json
from typing import Dict, Any, List, Optional, Union, Tuple
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain.retrievers import MultiQueryRetriever
from langchain_core.language_models import BaseLanguageModel

from .chroma_hybrid_retriever import ChromaHybridRetriever
from ..core.document_initializer import DocumentInitializer, get_document_initializer
from ..shared.text_utils import extract_keywords
from ..core.config_manager import get_config_manager

logger = logging.getLogger(__name__)


class AdvancedRetrievalManager:
    """高级检索管理器
    
    集成多种检索策略的高级检索管理器：
    - ChromaDB混合搜索（向量+关键词）
    - MultiQueryRetriever多视角检索
    - 智能查询重写
    - LangChain原生评分机制
    - 长上下文重排序
    
    特性：
    - 使用LangChain原生评分保证兼容性
    - 支持多种检索策略自动切换
    - 智能结果合并和去重
    - 可配置的检索参数
    - 性能监控和统计
    """
    
    def __init__(self, 
                 document_initializer: Optional[DocumentInitializer] = None,
                 llm: Optional[BaseLanguageModel] = None):
        """
        初始化高级检索管理器
        
        Args:
            document_initializer: 文档初始化器实例
            llm: 语言模型实例（用于MultiQueryRetriever）
        """
        self.document_initializer = document_initializer or get_document_initializer()
        self.llm = llm
        
        # 加载配置
        self.config = self._load_config()
        
        # 初始化检索器
        self.hybrid_retriever: Optional[ChromaHybridRetriever] = None
        self.multi_query_retriever: Optional[MultiQueryRetriever] = None
        
        # 统计信息
        self.stats = {
            'total_searches': 0,
            'hybrid_searches': 0,
            'multi_query_searches': 0,
            'vector_only_searches': 0,
            'keyword_only_searches': 0,
            'average_result_count': 0.0,
            'average_quality_score': 0.0
        }
        
        # 初始化检索器
        self._initialize_retrievers()
    
    def _load_config(self) -> Dict[str, Any]:
        """从配置中心加载配置"""
        try:
            config_manager = get_config_manager()
            # 获取高级检索配置，如果不存在则使用默认配置
            advanced_config = config_manager.get_section('advanced_retrieval')
            if advanced_config:
                logger.info("从配置中心加载高级检索配置")
                # 验证和补全配置
                validated_config = self._validate_and_merge_config(advanced_config)
                return validated_config
            else:
                logger.warning("配置中心未找到advanced_retrieval配置段，使用默认配置")
                return self._get_default_config()
        except Exception as e:
            logger.warning(f"从配置中心加载配置失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'hybrid_search': {
                'vector_weight': 0.7,
                'keyword_weight': 0.3,
                'min_score_threshold': 0.1,
                'max_results': 20,
                'enable_reranking': True,
                'enable_keyword_boost': True
            },
            'multi_query': {
                'enable': True,
                'num_queries': 3,
                'enable_query_rewrite': True,
                'max_query_length': 200,
                'temperature': 0.3,
                'max_docs_per_query': 10
            },
            'query_rewrite': {
                'enable_keyword_extraction': True,
                'enable_semantic_expansion': True,
                'max_keywords': 10,
                'similarity_threshold': 0.8
            },
            'reranking': {
                'enable': True,
                'enable_content_similarity': True,
                'enable_diversity_filter': True,
                'diversity_threshold': 0.85,
                'max_final_results': 10,
                'score_weights': {
                    'relevance': 0.4,
                    'content_similarity': 0.3,
                    'diversity': 0.3
                }
            },
            'fallback_strategy': {
                'enable_fallback': True,
                'min_results_threshold': 3,
                'enable_vector_fallback': True,
                'enable_keyword_fallback': True,
                'fallback_search_types': ['vector', 'keyword'],
                'expand_search_params': {
                    'increase_k_factor': 2.0,
                    'reduce_score_threshold': 0.5
                }
            },
            'performance': {
                'enable_monitoring': True,
                'cache_results': True,
                'cache_ttl_seconds': 300,
                'max_cache_size': 1000,
                'log_performance_metrics': True
            },
            'strategy_selection': {
                'auto_select': True,
                'query_length_thresholds': {
                    'short': 20,
                    'medium': 100,
                    'long': 200
                },
                'keyword_count_thresholds': {
                    'few': 3,
                    'many': 8
                },
                'default_strategy': 'hybrid'
            }
        }
    
    def _validate_and_merge_config(self, user_config: Dict[str, Any]) -> Dict[str, Any]:
        """验证用户配置并与默认配置合并
        
        Args:
            user_config: 用户提供的配置
            
        Returns:
            Dict[str, Any]: 验证并合并后的配置
        """
        default_config = self._get_default_config()
        
        # 深度合并配置
        merged_config = self._deep_merge_dict(default_config, user_config)
        
        # 验证配置值
        validated_config = self._validate_config_values(merged_config)
        
        return validated_config
    
    def _deep_merge_dict(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并字典
        
        Args:
            default: 默认配置
            user: 用户配置
            
        Returns:
            Dict[str, Any]: 合并后的配置
        """
        result = default.copy()
        
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge_dict(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """验证配置值的有效性
        
        Args:
            config: 待验证的配置
            
        Returns:
            Dict[str, Any]: 验证后的配置
        """
        try:
            # 验证hybrid_search配置
            if 'hybrid_search' in config:
                hs_config = config['hybrid_search']
                # 确保权重之和为1.0
                vector_weight = hs_config.get('vector_weight', 0.7)
                keyword_weight = hs_config.get('keyword_weight', 0.3)
                total_weight = vector_weight + keyword_weight
                if abs(total_weight - 1.0) > 0.01:  # 允许小的浮点误差
                    logger.warning(f"混合搜索权重之和不为1.0 ({total_weight})，自动调整")
                    hs_config['vector_weight'] = vector_weight / total_weight
                    hs_config['keyword_weight'] = keyword_weight / total_weight
                
                # 验证阈值范围
                if hs_config.get('min_score_threshold', 0) < 0:
                    hs_config['min_score_threshold'] = 0.0
                    logger.warning("min_score_threshold不能为负数，已设置为0.0")
                
                if hs_config.get('max_results', 20) <= 0:
                    hs_config['max_results'] = 20
                    logger.warning("max_results必须为正数，已设置为20")
            
            # 验证multi_query配置
            if 'multi_query' in config:
                mq_config = config['multi_query']
                if mq_config.get('num_queries', 3) <= 0:
                    mq_config['num_queries'] = 3
                    logger.warning("num_queries必须为正数，已设置为3")
                
                if mq_config.get('temperature', 0.3) < 0 or mq_config.get('temperature', 0.3) > 2.0:
                    mq_config['temperature'] = 0.3
                    logger.warning("temperature必须在0-2.0范围内，已设置为0.3")
            
            # 验证reranking配置
            if 'reranking' in config:
                rr_config = config['reranking']
                if 'score_weights' in rr_config:
                    weights = rr_config['score_weights']
                    total_weight = sum(weights.values())
                    if abs(total_weight - 1.0) > 0.01:
                        logger.warning(f"重排序权重之和不为1.0 ({total_weight})，自动调整")
                        for key in weights:
                            weights[key] = weights[key] / total_weight
                
                if rr_config.get('diversity_threshold', 0.85) < 0 or rr_config.get('diversity_threshold', 0.85) > 1.0:
                    rr_config['diversity_threshold'] = 0.85
                    logger.warning("diversity_threshold必须在0-1.0范围内，已设置为0.85")
            
            logger.info("配置验证完成")
            return config
            
        except Exception as e:
            logger.error(f"配置验证失败: {e}，使用默认配置")
            return self._get_default_config()
    
    def _initialize_retrievers(self):
        """初始化各种检索器"""
        try:
            # 确保向量库已初始化
            if not self.document_initializer.vector_store:
                logger.warning("向量库未初始化，某些检索功能可能不可用")
                return
            
            # 初始化混合检索器
            hybrid_config = self.config['hybrid_search']
            self.hybrid_retriever = ChromaHybridRetriever(
                vector_store=self.document_initializer.vector_store,
                vector_weight=hybrid_config['vector_weight'],
                keyword_weight=hybrid_config['keyword_weight'],
                min_score_threshold=hybrid_config['min_score_threshold'],
                enable_keyword_boost=hybrid_config['enable_keyword_boost']
            )
            
            # 初始化MultiQueryRetriever（如果有LLM）
            if self.llm and self.config['multi_query']['enable']:
                base_retriever = self.document_initializer.vector_store.as_retriever(
                    search_kwargs={'k': self.config['multi_query']['max_docs_per_query']}
                )
                self.multi_query_retriever = MultiQueryRetriever.from_llm(
                    retriever=base_retriever,
                    llm=self.llm
                )
            
            logger.info("检索器初始化完成")
            
        except Exception as e:
            logger.error(f"检索器初始化失败: {str(e)}")
    
    def intelligent_search(self, 
                         query: str, 
                         k: int = 5,
                         search_strategy: str = 'auto',
                         keywords: Optional[List[str]] = None,
                         enable_reranking: bool = True) -> Dict[str, Any]:
        """智能搜索：自动选择最佳检索策略
        
        Args:
            query: 查询文本
            k: 返回的文档数量
            search_strategy: 检索策略 ('auto', 'hybrid', 'multi_query', 'vector', 'keyword')
            keywords: 额外的关键词列表
            enable_reranking: 是否启用重排序
            
        Returns:
            包含文档、分数和元数据的详细结果
        """
        self.stats['total_searches'] += 1
        
        try:
            # 提取关键词（如果未提供）
            if not keywords:
                keywords = extract_keywords(query)
            
            # 自动选择检索策略
            if search_strategy == 'auto':
                search_strategy = self._select_optimal_strategy(query, keywords)
            
            # 执行检索
            documents, retrieval_metadata = self._execute_search(
                query, k, search_strategy, keywords
            )
            
            # 重排序（如果启用）
            if enable_reranking and self.config['reranking']['enable']:
                documents = self._rerank_documents(query, documents, keywords)
            
            # 计算质量分数
            quality_score = self._calculate_quality_score(query, documents)
            
            # 更新统计信息
            self._update_stats(len(documents), quality_score)
            
            # 构建结果
            result = {
                'documents': documents,
                'quality_score': quality_score,
                'query': query,
                'keywords': keywords,
                'search_strategy': search_strategy,
                'retrieval_metadata': retrieval_metadata,
                'reranking_enabled': enable_reranking,
                'total_results': len(documents),
                'search_id': self.stats['total_searches']
            }
            
            logger.info(f"智能搜索完成: 策略={search_strategy}, 结果数={len(documents)}, 质量分数={quality_score:.3f}")
            return result
            
        except Exception as e:
            logger.error(f"智能搜索失败: {str(e)}")
            # 返回空结果
            return {
                'documents': [],
                'quality_score': 0.0,
                'query': query,
                'keywords': keywords or [],
                'search_strategy': search_strategy,
                'error': str(e),
                'search_id': self.stats['total_searches']
            }
    
    def _select_optimal_strategy(self, query: str, keywords: List[str]) -> str:
        """自动选择最优检索策略"""
        # 基于查询特征选择策略
        query_length = len(query.split())
        keyword_count = len(keywords)
        
        # 复杂查询使用multi_query
        if (query_length > 10 or keyword_count > 5) and self.multi_query_retriever:
            return 'multi_query'
        
        # 有关键词的查询使用混合搜索
        if keyword_count > 0 and self.hybrid_retriever:
            return 'hybrid'
        
        # 默认使用向量搜索
        return 'vector'
    
    def _execute_search(self, 
                       query: str, 
                       k: int, 
                       strategy: str, 
                       keywords: List[str]) -> Tuple[List[Document], Dict[str, Any]]:
        """执行简化的检索策略：结合关键词查询和multi_query_retriever结果"""
        metadata = {'strategy': 'combined', 'fallback_used': False}
        all_documents = []
        
        # 检查向量库是否可用
        if not self.document_initializer.vector_store:
            logger.warning("向量库未初始化，无法执行检索")
            return [], metadata
        
        try:
            # 1. 使用 multi_query_retriever 获取结果
            if self.multi_query_retriever:
                self.stats['multi_query_searches'] += 1
                multi_query_docs = self.multi_query_retriever.get_relevant_documents(query)
                multi_query_docs = multi_query_docs[:k]  # 限制结果数量
                all_documents.extend(multi_query_docs)
                logger.debug(f"Multi-query检索获得 {len(multi_query_docs)} 个结果")

            # 2. 使用关键词查询获取结果
            if self.hybrid_retriever and keywords:
                self.stats['keyword_only_searches'] += 1
                keyword_results = self.hybrid_retriever.keyword_search(query, k, keywords)
                keyword_docs = [doc for doc, _ in keyword_results]
                all_documents.extend(keyword_docs)
                logger.debug(f"关键词检索获得 {len(keyword_docs)} 个结果")
            
            # 3. 如果没有专门的检索器，使用基础向量搜索
            if not all_documents and self.document_initializer.vector_store:
                self.stats['vector_only_searches'] += 1
                vector_docs = self.document_initializer.vector_store.similarity_search(query, k=k)
                all_documents.extend(vector_docs)
                logger.debug(f"基础向量检索获得 {len(vector_docs)} 个结果")
            
            # 4. 合并和去重文档
            documents = self._merge_and_deduplicate_documents(all_documents, k)
            logger.debug(f"合并去重后获得 {len(documents)} 个结果")
            
            # 5. 检查结果数量，如果太少则尝试回退策略
            if len(documents) < self.config['fallback_strategy']['min_results_threshold']:
                documents = self._apply_fallback_strategy(query, k, keywords, 'combined')
                metadata['fallback_used'] = True
            
            return documents, metadata
            
        except Exception as e:
            logger.error(f"简化检索策略执行失败: {str(e)}")
            # 回退到基础向量搜索
            try:
                if self.document_initializer.vector_store:
                    documents = self.document_initializer.vector_store.similarity_search(query, k=k)
                    metadata['fallback_used'] = True
                    metadata['error'] = str(e)
                    return documents, metadata
                else:
                    return [], metadata
            except Exception as fallback_error:
                logger.error(f"回退搜索也失败: {str(fallback_error)}")
                return [], metadata
    
    def _merge_and_deduplicate_documents(self, documents: List[Document], k: int) -> List[Document]:
        """合并和去重文档列表"""
        if not documents:
            return []
        
        # 使用文档内容的哈希值进行去重
        seen_hashes = set()
        unique_documents = []
        
        for doc in documents:
            # 创建文档的唯一标识（基于内容和来源）
            doc_hash = hash(doc.page_content + str(doc.metadata.get('source', '')))
            
            if doc_hash not in seen_hashes:
                seen_hashes.add(doc_hash)
                unique_documents.append(doc)
                
                # 限制结果数量
                if len(unique_documents) >= k:
                    break
        
        return unique_documents
    
    def _apply_fallback_strategy(self, 
                               query: str, 
                               k: int, 
                               keywords: List[str], 
                               original_strategy: str) -> List[Document]:
        """应用回退策略"""
        fallback_config = self.config['fallback_strategy']
        
        # 检查向量库是否可用
        if not self.document_initializer.vector_store:
            logger.warning("向量库未初始化，无法执行回退策略")
            return []
        
        # 尝试不同的回退策略
        if original_strategy != 'hybrid' and fallback_config['enable_vector_fallback']:
            try:
                if self.hybrid_retriever:
                    vector_results = self.hybrid_retriever.vector_search(query, k)
                    documents = [doc for doc, _ in vector_results]
                    if documents:
                        logger.info(f"回退到向量搜索成功，获得 {len(documents)} 个结果")
                        return documents
            except Exception as e:
                logger.warning(f"向量搜索回退失败: {str(e)}")
        
        if original_strategy != 'keyword' and fallback_config['enable_keyword_fallback']:
            try:
                if self.hybrid_retriever and keywords:
                    keyword_results = self.hybrid_retriever.keyword_search(query, k, keywords)
                    documents = [doc for doc, _ in keyword_results]
                    if documents:
                        logger.info(f"回退到关键词搜索成功，获得 {len(documents)} 个结果")
                        return documents
            except Exception as e:
                logger.warning(f"关键词搜索回退失败: {str(e)}")
        
        # 最后的回退：基础向量搜索
        try:
            if self.document_initializer.vector_store:
                documents = self.document_initializer.vector_store.similarity_search(query, k=k)
                logger.info(f"回退到基础向量搜索，获得 {len(documents)} 个结果")
                return documents
            else:
                logger.warning("向量库不可用，无法执行基础向量搜索")
                return []
        except Exception as e:
            logger.error(f"所有回退策略都失败: {str(e)}")
            return []
    
    def _rerank_documents(self, 
                         query: str, 
                         documents: List[Document], 
                         keywords: List[str]) -> List[Document]:
        """重排序文档"""
        if not documents:
            return documents
        
        try:
            rerank_config = self.config['reranking']
            
            # 基于多个因素重新排序
            scored_docs = []
            for doc in documents:
                score = self._calculate_rerank_score(query, doc, keywords)
                scored_docs.append((doc, score))
            
            # 按分数排序
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            # 应用多样性过滤
            if rerank_config['diversity_threshold'] > 0:
                scored_docs = self._apply_diversity_filter(scored_docs, rerank_config['diversity_threshold'])
            
            reranked_docs = [doc for doc, _ in scored_docs]
            
            logger.debug(f"文档重排序完成: 原始={len(documents)}, 重排序后={len(reranked_docs)}")
            return reranked_docs
            
        except Exception as e:
            logger.warning(f"文档重排序失败，返回原始结果: {str(e)}")
            return documents
    
    def _calculate_rerank_score(self, query: str, doc: Document, keywords: List[str]) -> float:
        """计算重排序分数"""
        score = 0.0
        
        # 基础分数：来自检索器的原始分数
        if 'retrieval_scores' in doc.metadata:
            retrieval_scores = doc.metadata['retrieval_scores']
            score += retrieval_scores.get('combined_score', 0.0) * 0.5
        
        # 查询相关性分数
        query_lower = query.lower()
        content_lower = doc.page_content.lower()
        
        # 查询词匹配
        query_words = query_lower.split()
        matched_words = sum(1 for word in query_words if word in content_lower)
        query_match_score = matched_words / len(query_words) if query_words else 0
        score += query_match_score * 0.3
        
        # 关键词匹配
        if keywords:
            keyword_matches = sum(1 for kw in keywords if kw.lower() in content_lower)
            keyword_score = keyword_matches / len(keywords)
            score += keyword_score * 0.2
        
        return score
    
    def _apply_diversity_filter(self, 
                              scored_docs: List[Tuple[Document, float]], 
                              threshold: float) -> List[Tuple[Document, float]]:
        """应用多样性过滤"""
        if not scored_docs:
            return scored_docs
        
        filtered_docs = [scored_docs[0]]  # 保留最高分的文档
        
        for doc, score in scored_docs[1:]:
            # 检查与已选文档的相似性
            is_diverse = True
            for selected_doc, _ in filtered_docs:
                similarity = self._calculate_content_similarity(doc.page_content, selected_doc.page_content)
                if similarity > threshold:
                    is_diverse = False
                    break
            
            if is_diverse:
                filtered_docs.append((doc, score))
        
        return filtered_docs
    
    def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """计算内容相似性（简单实现）"""
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_quality_score(self, query: str, documents: List[Document]) -> float:
        """计算检索质量分数"""
        if not documents:
            return 0.0
        
        # 基础分数：基于文档数量
        base_score = min(len(documents) / 5.0, 1.0)
        
        # 内容质量分数
        total_length = sum(len(doc.page_content) for doc in documents)
        avg_length = total_length / len(documents)
        
        # 理想长度为200-1000字符
        length_score = 1.0
        if avg_length < 100:
            length_score = avg_length / 100.0
        elif avg_length > 2000:
            length_score = max(0.5, 2000.0 / avg_length)
        
        # 多样性分数
        sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
        diversity_score = min(len(sources) / len(documents), 1.0)
        
        # 综合分数
        quality_score = base_score * 0.3 + length_score * 0.4 + diversity_score * 0.3
        return min(quality_score, 1.0)
    
    def _update_stats(self, result_count: int, quality_score: float):
        """更新统计信息"""
        total = self.stats['total_searches']
        
        # 更新平均结果数量
        current_avg_count = self.stats['average_result_count']
        new_avg_count = ((current_avg_count * (total - 1)) + result_count) / total
        self.stats['average_result_count'] = new_avg_count
        
        # 更新平均质量分数
        current_avg_quality = self.stats['average_quality_score']
        new_avg_quality = ((current_avg_quality * (total - 1)) + quality_score) / total
        self.stats['average_quality_score'] = new_avg_quality
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            **self.stats,
            'hybrid_retriever_available': self.hybrid_retriever is not None,
            'multi_query_retriever_available': self.multi_query_retriever is not None,
            'config': self.config
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(new_config)
        # 重新初始化检索器
        self._initialize_retrievers()
        logger.info("配置已更新，检索器已重新初始化")
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'total_searches': 0,
            'hybrid_searches': 0,
            'multi_query_searches': 0,
            'vector_only_searches': 0,
            'keyword_only_searches': 0,
            'average_result_count': 0.0,
            'average_quality_score': 0.0
        }
        logger.info("统计信息已重置")


# 全局高级检索管理器实例
_global_advanced_retrieval_manager: Optional[AdvancedRetrievalManager] = None


def get_advanced_retrieval_manager() -> AdvancedRetrievalManager:
    """获取全局高级检索管理器实例
    
    Returns:
        AdvancedRetrievalManager: 高级检索管理器实例
    """
    global _global_advanced_retrieval_manager
    
    if _global_advanced_retrieval_manager is None:
        _global_advanced_retrieval_manager = AdvancedRetrievalManager()
        logger.info("创建全局高级检索管理器实例")
    
    return _global_advanced_retrieval_manager


def reset_advanced_retrieval_manager():
    """重置全局高级检索管理器实例"""
    global _global_advanced_retrieval_manager
    _global_advanced_retrieval_manager = None
    logger.info("全局高级检索管理器实例已重置")