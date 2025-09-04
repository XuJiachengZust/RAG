"""检索器工厂模块

负责创建和管理各种类型的检索器，避免循环导入问题。
这个模块作为基础设施组件，不依赖于conversation_graph模块。
"""

import os
from typing import Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from .config_manager import ConfigManager
from .document_initializer import get_document_initializer, get_rules_retriever


def create_rules_retriever(rules_directory: str = "./rules", k: int = 5):
    """
    创建一个用于检索rules文件的retriever
    使用RetrievalManager来管理检索逻辑，与初始化逻辑分离
    
    Args:
        rules_directory: rules文件目录路径
        k: 返回的文档数量
    
    Returns:
        配置好的retriever
    """
    try:
        # 获取或创建文档初始化器（只在需要时初始化）
        config = {
            'rules_directory': rules_directory,
            'vector_store_path': './chroma_db_rules'
        }
        
        initializer = get_document_initializer(config)
        
        # 检查是否需要初始化（避免重复初始化）
        if not initializer.vector_store or initializer._should_reload():
            print("正在初始化文档库...")
            stats = initializer.initialize()
            
            if not stats.get('success', True) and stats.get('errors'):
                print(f"文档初始化警告: {len(stats['errors'])} 个错误")
                for error in stats['errors'][:3]:  # 只显示前3个错误
                    print(f"  - {error}")
        else:
            print("文档库已初始化，跳过重复初始化")
        
        # 尝试使用高级检索管理器
        try:
            from ..retrieval import get_advanced_retrieval_manager
            advanced_manager = get_advanced_retrieval_manager()
            return advanced_manager.hybrid_retriever
        except ImportError:
            # 回退到基础检索器
            return get_rules_retriever(k)
        
    except Exception as e:
        print(f"创建rules retriever失败: {str(e)}")
        # 回退到快捷方式
        try:
            return get_rules_retriever(k)
        except Exception:
            return None


def create_advanced_retriever(rules_directory: str = "./rules", 
                             search_type: str = "similarity",
                             k: int = 5,
                             search_kwargs: Dict[str, Any] = None):
    """
    创建高级检索器，支持多种检索策略
    
    注意：此函数使用旧的检索管理器，推荐使用 create_next_gen_retriever() 获得更强大的功能。
    
    Args:
        rules_directory: rules文件目录路径
        search_type: 检索类型 (similarity, mmr, similarity_score_threshold)
        k: 返回的文档数量
        search_kwargs: 额外的检索参数
    
    Returns:
        配置好的高级检索器
    """
    try:
        # 获取或创建文档初始化器
        config = {
            'rules_directory': rules_directory,
            'vector_store_path': './chroma_db_rules'
        }
        
        initializer = get_document_initializer(config)
        
        # 检查是否需要初始化
        if not initializer.vector_store or initializer._should_reload():
            print("正在初始化文档库...")
            stats = initializer.initialize()
            
            if not stats.get('success', True) and stats.get('errors'):
                print(f"文档初始化警告: {len(stats['errors'])} 个错误")
        else:
            print("文档库已初始化，跳过重复初始化")
        
        # 尝试使用高级检索管理器
        try:
            from ..retrieval import get_advanced_retrieval_manager
            advanced_manager = get_advanced_retrieval_manager()
            
            # 根据检索类型返回相应的检索器
            if search_type == "mmr":
                return advanced_manager.hybrid_retriever  # 混合检索包含MMR功能
            elif search_type == "similarity_score_threshold":
                return advanced_manager.hybrid_retriever  # 使用混合检索的评分机制
            else:
                return advanced_manager.hybrid_retriever  # 默认使用混合检索
        except ImportError:
            # 回退到基础检索器
            return get_rules_retriever(k)
        
    except Exception as e:
        print(f"创建高级检索器失败: {str(e)}")
        return None


def create_next_gen_retriever(rules_directory: str = "./rules",
                             strategy: str = "hybrid",
                             k: int = 5,
                             **kwargs):
    """
    创建下一代高级检索器，使用新的 AdvancedRetrievalManager
    
    提供更强大的检索功能：
    - ChromaDB混合搜索（向量+关键词）
    - MultiQueryRetriever多视角检索
    - LangChain原生评分机制
    - 智能查询重写和结果重排序
    
    Args:
        rules_directory: rules文件目录路径
        strategy: 检索策略 ("hybrid", "vector", "keyword", "multi_query")
        k: 返回的文档数量
        **kwargs: 其他检索参数
    
    Returns:
        配置好的下一代检索器
    
    示例:
        # 使用混合搜索
        retriever = create_next_gen_retriever(strategy="hybrid", k=5)
        
        # 使用多视角检索
        retriever = create_next_gen_retriever(strategy="multi_query", k=10)
    """
    try:
        # 获取或创建文档初始化器
        config = {
            'rules_directory': rules_directory,
            'vector_store_path': './chroma_db_rules'
        }
        
        initializer = get_document_initializer(config)
        
        # 检查是否需要初始化
        if not initializer.vector_store or initializer._should_reload():
            print("正在初始化文档库...")
            stats = initializer.initialize()
            
            if not stats.get('success', True) and stats.get('errors'):
                print(f"文档初始化警告: {len(stats['errors'])} 个错误")
        else:
            print("文档库已初始化，跳过重复初始化")
        
        # 尝试使用高级检索管理器
        try:
            from ..retrieval import get_advanced_retrieval_manager
            advanced_manager = get_advanced_retrieval_manager()
            
            # 根据策略返回不同的检索器
            if strategy == "hybrid":
                return advanced_manager.get_hybrid_retriever(k=k, **kwargs)
            elif strategy == "multi_query":
                return advanced_manager.get_multi_query_retriever(k=k, **kwargs)
            elif strategy == "vector":
                return advanced_manager.get_vector_retriever(k=k, **kwargs)
            elif strategy == "keyword":
                return advanced_manager.get_keyword_retriever(k=k, **kwargs)
            else:
                # 默认使用混合检索
                return advanced_manager.get_hybrid_retriever(k=k, **kwargs)
        except ImportError:
            # 回退到旧版本
            return create_advanced_retriever(rules_directory, "similarity", k, kwargs)
        
    except ImportError:
        print("新的高级检索管理器尚未可用，回退到旧版本")
        return create_advanced_retriever(rules_directory, "similarity", k, kwargs)
    except Exception as e:
        print(f"创建下一代检索器失败: {str(e)}")
        return None


def retrieve_documents_with_quality(query: str,
                                   rules_directory: str = "./rules",
                                   k: int = 5,
                                   quality_threshold: float = 0.0) -> Dict[str, Any]:
    """
    检索文档并返回质量评估结果
    
    Args:
        query: 查询文本
        rules_directory: rules文件目录路径
        k: 返回的文档数量
        quality_threshold: 质量阈值
    
    Returns:
        包含文档、质量分数和元数据的字典
    """
    try:
        # 获取或创建文档初始化器
        config = {
            'rules_directory': rules_directory,
            'vector_store_path': './chroma_db_rules'
        }
        
        initializer = get_document_initializer(config)
        
        # 检查是否需要初始化
        if not initializer.vector_store or initializer._should_reload():
            print("正在初始化文档库...")
            initializer.initialize()
        
        # 使用新的高级检索管理器进行检索
        from ..retrieval import get_advanced_retrieval_manager
        advanced_manager = get_advanced_retrieval_manager()
        search_result = advanced_manager.intelligent_search(
            query=query,
            k=k
        )
        
        # 过滤低质量文档
        filtered_docs = []
        for doc in search_result['documents']:
            if search_result['quality_score'] >= quality_threshold:
                filtered_docs.append(doc)
        
        return {
            'documents': filtered_docs,
            'quality_score': search_result['quality_score'],
            'query': query,
            'strategy': search_result.get('strategy', 'unknown')
        }
        
    except Exception as e:
        print(f"文档检索失败: {str(e)}")
        return {
            'documents': [],
            'quality_score': 0.0,
            'query': query,
            'error': str(e)
        }