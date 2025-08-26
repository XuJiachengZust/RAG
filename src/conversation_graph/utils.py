"""自我纠错RAG系统的工具函数

包含检索器创建、质量评估、文本处理等辅助功能。
"""

import os
import re
import time
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.core.config_manager import ConfigManager
from .document_initializer import get_document_initializer, get_rules_retriever
from .retrieval_manager import get_retrieval_manager


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
        
        # 使用RetrievalManager获取检索器
        retrieval_manager = get_retrieval_manager(initializer)
        return retrieval_manager.get_retriever(k)
        
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
        
        # 使用RetrievalManager获取高级检索器
        retrieval_manager = get_retrieval_manager(initializer)
        return retrieval_manager.get_retriever(k, search_type, search_kwargs)
        
    except Exception as e:
        print(f"创建高级检索器失败: {str(e)}")
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
        
        # 使用RetrievalManager进行检索
        retrieval_manager = get_retrieval_manager(initializer)
        return retrieval_manager.retrieve_documents(
            query=query,
            k=k,
            quality_threshold=quality_threshold
        )
        
    except Exception as e:
        print(f"文档检索失败: {str(e)}")
        return {
            'documents': [],
            'quality_score': 0.0,
            'query': query,
            'error': str(e)
        }


def evaluate_answer_quality(query: str, context: str, answer: str) -> float:
    """评估答案质量
    
    基于答案长度、相关性、基于上下文程度等因素评估答案质量。
    
    Args:
        query: 用户查询
        context: 上下文信息
        answer: 生成的答案
        
    Returns:
        质量分数 (0-1)
    """
    if not answer or not answer.strip():
        return 0.0
    
    # 1. 答案长度评分 (理想长度为50-300词)
    answer_length = len(answer.split())
    if 50 <= answer_length <= 300:
        length_score = 1.0
    elif answer_length < 50:
        length_score = answer_length / 50
    else:
        length_score = max(0.6, 300 / answer_length)
    
    # 2. 查询相关性评分
    query_terms = set(clean_text(query).lower().split())
    answer_terms = set(clean_text(answer).lower().split())
    
    if query_terms:
        relevance_score = len(query_terms.intersection(answer_terms)) / len(query_terms)
    else:
        relevance_score = 0.0
    
    # 3. 上下文基础评分
    if context and context.strip():
        context_terms = set(clean_text(context).lower().split())
        if context_terms and answer_terms:
            context_score = len(answer_terms.intersection(context_terms)) / len(answer_terms)
        else:
            context_score = 0.0
    else:
        context_score = 0.0
    
    # 4. 答案完整性评分 (检查是否包含常见的结束词)
    completeness_indicators = ['总结', '综上', '因此', '所以', '总的来说', '最后', '结论']
    has_conclusion = any(indicator in answer for indicator in completeness_indicators)
    completeness_score = 1.0 if has_conclusion else 0.7
    
    # 5. 综合评分
    # 权重分配：长度20%，相关性30%，上下文30%，完整性20%
    final_score = (
        length_score * 0.2 +
        relevance_score * 0.3 +
        context_score * 0.3 +
        completeness_score * 0.2
    )
    
    return min(final_score, 1.0)


def clean_text(text: str) -> str:
    """清理文本，移除特殊字符和多余空格
    
    Args:
        text: 原始文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除特殊字符，保留中英文、数字和基本标点
    cleaned = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()\[\]{}"\'-]', ' ', text)
    
    # 移除多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    return cleaned.strip()


def extract_keywords_simple(text: str, max_keywords: int = 10) -> List[str]:
    """简单的关键词提取（基于词频统计）
    
    Args:
        text: 输入文本
        max_keywords: 最大关键词数量
        
    Returns:
        关键词列表
    """
    if not text:
        return []
    
    # 简单的关键词提取：移除停用词，按词频排序
    stop_words = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
        '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
        'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had'
    }
    
    # 清理文本并分词
    cleaned = clean_text(text).lower()
    words = cleaned.split()
    
    # 过滤停用词和短词
    keywords = [word for word in words if len(word) > 2 and word not in stop_words]
    
    # 统计词频
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # 按频率排序并返回前N个
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    
    return [word for word, freq in sorted_keywords[:max_keywords]]


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """智能关键词提取（基于LLM）
    
    使用LLM进行智能关键词提取，当LLM不可用时自动回退到简单实现。
    
    Args:
        text: 输入文本
        max_keywords: 最大关键词数量
        
    Returns:
        关键词列表
    """
    if not text or not text.strip():
        return []
    
    try:
        # 尝试使用LLM进行关键词提取
        llm_client = get_llm_client()
        if llm_client:
            keywords = _extract_keywords_with_llm(text, max_keywords, llm_client)
            if keywords:  # 如果LLM成功返回关键词
                return keywords
    except Exception as e:
        print(f"LLM关键词提取失败，回退到简单实现: {e}")
    
    # 回退到简单实现
    return extract_keywords_simple(text, max_keywords)


def _extract_keywords_with_llm(text: str, max_keywords: int, llm_client) -> List[str]:
    """使用LLM提取关键词的内部函数
    
    Args:
        text: 输入文本
        max_keywords: 最大关键词数量
        llm_client: LLM客户端
        
    Returns:
        关键词列表，失败时返回空列表
    """
    try:
        # 构建提示词
        prompt = f"""请从以下文本中提取最重要的关键词。要求：
1. 提取最多{max_keywords}个关键词
2. 关键词应该是名词、动词或重要的形容词
3. 优先选择专业术语、核心概念和重要实体
4. 忽略常见的停用词（如：的、了、在、是等）
5. 按重要性排序
6. 只返回关键词列表，每行一个，不要其他解释

文本内容：
{text[:1000]}  # 限制文本长度避免token过多

关键词："""
        
        # 调用LLM
        from langchain_core.messages import HumanMessage
        response = llm_client.invoke([HumanMessage(content=prompt)])
        
        if response and hasattr(response, 'content'):
            # 解析LLM返回的关键词
            keywords = _parse_keywords_from_response(response.content, max_keywords)
            return keywords
        
    except Exception as e:
        print(f"LLM关键词提取过程中发生错误: {e}")
    
    return []


def _parse_keywords_from_response(response_text: str, max_keywords: int) -> List[str]:
    """解析LLM响应中的关键词
    
    Args:
        response_text: LLM的响应文本
        max_keywords: 最大关键词数量
        
    Returns:
        解析出的关键词列表
    """
    if not response_text:
        return []
    
    try:
        # 按行分割并清理
        lines = response_text.strip().split('\n')
        keywords = []
        
        for line in lines:
            # 清理每行，移除序号、标点等
            cleaned_line = re.sub(r'^\d+[.、]\s*', '', line.strip())
            cleaned_line = re.sub(r'[^\w\u4e00-\u9fff\s]', '', cleaned_line)
            cleaned_line = cleaned_line.strip()
            
            # 如果是有效的关键词
            if cleaned_line and len(cleaned_line) > 1:
                # 可能包含多个词，用空格或逗号分割
                words = re.split(r'[,，\s]+', cleaned_line)
                for word in words:
                    word = word.strip()
                    if word and len(word) > 1 and word not in keywords:
                        keywords.append(word)
                        if len(keywords) >= max_keywords:
                            break
            
            if len(keywords) >= max_keywords:
                break
        
        return keywords[:max_keywords]
        
    except Exception as e:
        print(f"解析关键词响应时发生错误: {e}")
        return []


def parse_relevance_score(response_text: str) -> float:
    """解析LLM返回的相关性评分
    
    Args:
        response_text: LLM的响应文本
        
    Returns:
        相关性分数 (0-1)
    """
    try:
        # 查找数字模式
        import re
        
        # 查找0-1之间的小数
        decimal_match = re.search(r'0\.[0-9]+', response_text)
        if decimal_match:
            return float(decimal_match.group())
        
        # 查找百分比
        percent_match = re.search(r'([0-9]+)%', response_text)
        if percent_match:
            return float(percent_match.group(1)) / 100
        
        # 查找"是"或"否"
        if '是' in response_text or 'yes' in response_text.lower():
            return 0.8
        elif '否' in response_text or 'no' in response_text.lower():
            return 0.2
        
        # 默认中等相关性
        return 0.5
        
    except Exception:
        return 0.5


def parse_validation_result(response_text: str) -> Dict[str, Any]:
    """解析答案验证结果
    
    Args:
        response_text: LLM的验证响应
        
    Returns:
        包含分数、是否需要纠错、反馈的字典
    """
    try:
        import re
        
        result = {
            "score": 0.5,
            "needs_correction": True,
            "feedback": response_text
        }
        
        # 解析分数
        decimal_match = re.search(r'0\.[0-9]+', response_text)
        if decimal_match:
            result["score"] = float(decimal_match.group())
        
        # 判断是否需要纠错
        if result["score"] >= 0.7:
            result["needs_correction"] = False
        elif '不需要' in response_text or 'no need' in response_text.lower():
            result["needs_correction"] = False
        
        return result
        
    except Exception:
        return {
            "score": 0.5,
            "needs_correction": True,
            "feedback": "解析验证结果时发生错误"
        }


def format_documents_for_context(documents: List[Document], max_length: int = 2000) -> str:
    """格式化文档为上下文字符串
    
    Args:
        documents: 文档列表
        max_length: 最大长度限制
        
    Returns:
        格式化的上下文字符串
    """
    if not documents:
        return ""
    
    context_parts = []
    current_length = 0
    
    for i, doc in enumerate(documents):
        doc_text = doc.page_content.strip()
        
        # 添加文档标识
        doc_header = f"\n=== 文档 {i+1} ===\n"
        doc_content = doc_header + doc_text
        
        # 检查长度限制
        if current_length + len(doc_content) > max_length:
            # 截断最后一个文档
            remaining_length = max_length - current_length - len(doc_header)
            if remaining_length > 100:  # 至少保留100字符
                truncated_content = doc_header + doc_text[:remaining_length] + "..."
                context_parts.append(truncated_content)
            break
        
        context_parts.append(doc_content)
        current_length += len(doc_content)
    
    return "\n".join(context_parts)


def get_llm_client(model_name: str = None, node_type: str = None):
    """获取LLM客户端
    
    Args:
        model_name: 模型名称（可选，如果提供则直接使用）
        node_type: 节点类型（generation, validation, correction, rewrite, grading）
        
    Returns:
        配置好的LLM客户端
    """
    try:
        from langchain_openai import ChatOpenAI
        
        config = ConfigManager().get_config()
        
        # 如果没有指定模型名称，根据节点类型从配置中获取
        if model_name is None:
            if node_type:
                node_models = config.get("node_models", {})
                model_name = node_models.get(f"{node_type}_model", "gpt-3.5-turbo")
            else:
                # 默认使用配置中的通用模型
                model_name = config.get("model", {}).get("model_name", "gpt-3.5-turbo")
        
        return ChatOpenAI(
            openai_api_key=config.get("api", {}).get("openai_api_key"),
            model=model_name,
            temperature=config.get("model_params", {}).get("temperature", 0.1),
            max_tokens=config.get("model_params", {}).get("max_tokens", 1000)
        )
        
    except Exception as e:
        print(f"创建LLM客户端时发生错误: {e}")
        return None