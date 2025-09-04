"""自我纠错RAG系统的工具函数

包含质量评估、文本处理等辅助功能。
检索器创建功能已移动到 src.core.retriever_factory 模块。
"""

import os
import re
import time
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from src.core.config_manager import ConfigManager
# 导入独立的文本处理工具
from src.shared.text_utils import clean_text


# 检索器创建函数已移动到 src.core.retriever_factory 模块
# 请使用以下导入：
# from src.core.retriever_factory import create_rules_retriever, create_advanced_retriever, create_next_gen_retriever








def retrieve_documents_with_quality(query: str,
                                    rules_directory: str = "./rules",
                                    k: int = 5,
                                    quality_threshold: float = 0.7,
                                    use_advanced: bool = True) -> Dict[str, Any]:
    """
    检索文档并进行质量评估
    
    Args:
        query: 查询字符串
        rules_directory: rules文件目录路径
        k: 返回的文档数量
        quality_threshold: 质量阈值
        use_advanced: 是否使用高级检索器
    
    Returns:
        包含文档和质量评估的字典
    """
    try:
        # 使用新的检索器工厂
        from src.core.retriever_factory import create_next_gen_retriever
        from src.core.document_initializer import get_document_initializer
        
        # 获取文档初始化器
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
        
        # 使用高级检索管理器
        try:
            from src.retrieval import get_advanced_retrieval_manager
            advanced_manager = get_advanced_retrieval_manager()
            
            # 执行检索
            result = advanced_manager.intelligent_search(
                query=query,
                k=k,
                quality_threshold=quality_threshold
            )
            documents = result.get('documents', [])
            
            # 评估整体质量
            overall_quality = evaluate_documents_quality(query, documents)
            
            return {
                'documents': documents,
                'query': query,
                'quality_score': overall_quality,
                'retrieval_stats': {
                    'total_retrieved': len(documents),
                    'quality_threshold': quality_threshold,
                    'use_advanced': use_advanced
                }
            }
        except ImportError:
            # 回退到基础检索
            retriever = create_next_gen_retriever(rules_directory, k=k)
            if retriever:
                documents = retriever.get_relevant_documents(query)
            else:
                documents = []
            
            # 简单质量评估
            quality_score = sum(1 for doc in documents if len(doc.page_content) > 50) / len(documents) if documents else 0
            
            return {
                'documents': documents,
                'query': query,
                'quality_score': quality_score,
                'retrieval_stats': {
                    'total_retrieved': len(documents),
                    'quality_threshold': quality_threshold,
                    'use_advanced': False
                }
            }
        
    except Exception as e:
        print(f"文档检索失败: {str(e)}")
        return {
            'documents': [],
            'query': query,
            'quality_score': 0.0,
            'retrieval_stats': {
                'total_retrieved': 0,
                'quality_threshold': quality_threshold,
                'use_advanced': use_advanced,
                'error': str(e)
            }
        }


def evaluate_documents_quality(query: str, documents: list) -> float:
    """评估检索文档质量
    
    基于文档相关性、内容长度、多样性等因素评估检索文档的整体质量。
    
    Args:
        query: 用户查询
        documents: 检索到的文档列表
        
    Returns:
        质量分数 (0-1)
    """
    if not documents:
        return 0.0
    
    total_score = 0.0
    
    for doc in documents:
        doc_content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
        
        # 1. 文档长度评分 (理想长度为100-1000字符)
        content_length = len(doc_content)
        if 100 <= content_length <= 1000:
            length_score = 1.0
        elif content_length < 100:
            length_score = content_length / 100
        else:
            length_score = max(0.6, 1000 / content_length)
        
        # 2. 查询相关性评分
        try:
            query_terms = set(clean_text(query).lower().split())
            doc_terms = set(clean_text(doc_content).lower().split())
            
            if query_terms and doc_terms:
                relevance_score = len(query_terms.intersection(doc_terms)) / len(query_terms)
            else:
                relevance_score = 0.0
        except:
            relevance_score = 0.0
        
        # 3. 内容质量评分 (基于内容丰富度)
        if doc_content and doc_content.strip():
            quality_score = min(1.0, len(doc_content.split()) / 50)  # 50词为满分
        else:
            quality_score = 0.0
        
        # 综合评分：长度30%，相关性50%，质量20%
        doc_score = (
            length_score * 0.3 +
            relevance_score * 0.5 +
            quality_score * 0.2
        )
        
        total_score += doc_score
    
    # 返回平均分数
    return min(total_score / len(documents), 1.0)


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


# 注意：clean_text 函数已移动到 src.shared.text_utils 模块
# 这里保留一个简化版本以保持向后兼容性
def clean_text_legacy(text: str) -> str:
    """清理文本，移除特殊字符和多余空格（向后兼容版本）
    
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
        prompt = f"""请从以下文本中提取最重要功能关键词要求：
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