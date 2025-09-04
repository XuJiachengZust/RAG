"""查询预处理节点

负责清理和标准化用户查询，提取关键信息。
"""

import logging
from typing import Dict, Any
from ..state import SelfCorrectiveRAGState
from ..utils import clean_text, extract_keywords

logger = logging.getLogger(__name__)


def preprocess_query_node(state: SelfCorrectiveRAGState) -> Dict[str, Any]:
    """预处理用户查询节点
    
    功能：
    1. 清理和标准化查询文本
    2. 提取关键词
    3. 检测查询意图
    4. 设置处理参数
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态字典
    """
    try:
        query = state.get("query", "")
        
        if not query or not query.strip():
            return {
                "error_message": "查询不能为空",
                "final_answer": "请提供有效的查询内容。",
                "confidence_score": 0.0
            }
        
        # 1. 清理查询文本
        cleaned_query = clean_text(query)
        
        # 2. 提取关键词
        keywords = extract_keywords(cleaned_query, max_keywords=8)
        logging.info(f"预处理节点 - 提取的关键词: {keywords}")
        
        # 3. 检测查询类型和意图
        query_intent = detect_query_intent(cleaned_query)
        
        # 4. 设置查询复杂度
        complexity_level = assess_query_complexity(cleaned_query, keywords)
        
        # 5. 根据复杂度调整处理参数
        adjusted_params = adjust_processing_parameters(state, complexity_level)
        
        # 返回更新的状态
        updates = {
            "query": cleaned_query,
            "keywords": keywords,
            "query_intent": query_intent,
            "complexity_level": complexity_level,
            "processing_start_time": state.get("timestamp")
        }
        
        # 合并调整后的参数
        updates.update(adjusted_params)
        
        return updates
        
    except Exception as e:
        return {
            "error_message": f"查询预处理失败: {str(e)}",
            "final_answer": "查询预处理过程中发生错误，请重试。",
            "confidence_score": 0.0
        }


def detect_query_intent(query: str) -> str:
    """检测查询意图
    
    Args:
        query: 清理后的查询文本
        
    Returns:
        查询意图类型
    """
    query_lower = query.lower()
    
    # 定义意图关键词
    intent_patterns = {
        "definition": ["什么是", "定义", "含义", "概念", "what is", "define", "meaning"],
        "how_to": ["如何", "怎么", "怎样", "方法", "步骤", "how to", "how do", "steps"],
        "comparison": ["比较", "对比", "区别", "差异", "vs", "versus", "compare", "difference"],
        "best_practice": ["最佳实践", "最好的", "推荐", "建议", "best practice", "recommend", "suggest"],
        "troubleshooting": ["问题", "错误", "故障", "解决", "修复", "problem", "error", "issue", "fix"],
        "list": ["列出", "列表", "有哪些", "包括", "list", "what are", "include"],
        "explanation": ["解释", "说明", "为什么", "原因", "explain", "why", "reason"]
    }
    
    # 检测意图
    for intent, patterns in intent_patterns.items():
        if any(pattern in query_lower for pattern in patterns):
            return intent
    
    # 默认为一般查询
    return "general"


def assess_query_complexity(query: str, keywords: list) -> str:
    """评估查询复杂度
    
    Args:
        query: 查询文本
        keywords: 关键词列表
        
    Returns:
        复杂度级别: simple, medium, complex
    """
    # 计算复杂度指标
    word_count = len(query.split())
    keyword_count = len(keywords)
    
    # 检查是否包含复杂结构
    complex_indicators = [
        "并且", "或者", "但是", "然而", "因此", "所以", "如果", "当",
        "and", "or", "but", "however", "therefore", "if", "when", "while"
    ]
    
    has_complex_structure = any(indicator in query.lower() for indicator in complex_indicators)
    
    # 检查是否包含多个问题
    question_markers = ["?", "？", "吗", "呢"]
    question_count = sum(query.count(marker) for marker in question_markers)
    
    # 评估复杂度
    complexity_score = 0
    
    # 词数评分
    if word_count > 20:
        complexity_score += 2
    elif word_count > 10:
        complexity_score += 1
    
    # 关键词评分
    if keyword_count > 6:
        complexity_score += 2
    elif keyword_count > 3:
        complexity_score += 1
    
    # 结构复杂性评分
    if has_complex_structure:
        complexity_score += 1
    
    # 多问题评分
    if question_count > 1:
        complexity_score += 1
    
    # 确定复杂度级别
    if complexity_score >= 4:
        return "complex"
    elif complexity_score >= 2:
        return "medium"
    else:
        return "simple"


def adjust_processing_parameters(state: SelfCorrectiveRAGState, complexity_level: str) -> Dict[str, Any]:
    """根据查询复杂度调整处理参数
    
    Args:
        state: 当前状态
        complexity_level: 复杂度级别
        
    Returns:
        调整后的参数字典
    """
    base_params = {
        "max_retrieval_attempts": state.get("max_retrieval_attempts", 3),
        "max_generation_attempts": state.get("max_generation_attempts", 2),
        "quality_threshold": state.get("quality_threshold", 0.7)
    }
    
    # 从 state 中获取配置的 retrieval_k 基础值
    base_retrieval_k = state.get("retrieval_k", 5)
    logger.info(f"预处理节点 - 基础 retrieval_k: {base_retrieval_k}, 查询复杂度: {complexity_level}")
    
    if complexity_level == "complex":
        # 复杂查询：增加检索文档数量和重试次数
        adjusted_retrieval_k = min(base_retrieval_k + 2, 15)  # 增加2个文档，最大15
        logger.info(f"复杂查询 - 调整后 retrieval_k: {adjusted_retrieval_k}")
    elif complexity_level == "medium":
        # 中等查询：保持基础值
        adjusted_retrieval_k = base_retrieval_k
        logger.info(f"中等查询 - 保持 retrieval_k: {adjusted_retrieval_k}")
    else:  # simple
        # 简单查询：减少检索文档数量
        adjusted_retrieval_k = max(base_retrieval_k - 2, 1)  # 减少2个文档，最小1
        logger.info(f"简单查询 - 调整后 retrieval_k: {adjusted_retrieval_k}")
    
    # 根据复杂度调整参数并返回
    if complexity_level == "complex":
        # 复杂查询：增加尝试次数，降低质量阈值
        return {
            "max_retrieval_attempts": min(base_params["max_retrieval_attempts"] + 1, 5),
            "max_generation_attempts": min(base_params["max_generation_attempts"] + 1, 3),
            "quality_threshold": max(base_params["quality_threshold"] - 0.1, 0.5),
            "retrieval_k": adjusted_retrieval_k
        }
    elif complexity_level == "medium":
        # 中等复杂度：保持基础配置值
        return {
            "max_retrieval_attempts": base_params["max_retrieval_attempts"],
            "max_generation_attempts": base_params["max_generation_attempts"],
            "quality_threshold": base_params["quality_threshold"],
            "retrieval_k": adjusted_retrieval_k
        }
    else:
        # 简单查询：可以使用更严格的质量要求
        return {
            "max_retrieval_attempts": max(base_params["max_retrieval_attempts"] - 1, 2),
            "max_generation_attempts": base_params["max_generation_attempts"],
            "quality_threshold": min(base_params["quality_threshold"] + 0.1, 0.9),
            "retrieval_k": adjusted_retrieval_k
        }


def enhance_query_with_context(query: str, keywords: list, intent: str) -> str:
    """根据意图和关键词增强查询
    
    Args:
        query: 原始查询
        keywords: 关键词列表
        intent: 查询意图
        
    Returns:
        增强后的查询
    """
    # 根据意图添加上下文提示
    intent_enhancements = {
        "definition": "请提供详细的定义和解释",
        "how_to": "请提供具体的步骤和方法",
        "comparison": "请比较不同选项的优缺点",
        "best_practice": "请提供最佳实践和推荐方案",
        "troubleshooting": "请提供问题解决方案",
        "list": "请提供完整的列表和说明",
        "explanation": "请提供详细的解释和原因"
    }
    
    enhancement = intent_enhancements.get(intent, "")
    
    if enhancement:
        enhanced_query = f"{query}。{enhancement}。"
    else:
        enhanced_query = query
    
    # 如果关键词很重要，可以在查询中强调
    if len(keywords) > 0:
        key_terms = ", ".join(keywords[:3])  # 取前3个关键词
        enhanced_query += f" 重点关注: {key_terms}。"
    
    return enhanced_query


def validate_query_safety(query: str) -> tuple[bool, str]:
    """验证查询安全性
    
    Args:
        query: 查询文本
        
    Returns:
        (是否安全, 错误信息)
    """
    # 检查恶意内容
    malicious_patterns = [
        "<script", "javascript:", "eval(", "exec(",
        "rm -rf", "del /", "format c:",
        "DROP TABLE", "DELETE FROM", "UPDATE SET"
    ]
    
    query_lower = query.lower()
    
    for pattern in malicious_patterns:
        if pattern.lower() in query_lower:
            return False, f"查询包含潜在的恶意内容: {pattern}"
    
    # 检查查询长度
    if len(query) > 1000:
        return False, "查询过长，请缩短查询内容"
    
    return True, ""