"""路由决策节点

负责在LangGraph中进行条件路由决策。
"""

import logging
from typing import Dict, Any, Literal

from ..state import SelfCorrectiveRAGState

logger = logging.getLogger(__name__)


def should_retrieve_documents(state: SelfCorrectiveRAGState) -> Literal["retrieve", "generate_direct"]:
    """决定是否需要检索文档
    
    Args:
        state: 当前状态
        
    Returns:
        路由决策："retrieve" 或 "generate_direct"
    """
    query = state.get("query", "")
    preprocessed_query = state.get("preprocessed_query", "")
    query_intent = state.get("query_intent", "general")
    
    # 检查查询是否为空
    if not query and not preprocessed_query:
        return "generate_direct"
    
    # 检查是否是简单的问候或感谢
    simple_patterns = [
        "你好", "hello", "hi", "谢谢", "thank you", "再见", "bye",
        "没事", "算了", "不用了", "取消"
    ]
    
    query_lower = query.lower()
    if any(pattern in query_lower for pattern in simple_patterns):
        return "generate_direct"
    
    # 检查查询长度
    if len(query.strip()) < 3:
        return "generate_direct"
    
    # 对于需要文档支持的意图，必须检索
    document_required_intents = [
        "definition", "how_to", "best_practice", "troubleshooting", 
        "explanation", "comparison", "list"
    ]
    
    if query_intent in document_required_intents:
        return "retrieve"
    
    # 默认进行检索
    return "retrieve"


def should_rewrite_query(state: SelfCorrectiveRAGState) -> Literal["rewrite", "generate"]:
    """决定是否需要重写查询
    
    Args:
        state: 当前状态
        
    Returns:
        路由决策："rewrite" 或 "generate"
    """
    graded_documents = state.get("graded_documents", [])
    retrieval_attempts = state.get("retrieval_attempts", 0)
    max_retrieval_attempts = state.get("max_retrieval_attempts", 3)
    retrieval_score = state.get("retrieval_score", 0.0)
    quality_threshold = state.get("quality_threshold", 0.7)
    
    # 打印调试信息
    logger.info(f"should_rewrite_query - 状态参数: graded_documents={len(graded_documents)}, "
                f"retrieval_attempts={retrieval_attempts}, max_retrieval_attempts={max_retrieval_attempts}, "
                f"retrieval_score={retrieval_score}, quality_threshold={quality_threshold}")
    
    # 检查是否已达到最大尝试次数
    if retrieval_attempts >= max_retrieval_attempts:
        return "generate"
    
    # 检查是否有相关文档
    relevant_docs = [item for item in graded_documents if item.get("is_relevant", False)]
    
    # 如果没有相关文档，需要重写查询
    if not relevant_docs:
        return "rewrite"
    
    # 如果检索质量低于阈值，需要重写查询
    if retrieval_score < quality_threshold:
        return "rewrite"
    
    # 如果相关文档数量太少，可能需要重写
    if len(relevant_docs) < 2 and retrieval_attempts < 2:
        return "rewrite"
    
    # 否则继续生成答案
    return "generate"


def should_validate_answer(state: SelfCorrectiveRAGState) -> Literal["validate", "finalize"]:
    """决定是否需要验证答案
    
    Args:
        state: 当前状态
        
    Returns:
        路由决策："validate" 或 "finalize"
    """
    generated_answer = state.get("generated_answer", "")
    answer_quality_score = state.get("answer_quality_score", 0.0)
    complexity_level = state.get("complexity_level", "medium")
    query_intent = state.get("query_intent", "general")
    
    # 如果没有生成答案，跳过验证
    if not generated_answer:
        return "finalize"
    
    # 如果答案质量已经很高，可能跳过验证
    if answer_quality_score >= 0.9:
        return "finalize"
    
    # 对于简单查询，可能跳过验证
    if complexity_level == "simple" and len(generated_answer) < 100:
        return "finalize"
    
    # 对于重要的查询意图，必须验证
    critical_intents = ["best_practice", "troubleshooting", "how_to"]
    if query_intent in critical_intents:
        return "validate"
    
    # 对于复杂查询，必须验证
    if complexity_level == "complex":
        return "validate"
    
    # 默认进行验证
    return "validate"


def should_correct_answer(state: SelfCorrectiveRAGState) -> Literal["correct", "finalize"]:
    """决定是否需要纠正答案
    
    Args:
        state: 当前状态
        
    Returns:
        路由决策："correct" 或 "finalize"
    """
    validation_passed = state.get("validation_passed", True)
    validation_score = state.get("validation_score", 1.0)
    correction_attempts = state.get("correction_attempts", 0)
    max_correction_attempts = state.get("max_correction_attempts", 2)
    needs_correction = state.get("needs_correction", False)
    
    # 如果验证通过，不需要纠正
    if validation_passed and not needs_correction:
        return "finalize"
    
    # 如果已达到最大纠正次数，停止纠正
    if correction_attempts >= max_correction_attempts:
        return "finalize"
    
    # 如果验证分数太低，需要纠正
    if validation_score < 0.5:
        return "correct"
    
    # 如果明确需要纠正，进行纠正
    if needs_correction:
        return "correct"
    
    # 默认完成
    return "finalize"


def route_by_query_complexity(state: SelfCorrectiveRAGState) -> Literal["simple_path", "complex_path"]:
    """根据查询复杂度进行路由
    
    Args:
        state: 当前状态
        
    Returns:
        路由决策："simple_path" 或 "complex_path"
    """
    complexity_level = state.get("complexity_level", "medium")
    query_intent = state.get("query_intent", "general")
    query = state.get("query", "")
    
    # 简单查询的条件
    if complexity_level == "simple":
        return "simple_path"
    
    # 复杂查询的条件
    if complexity_level == "complex":
        return "complex_path"
    
    # 根据查询意图判断
    simple_intents = ["definition", "list"]
    complex_intents = ["comparison", "best_practice", "troubleshooting", "explanation"]
    
    if query_intent in simple_intents:
        return "simple_path"
    elif query_intent in complex_intents:
        return "complex_path"
    
    # 根据查询长度判断
    if len(query.split()) <= 5:
        return "simple_path"
    else:
        return "complex_path"


def route_by_retrieval_quality(state: SelfCorrectiveRAGState) -> Literal["high_quality", "low_quality", "no_documents"]:
    """根据检索质量进行路由
    
    Args:
        state: 当前状态
        
    Returns:
        路由决策："high_quality", "low_quality", 或 "no_documents"
    """
    graded_documents = state.get("graded_documents", [])
    retrieval_quality = state.get("retrieval_quality", 0.0)
    
    # 检查是否有文档
    if not graded_documents:
        return "no_documents"
    
    # 检查相关文档数量
    relevant_docs = [item for item in graded_documents if item.get("is_relevant", False)]
    
    if not relevant_docs:
        return "no_documents"
    
    # 根据检索质量分数判断
    if retrieval_quality >= 0.7:
        return "high_quality"
    else:
        return "low_quality"


def route_by_answer_quality(state: SelfCorrectiveRAGState) -> Literal["high_quality", "medium_quality", "low_quality"]:
    """根据答案质量进行路由
    
    Args:
        state: 当前状态
        
    Returns:
        路由决策："high_quality", "medium_quality", 或 "low_quality"
    """
    answer_quality_score = state.get("answer_quality_score", 0.0)
    validation_score = state.get("validation_score", 0.0)
    
    # 综合考虑答案质量和验证分数
    combined_score = (answer_quality_score + validation_score) / 2
    
    if combined_score >= 0.8:
        return "high_quality"
    elif combined_score >= 0.5:
        return "medium_quality"
    else:
        return "low_quality"


def should_continue_processing(state: SelfCorrectiveRAGState) -> Literal["continue", "stop"]:
    """决定是否继续处理
    
    Args:
        state: 当前状态
        
    Returns:
        路由决策："continue" 或 "stop"
    """
    error_message = state.get("error_message", "")
    final_answer = state.get("final_answer", "")
    max_total_attempts = state.get("max_total_attempts", 10)
    
    # 计算总尝试次数
    total_attempts = (
        state.get("retrieval_attempts", 0) +
        state.get("generation_attempts", 0) +
        state.get("correction_attempts", 0)
    )
    
    # 如果有严重错误，停止处理
    if error_message and "严重" in error_message:
        return "stop"
    
    # 如果已有最终答案，停止处理
    if final_answer:
        return "stop"
    
    # 如果超过最大尝试次数，停止处理
    if total_attempts >= max_total_attempts:
        return "stop"
    
    # 否则继续处理
    return "continue"


def route_error_handling(state: SelfCorrectiveRAGState) -> Literal["retry", "fallback", "abort"]:
    """错误处理路由
    
    Args:
        state: 当前状态
        
    Returns:
        路由决策："retry", "fallback", 或 "abort"
    """
    error_message = state.get("error_message", "")
    error_count = state.get("error_count", 0)
    max_error_count = state.get("max_error_count", 3)
    
    # 如果没有错误，不需要错误处理
    if not error_message:
        return "retry"
    
    # 如果错误次数超过限制，中止处理
    if error_count >= max_error_count:
        return "abort"
    
    # 根据错误类型决定处理方式
    if "网络" in error_message or "连接" in error_message:
        return "retry"
    elif "模型" in error_message or "API" in error_message:
        return "fallback"
    elif "配置" in error_message or "参数" in error_message:
        return "abort"
    else:
        return "retry"


def get_next_node(state: SelfCorrectiveRAGState) -> str:
    """获取下一个节点名称（通用路由函数）
    
    Args:
        state: 当前状态
        
    Returns:
        下一个节点名称
    """
    current_node = state.get("current_node", "preprocess")
    
    # 定义节点流程
    node_flow = {
        "preprocess": "retrieve",
        "retrieve": "grade",
        "grade": "generate",
        "rewrite": "retrieve",
        "generate": "validate",
        "validate": "correct",
        "correct": "finalize",
        "finalize": "END"
    }
    
    return node_flow.get(current_node, "END")


def create_conditional_map(state: SelfCorrectiveRAGState) -> Dict[str, str]:
    """创建条件映射（用于复杂路由逻辑）
    
    Args:
        state: 当前状态
        
    Returns:
        条件到节点的映射
    """
    # 这个函数可以用于更复杂的条件路由逻辑
    # 根据多个状态变量的组合来决定路由
    
    conditions = {
        "has_documents": len(state.get("graded_documents", [])) > 0,
        "has_relevant_docs": any(
            item.get("is_relevant", False) 
            for item in state.get("graded_documents", [])
        ),
        "high_quality": state.get("retrieval_quality", 0.0) >= 0.7,
        "validation_passed": state.get("validation_passed", False),
        "needs_correction": state.get("needs_correction", False),
        "max_attempts_reached": (
            state.get("retrieval_attempts", 0) >= 
            state.get("max_retrieval_attempts", 3)
        )
    }
    
    # 根据条件组合决定路由
    if not conditions["has_documents"] and not conditions["max_attempts_reached"]:
        return {"next": "rewrite"}
    elif not conditions["has_relevant_docs"] and not conditions["max_attempts_reached"]:
        return {"next": "rewrite"}
    elif not conditions["high_quality"] and not conditions["max_attempts_reached"]:
        return {"next": "rewrite"}
    elif conditions["needs_correction"]:
        return {"next": "correct"}
    elif not conditions["validation_passed"]:
        return {"next": "validate"}
    else:
        return {"next": "finalize"}