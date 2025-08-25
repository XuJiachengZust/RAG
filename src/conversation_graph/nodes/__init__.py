"""自我纠错RAG系统的节点实现

本模块包含LangGraph中各个节点的实现，包括：
- 查询预处理节点
- 智能检索节点
- 文档相关性评分节点
- 查询重写节点
- 答案生成节点
- 答案验证节点
- 答案纠错节点
- 路由决策节点
"""

# 导入所有节点函数
try:
    from .preprocessing import preprocess_query_node
except ImportError:
    preprocess_query_node = None

try:
    from .retrieval import intelligent_retrieval_node, grade_documents_node
except ImportError:
    intelligent_retrieval_node = None
    grade_documents_node = None

try:
    from .rewrite import rewrite_query_node
except ImportError:
    rewrite_query_node = None

try:
    from .generation import generate_answer_node
except ImportError:
    generate_answer_node = None

try:
    from .validation import validate_answer_node, correct_answer_node
except ImportError:
    validate_answer_node = None
    correct_answer_node = None

try:
    from .routing import (
        should_retrieve_documents,
        should_rewrite_query,
        should_validate_answer,
        should_correct_answer,
        route_by_query_complexity,
        route_by_retrieval_quality,
        route_by_answer_quality
    )
except ImportError:
    should_retrieve_documents = None
    should_rewrite_query = None
    should_validate_answer = None
    should_correct_answer = None
    route_by_query_complexity = None
    route_by_retrieval_quality = None
    route_by_answer_quality = None

__all__ = [
    # 处理节点
    "preprocess_query_node",
    "intelligent_retrieval_node",
    "grade_documents_node",
    "rewrite_query_node",
    "generate_answer_node",
    "validate_answer_node",
    "correct_answer_node",
    
    # 路由函数
    "should_retrieve_documents",
    "should_rewrite_query",
    "should_validate_answer",
    "should_correct_answer",
    "route_by_query_complexity",
    "route_by_retrieval_quality",
    "route_by_answer_quality"
]