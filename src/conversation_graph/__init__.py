"""自我纠错RAG对话图模块

这个模块实现了基于LangGraph的自我纠错RAG系统，包括：
- 智能文档检索
- 查询重写机制
- 答案质量验证
- 自动纠错功能
"""

from .state import SelfCorrectiveRAGState
from .main_langgraph import SelfCorrectiveRAGGraph
from .nodes import (
    preprocess_query_node,
    intelligent_retrieval_node,
    grade_documents_node,
    rewrite_query_node,
    generate_answer_node,
    validate_answer_node,
    correct_answer_node
)
from .utils import (
    evaluate_answer_quality
)
# 检索器创建函数已移动到 src.core.retriever_factory
from ..core.retriever_factory import (
    create_rules_retriever,
    create_advanced_retriever,
    create_next_gen_retriever
)
# 旧的检索管理器已被移除，质量计算功能已集成到新的高级检索管理器中

__all__ = [
    'SelfCorrectiveRAGState',
    'SelfCorrectiveRAGGraph',
    'preprocess_query_node',
    'intelligent_retrieval_node',
    'grade_documents_node',
    'rewrite_query_node',
    'generate_answer_node',
    'validate_answer_node',
    'correct_answer_node',
    'create_rules_retriever',
    'create_advanced_retriever',
    'create_next_gen_retriever',
    # 'calculate_retrieval_quality',  # 已集成到新的高级检索管理器中
    'evaluate_answer_quality'
]