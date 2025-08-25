"""自我纠错RAG系统的状态定义

定义了LangGraph中使用的状态结构，包括查询处理、文档检索、
答案生成和质量控制等各个阶段的状态信息。
"""

from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from datetime import datetime


class SelfCorrectiveRAGState(TypedDict):
    """自我纠错RAG系统的状态定义
    
    这个状态类包含了整个RAG流程中需要跟踪的所有信息，
    包括查询处理、文档检索、答案生成和质量控制等。
    """
    
    # === 基础状态 ===
    messages: List[BaseMessage]  # 对话消息历史
    query: str  # 当前处理的查询
    original_query: str  # 原始用户查询
    keywords: Optional[List[str]]  # 提取的关键词
    
    # === 检索状态 ===
    retrieved_documents: List[Document]  # 检索到的原始文档
    graded_documents: List[Dict[str, Any]]  # 评分后的文档列表
    retrieval_score: float  # 检索质量分数 (0-1)
    retrieval_attempts: int  # 检索尝试次数
    
    # === 生成状态 ===
    generated_answer: str  # 生成的答案
    answer_quality_score: float  # 答案质量分数 (0-1)
    generation_attempts: int  # 生成尝试次数
    validation_feedback: Optional[str]  # 验证反馈信息
    
    # === 纠错状态 ===
    needs_query_rewrite: bool  # 是否需要重写查询
    needs_answer_correction: bool  # 是否需要纠错答案
    rewritten_queries: List[str]  # 重写的查询历史
    correction_history: List[Dict[str, Any]]  # 纠错历史记录
    
    # === 控制参数 ===
    max_retrieval_attempts: int  # 最大检索尝试次数
    max_generation_attempts: int  # 最大生成尝试次数
    quality_threshold: float  # 质量阈值 (0-1)
    
    # === 结果状态 ===
    final_answer: str  # 最终答案
    confidence_score: float  # 最终置信度分数
    processing_time: Optional[float]  # 处理时间（秒）
    
    # === 元数据 ===
    session_id: Optional[str]  # 会话ID
    timestamp: Optional[str]  # 时间戳
    error_message: Optional[str]  # 错误信息


class DocumentGrade(TypedDict):
    """文档评分结果"""
    document: Document  # 文档对象
    relevance_score: float  # 相关性分数 (0-1)
    is_relevant: bool  # 是否相关
    reasoning: Optional[str]  # 评分理由


class CorrectionRecord(TypedDict):
    """纠错记录"""
    original_content: str  # 原始内容
    corrected_content: str  # 纠错后内容
    correction_type: str  # 纠错类型 (query_rewrite/answer_correction)
    feedback: str  # 反馈信息
    timestamp: str  # 时间戳
    quality_improvement: float  # 质量提升分数


class ValidationResult(TypedDict):
    """验证结果"""
    score: float  # 质量分数 (0-1)
    needs_correction: bool  # 是否需要纠错
    feedback: str  # 详细反馈
    issues: List[str]  # 发现的问题列表
    suggestions: List[str]  # 改进建议


def initialize_state(
    query: str,
    session_id: Optional[str] = None,
    max_retrieval_attempts: int = 3,
    max_generation_attempts: int = 2,
    quality_threshold: float = 0.7
) -> SelfCorrectiveRAGState:
    """创建初始状态
    
    Args:
        query: 用户查询
        session_id: 会话ID
        max_retrieval_attempts: 最大检索尝试次数
        max_generation_attempts: 最大生成尝试次数
        quality_threshold: 质量阈值
        
    Returns:
        初始化的状态对象
    """
    return SelfCorrectiveRAGState(
        # 基础状态
        messages=[],
        query=query,
        original_query=query,
        keywords=None,
        
        # 检索状态
        retrieved_documents=[],
        graded_documents=[],
        retrieval_score=0.0,
        retrieval_attempts=0,
        
        # 生成状态
        generated_answer="",
        answer_quality_score=0.0,
        generation_attempts=0,
        validation_feedback=None,
        
        # 纠错状态
        needs_query_rewrite=False,
        needs_answer_correction=False,
        rewritten_queries=[],
        correction_history=[],
        
        # 控制参数
        max_retrieval_attempts=max_retrieval_attempts,
        max_generation_attempts=max_generation_attempts,
        quality_threshold=quality_threshold,
        
        # 结果状态
        final_answer="",
        confidence_score=0.0,
        processing_time=None,
        
        # 元数据
        session_id=session_id,
        timestamp=datetime.now().isoformat(),
        error_message=None
    )


def update_state_with_error(
    state: SelfCorrectiveRAGState,
    error_message: str
) -> SelfCorrectiveRAGState:
    """更新状态中的错误信息
    
    Args:
        state: 当前状态
        error_message: 错误信息
        
    Returns:
        更新后的状态
    """
    updated_state = state.copy()
    updated_state["error_message"] = error_message
    updated_state["final_answer"] = f"处理过程中发生错误: {error_message}"
    updated_state["confidence_score"] = 0.0
    return updated_state


def finalize_state(
    state: SelfCorrectiveRAGState,
    processing_start_time: float
) -> SelfCorrectiveRAGState:
    """完成状态处理
    
    Args:
        state: 当前状态
        processing_start_time: 处理开始时间
        
    Returns:
        最终状态
    """
    import time
    
    updated_state = state.copy()
    updated_state["final_answer"] = state["generated_answer"]
    updated_state["confidence_score"] = state["answer_quality_score"]
    updated_state["processing_time"] = time.time() - processing_start_time
    
    return updated_state