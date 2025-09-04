"""自我纠错RAG系统的主要LangGraph实现

这个模块包含了完整的自我纠错RAG图的构建和执行逻辑。
"""

import time
import os
from typing import Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import RunnableConfig

from .state import SelfCorrectiveRAGState, initialize_state, finalize_state, update_state_with_error
from .nodes.preprocessing import preprocess_query_node
from .nodes.retrieval import intelligent_retrieval_node, grade_documents_node
from .nodes.rewrite import rewrite_query_node
from .nodes.generation import generate_answer_node
from .nodes.validation import validate_answer_node, correct_answer_node
from .nodes.routing import (
    should_retrieve_documents,
    should_rewrite_query,
    should_validate_answer,
    should_correct_answer
)
from .utils import get_llm_client
from src.core.config_manager import ConfigManager


class SelfCorrectiveRAGGraph:
    """自我纠错RAG图管理器
    
    负责构建和管理整个自我纠错RAG流程的LangGraph。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化RAG图
        
        Args:
            config: 可选的配置字典，如果不提供则使用默认配置
        """
        self.config = config or self._load_default_config()
        self._setup_langsmith()
        self.graph = None
        self._build_graph()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """加载默认配置"""
        try:
            config_manager = ConfigManager()
            base_config = config_manager.get_config()
            
            # 提取自我纠错RAG相关配置
            rag_config = base_config.get("self_corrective_rag", {})
            
            # 设置默认值
            default_config = {
                "max_retrieval_attempts": rag_config.get("max_retrieval_attempts", 3),
                "max_generation_attempts": rag_config.get("max_generation_attempts", 2),
                "quality_threshold": rag_config.get("quality_threshold", 0.7),
                "rules_directory": rag_config.get("rules_directory", "./rules"),
                "enable_query_rewriting": rag_config.get("enable_query_rewriting", True),
                "enable_answer_correction": rag_config.get("enable_answer_correction", True),
                "grading_model": rag_config.get("grading_model", "gpt-3.5-turbo"),
                "generation_model": rag_config.get("generation_model", "gpt-4"),
                "validation_model": rag_config.get("validation_model", "gpt-3.5-turbo")
            }
            
            return default_config
            
        except Exception as e:
            print(f"加载配置时发生错误，使用默认配置: {e}")
            return {
                "max_retrieval_attempts": 3,
                "max_generation_attempts": 2,
                "quality_threshold": 0.7,
                "rules_directory": "./rules",
                "enable_query_rewriting": True,
                "enable_answer_correction": True,
                "grading_model": "gpt-3.5-turbo",
                "generation_model": "gpt-4",
                "validation_model": "gpt-3.5-turbo"
            }
    
    def _setup_langsmith(self):
        """设置LangSmith追踪"""
        try:
            # 从环境变量获取LangSmith配置
            langsmith_api_key = os.getenv('LANGSMITH_API_KEY')
            
            if langsmith_api_key:
                # 设置LangSmith环境变量
                os.environ['LANGCHAIN_TRACING_V2'] = 'true'
                os.environ['LANGCHAIN_API_KEY'] = langsmith_api_key
                os.environ['LANGCHAIN_PROJECT'] = 'self-corrective-rag'
                print("[成功] LangSmith追踪已启用")
            else:
                print("[警告] 未找到LANGSMITH_API_KEY，跳过LangSmith配置")
                
        except Exception as e:
            print(f"[错误] 设置LangSmith时发生错误: {e}")
    
    def _build_graph(self):
        """构建LangGraph"""
        try:
            # 创建状态图
            workflow = StateGraph(SelfCorrectiveRAGState)
            
            # 添加节点
            workflow.add_node("preprocess", preprocess_query_node)
            workflow.add_node("retrieve", intelligent_retrieval_node)
            workflow.add_node("grade", grade_documents_node)
            workflow.add_node("rewrite", rewrite_query_node)
            workflow.add_node("generate", generate_answer_node)
            workflow.add_node("validate", validate_answer_node)
            workflow.add_node("correct", correct_answer_node)
            
            # 设置入口点
            workflow.set_entry_point("preprocess")
            
            # 添加条件边
            workflow.add_conditional_edges(
                "preprocess",
                should_retrieve_documents,
                {
                    "retrieve": "retrieve",
                    "generate_direct": "generate"
                }
            )
            
            workflow.add_edge("retrieve", "grade")
            
            workflow.add_conditional_edges(
                "grade",
                should_rewrite_query,
                {
                    "rewrite": "rewrite",
                    "generate": "generate"
                }
            )
            
            workflow.add_edge("rewrite", "retrieve")
            
            workflow.add_conditional_edges(
                "generate",
                should_validate_answer,
                {
                    "validate": "validate",
                    "finalize": END
                }
            )
            
            workflow.add_conditional_edges(
                "validate",
                should_correct_answer,
                {
                    "correct": "correct",
                    "finalize": END
                }
            )
            
            workflow.add_edge("correct", END)
            
            # 编译图
            self.graph = workflow.compile()
            
            print("自我纠错RAG图构建成功")
            
        except Exception as e:
            print(f"构建RAG图时发生错误: {e}")
            raise
    
    def invoke(
        self,
        query: str,
        session_id: Optional[str] = None,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """执行RAG查询
        
        Args:
            query: 用户查询
            session_id: 会话ID
            config: 运行时配置
            
        Returns:
            包含答案和元数据的结果字典
        """
        if not self.graph:
            raise RuntimeError("RAG图未正确初始化")
        
        start_time = time.time()
        
        try:
            # 创建初始状态
            initial_state = initialize_state(
                query=query,
                session_id=session_id,
                max_retrieval_attempts=self.config["max_retrieval_attempts"],
                max_generation_attempts=self.config["max_generation_attempts"],
                quality_threshold=self.config["quality_threshold"]
            )
            
            # 执行图
            result_state = self.graph.invoke(initial_state, config=config)
            
            # 完成状态处理
            final_state = finalize_state(result_state, start_time)
            
            # 返回结果
            return self._format_result(final_state)
            
        except Exception as e:
            print(f"执行RAG查询时发生错误: {e}")
            
            # 创建错误状态
            error_state = initialize_state(query, session_id)
            error_state = update_state_with_error(error_state, str(e))
            error_state = finalize_state(error_state, start_time)
            
            return self._format_result(error_state)
    
    async def ainvoke(
        self,
        query: str,
        session_id: Optional[str] = None,
        config: Optional[RunnableConfig] = None
    ) -> Dict[str, Any]:
        """异步执行RAG查询
        
        Args:
            query: 用户查询
            session_id: 会话ID
            config: 运行时配置
            
        Returns:
            包含答案和元数据的结果字典
        """
        if not self.graph:
            raise RuntimeError("RAG图未正确初始化")
        
        start_time = time.time()
        
        try:
            # 创建初始状态
            initial_state = initialize_state(
                query=query,
                session_id=session_id,
                max_retrieval_attempts=self.config["max_retrieval_attempts"],
                max_generation_attempts=self.config["max_generation_attempts"],
                quality_threshold=self.config["quality_threshold"]
            )
            
            # 异步执行图
            result_state = await self.graph.ainvoke(initial_state, config=config)
            
            # 完成状态处理
            final_state = finalize_state(result_state, start_time)
            
            # 返回结果
            return self._format_result(final_state)
            
        except Exception as e:
            print(f"异步执行RAG查询时发生错误: {e}")
            
            # 创建错误状态
            error_state = initialize_state(query, session_id)
            error_state = update_state_with_error(error_state, str(e))
            error_state = finalize_state(error_state, start_time)
            
            return self._format_result(error_state)
    
    def _format_result(self, state: SelfCorrectiveRAGState) -> Dict[str, Any]:
        """格式化返回结果为JSON结构
        
        Args:
            state: 最终状态
            
        Returns:
            包含LLM回答和参考数据chunk的JSON结构
        """
        # 构建参考数据块
        reference_chunks = []
        
        # 处理检索到的文档
        retrieved_docs = state.get("retrieved_documents", [])
        graded_docs = state.get("graded_documents", [])
        
        # 创建文档ID到评分的映射
        doc_scores = {}
        for graded_doc in graded_docs:
            if isinstance(graded_doc, dict) and "document" in graded_doc:
                doc_id = id(graded_doc["document"])
                doc_scores[doc_id] = {
                    "relevance_score": graded_doc.get("relevance_score", 0.0),
                    "is_relevant": graded_doc.get("is_relevant", False),
                    "reasoning": graded_doc.get("reasoning", "")
                }
        
        # 构建参考数据块列表
        for i, doc in enumerate(retrieved_docs):
            doc_id = id(doc)
            score_info = doc_scores.get(doc_id, {
                "relevance_score": 0.0,
                "is_relevant": False,
                "reasoning": "未评分"
            })
            
            chunk = {
                "chunk_id": i + 1,
                "content": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "source": doc.metadata.get("source", "未知来源"),
                "relevance_score": score_info["relevance_score"],
                "is_relevant": score_info["is_relevant"],
                "metadata": {
                    "file_path": doc.metadata.get("file_path", ""),
                    "page": doc.metadata.get("page", ""),
                    "section": doc.metadata.get("section", ""),
                    "chunk_index": doc.metadata.get("chunk_index", i),
                    "reasoning": score_info["reasoning"]
                }
            }
            reference_chunks.append(chunk)
        
        return {
            "answer": state["final_answer"],
            "reference_chunks": reference_chunks,
            "confidence_score": state["confidence_score"],
            "processing_time": state.get("processing_time", 0),
            "metadata": {
                "original_query": state.get("original_query"),
                "rewritten_queries": state.get("rewritten_queries", []),
                "retrieval_score": state.get("retrieval_score", 0),
                "answer_quality_score": state.get("answer_quality_score", 0),
                "documents_count": len(retrieved_docs),
                "relevant_documents_count": len([
                    doc for doc in graded_docs
                    if doc.get("is_relevant", False)
                ]),
                "retrieval_attempts": state.get("retrieval_attempts", 0),
                "generation_attempts": state.get("generation_attempts", 0),
                "correction_history": state.get("correction_history", []),
                "session_id": state.get("session_id"),
                "timestamp": state.get("timestamp"),
                "error_message": state.get("error_message")
            }
        }
    
    def get_graph_visualization(self) -> str:
        """获取图的可视化表示
        
        Returns:
            图的Mermaid格式字符串
        """
        return """
        graph TD
            A[用户查询] --> B[查询预处理]
            B --> C[文档检索]
            C --> D[文档评分]
            D --> E{检索质量评估}
            E -->|质量不佳| F[查询重写]
            E -->|质量良好| G[答案生成]
            F --> C
            G --> H[答案验证]
            H --> I{答案质量评估}
            I -->|质量不佳| J[答案纠错]
            I -->|质量良好| K[返回结果]
            J --> H
            
            style A fill:#e1f5fe
            style K fill:#c8e6c9
            style E fill:#fff3e0
            style I fill:#fff3e0
        """
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置
        
        Returns:
            配置字典
        """
        return self.config.copy()
    
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置
        
        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        # 重新构建图以应用新配置
        self._build_graph()


def create_self_corrective_rag_graph(config: Optional[Dict[str, Any]] = None) -> SelfCorrectiveRAGGraph:
    """创建自我纠错RAG图的便捷函数
    
    Args:
        config: 可选的配置字典
        
    Returns:
        配置好的RAG图实例
    """
    return SelfCorrectiveRAGGraph(config)


# 示例使用
if __name__ == "__main__":
    # 创建RAG图
    rag_graph = create_self_corrective_rag_graph()
    
    # 执行查询
    result = rag_graph.invoke("什么是数据处理的最佳实践？")
    
    print(f"答案: {result['answer']}")
    print(f"置信度: {result['confidence_score']}")
    print(f"处理时间: {result['processing_time']:.2f}秒")
    print(f"检索尝试次数: {result['retrieval_attempts']}")
    print(f"生成尝试次数: {result['generation_attempts']}")