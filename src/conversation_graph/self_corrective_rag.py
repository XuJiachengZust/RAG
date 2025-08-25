"""自我纠错RAG系统

这个模块实现了基于LangGraph的自我纠错RAG系统，包括：
- 查询预处理和重写
- 智能文档检索
- 相关性评分和验证
- 答案生成和自我纠错
- 条件路由和流程控制
"""

import logging
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

from .state import SelfCorrectiveRAGState
from .nodes.preprocessing import preprocess_query_node
from .nodes.retrieval import intelligent_retrieval_node
from .nodes.generation import generate_answer_node
from .nodes.validation import validate_answer_node
from .nodes.rewrite import rewrite_query_node
from .nodes.routing import should_rewrite_query, should_continue_processing
from .document_initializer import DocumentInitializer
from ..core.config_manager import ConfigManager

logger = logging.getLogger(__name__)

class SelfCorrectiveRAG:
    """自我纠错RAG系统主类"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化自我纠错RAG系统
        
        Args:
            config: 配置参数
        """
        self.config_manager = ConfigManager()
        self.config = config or self._get_default_config()
        
        # 初始化文档系统
        self.document_initializer = DocumentInitializer({
            'rules_directory': self.config.get('rules_directory', './rules'),
            'force_reload': self.config.get('force_reload', False)
        })
        
        # 初始化各个节点
        self._initialize_nodes()
        
        # 构建工作流图
        self.workflow = self._build_workflow()
        
        logger.info("自我纠错RAG系统初始化完成")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'max_retries': 3,
            'relevance_threshold': 0.7,
            'rules_directory': './rules',
            'force_reload': False,
            'enable_query_rewrite': True,
            'enable_answer_validation': True,
            'temperature': 0.1,
            'max_tokens': 1000
        }
    
    def _initialize_nodes(self):
        """初始化文档系统"""
        try:
            # 确保文档已初始化
            self.document_initializer.initialize()
            self.vector_store = self.document_initializer.vector_store
            
            logger.info("文档系统初始化完成")
            
        except Exception as e:
            logger.error(f"文档系统初始化失败: {e}")
            raise
    
    def _build_workflow(self) -> StateGraph:
        """构建LangGraph工作流"""
        # 创建状态图
        workflow = StateGraph(SelfCorrectiveRAGState)
        
        # 添加节点
        workflow.add_node("preprocess", self._preprocess_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("validate", self._validate_node)
        workflow.add_node("rewrite", self._rewrite_node)
        
        # 设置入口点
        workflow.set_entry_point("preprocess")
        
        # 添加边和条件路由
        workflow.add_edge("preprocess", "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")
        
        # 条件路由：验证后决定是否需要重写查询
        workflow.add_conditional_edges(
            "validate",
            self._should_rewrite_query,
            {
                "rewrite": "rewrite",
                "end": END
            }
        )
        
        # 重写后回到检索
        workflow.add_edge("rewrite", "retrieve")
        
        return workflow.compile()
    
    def _preprocess_node(self, state: SelfCorrectiveRAGState) -> SelfCorrectiveRAGState:
        """查询预处理节点"""
        try:
            result = preprocess_query_node(state)
            logger.debug(f"查询预处理完成")
            return result
        except Exception as e:
            logger.error(f"查询预处理失败: {e}")
            state["error"] = str(e)
            return state
    
    def _retrieve_node(self, state: SelfCorrectiveRAGState) -> SelfCorrectiveRAGState:
        """智能检索节点"""
        try:
            result = intelligent_retrieval_node(state)
            logger.debug(f"检索完成，获得 {len(result.get('documents', []))} 个文档")
            return result
        except Exception as e:
            logger.error(f"文档检索失败: {e}")
            state["error"] = str(e)
            return state
    
    def _generate_node(self, state: SelfCorrectiveRAGState) -> SelfCorrectiveRAGState:
        """答案生成节点"""
        try:
            result = generate_answer_node(state)
            logger.debug(f"答案生成完成")
            return result
        except Exception as e:
            logger.error(f"答案生成失败: {e}")
            state["error"] = str(e)
            return state
    
    def _validate_node(self, state: SelfCorrectiveRAGState) -> SelfCorrectiveRAGState:
        """答案验证节点"""
        try:
            result = validate_answer_node(state)
            logger.debug(f"答案验证完成，相关性分数: {result.get('relevance_score', 0)}")
            return result
        except Exception as e:
            logger.error(f"答案验证失败: {e}")
            state["error"] = str(e)
            return state
    
    def _rewrite_node(self, state: SelfCorrectiveRAGState) -> SelfCorrectiveRAGState:
        """查询重写节点"""
        try:
            result = rewrite_query_node(state)
            logger.debug(f"查询重写完成")
            return result
        except Exception as e:
            logger.error(f"查询重写失败: {e}")
            state["error"] = str(e)
            return state
    
    def _should_rewrite_query(self, state: SelfCorrectiveRAGState) -> str:
        """决定是否需要重写查询"""
        return should_rewrite_query(state)
    
    def process_query(self, query: str, conversation_history: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """处理用户查询
        
        Args:
            query: 用户查询
            conversation_history: 对话历史
            
        Returns:
            包含答案和元数据的字典
        """
        try:
            # 初始化状态
            initial_state = SelfCorrectiveRAGState(
                messages=[],
                query=query,
                original_query=query,
                keywords=None,
                retrieved_documents=[],
                graded_documents=[],
                retrieval_score=0.0,
                retrieval_attempts=0,
                generated_answer="",
                answer_quality_score=0.0,
                generation_attempts=0,
                validation_feedback=None,
                needs_query_rewrite=False,
                needs_answer_correction=False,
                rewritten_queries=[],
                correction_history=[],
                max_retrieval_attempts=self.config.get('max_retries', 3),
                max_generation_attempts=self.config.get('max_retries', 3),
                quality_threshold=self.config.get('relevance_threshold', 0.7),
                final_answer="",
                confidence_score=0.0,
                processing_time=0.0,
                error_message=None,
                metadata={}
            )
            
            logger.info(f"开始处理查询: {query}")
            
            # 运行工作流
            final_state = self.workflow.invoke(initial_state)
            
            # 构建返回结果
            result = {
                "answer": final_state.get("final_answer", ""),
                "relevance_score": final_state.get("answer_quality_score", 0.0),
                "documents": final_state.get("retrieved_documents", []),
                "retry_count": final_state.get("retrieval_attempts", 0) + final_state.get("generation_attempts", 0),
                "rewritten_query": final_state.get("rewritten_queries", [])[-1] if final_state.get("rewritten_queries") else "",
                "metadata": final_state.get("metadata", {}),
                "error": final_state.get("error_message")
            }
            
            if result["error"]:
                logger.error(f"查询处理失败: {result['error']}")
            else:
                logger.info(f"查询处理完成，重试次数: {result['retry_count']}")
            
            return result
            
        except Exception as e:
            logger.error(f"查询处理异常: {e}")
            return {
                "answer": "抱歉，处理您的查询时出现了错误。",
                "relevance_score": 0.0,
                "documents": [],
                "retry_count": 0,
                "rewritten_query": "",
                "metadata": {},
                "error": str(e)
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        try:
            vector_store = self.document_initializer.vector_store
            doc_count = len(vector_store.get()['ids']) if vector_store else 0
            
            return {
                "document_count": doc_count,
                "config": self.config,
                "nodes_initialized": True
            }
        except Exception as e:
            logger.error(f"获取统计信息失败: {e}")
            return {
                "document_count": 0,
                "config": self.config,
                "nodes_initialized": False,
                "error": str(e)
            }