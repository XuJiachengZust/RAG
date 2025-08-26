"""答案生成节点

负责基于检索到的文档生成高质量答案。
"""

from typing import Dict, Any, List, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
import re
import logging

from ..state import SelfCorrectiveRAGState
from ..utils import get_llm_client, evaluate_answer_quality, format_documents_for_context
from ...core.prompt_manager import prompt_manager


def generate_answer_node(state: SelfCorrectiveRAGState) -> Dict[str, Any]:
    """答案生成节点
    
    功能：
    1. 基于相关文档生成答案
    2. 根据查询意图调整生成策略
    3. 评估答案质量
    4. 记录生成尝试次数
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态字典
    """
    try:
        query = state.get("query", "")
        graded_documents = state.get("graded_documents", [])
        query_intent = state.get("query_intent", "general")
        complexity_level = state.get("complexity_level", "medium")
        
        if not query:
            return {
                "error_message": "查询为空，无法生成答案",
                "generated_answer": "请提供有效的查询。",
                "answer_quality_score": 0.0,
                "generation_attempts": state.get("generation_attempts", 0) + 1
            }
        
        # 筛选相关文档
        relevant_docs = [
            item["document"] for item in graded_documents 
            if item.get("is_relevant", False)
        ]
        
        if not relevant_docs:
            return generate_no_documents_response(state, query)
        
        # 获取生成模型
        generator_llm = get_llm_client(node_type="generation")
        
        if not generator_llm:
            return {
                "error_message": "无法获取答案生成模型",
                "generated_answer": "系统暂时无法生成答案，请稍后重试。",
                "answer_quality_score": 0.0,
                "generation_attempts": state.get("generation_attempts", 0) + 1
            }
        
        # 构建上下文
        context = format_documents_for_context(relevant_docs, max_length=3000)
        
        # 根据意图选择生成策略
        generation_prompt = create_generation_prompt(query_intent, complexity_level)
        
        # 生成答案
        response = generator_llm.invoke(
            generation_prompt.format_messages(
                context=context,
                query=query,
                intent=query_intent
            )
        )
        
        generated_answer = response.content.strip()
        
        # 后处理答案
        processed_answer = post_process_answer(generated_answer, query_intent)
        
        # 评估答案质量
        quality_score = evaluate_answer_quality(query, context, processed_answer)
        
        # 记录生成尝试
        generation_attempts = state.get("generation_attempts", 0) + 1
        
        return {
            "generated_answer": processed_answer,
            "answer_quality_score": quality_score,
            "generation_attempts": generation_attempts,
            "generation_metadata": {
                "context_length": len(context),
                "relevant_docs_count": len(relevant_docs),
                "query_intent": query_intent,
                "complexity_level": complexity_level,
                "answer_length": len(processed_answer)
            }
        }
        
    except Exception as e:
        return {
            "error_message": f"答案生成失败: {str(e)}",
            "generated_answer": "生成答案时发生错误，请重试。",
            "answer_quality_score": 0.0,
            "generation_attempts": state.get("generation_attempts", 0) + 1
        }


def create_generation_prompt(query_intent: str, complexity_level: str) -> ChatPromptTemplate:
    """根据查询意图创建生成提示
    
    Args:
        query_intent: 查询意图
        complexity_level: 复杂度级别
        
    Returns:
        生成提示模板
    """
    try:
        # 从配置中获取基础系统提示
        base_system_prompt = prompt_manager.get_prompt('generation', 'base_system_prompt')
        
        # 获取意图特定的提示
        intent_prompt = prompt_manager.get_prompt('generation', f'intent_prompts.{query_intent}', default='')
        
        # 获取复杂度调整提示
        complexity_prompt = prompt_manager.get_prompt('generation', f'complexity_adjustments.{complexity_level}', default='')
        
        # 获取用户模板
        user_template = prompt_manager.get_prompt('generation', 'user_template')
        
        # 组合系统提示
        system_prompt = base_system_prompt
        if intent_prompt:
            system_prompt += f"\n\n{intent_prompt}"
        if complexity_prompt:
            system_prompt += f"\n\n{complexity_prompt}"
        
        # 创建提示模板
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        
    except Exception as e:
        # 如果配置加载失败，使用默认的简化版本
        print(f"Warning: Failed to load generation prompts from config: {e}")
        return ChatPromptTemplate.from_messages([
            ("system", "你是一个专业的知识助手，能够基于提供的上下文信息准确回答用户问题。请确保答案准确、完整、有条理。"),
            ("human", "上下文信息：\n{context}\n\n用户问题：{query}\n\n请基于上述上下文信息回答用户问题。")
        ])


def generate_no_documents_response(state: SelfCorrectiveRAGState, query: str) -> Dict[str, Any]:
    """当没有相关文档时生成响应
    
    Args:
        state: 当前状态
        query: 用户查询
        
    Returns:
        响应状态字典
    """
    # 分析为什么没有找到相关文档
    retrieval_attempts = state.get("retrieval_attempts", 0)
    
    if retrieval_attempts >= state.get("max_retrieval_attempts", 3):
        # 已达到最大尝试次数
        answer = (
            f"抱歉，经过多次尝试，我没有找到与您的问题 \"{query}\" 相关的信息。\n\n"
            "可能的原因：\n"
            "1. 您询问的内容不在当前的知识库范围内\n"
            "2. 问题的表述可能需要调整\n"
            "3. 相关文档可能尚未添加到系统中\n\n"
            "建议：\n"
            "- 尝试使用不同的关键词重新提问\n"
            "- 将复杂问题分解为更具体的子问题\n"
            "- 检查问题中是否有拼写错误"
        )
    else:
        # 还可以继续尝试
        answer = (
            f"暂时没有找到与 \"{query}\" 直接相关的信息。\n"
            "系统将尝试重新组织查询以获得更好的结果。"
        )
    
    return {
        "generated_answer": answer,
        "answer_quality_score": 0.3,  # 给一个较低但非零的分数
        "generation_attempts": state.get("generation_attempts", 0) + 1,
        "needs_query_rewrite": retrieval_attempts < state.get("max_retrieval_attempts", 3)
    }


def post_process_answer(answer: str, query_intent: str) -> str:
    """后处理生成的答案
    
    Args:
        answer: 原始答案
        query_intent: 查询意图
        
    Returns:
        处理后的答案
    """
    if not answer or not answer.strip():
        return "抱歉，无法生成有效的答案。"
    
    processed = answer.strip()
    
    # 根据意图进行特定的后处理
    if query_intent == "how_to":
        # 确保步骤清晰
        processed = ensure_step_formatting(processed)
    elif query_intent == "list":
        # 确保列表格式
        processed = ensure_list_formatting(processed)
    elif query_intent == "definition":
        # 确保定义清晰
        processed = ensure_definition_clarity(processed)
    
    # 通用后处理
    processed = improve_readability(processed)
    
    return processed


def ensure_step_formatting(text: str) -> str:
    """确保步骤格式清晰"""
    lines = text.split('\n')
    processed_lines = []
    step_counter = 1
    
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append("")
            continue
        
        # 检查是否是步骤行
        if any(line.startswith(prefix) for prefix in ['1.', '2.', '3.', '4.', '5.', '•', '-']):
            processed_lines.append(line)
        elif line.startswith(('首先', '然后', '接下来', '最后', '第一', '第二', '第三')):
            processed_lines.append(f"{step_counter}. {line}")
            step_counter += 1
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def ensure_list_formatting(text: str) -> str:
    """确保列表格式清晰"""
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append("")
            continue
        
        # 如果是列表项但没有标记，添加标记
        if not line.startswith(('•', '-', '*', '1.', '2.', '3.')) and len(line) < 100:
            if any(keyword in line for keyword in ['包括', '有', '是', '为']):
                processed_lines.append(f"• {line}")
            else:
                processed_lines.append(line)
        else:
            processed_lines.append(line)
    
    return '\n'.join(processed_lines)


def ensure_definition_clarity(text: str) -> str:
    """确保定义清晰"""
    # 如果答案没有明确的定义开头，添加一个
    if not any(text.startswith(prefix) for prefix in ['是指', '是', '指的是', '定义为']):
        # 查找第一个句子
        sentences = text.split('。')
        if sentences and len(sentences[0]) < 200:
            # 如果第一句话不太长，可能是定义
            return text
    
    return text


def improve_readability(text: str) -> str:
    """改善可读性"""
    # 确保段落间有适当的空行
    lines = text.split('\n')
    processed_lines = []
    
    for i, line in enumerate(lines):
        processed_lines.append(line)
        
        # 在标题后添加空行
        if line.strip() and (
            line.endswith('：') or 
            line.endswith(':') or
            any(line.startswith(prefix) for prefix in ['## ', '### ', '#### '])
        ):
            if i + 1 < len(lines) and lines[i + 1].strip():
                processed_lines.append("")
    
    # 移除多余的空行
    final_lines = []
    prev_empty = False
    
    for line in processed_lines:
        if not line.strip():
            if not prev_empty:
                final_lines.append(line)
            prev_empty = True
        else:
            final_lines.append(line)
            prev_empty = False
    
    return '\n'.join(final_lines).strip()


def validate_answer_completeness(answer: str, query: str) -> tuple[bool, List[str]]:
    """验证答案完整性
    
    Args:
        answer: 生成的答案
        query: 原始查询
        
    Returns:
        (是否完整, 缺失的方面列表)
    """
    issues = []
    
    # 检查答案长度
    if len(answer.split()) < 20:
        issues.append("答案过于简短")
    
    # 检查是否回答了问题
    query_lower = query.lower()
    answer_lower = answer.lower()
    
    # 检查关键词覆盖
    query_words = set(query_lower.split())
    answer_words = set(answer_lower.split())
    
    coverage = len(query_words.intersection(answer_words)) / len(query_words) if query_words else 0
    
    if coverage < 0.3:
        issues.append("答案与问题相关性不足")
    
    # 检查是否有结论
    conclusion_indicators = ['总结', '综上', '因此', '所以', '总的来说', '最后']
    has_conclusion = any(indicator in answer for indicator in conclusion_indicators)
    
    if len(answer.split()) > 100 and not has_conclusion:
        issues.append("缺少总结或结论")
    
    return len(issues) == 0, issues