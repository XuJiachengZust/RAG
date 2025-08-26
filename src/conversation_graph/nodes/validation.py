"""答案验证和纠错节点

负责验证生成答案的质量并进行必要的纠错。
"""

from typing import Dict, Any, List, Tuple
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from ..state import SelfCorrectiveRAGState
from ..utils import get_llm_client, evaluate_answer_quality, parse_validation_result
from ...core.prompt_manager import prompt_manager


def validate_answer_node(state: SelfCorrectiveRAGState) -> Dict[str, Any]:
    """答案验证节点
    
    功能：
    1. 验证答案的准确性和完整性
    2. 检查答案是否回答了用户问题
    3. 评估答案质量
    4. 决定是否需要纠错
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态字典
    """
    try:
        query = state.get("query", "")
        generated_answer = state.get("generated_answer", "")
        graded_documents = state.get("graded_documents", [])
        quality_threshold = state.get("quality_threshold", 0.7)
        
        if not generated_answer:
            return {
                "validation_passed": False,
                "validation_score": 0.0,
                "validation_feedback": "没有生成的答案需要验证",
                "needs_correction": True
            }
        
        # 获取验证模型
        validator_llm = get_llm_client(node_type="validation")
        
        if not validator_llm:
            # 使用简单验证作为后备
            return simple_answer_validation(state)
        
        # 准备验证上下文
        relevant_docs = [
            item["document"] for item in graded_documents 
            if item.get("is_relevant", False)
        ]
        
        context = "\n\n".join([
            f"文档 {i+1}: {doc.page_content[:500]}..."
            for i, doc in enumerate(relevant_docs[:3])
        ])
        
        # 执行多维度验证
        validation_results = perform_comprehensive_validation(
            validator_llm, query, generated_answer, context
        )
        
        # 计算综合验证分数
        overall_score = calculate_validation_score(validation_results)
        
        # 生成验证反馈
        feedback = generate_validation_feedback(validation_results)
        
        # 决定是否需要纠错
        needs_correction = overall_score < quality_threshold
        
        return {
            "validation_passed": not needs_correction,
            "validation_score": overall_score,
            "validation_feedback": feedback,
            "validation_details": validation_results,
            "needs_correction": needs_correction
        }
        
    except Exception as e:
        return {
            "validation_passed": False,
            "validation_score": 0.0,
            "validation_feedback": f"验证过程出错: {str(e)}",
            "needs_correction": True,
            "error_message": f"答案验证失败: {str(e)}"
        }


def correct_answer_node(state: SelfCorrectiveRAGState) -> Dict[str, Any]:
    """答案纠错节点
    
    功能：
    1. 基于验证反馈纠正答案
    2. 改进答案的准确性和完整性
    3. 确保答案符合质量标准
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态字典
    """
    try:
        query = state.get("query", "")
        generated_answer = state.get("generated_answer", "")
        validation_feedback = state.get("validation_feedback", "")
        validation_details = state.get("validation_details", {})
        graded_documents = state.get("graded_documents", [])
        correction_attempts = state.get("correction_attempts", 0)
        max_correction_attempts = state.get("max_correction_attempts", 2)
        
        if correction_attempts >= max_correction_attempts:
            return {
                "final_answer": generated_answer,
                "correction_applied": False,
                "correction_feedback": "已达到最大纠错次数，使用当前答案",
                "correction_attempts": correction_attempts
            }
        
        # 获取纠错模型
        corrector_llm = get_llm_client(node_type="correction")
        
        if not corrector_llm:
            return {
                "final_answer": generated_answer,
                "correction_applied": False,
                "correction_feedback": "无法获取纠错模型，使用原始答案",
                "correction_attempts": correction_attempts + 1
            }
        
        # 准备纠错上下文
        relevant_docs = [
            item["document"] for item in graded_documents 
            if item.get("is_relevant", False)
        ]
        
        context = "\n\n".join([
            f"文档 {i+1}: {doc.page_content[:800]}..."
            for i, doc in enumerate(relevant_docs[:5])
        ])
        
        # 执行答案纠错
        corrected_answer = perform_answer_correction(
            corrector_llm, query, generated_answer, context, validation_feedback, validation_details
        )
        
        # 验证纠错效果
        improvement_score = evaluate_correction_improvement(
            query, generated_answer, corrected_answer, context
        )
        
        correction_applied = improvement_score > 0.1  # 如果改进明显才应用纠错
        
        final_answer = corrected_answer if correction_applied else generated_answer
        
        return {
            "final_answer": final_answer,
            "corrected_answer": corrected_answer,
            "correction_applied": correction_applied,
            "correction_feedback": f"纠错{'成功' if correction_applied else '未显著改善'}，改进分数: {improvement_score:.2f}",
            "correction_attempts": correction_attempts + 1,
            "improvement_score": improvement_score
        }
        
    except Exception as e:
        return {
            "final_answer": state.get("generated_answer", "生成答案时发生错误"),
            "correction_applied": False,
            "correction_feedback": f"纠错过程出错: {str(e)}",
            "correction_attempts": state.get("correction_attempts", 0) + 1,
            "error_message": f"答案纠错失败: {str(e)}"
        }


def perform_comprehensive_validation(
    validator_llm, query: str, answer: str, context: str
) -> Dict[str, Any]:
    """执行全面的答案验证
    
    Args:
        validator_llm: 验证模型
        query: 用户查询
        answer: 生成的答案
        context: 相关文档上下文
        
    Returns:
        验证结果字典
    """
    try:
        # 从配置中获取验证prompt
        system_prompt = prompt_manager.get_prompt('validation', 'system_prompt')
        user_template = prompt_manager.get_prompt('validation', 'user_template')
        
        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        
    except Exception as e:
        # 如果配置加载失败，使用默认prompt
        print(f"Warning: Failed to load validation prompts from config: {e}")
        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是一个专业的答案质量评估专家。请从以下维度评估答案质量：\n"
             "1. 准确性：答案是否基于提供的上下文，没有错误信息\n"
             "2. 完整性：答案是否完整回答了用户问题\n"
             "3. 相关性：答案是否与问题直接相关\n"
             "4. 清晰度：答案是否表达清晰，易于理解\n"
             "5. 结构性：答案是否组织良好，逻辑清晰\n\n"
             "请为每个维度给出1-10的评分，并提供具体的改进建议。\n"
             "输出格式：\n"
             "准确性: [分数] - [评价]\n"
             "完整性: [分数] - [评价]\n"
             "相关性: [分数] - [评价]\n"
             "清晰度: [分数] - [评价]\n"
             "结构性: [分数] - [评价]\n"
             "总体评价: [总结]\n"
             "改进建议: [具体建议]"),
            ("human", 
             "上下文信息：\n{context}\n\n"
             "用户问题：{query}\n\n"
             "生成的答案：{answer}\n\n"
             "请评估这个答案的质量。")
        ])
    
    try:
        response = validator_llm.invoke(
            validation_prompt.format_messages(
                context=context,
                query=query,
                answer=answer
            )
        )
        
        # 解析验证结果
        validation_text = response.content
        return parse_validation_response(validation_text)
        
    except Exception as e:
        return {
            "accuracy": 5.0,
            "completeness": 5.0,
            "relevance": 5.0,
            "clarity": 5.0,
            "structure": 5.0,
            "overall_feedback": f"验证过程出错: {str(e)}",
            "improvement_suggestions": ["请检查系统配置"]
        }


def parse_validation_response(validation_text: str) -> Dict[str, Any]:
    """解析验证响应
    
    Args:
        validation_text: 验证响应文本
        
    Returns:
        解析后的验证结果
    """
    result = {
        "accuracy": 5.0,
        "completeness": 5.0,
        "relevance": 5.0,
        "clarity": 5.0,
        "structure": 5.0,
        "overall_feedback": "",
        "improvement_suggestions": []
    }
    
    lines = validation_text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 解析各维度分数
        if line.startswith('准确性:'):
            result['accuracy'] = extract_score(line)
        elif line.startswith('完整性:'):
            result['completeness'] = extract_score(line)
        elif line.startswith('相关性:'):
            result['relevance'] = extract_score(line)
        elif line.startswith('清晰度:'):
            result['clarity'] = extract_score(line)
        elif line.startswith('结构性:'):
            result['structure'] = extract_score(line)
        elif line.startswith('总体评价:'):
            result['overall_feedback'] = line.replace('总体评价:', '').strip()
        elif line.startswith('改进建议:'):
            suggestions = line.replace('改进建议:', '').strip()
            result['improvement_suggestions'] = [s.strip() for s in suggestions.split('；') if s.strip()]
    
    return result


def extract_score(line: str) -> float:
    """从文本行中提取分数
    
    Args:
        line: 包含分数的文本行
        
    Returns:
        提取的分数
    """
    import re
    
    # 查找数字
    numbers = re.findall(r'\d+(?:\.\d+)?', line)
    
    if numbers:
        try:
            score = float(numbers[0])
            return max(1.0, min(10.0, score))  # 确保分数在1-10范围内
        except ValueError:
            pass
    
    return 5.0  # 默认分数


def calculate_validation_score(validation_results: Dict[str, Any]) -> float:
    """计算综合验证分数
    
    Args:
        validation_results: 验证结果
        
    Returns:
        综合分数 (0-1)
    """
    # 各维度权重
    weights = {
        'accuracy': 0.3,      # 准确性最重要
        'completeness': 0.25, # 完整性次之
        'relevance': 0.2,     # 相关性
        'clarity': 0.15,      # 清晰度
        'structure': 0.1      # 结构性
    }
    
    weighted_score = 0.0
    
    for dimension, weight in weights.items():
        score = validation_results.get(dimension, 5.0)
        weighted_score += (score / 10.0) * weight
    
    return weighted_score


def generate_validation_feedback(validation_results: Dict[str, Any]) -> str:
    """生成验证反馈
    
    Args:
        validation_results: 验证结果
        
    Returns:
        反馈文本
    """
    feedback_parts = []
    
    # 添加总体评价
    overall_feedback = validation_results.get('overall_feedback', '')
    if overall_feedback:
        feedback_parts.append(f"总体评价：{overall_feedback}")
    
    # 添加各维度评分
    dimensions = {
        'accuracy': '准确性',
        'completeness': '完整性',
        'relevance': '相关性',
        'clarity': '清晰度',
        'structure': '结构性'
    }
    
    scores = []
    for key, name in dimensions.items():
        score = validation_results.get(key, 5.0)
        scores.append(f"{name}: {score:.1f}/10")
    
    feedback_parts.append("各维度评分：" + "，".join(scores))
    
    # 添加改进建议
    suggestions = validation_results.get('improvement_suggestions', [])
    if suggestions:
        feedback_parts.append("改进建议：" + "；".join(suggestions))
    
    return "\n".join(feedback_parts)


def simple_answer_validation(state: SelfCorrectiveRAGState) -> Dict[str, Any]:
    """简单的答案验证（后备方案）
    
    Args:
        state: 当前状态
        
    Returns:
        验证结果
    """
    query = state.get("query", "")
    generated_answer = state.get("generated_answer", "")
    
    # 基本检查
    issues = []
    
    if len(generated_answer.strip()) < 20:
        issues.append("答案过于简短")
    
    if "抱歉" in generated_answer and "无法" in generated_answer:
        issues.append("答案表明无法回答问题")
    
    # 关键词匹配检查
    query_words = set(query.lower().split())
    answer_words = set(generated_answer.lower().split())
    
    overlap = len(query_words.intersection(answer_words))
    coverage = overlap / len(query_words) if query_words else 0
    
    if coverage < 0.2:
        issues.append("答案与问题相关性不足")
    
    # 计算简单分数
    base_score = 0.8
    penalty = len(issues) * 0.2
    validation_score = max(0.1, base_score - penalty)
    
    needs_correction = validation_score < state.get("quality_threshold", 0.7)
    
    feedback = "简单验证结果：" + ("；".join(issues) if issues else "基本符合要求")
    
    return {
        "validation_passed": not needs_correction,
        "validation_score": validation_score,
        "validation_feedback": feedback,
        "needs_correction": needs_correction
    }


def perform_answer_correction(
    corrector_llm, query: str, original_answer: str, context: str, 
    validation_feedback: str, validation_details: Dict[str, Any]
) -> str:
    """执行答案纠错
    
    Args:
        corrector_llm: 纠错模型
        query: 用户查询
        original_answer: 原始答案
        context: 相关文档上下文
        validation_feedback: 验证反馈
        validation_details: 验证详情
        
    Returns:
        纠正后的答案
    """
    try:
        # 从配置中获取纠错prompt
        system_prompt = prompt_manager.get_prompt('correction', 'system_prompt')
        user_template = prompt_manager.get_prompt('correction', 'user_template')
        
        correction_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        
    except Exception as e:
        # 如果配置加载失败，使用默认prompt
        print(f"Warning: Failed to load correction prompts from config: {e}")
        correction_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "你是一个专业的答案改进专家。请基于验证反馈改进原始答案。\n"
             "改进原则：\n"
             "1. 保持答案的核心内容和准确性\n"
             "2. 根据反馈改进答案的不足之处\n"
             "3. 确保答案完整回答用户问题\n"
             "4. 改善答案的清晰度和结构\n"
             "5. 只基于提供的上下文信息\n\n"
             "请输出改进后的答案，不要包含解释或元信息。"),
            ("human", 
             "上下文信息：\n{context}\n\n"
             "用户问题：{query}\n\n"
             "原始答案：{original_answer}\n\n"
             "验证反馈：{validation_feedback}\n\n"
             "请基于上述信息改进答案。")
        ])
    
    try:
        response = corrector_llm.invoke(
            correction_prompt.format_messages(
                context=context,
                query=query,
                original_answer=original_answer,
                validation_feedback=validation_feedback
            )
        )
        
        corrected_answer = response.content.strip()
        
        # 确保纠正后的答案不为空
        if not corrected_answer or len(corrected_answer.strip()) < 10:
            return original_answer
        
        return corrected_answer
        
    except Exception as e:
        print(f"答案纠错失败: {str(e)}")
        return original_answer


def evaluate_correction_improvement(
    query: str, original_answer: str, corrected_answer: str, context: str
) -> float:
    """评估纠错改进效果
    
    Args:
        query: 用户查询
        original_answer: 原始答案
        corrected_answer: 纠正后答案
        context: 上下文
        
    Returns:
        改进分数 (0-1)
    """
    # 简单的改进评估
    improvements = 0
    total_checks = 5
    
    # 1. 长度改进
    if len(corrected_answer) > len(original_answer) * 1.1:
        improvements += 1
    
    # 2. 结构改进（检查是否有更多的段落或列表）
    original_structure = original_answer.count('\n') + original_answer.count('•') + original_answer.count('-')
    corrected_structure = corrected_answer.count('\n') + corrected_answer.count('•') + corrected_answer.count('-')
    
    if corrected_structure > original_structure:
        improvements += 1
    
    # 3. 关键词覆盖改进
    query_words = set(query.lower().split())
    
    original_coverage = len(query_words.intersection(set(original_answer.lower().split())))
    corrected_coverage = len(query_words.intersection(set(corrected_answer.lower().split())))
    
    if corrected_coverage > original_coverage:
        improvements += 1
    
    # 4. 上下文利用改进
    context_words = set(context.lower().split())
    
    original_context_use = len(context_words.intersection(set(original_answer.lower().split())))
    corrected_context_use = len(context_words.intersection(set(corrected_answer.lower().split())))
    
    if corrected_context_use > original_context_use:
        improvements += 1
    
    # 5. 完整性改进（检查是否有结论性语句）
    conclusion_words = ['总结', '综上', '因此', '所以', '总的来说', '最后']
    
    original_has_conclusion = any(word in original_answer for word in conclusion_words)
    corrected_has_conclusion = any(word in corrected_answer for word in conclusion_words)
    
    if corrected_has_conclusion and not original_has_conclusion:
        improvements += 1
    
    return improvements / total_checks