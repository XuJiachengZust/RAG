"""查询重写节点

负责在检索效果不佳时重写用户查询以改善检索结果。
"""

from typing import Dict, Any, List
from langchain_core.prompts import ChatPromptTemplate

from ..state import SelfCorrectiveRAGState
from ..utils import get_llm_client, extract_keywords, clean_text
from ...core.prompt_manager import prompt_manager


def rewrite_query_node(state: SelfCorrectiveRAGState) -> Dict[str, Any]:
    """查询重写节点
    
    功能：
    1. 分析原查询的检索失败原因
    2. 生成改进的查询版本
    3. 扩展查询关键词
    4. 调整查询策略
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态字典
    """
    try:
        original_query = state.get("query", "")
        retrieval_attempts = state.get("retrieval_attempts", 0)
        graded_documents = state.get("graded_documents", [])
        query_intent = state.get("query_intent", "general")
        complexity_level = state.get("complexity_level", "medium")
        
        if not original_query:
            return {
                "error_message": "原始查询为空，无法重写",
                "rewritten_query": original_query,
                "rewrite_strategy": "none"
            }
        
        # 分析检索失败原因
        failure_analysis = analyze_retrieval_failure(state)
        
        # 获取查询重写模型
        rewriter_llm = get_llm_client(node_type="rewrite")
        
        if not rewriter_llm:
            # 使用简单重写策略作为后备
            return simple_query_rewrite(state, failure_analysis)
        
        # 根据失败原因选择重写策略
        rewrite_strategy = select_rewrite_strategy(failure_analysis, retrieval_attempts)
        
        # 执行查询重写
        rewritten_query = perform_query_rewrite(
            rewriter_llm, original_query, rewrite_strategy, failure_analysis, 
            query_intent, complexity_level
        )
        
        # 验证重写效果
        rewrite_quality = evaluate_rewrite_quality(original_query, rewritten_query)
        
        # 如果重写质量不佳，使用备选策略
        if rewrite_quality < 0.3:
            rewritten_query = fallback_query_rewrite(original_query, rewrite_strategy)
        
        return {
            "rewritten_query": rewritten_query,
            "original_query": original_query,
            "rewrite_strategy": rewrite_strategy,
            "rewrite_quality": rewrite_quality,
            "failure_analysis": failure_analysis,
            "query": rewritten_query,  # 更新当前查询
            "retrieval_attempts": retrieval_attempts + 1
        }
        
    except Exception as e:
        return {
            "error_message": f"查询重写失败: {str(e)}",
            "rewritten_query": state.get("query", ""),
            "rewrite_strategy": "error",
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1
        }


def analyze_retrieval_failure(state: SelfCorrectiveRAGState) -> Dict[str, Any]:
    """分析检索失败原因
    
    Args:
        state: 当前状态
        
    Returns:
        失败分析结果
    """
    analysis = {
        "no_documents_found": False,
        "low_relevance_scores": False,
        "insufficient_keywords": False,
        "too_specific": False,
        "too_general": False,
        "domain_mismatch": False,
        "language_issues": False
    }
    
    query = state.get("query", "")
    graded_documents = state.get("graded_documents", [])
    retrieved_documents = state.get("retrieved_documents", [])
    
    # 检查是否找到文档
    if not retrieved_documents:
        analysis["no_documents_found"] = True
    
    # 检查相关性分数
    if graded_documents:
        relevant_count = sum(1 for item in graded_documents if item.get("is_relevant", False))
        if relevant_count == 0:
            analysis["low_relevance_scores"] = True
    
    # 检查查询特征
    query_words = query.split()
    
    # 关键词不足
    if len(query_words) < 3:
        analysis["insufficient_keywords"] = True
    
    # 过于具体（包含很多专业术语或特定名词）
    if len(query_words) > 10 or any(len(word) > 15 for word in query_words):
        analysis["too_specific"] = True
    
    # 过于宽泛（只有通用词汇）
    common_words = {'什么', '如何', '怎么', '为什么', '哪些', '介绍', '说明', '解释'}
    if len(set(query_words).intersection(common_words)) / len(query_words) > 0.5:
        analysis["too_general"] = True
    
    # 语言问题（混合中英文等）
    has_chinese = any('\u4e00' <= char <= '\u9fff' for char in query)
    has_english = any(char.isalpha() and ord(char) < 128 for char in query)
    
    if has_chinese and has_english:
        analysis["language_issues"] = True
    
    return analysis


def select_rewrite_strategy(failure_analysis: Dict[str, Any], retrieval_attempts: int) -> str:
    """选择重写策略
    
    Args:
        failure_analysis: 失败分析结果
        retrieval_attempts: 检索尝试次数
        
    Returns:
        重写策略名称
    """
    # 根据失败原因和尝试次数选择策略
    if failure_analysis["no_documents_found"]:
        if retrieval_attempts == 1:
            return "expand_keywords"
        else:
            return "generalize_query"
    
    elif failure_analysis["low_relevance_scores"]:
        if retrieval_attempts == 1:
            return "rephrase_query"
        else:
            return "decompose_query"
    
    elif failure_analysis["insufficient_keywords"]:
        return "expand_keywords"
    
    elif failure_analysis["too_specific"]:
        return "generalize_query"
    
    elif failure_analysis["too_general"]:
        return "add_specificity"
    
    elif failure_analysis["language_issues"]:
        return "normalize_language"
    
    else:
        # 默认策略，根据尝试次数递进
        strategies = ["rephrase_query", "expand_keywords", "generalize_query"]
        return strategies[min(retrieval_attempts - 1, len(strategies) - 1)]


def perform_query_rewrite(
    rewriter_llm, original_query: str, strategy: str, failure_analysis: Dict[str, Any],
    query_intent: str, complexity_level: str
) -> str:
    """执行查询重写
    
    Args:
        rewriter_llm: 重写模型
        original_query: 原始查询
        strategy: 重写策略
        failure_analysis: 失败分析
        query_intent: 查询意图
        complexity_level: 复杂度级别
        
    Returns:
        重写后的查询
    """
    # 构建重写提示
    rewrite_prompt = create_rewrite_prompt(strategy, query_intent)
    
    # 准备上下文信息
    context_info = {
        "original_query": original_query,
        "strategy": strategy,
        "intent": query_intent,
        "complexity": complexity_level,
        "failure_reasons": [k for k, v in failure_analysis.items() if v]
    }
    
    try:
        response = rewriter_llm.invoke(
            rewrite_prompt.format_messages(**context_info)
        )
        
        rewritten_query = response.content.strip()
        
        # 后处理重写结果
        rewritten_query = post_process_rewritten_query(rewritten_query, original_query)
        
        return rewritten_query
        
    except Exception as e:
        print(f"LLM查询重写失败: {str(e)}")
        return fallback_query_rewrite(original_query, strategy)


def create_rewrite_prompt(strategy: str, query_intent: str) -> ChatPromptTemplate:
    """创建重写提示模板
    
    Args:
        strategy: 重写策略
        query_intent: 查询意图
        
    Returns:
        提示模板
    """
    try:
        # 从配置中获取重写prompt
        base_system_prompt = prompt_manager.get_prompt('rewrite', 'base_system_prompt')
        strategy_prompt = prompt_manager.get_prompt('rewrite', f'strategy_prompts.{strategy}')
        intent_prompt = prompt_manager.get_prompt('rewrite', f'intent_prompts.{query_intent}')
        output_instruction = prompt_manager.get_prompt('rewrite', 'output_instruction')
        user_template = prompt_manager.get_prompt('rewrite', 'user_template')
        
        # 组合系统提示
        system_prompt = base_system_prompt + strategy_prompt + intent_prompt + output_instruction
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", user_template)
        ])
        
    except Exception as e:
        # 如果配置加载失败，使用默认prompt
        print(f"Warning: Failed to load rewrite prompts from config: {e}")
        base_system_prompt = (
            "你是一个专业的查询优化专家，擅长改写用户查询以提高信息检索效果。\n"
            "请根据指定的策略重写用户查询，确保重写后的查询能够更好地匹配相关文档。\n\n"
            "重写原则：\n"
            "1. 保持原始查询的核心意图\n"
            "2. 使用更准确和具体的词汇\n"
            "3. 避免过于复杂或模糊的表达\n"
            "4. 考虑同义词和相关术语\n"
            "5. 确保查询在目标领域内有意义"
        )
        
        # 策略特定的指导
        strategy_prompts = {
            "expand_keywords": (
                "\n\n当前策略：扩展关键词\n"
                "- 添加相关的同义词和近义词\n"
                "- 包含更多描述性词汇\n"
                "- 考虑不同的表达方式\n"
                "- 保持查询的核心主题"
            ),
            "rephrase_query": (
                "\n\n当前策略：重新表述\n"
                "- 使用不同的句式结构\n"
                "- 替换关键词为更常用的表达\n"
                "- 简化复杂的表述\n"
                "- 使查询更加自然和直接"
            ),
            "generalize_query": (
                "\n\n当前策略：泛化查询\n"
                "- 移除过于具体的限定词\n"
                "- 使用更通用的概念\n"
                "- 扩大查询的覆盖范围\n"
                "- 关注核心主题而非细节"
            ),
            "add_specificity": (
                "\n\n当前策略：增加具体性\n"
                "- 添加更具体的描述词\n"
                "- 明确查询的具体方面\n"
                "- 包含相关的技术术语\n"
                "- 缩小查询范围以提高精确度"
            ),
            "decompose_query": (
                "\n\n当前策略：分解查询\n"
                "- 将复杂查询分解为核心部分\n"
                "- 专注于最重要的信息需求\n"
                "- 简化查询结构\n"
                "- 突出主要关键词"
            ),
            "normalize_language": (
                "\n\n当前策略：语言规范化\n"
                "- 统一语言使用（中文或英文）\n"
                "- 规范术语表达\n"
                "- 修正语法和拼写\n"
                "- 使用标准的表达方式"
            )
        }
        
        # 意图特定的指导
        intent_prompts = {
            "definition": "\n\n查询意图：定义类问题\n重写时确保查询明确要求定义或解释。",
            "how_to": "\n\n查询意图：操作指导\n重写时确保查询明确要求步骤或方法。",
            "comparison": "\n\n查询意图：比较分析\n重写时确保查询明确要求比较不同选项。",
            "best_practice": "\n\n查询意图：最佳实践\n重写时确保查询明确要求推荐做法。",
            "troubleshooting": "\n\n查询意图：问题解决\n重写时确保查询明确描述问题和解决需求。",
            "list": "\n\n查询意图：列表枚举\n重写时确保查询明确要求列举或枚举。",
            "explanation": "\n\n查询意图：深入解释\n重写时确保查询明确要求详细解释。"
        }
        
        # 组合系统提示
        system_prompt = base_system_prompt
        system_prompt += strategy_prompts.get(strategy, "")
        system_prompt += intent_prompts.get(query_intent, "")
        system_prompt += "\n\n请只输出重写后的查询，不要包含解释或其他内容。"
        
        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", 
             "原始查询：{original_query}\n"
             "重写策略：{strategy}\n"
             "查询意图：{intent}\n"
             "复杂度：{complexity}\n"
             "检索失败原因：{failure_reasons}\n\n"
             "请重写这个查询。")
        ])


def simple_query_rewrite(state: SelfCorrectiveRAGState, failure_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """简单查询重写（后备方案）
    
    Args:
        state: 当前状态
        failure_analysis: 失败分析
        
    Returns:
        重写结果
    """
    original_query = state.get("query", "")
    retrieval_attempts = state.get("retrieval_attempts", 0)
    
    # 选择简单重写策略
    if failure_analysis["insufficient_keywords"]:
        strategy = "expand_keywords"
        rewritten_query = expand_query_keywords(original_query)
    elif failure_analysis["too_specific"]:
        strategy = "generalize"
        rewritten_query = generalize_query(original_query)
    elif failure_analysis["too_general"]:
        strategy = "add_specificity"
        rewritten_query = add_query_specificity(original_query)
    else:
        strategy = "rephrase"
        rewritten_query = simple_rephrase(original_query)
    
    return {
        "rewritten_query": rewritten_query,
        "original_query": original_query,
        "rewrite_strategy": f"simple_{strategy}",
        "rewrite_quality": 0.5,  # 中等质量
        "failure_analysis": failure_analysis,
        "query": rewritten_query,
        "retrieval_attempts": retrieval_attempts + 1
    }


def expand_query_keywords(query: str) -> str:
    """扩展查询关键词"""
    # 添加常见的同义词和相关词
    expansions = {
        "方法": "方法 方式 步骤 流程",
        "问题": "问题 故障 错误 异常",
        "配置": "配置 设置 参数 选项",
        "安装": "安装 部署 配置 设置",
        "使用": "使用 操作 应用 执行",
        "优化": "优化 改进 提升 增强",
        "管理": "管理 控制 维护 监控"
    }
    
    expanded_query = query
    for key, expansion in expansions.items():
        if key in query:
            expanded_query = expanded_query.replace(key, expansion)
            break
    
    return expanded_query


def generalize_query(query: str) -> str:
    """泛化查询"""
    # 移除过于具体的修饰词
    specific_words = ['具体', '详细', '完整', '全面', '深入', '专业']
    
    words = query.split()
    filtered_words = [word for word in words if word not in specific_words]
    
    return ' '.join(filtered_words) if filtered_words else query


def add_query_specificity(query: str) -> str:
    """增加查询具体性"""
    # 添加常见的具体化词汇
    if "如何" in query or "怎么" in query:
        return f"{query} 详细步骤"
    elif "什么" in query:
        return f"{query} 定义 概念"
    elif "为什么" in query:
        return f"{query} 原因 解释"
    else:
        return f"{query} 详细信息"


def simple_rephrase(query: str) -> str:
    """简单重新表述"""
    # 简单的同义词替换
    replacements = {
        "如何": "怎样",
        "怎么": "如何",
        "什么是": "什么叫",
        "为什么": "为何",
        "哪些": "什么",
        "方法": "方式",
        "步骤": "流程"
    }
    
    rephrased = query
    for old, new in replacements.items():
        if old in rephrased:
            rephrased = rephrased.replace(old, new, 1)
            break
    
    return rephrased


def fallback_query_rewrite(original_query: str, strategy: str) -> str:
    """后备查询重写
    
    Args:
        original_query: 原始查询
        strategy: 重写策略
        
    Returns:
        重写后的查询
    """
    if strategy == "expand_keywords":
        return expand_query_keywords(original_query)
    elif strategy == "generalize_query":
        return generalize_query(original_query)
    elif strategy == "add_specificity":
        return add_query_specificity(original_query)
    else:
        return simple_rephrase(original_query)


def post_process_rewritten_query(rewritten_query: str, original_query: str) -> str:
    """后处理重写的查询
    
    Args:
        rewritten_query: 重写后的查询
        original_query: 原始查询
        
    Returns:
        处理后的查询
    """
    # 清理查询
    processed = clean_text(rewritten_query)
    
    # 确保查询不为空
    if not processed or len(processed.strip()) < 3:
        return original_query
    
    # 限制查询长度
    if len(processed) > 200:
        processed = processed[:200].rsplit(' ', 1)[0]
    
    # 移除重复的词汇
    words = processed.split()
    unique_words = []
    seen = set()
    
    for word in words:
        if word.lower() not in seen:
            unique_words.append(word)
            seen.add(word.lower())
    
    return ' '.join(unique_words)


def evaluate_rewrite_quality(original_query: str, rewritten_query: str) -> float:
    """评估重写质量
    
    Args:
        original_query: 原始查询
        rewritten_query: 重写后的查询
        
    Returns:
        质量分数 (0-1)
    """
    if not rewritten_query or rewritten_query == original_query:
        return 0.0
    
    score = 0.0
    
    # 1. 长度合理性 (0.2)
    length_ratio = len(rewritten_query) / len(original_query) if original_query else 1
    if 0.8 <= length_ratio <= 2.0:
        score += 0.2
    
    # 2. 关键词保留 (0.3)
    original_words = set(original_query.lower().split())
    rewritten_words = set(rewritten_query.lower().split())
    
    if original_words:
        keyword_retention = len(original_words.intersection(rewritten_words)) / len(original_words)
        score += keyword_retention * 0.3
    
    # 3. 新词汇添加 (0.2)
    new_words = rewritten_words - original_words
    if new_words:
        score += min(len(new_words) / 5, 1.0) * 0.2
    
    # 4. 语言流畅性 (0.3)
    # 简单检查：没有重复词汇，有合理的结构
    words = rewritten_query.split()
    unique_words = set(words)
    
    if len(unique_words) == len(words):  # 没有重复
        score += 0.15
    
    if any(char in rewritten_query for char in '，。？！'):  # 有标点符号
        score += 0.15
    
    return min(score, 1.0)