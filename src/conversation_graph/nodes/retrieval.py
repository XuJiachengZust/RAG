"""文档检索和评分节点

负责智能文档检索和相关性评分。
"""

from typing import Dict, Any, List
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from ..state import SelfCorrectiveRAGState, DocumentGrade
from ..utils import create_rules_retriever, get_llm_client, parse_relevance_score
from ..retrieval_manager import calculate_retrieval_quality


def intelligent_retrieval_node(state: SelfCorrectiveRAGState) -> Dict[str, Any]:
    """智能文档检索节点
    
    功能：
    1. 使用多种检索策略
    2. 基于查询复杂度调整检索参数
    3. 计算检索质量分数
    4. 记录检索尝试次数
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态字典
    """
    try:
        query = state.get("query", "")
        keywords = state.get("keywords", [])
        complexity_level = state.get("complexity_level", "medium")
        retrieval_k = state.get("retrieval_k", 5)
        
        if not query:
            return {
                "error_message": "查询为空，无法进行检索",
                "retrieved_documents": [],
                "retrieval_score": 0.0,
                "retrieval_attempts": state.get("retrieval_attempts", 0) + 1
            }
        
        # 创建检索器
        retriever = create_rules_retriever(k=retrieval_k)
        
        if not retriever:
            return {
                "error_message": "无法创建文档检索器",
                "retrieved_documents": [],
                "retrieval_score": 0.0,
                "retrieval_attempts": state.get("retrieval_attempts", 0) + 1
            }
        
        # 执行基础检索
        documents = retriever.invoke(query)
        
        # 如果基础检索结果不足，尝试关键词检索
        if len(documents) < 3 and keywords:
            keyword_query = " ".join(keywords)
            additional_docs = retriever.invoke(keyword_query)
            
            # 合并文档，去重
            seen_content = {doc.page_content for doc in documents}
            for doc in additional_docs:
                if doc.page_content not in seen_content:
                    documents.append(doc)
                    seen_content.add(doc.page_content)
        
        # 根据复杂度进行文档过滤和排序
        filtered_documents = filter_and_rank_documents(
            documents, query, keywords, complexity_level
        )
        
        # 计算检索质量分数
        retrieval_score = calculate_retrieval_quality(query, filtered_documents)
        
        # 记录检索尝试
        retrieval_attempts = state.get("retrieval_attempts", 0) + 1
        
        return {
            "retrieved_documents": filtered_documents,
            "retrieval_score": retrieval_score,
            "retrieval_attempts": retrieval_attempts,
            "retrieval_metadata": {
                "original_doc_count": len(documents),
                "filtered_doc_count": len(filtered_documents),
                "retrieval_strategy": "hybrid" if keywords else "semantic",
                "complexity_level": complexity_level
            }
        }
        
    except Exception as e:
        return {
            "error_message": f"文档检索失败: {str(e)}",
            "retrieved_documents": [],
            "retrieval_score": 0.0,
            "retrieval_attempts": state.get("retrieval_attempts", 0) + 1
        }


def grade_documents_node(state: SelfCorrectiveRAGState) -> Dict[str, Any]:
    """文档相关性评分节点
    
    功能：
    1. 使用LLM评估每个文档的相关性
    2. 计算相关性分数
    3. 过滤不相关文档
    4. 提供评分理由
    
    Args:
        state: 当前状态
        
    Returns:
        更新后的状态字典
    """
    try:
        query = state.get("query", "")
        documents = state.get("retrieved_documents", [])
        
        if not documents:
            return {
                "graded_documents": [],
                "retrieval_score": 0.0,
                "relevant_doc_count": 0
            }
        
        # 获取评分模型
        grader_llm = get_llm_client("gpt-3.5-turbo")
        
        if not grader_llm:
            # 如果无法获取LLM，使用简单的文本匹配评分
            graded_documents = simple_document_grading(query, documents)
        else:
            # 使用LLM进行详细评分
            graded_documents = llm_document_grading(query, documents, grader_llm)
        
        # 计算相关文档数量和平均分数
        relevant_docs = [doc for doc in graded_documents if doc["is_relevant"]]
        relevant_count = len(relevant_docs)
        
        if graded_documents:
            avg_score = sum(doc["relevance_score"] for doc in graded_documents) / len(graded_documents)
        else:
            avg_score = 0.0
        
        return {
            "graded_documents": graded_documents,
            "retrieval_score": avg_score,
            "relevant_doc_count": relevant_count,
            "grading_metadata": {
                "total_documents": len(documents),
                "relevant_documents": relevant_count,
                "relevance_rate": relevant_count / len(documents) if documents else 0,
                "average_score": avg_score
            }
        }
        
    except Exception as e:
        return {
            "error_message": f"文档评分失败: {str(e)}",
            "graded_documents": [],
            "retrieval_score": 0.0,
            "relevant_doc_count": 0
        }


def filter_and_rank_documents(
    documents: List[Document],
    query: str,
    keywords: List[str],
    complexity_level: str
) -> List[Document]:
    """过滤和排序文档
    
    Args:
        documents: 原始文档列表
        query: 查询文本
        keywords: 关键词列表
        complexity_level: 复杂度级别
        
    Returns:
        过滤和排序后的文档列表
    """
    if not documents:
        return []
    
    # 计算每个文档的相关性分数
    scored_docs = []
    
    for doc in documents:
        score = calculate_document_relevance(doc, query, keywords)
        scored_docs.append((doc, score))
    
    # 按分数排序
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 根据复杂度确定返回数量
    max_docs = {
        "simple": 3,
        "medium": 5,
        "complex": 7
    }.get(complexity_level, 5)
    
    # 过滤低分文档
    min_score = 0.3
    filtered_docs = [
        doc for doc, score in scored_docs[:max_docs]
        if score >= min_score
    ]
    
    return filtered_docs


def calculate_document_relevance(
    document: Document,
    query: str,
    keywords: List[str]
) -> float:
    """计算文档相关性分数
    
    Args:
        document: 文档对象
        query: 查询文本
        keywords: 关键词列表
        
    Returns:
        相关性分数 (0-1)
    """
    doc_text = document.page_content.lower()
    query_lower = query.lower()
    
    # 1. 查询词匹配分数
    query_words = set(query_lower.split())
    doc_words = set(doc_text.split())
    
    if query_words:
        query_match_score = len(query_words.intersection(doc_words)) / len(query_words)
    else:
        query_match_score = 0.0
    
    # 2. 关键词匹配分数
    if keywords:
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in doc_text)
        keyword_score = keyword_matches / len(keywords)
    else:
        keyword_score = 0.0
    
    # 3. 文档长度分数（适中长度更好）
    doc_length = len(doc_text.split())
    if 50 <= doc_length <= 500:
        length_score = 1.0
    elif doc_length < 50:
        length_score = doc_length / 50
    else:
        length_score = max(0.5, 500 / doc_length)
    
    # 4. 文档质量分数（基于结构化程度）
    quality_indicators = ['。', '：', '\n', '1.', '2.', '•', '-']
    quality_score = min(1.0, sum(doc_text.count(indicator) for indicator in quality_indicators) / 10)
    
    # 综合评分
    final_score = (
        query_match_score * 0.4 +
        keyword_score * 0.3 +
        length_score * 0.2 +
        quality_score * 0.1
    )
    
    return min(final_score, 1.0)


def simple_document_grading(query: str, documents: List[Document]) -> List[Dict[str, Any]]:
    """简单的文档评分（不使用LLM）
    
    Args:
        query: 查询文本
        documents: 文档列表
        
    Returns:
        评分结果列表
    """
    graded_documents = []
    
    for doc in documents:
        relevance_score = calculate_document_relevance(doc, query, [])
        is_relevant = relevance_score >= 0.5
        
        graded_documents.append({
            "document": doc,
            "relevance_score": relevance_score,
            "is_relevant": is_relevant,
            "reasoning": f"基于文本匹配的相关性分数: {relevance_score:.2f}"
        })
    
    return graded_documents


def llm_document_grading(
    query: str,
    documents: List[Document],
    grader_llm
) -> List[Dict[str, Any]]:
    """使用LLM进行文档评分
    
    Args:
        query: 查询文本
        documents: 文档列表
        grader_llm: 评分LLM
        
    Returns:
        评分结果列表
    """
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", 
         "你是一个文档相关性评估专家。你需要评估给定文档是否满足用户查询意图。\n"
         "请给出0-1之间的相关性分数，并简要说明理由。\n"
         "评分标准：\n"
         "- 0.8-1.0: 高度相关，直接回答查询\n"
         "- 0.6-0.8: 相关，包含有用信息\n"
         "- 0.4-0.6: 部分相关，有一些相关内容\n"
         "- 0.2-0.4: 弱相关，只有少量相关信息\n"
         "- 0.0-0.2: 不相关，与查询无关"),
        ("human", 
         "查询: {query}\n\n"
         "文档内容: {document}\n\n"
         "请评估这个文档是否满足用户查询意图，给出分数（0-1）和理由。")
    ])
    
    graded_documents = []
    
    for doc in documents:
        try:
            # 截断过长的文档
            doc_content = doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content
            
            response = grader_llm.invoke(
                grader_prompt.format_messages(
                    query=query,
                    document=doc_content
                )
            )
            
            # 解析响应
            relevance_score = parse_relevance_score(response.content)
            is_relevant = relevance_score >= 0.6
            
            graded_documents.append({
                "document": doc,
                "relevance_score": relevance_score,
                "is_relevant": is_relevant,
                "reasoning": response.content[:200] + "..." if len(response.content) > 200 else response.content
            })
            
        except Exception as e:
            # 如果LLM评分失败，使用简单评分作为fallback
            relevance_score = calculate_document_relevance(doc, query, [])
            is_relevant = relevance_score >= 0.5
            
            graded_documents.append({
                "document": doc,
                "relevance_score": relevance_score,
                "is_relevant": is_relevant,
                "reasoning": f"LLM评分失败，使用简单评分: {str(e)[:100]}"
            })
    
    return graded_documents


def enhance_retrieval_with_metadata(documents: List[Document]) -> List[Document]:
    """使用元数据增强检索结果
    
    Args:
        documents: 原始文档列表
        
    Returns:
        增强后的文档列表
    """
    enhanced_docs = []
    
    for doc in documents:
        # 添加文档元数据
        metadata = doc.metadata.copy()
        
        # 计算文档统计信息
        content = doc.page_content
        metadata.update({
            "word_count": len(content.split()),
            "char_count": len(content),
            "paragraph_count": content.count('\n\n') + 1,
            "has_structure": any(indicator in content for indicator in ['1.', '2.', '•', '-', '：'])
        })
        
        # 创建增强文档
        enhanced_doc = Document(
            page_content=content,
            metadata=metadata
        )
        
        enhanced_docs.append(enhanced_doc)
    
    return enhanced_docs