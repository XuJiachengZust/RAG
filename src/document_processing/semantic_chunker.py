from typing import Dict, List, Optional, Any, Union
import logging
from pathlib import Path

# LangChain imports
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import HTMLHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# 本地导入
from .structured_chunker import StructuredChunker

logger = logging.getLogger(__name__)

class HybridSemanticChunker:
    """
    混合语义分块器 - 支持多种文件类型的智能分块策略
    - PDF/Word: 转换为Markdown → 结构化分块 → 语义分块
    - HTML: 直接使用LangChain原生HTML分块器
    - 其他: 纯语义分块
    """
    
    def __init__(self, embeddings: Embeddings, config: Optional[Dict[str, Any]] = None):
        """
        初始化混合语义分块器
        
        Args:
            embeddings: 嵌入模型
            config: 配置字典
        """
        self.embeddings = embeddings
        self.config = config or {}
        
        # 配置参数 - 确保数值类型正确
        self.breakpoint_threshold_type = self.config.get('breakpoint_threshold_type', 'percentile')
        self.breakpoint_threshold_amount = float(self.config.get('breakpoint_threshold_amount', 95.0))
        self.min_chunk_size = int(self.config.get('min_chunk_size', 100))
        self.max_chunk_size = int(self.config.get('max_chunk_size', 2000))
        self.enable_structured_chunking = self.config.get('enable_structured_chunking', True)
        self.semantic_chunk_threshold = int(self.config.get('semantic_chunk_threshold', 500))
        
        # 初始化结构化分块器
        if self.enable_structured_chunking:
            self.structured_chunker = StructuredChunker(config)
        else:
            self.structured_chunker = None
        
        # 初始化语义分块器
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount
        )
        
        # 初始化HTML分块器
        self.html_chunker = HTMLHeaderTextSplitter(
            headers_to_split_on=[
                ("h1", "Header 1"),
                ("h2", "Header 2"),
                ("h3", "Header 3"),
                ("h4", "Header 4"),
                ("h5", "Header 5"),
                ("h6", "Header 6"),
            ],
            return_each_element=False
        )
        
        # 初始化备用文本分块器
        self.text_chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=int(self.max_chunk_size * 0.1),
            length_function=len
        )
    
    def chunk_documents(self, content: str, source_path: Optional[str] = None, content_type: str = 'markdown') -> List[Document]:
        """
        对文档进行智能分块，根据文件类型选择最佳分块策略
        
        Args:
            content: 文档内容（可以是Markdown、HTML等）
            source_path: 源文件路径
            content_type: 内容类型 ('markdown', 'html', 'text')
            
        Returns:
            分块后的文档列表
        """
        # 根据源文件路径推断文件类型
        if source_path:
            file_ext = Path(source_path).suffix.lower()
            if file_ext == '.html' or file_ext == '.htm':
                content_type = 'html'
            elif file_ext in ['.docx', '.pdf']:
                content_type = 'markdown'  # 这些文件已转换为Markdown
            elif content_type == 'markdown':
                content_type = 'markdown'
            else:
                content_type = 'text'
        
        # 根据内容类型选择分块策略
        if content_type == 'html':
            return self._chunk_html_content(content, source_path)
        elif content_type == 'markdown':
            return self._chunk_markdown_content(content, source_path)
        else:
            return self._chunk_text_content(content, source_path)
    
    def _chunk_html_content(self, html_content: str, source_path: Optional[str] = None) -> List[Document]:
        """
        使用LangChain原生HTML分块器处理HTML内容
        
        Args:
            html_content: HTML内容
            source_path: 源文件路径
            
        Returns:
            分块后的文档列表
        """
        try:
            # 使用HTML分块器进行分块
            html_docs = self.html_chunker.split_text(html_content)
            
            documents = []
            for i, doc in enumerate(html_docs):
                # 检查块大小，如果过大则进一步分割
                if len(doc.page_content) > self.max_chunk_size:
                    sub_chunks = self.text_chunker.split_text(doc.page_content)
                    for j, sub_chunk in enumerate(sub_chunks):
                        sub_doc = Document(
                            page_content=sub_chunk,
                            metadata={
                                **doc.metadata,
                                'chunk_index': f"{i}_{j}",
                                'source': source_path or 'unknown',
                                'chunk_method': 'html_native_sub',
                                'chunk_type': 'html_sub',
                                'char_count': len(sub_chunk),
                                'word_count': len(sub_chunk.split()),
                                'parent_chunk_index': str(i)
                            }
                        )
                        documents.append(sub_doc)
                else:
                    # 直接使用HTML分块结果
                    doc.metadata.update({
                        'chunk_index': str(i),
                        'source': source_path or 'unknown',
                        'chunk_method': 'html_native',
                        'chunk_type': 'html',
                        'char_count': len(doc.page_content),
                        'word_count': len(doc.page_content.split())
                    })
                    documents.append(doc)
            
            logger.info(f"HTML原生分块完成: {len(documents)} 个块")
            return documents
            
        except Exception as e:
            logger.error(f"HTML分块失败: {str(e)}")
            # 降级到文本分块
            return self._chunk_text_content(html_content, source_path)
    
    def _chunk_markdown_content(self, markdown_content: str, source_path: Optional[str] = None) -> List[Document]:
        """
        处理Markdown内容：结构化分块 + 语义分块
        
        Args:
            markdown_content: Markdown内容
            source_path: 源文件路径
            
        Returns:
            分块后的文档列表
        """
        try:
            if self.enable_structured_chunking and self.structured_chunker:
                # 先进行结构化分块
                structured_docs = self.structured_chunker.chunk_markdown(markdown_content, source_path)
                
                # 对每个结构化块进行语义分块
                final_documents = []
                
                for struct_doc in structured_docs:
                    # 检查块大小是否需要语义分块
                    if len(struct_doc.page_content) > self.semantic_chunk_threshold:
                        semantic_docs = self._apply_semantic_chunking(struct_doc)
                        final_documents.extend(semantic_docs)
                    else:
                        # 块足够小，直接使用
                        struct_doc.metadata['chunk_method'] = 'structured_only'
                        final_documents.append(struct_doc)
                
                logger.info(f"Markdown混合分块完成: {len(final_documents)} 个块 (从 {len(structured_docs)} 个结构化块)")
                return final_documents
            
            else:
                # 仅使用语义分块
                return self._chunk_text_content(markdown_content, source_path)
                
        except Exception as e:
            logger.error(f"Markdown分块失败: {str(e)}")
            # 降级到纯语义分块
            return self._chunk_text_content(markdown_content, source_path)
    
    def _chunk_text_content(self, content: str, source_path: Optional[str] = None) -> List[Document]:
        """
        纯语义分块（用于普通文本或降级处理）
        
        Args:
            content: 文本内容
            source_path: 源文件路径
            
        Returns:
            分块后的文档列表
        """
        logger.info("使用纯语义分块")
        
        try:
            chunks = self.semantic_chunker.split_text(content)
            documents = []
            
            for i, chunk in enumerate(chunks):
                # 检查块大小
                if len(chunk) < self.min_chunk_size and documents:
                    # 块太小，与前一个块合并
                    if len(documents[-1].page_content) + len(chunk) <= self.max_chunk_size:
                        documents[-1].page_content += "\n\n" + chunk
                        documents[-1].metadata['char_count'] = len(documents[-1].page_content)
                        documents[-1].metadata['word_count'] = len(documents[-1].page_content.split())
                        continue
                
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'chunk_index': str(i),
                        'source': source_path or 'unknown',
                        'chunk_method': 'semantic_only',
                        'chunk_type': 'semantic',
                        'char_count': len(chunk),
                        'word_count': len(chunk.split()),
                        'is_semantic_chunk': True
                    }
                )
                documents.append(doc)
            
            logger.info(f"纯语义分块完成: {len(documents)} 个块")
            return documents
            
        except Exception as e:
            logger.error(f"语义分块失败: {str(e)}")
            # 最后的降级方案：使用文本分块器
            return self._fallback_text_chunking(content, source_path)
    
    def _apply_semantic_chunking(self, document: Document) -> List[Document]:
        """
        对单个文档应用语义分块
        
        Args:
            document: 要分块的文档
            
        Returns:
            语义分块后的文档列表
        """
        try:
            # 使用语义分块器分割文档内容
            semantic_chunks = self.semantic_chunker.split_text(document.page_content)
            
            semantic_docs = []
            for i, chunk in enumerate(semantic_chunks):
                # 检查块大小
                if len(chunk) < self.min_chunk_size:
                    # 块太小，尝试与前一个块合并
                    if semantic_docs and len(semantic_docs[-1].page_content) + len(chunk) <= self.max_chunk_size:
                        semantic_docs[-1].page_content += "\n\n" + chunk
                        semantic_docs[-1].metadata['char_count'] = len(semantic_docs[-1].page_content)
                        semantic_docs[-1].metadata['word_count'] = len(semantic_docs[-1].page_content.split())
                        continue
                
                # 创建新的语义块文档
                semantic_doc = Document(
                    page_content=chunk,
                    metadata={
                        **document.metadata,  # 继承原始元数据
                        'semantic_chunk_index': i,
                        'parent_chunk_index': document.metadata.get('chunk_index', 'unknown'),
                        'chunk_method': 'structured_semantic',
                        'char_count': len(chunk),
                        'word_count': len(chunk.split()),
                        'is_semantic_chunk': True
                    }
                )
                semantic_docs.append(semantic_doc)
            
            # 更新语义块的总数信息
            for i, doc in enumerate(semantic_docs):
                doc.metadata.update({
                    'semantic_total_chunks': len(semantic_docs),
                    'semantic_chunk_position': i + 1
                })
            
            return semantic_docs
            
        except Exception as e:
            logger.warning(f"语义分块失败，保持原始块: {str(e)}")
            document.metadata['chunk_method'] = 'structured_fallback'
            return [document]
    
    def _fallback_text_chunking(self, content: str, source_path: Optional[str] = None) -> List[Document]:
        """
        降级文本分块（当其他分块方法失败时使用）
        
        Args:
            content: 文本内容
            source_path: 源文件路径
            
        Returns:
            分块后的文档列表
        """
        logger.info("使用降级文本分块")
        
        try:
            chunks = self.text_chunker.split_text(content)
            documents = []
            
            for i, chunk in enumerate(chunks):
                # 检查块大小
                if len(chunk) < self.min_chunk_size and documents:
                    # 块太小，与前一个块合并
                    if len(documents[-1].page_content) + len(chunk) <= self.max_chunk_size:
                        documents[-1].page_content += "\n\n" + chunk
                        documents[-1].metadata['char_count'] = len(documents[-1].page_content)
                        documents[-1].metadata['word_count'] = len(documents[-1].page_content.split())
                        continue
                
                doc = Document(
                    page_content=chunk,
                    metadata={
                        'chunk_index': str(i),
                        'source': source_path or 'unknown',
                        'chunk_method': 'text_fallback',
                        'char_count': len(chunk),
                        'word_count': len(chunk.split()),
                        'is_semantic_chunk': False
                    }
                )
                documents.append(doc)
            
            # 更新总数信息
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'total_chunks': len(documents),
                    'chunk_position': i + 1
                })
            
            return documents
            
        except Exception as e:
            logger.error(f"降级文本分块失败: {str(e)}")
            raise
    
    def get_chunking_statistics(self, documents: List[Document]) -> Dict[str, Any]:
        """
        获取分块统计信息
        
        Args:
            documents: 文档列表
            
        Returns:
            统计信息字典
        """
        if not documents:
            return {}
        
        char_counts = [doc.metadata.get('char_count', 0) for doc in documents]
        word_counts = [doc.metadata.get('word_count', 0) for doc in documents]
        
        # 统计不同分块方法的数量
        method_counts = {}
        for doc in documents:
            method = doc.metadata.get('chunk_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        stats = {
            'total_chunks': len(documents),
            'total_characters': sum(char_counts),
            'total_words': sum(word_counts),
            'avg_chunk_size': sum(char_counts) / len(char_counts) if char_counts else 0,
            'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
            'min_chunk_size': min(char_counts) if char_counts else 0,
            'max_chunk_size': max(char_counts) if char_counts else 0,
            'chunking_methods': method_counts,
            'semantic_chunks': len([d for d in documents if d.metadata.get('is_semantic_chunk', False)]),
            'structured_chunks': len([d for d in documents if 'structured' in d.metadata.get('chunk_method', '')])
        }
        
        return stats
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        
        # 更新参数 - 确保数值类型正确
        self.breakpoint_threshold_type = self.config.get('breakpoint_threshold_type', 'percentile')
        self.breakpoint_threshold_amount = float(self.config.get('breakpoint_threshold_amount', 95.0))
        self.min_chunk_size = int(self.config.get('min_chunk_size', 100))
        self.max_chunk_size = int(self.config.get('max_chunk_size', 2000))
        self.enable_structured_chunking = self.config.get('enable_structured_chunking', True)
        self.semantic_chunk_threshold = int(self.config.get('semantic_chunk_threshold', 500))
        
        # 重新初始化语义分块器
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount
        )
        
        # 更新结构化分块器配置
        if self.structured_chunker:
            self.structured_chunker.update_config(new_config)
        
        logger.info(f"混合语义分块器配置已更新")
    
    def set_embeddings(self, embeddings: Embeddings) -> None:
        """
        更新嵌入模型
        
        Args:
            embeddings: 新的嵌入模型
        """
        self.embeddings = embeddings
        
        # 重新初始化语义分块器
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount
        )
        
        logger.info("嵌入模型已更新")