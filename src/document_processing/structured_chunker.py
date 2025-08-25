from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

# LangChain imports
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class StructuredChunker:
    """
    结构化分块器 - 基于Markdown标题结构进行文档分块
    直接使用MarkdownHeaderTextSplitter对Markdown内容进行结构化分块，保持表格等格式完整性
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化结构化分块器
        
        Args:
            config: 配置字典
        """
        self.config = config or {}
        
        # 配置参数
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        self.return_each_line = self.config.get('return_each_line', False)
        
        # 设置要分割的标题层级
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
            ("####", "Header 4"),
            ("#####", "Header 5"),
            ("######", "Header 6"),
        ]
        
        # 初始化Markdown分块器
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=self.headers_to_split_on,
            return_each_line=self.return_each_line,
            strip_headers=False  # 保留标题以维持上下文
        )
        
        # 备用分块器，用于处理过长的块
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
    
    def chunk_markdown(self, markdown_content: str, source_path: Optional[str] = None) -> List[Document]:
        """
        对Markdown内容进行结构化分块
        
        Args:
            markdown_content: Markdown内容
            source_path: 源文件路径（用于元数据）
            
        Returns:
            分块后的文档列表
        """
        try:
            # 直接使用Markdown分块器进行结构化分块
            markdown_header_splits = self.markdown_splitter.split_text(markdown_content)
            
            # 处理分块结果
            documents = []
            for i, split in enumerate(markdown_header_splits):
                # 检查块大小，如果过大则进一步分割
                if len(split.page_content) > self.chunk_size * 1.5:
                    # 使用文本分块器进一步分割
                    sub_splits = self.text_splitter.split_text(split.page_content)
                    
                    for j, sub_split in enumerate(sub_splits):
                        # 创建文档对象
                        doc = Document(
                            page_content=sub_split,
                            metadata={
                                **split.metadata,
                                'chunk_index': f"{i}_{j}",
                                'sub_chunk': True,
                                'source': source_path or 'unknown',
                                'chunk_type': 'structured_sub',
                                'char_count': len(sub_split),
                                'word_count': len(sub_split.split())
                            }
                        )
                        documents.append(doc)
                else:
                    # 直接使用原始分块
                    doc = Document(
                        page_content=split.page_content,
                        metadata={
                            **split.metadata,
                            'chunk_index': str(i),
                            'sub_chunk': False,
                            'source': source_path or 'unknown',
                            'chunk_type': 'structured',
                            'char_count': len(split.page_content),
                            'word_count': len(split.page_content.split())
                        }
                    )
                    documents.append(doc)
            
            # 添加总体元数据
            total_chunks = len(documents)
            for i, doc in enumerate(documents):
                doc.metadata.update({
                    'total_chunks': total_chunks,
                    'chunk_position': i + 1
                })
            
            logger.info(f"Markdown结构化分块完成: {len(documents)} 个块")
            return documents
            
        except Exception as e:
            logger.error(f"Markdown结构化分块失败: {str(e)}")
            # 降级到简单文本分块
            return self._fallback_text_chunking(markdown_content, source_path)
    

    
    def _fallback_text_chunking(self, content: str, source_path: Optional[str] = None) -> List[Document]:
        """
        备用文本分块方法
        
        Args:
            content: 文本内容
            source_path: 源文件路径
            
        Returns:
            分块后的文档列表
        """
        logger.info("使用备用文本分块方法")
        
        chunks = self.text_splitter.split_text(content)
        documents = []
        
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    'chunk_index': str(i),
                    'source': source_path or 'unknown',
                    'chunk_type': 'fallback_text',
                    'char_count': len(chunk),
                    'word_count': len(chunk.split()),
                    'total_chunks': len(chunks),
                    'chunk_position': i + 1
                }
            )
            documents.append(doc)
        
        return documents
    
    def get_chunk_statistics(self, documents: List[Document]) -> Dict[str, Any]:
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
        
        stats = {
            'total_chunks': len(documents),
            'total_characters': sum(char_counts),
            'total_words': sum(word_counts),
            'avg_chunk_size': sum(char_counts) / len(char_counts) if char_counts else 0,
            'avg_word_count': sum(word_counts) / len(word_counts) if word_counts else 0,
            'min_chunk_size': min(char_counts) if char_counts else 0,
            'max_chunk_size': max(char_counts) if char_counts else 0,
            'structured_chunks': len([d for d in documents if d.metadata.get('chunk_type') == 'structured']),
            'sub_chunks': len([d for d in documents if d.metadata.get('sub_chunk', False)])
        }
        
        return stats
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        更新配置
        
        Args:
            new_config: 新的配置字典
        """
        self.config.update(new_config)
        
        # 重新初始化分块器
        self.chunk_size = self.config.get('chunk_size', 1000)
        self.chunk_overlap = self.config.get('chunk_overlap', 200)
        self.return_each_line = self.config.get('return_each_line', False)
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        logger.info(f"结构化分块器配置已更新: chunk_size={self.chunk_size}, chunk_overlap={self.chunk_overlap}")