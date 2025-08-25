import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
from .metadata_extractor import ChunkMetadata, DocumentContent

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class EnhancedChunkMetadata(ChunkMetadata):
    """
    简化的分块元数据，包含基本的上下文信息
    """
    # 继承基础元数据字段
    # document_name: str
    # page_number: Optional[int] = None
    # start_page: Optional[int] = None
    # end_page: Optional[int] = None
    # title: Optional[str] = None
    # title_level: Optional[int] = None
    # title_path: Optional[List[str]] = None
    
    # 简化的新增字段
    chunk_index: Optional[int] = None  # 分块在文档中的索引
    total_chunks: Optional[int] = None  # 文档总分块数
    word_count: Optional[int] = None  # 词数统计
    char_count: Optional[int] = None  # 字符数统计
    
class MetadataEnhancer:
    """
    元数据增强器
    负责增强分块的元数据信息，包括标题层级关联、页码范围确定等
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化元数据增强器
        
        Args:
            config: 配置参数
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 简化的配置参数
        self.enable_word_count = self.config.get('enable_word_count', True)
    
    def enhance_chunk_metadata(
        self, 
        chunk_text: str, 
        base_metadata: ChunkMetadata,
        document_content: DocumentContent,
        chunk_index: int,
        total_chunks: int
    ) -> EnhancedChunkMetadata:
        """
        增强单个分块的元数据
        
        Args:
            chunk_text: 分块文本内容
            base_metadata: 基础元数据
            document_content: 文档内容结构
            chunk_index: 分块索引
            total_chunks: 总分块数
            
        Returns:
            EnhancedChunkMetadata: 增强后的元数据
        """
        try:
            # 创建简化的元数据对象
            enhanced_metadata = EnhancedChunkMetadata(
                document_name=base_metadata.document_name,
                page_number=base_metadata.page_number,
                start_page=base_metadata.start_page,
                end_page=base_metadata.end_page,
                title=base_metadata.title,
                title_level=base_metadata.title_level,
                title_path=base_metadata.title_path or [],
                chunk_index=chunk_index,
                total_chunks=total_chunks
            )
            
            # 添加基本的词数统计
            if self.enable_word_count:
                enhanced_metadata.word_count = self._count_words(chunk_text)
                enhanced_metadata.char_count = len(chunk_text)
            
            return enhanced_metadata
        except Exception as e:
            self.logger.error(f"元数据增强失败: {str(e)}")
            # 返回基础元数据
            return EnhancedChunkMetadata(
                document_name=base_metadata.document_name,
                page_number=base_metadata.page_number,
                start_page=base_metadata.start_page,
                end_page=base_metadata.end_page,
                title=base_metadata.title,
                title_level=base_metadata.title_level,
                title_path=base_metadata.title_path or [],
                chunk_index=chunk_index,
                total_chunks=total_chunks,
                word_count=len(chunk_text.split()) if chunk_text else 0,
                char_count=len(chunk_text) if chunk_text else 0
            )
    
    def _count_words(self, text: str) -> int:
        """
        统计文本中的词数
        
        Args:
            text: 输入文本
            
        Returns:
            int: 词数
        """
        if not text:
            return 0
        
        # 简单的词数统计
        words = text.split()
        return len(words)
    
    def validate_metadata(self, metadata: EnhancedChunkMetadata) -> bool:
        """
        验证元数据的有效性
        
        Args:
            metadata: 元数据对象
            
        Returns:
            bool: 元数据是否有效
        """
        try:
            # 基本验证
            if not metadata.document_name:
                return False
            
            # 检查数值字段的合理性
            if metadata.chunk_index is not None and metadata.chunk_index < 0:
                return False
                
            if metadata.total_chunks is not None and metadata.total_chunks < 1:
                return False
                
            if metadata.word_count is not None and metadata.word_count < 0:
                return False
                
            if metadata.char_count is not None and metadata.char_count < 0:
                return False
            
            return True
        except Exception:
            return False
    
    def to_dict(self, metadata: EnhancedChunkMetadata) -> Dict[str, Any]:
        """
        将元数据转换为字典格式
        
        Args:
            metadata: 元数据对象
            
        Returns:
            Dict[str, Any]: 字典格式的元数据
        """
        from dataclasses import asdict
        return asdict(metadata)