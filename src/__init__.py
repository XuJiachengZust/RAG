"""智能文档处理系统

该系统提供文档加载、处理、分块和元数据提取等功能。
"""

# 核心模块
from .core import ConfigManager, DocumentLoader

# 文档处理器
from .processors import excel_processor

# 元数据处理
from .metadata import (
    MetadataExtractor,
    ChunkMetadata,
    DocumentContent,
    MetadataEnhancer,
    EnhancedChunkMetadata,
    MetadataAwareSemanticChunker
)

# 文档加载器工厂
# LoaderFactory已被新的DocumentConverter替代

__all__ = [
    # 核心模块
    'ConfigManager',
    'DocumentLoader',
    
    # 文档处理器
    # 'EnhancedDocumentProcessor', # 已被新的文档处理逻辑替代
    'ExcelProcessor',
    
    # 元数据处理
    'MetadataExtractor',
    'ChunkMetadata',
    'DocumentContent',
    'MetadataEnhancer',
    'EnhancedChunkMetadata',
    'MetadataAwareSemanticChunker',
    
    # 加载器工厂
    # 'LoaderFactory' # 已被DocumentConverter替代
]