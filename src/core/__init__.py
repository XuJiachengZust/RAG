"""核心模块

提供配置管理、文档加载等核心功能。
"""

from .config_manager import ConfigManager, get_config_manager, get_config, get_config_section

# 临时的DocumentLoader类，用于兼容现有导入
class DocumentLoader:
    """文档加载器占位符类
    
    这是一个临时类，用于兼容现有的导入语句。
    实际的文档加载功能在document_processing模块中实现。
    """
    
    def __init__(self):
        pass
    
    def load_document(self, file_path: str):
        """加载文档的占位符方法"""
        raise NotImplementedError("DocumentLoader功能已迁移到document_processing模块")

__all__ = [
    'ConfigManager',
    'DocumentLoader',
    'get_config_manager',
    'get_config',
    'get_config_section'
]