from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """
    分块元数据基础类
    """
    chunk_index: int = 0
    total_chunks: int = 0
    document_name: Optional[str] = None
    page_number: Optional[int] = None
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    title: Optional[str] = None
    title_level: Optional[int] = None
    title_path: Optional[List[str]] = None
    line_number: Optional[int] = None
    heading_level: Optional[int] = None
    heading_text: Optional[str] = None
    content_type: Optional[str] = None
    word_count: int = 0
    char_count: int = 0
    confidence_score: float = 1.0

@dataclass
class DocumentContent:
    """
    文档内容结构
    """
    text: str  # 文档文本内容
    page_mapping: Dict[int, int]  # 行号到页码的映射
    title_hierarchy: List[Dict[str, Any]]  # 标题层级信息
    document_name: str  # 文档名称
    file_path: str  # 文件路径
    total_pages: int  # 总页数

class MetadataExtractor(ABC):
    """
    元数据提取器抽象基类
    定义统一的元数据提取接口
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def extract_document_content(self, file_path: str) -> DocumentContent:
        """
        提取文档内容和基础元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            DocumentContent: 文档内容结构
        """
        pass
    
    @abstractmethod
    def convert_to_markdown(self, file_path: str, output_dir: Optional[str] = None) -> str:
        """
        将文档转换为Markdown格式
        
        Args:
            file_path: 文件路径
            output_dir: 输出目录（可选）
            
        Returns:
            str: Markdown内容
        """
        pass
    
    def extract_title_hierarchy(self, text: str) -> List[Dict[str, Any]]:
        """
        提取文档的标题层级结构
        
        Args:
            text: 文档文本
            
        Returns:
            List[Dict]: 标题层级信息列表
        """
        title_hierarchy = []
        lines = text.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line_stripped = line.strip()
            if not line_stripped:
                continue
                
            title_info = self._identify_title(line_stripped)
            if title_info:
                level, title_number, title_text = title_info
                title_hierarchy.append({
                    'level': level,
                    'title_number': title_number,
                    'title_text': title_text,
                    'full_title': line_stripped,
                    'line_number': line_num
                })
        
        return title_hierarchy
    
    def _identify_title(self, text: str) -> Optional[Tuple[int, str, str]]:
        """
        识别标题级别和内容
        
        Args:
            text: 文本行
            
        Returns:
            Optional[Tuple]: (级别, 编号, 标题文本) 或 None
        """
        # 标题识别模式
        title_patterns = {
            1: [  # 一级标题
                r'^(\d+)、\s*([^：:]+(?:(?!：|:)[^\n])*)$',  # 1、标题
                r'^(\d+)\.\s*([^：:]+(?:(?!：|:)[^\n])*)$',  # 1. 标题
                r'^(第[一二三四五六七八九十\d]+章)\s*([^：:]+(?:(?!：|:)[^\n])*)$',  # 第一章 标题
                r'^([一二三四五六七八九十])、\s*([^：:]+(?:(?!：|:)[^\n])*)$',  # 一、标题
            ],
            2: [  # 二级标题
                r'^(\d+\.\d+)、\s*([^：:]+(?:(?!：|:)[^\n])*)$',  # 1.1、标题
                r'^(\d+\.\d+)\.\s*([^：:]+(?:(?!：|:)[^\n])*)$',  # 1.1. 标题
                r'^(\d+\.\d+)\s+([^：:]+(?:(?!：|:)[^\n])*)$',  # 1.1 标题
            ],
            3: [  # 三级标题
                r'^(\d+\.\d+\.\d+)、\s*([^：:]+(?:(?!：|:)[^\n])*)$',  # 1.1.1、标题
                r'^(\d+\.\d+\.\d+)\.\s*([^：:]+(?:(?!：|:)[^\n])*)$',  # 1.1.1. 标题
                r'^(\d+\.\d+\.\d+)\s+([^：:]+(?:(?!：|:)[^\n])*)$',  # 1.1.1 标题
            ]
        }
        
        # Markdown标题模式
        markdown_patterns = {
            1: r'^#\s+(.+)$',      # # 标题
            2: r'^##\s+(.+)$',     # ## 标题
            3: r'^###\s+(.+)$',    # ### 标题
            4: r'^####\s+(.+)$',   # #### 标题
            5: r'^#####\s+(.+)$',  # ##### 标题
            6: r'^######\s+(.+)$'  # ###### 标题
        }
        
        # 检查Markdown标题
        for level, pattern in markdown_patterns.items():
            import re
            match = re.match(pattern, text)
            if match:
                title_text = match.group(1).strip()
                return (level, '', title_text)
        
        # 检查编号标题
        for level, patterns in title_patterns.items():
            for pattern in patterns:
                import re
                match = re.match(pattern, text)
                if match:
                    title_number = match.group(1)
                    title_text = match.group(2).strip()
                    return (level, title_number, title_text)
        
        return None
    
    def get_document_name(self, file_path: str) -> str:
        """
        从文件路径提取文档名称
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文档名称（不含扩展名）
        """
        return os.path.splitext(os.path.basename(file_path))[0]
    
    def validate_file_exists(self, file_path: str) -> bool:
        """
        验证文件是否存在
        
        Args:
            file_path: 文件路径
            
        Returns:
            bool: 文件是否存在
        """
        if not os.path.exists(file_path):
            self.logger.error(f"文件不存在: {file_path}")
            return False
        return True