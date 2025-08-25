"""共享工具函数模块

包含在LangGraph重构中使用的通用工具函数。
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)

def generate_conversation_id() -> str:
    """生成唯一的对话ID"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    return f"conv_{timestamp}_{unique_id}"

def load_documents_from_directory(directory: str, extensions: List[str] = None) -> List[str]:
    """从目录加载指定扩展名的文档路径
    
    Args:
        directory: 目录路径
        extensions: 文件扩展名列表，如 ['.pdf', '.docx', '.txt']
    
    Returns:
        文档路径列表
    """
    if extensions is None:
        extensions = ['.pdf', '.docx', '.txt']
    
    document_paths = []
    
    if not os.path.exists(directory):
        logger.warning(f"目录不存在: {directory}")
        return document_paths
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_ext = os.path.splitext(file)[1].lower()
            
            if file_ext in extensions:
                document_paths.append(file_path)
                logger.info(f"找到文档: {file_path}")
    
    logger.info(f"从目录 {directory} 加载了 {len(document_paths)} 个文档")
    return document_paths

def save_json_file(data: Dict[Any, Any], file_path: str) -> bool:
    """保存数据到JSON文件
    
    Args:
        data: 要保存的数据
        file_path: 文件路径
    
    Returns:
        是否保存成功
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存到: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"保存JSON文件失败: {file_path} - {str(e)}")
        return False

def load_json_file(file_path: str) -> Optional[Dict[Any, Any]]:
    """从JSON文件加载数据
    
    Args:
        file_path: 文件路径
    
    Returns:
        加载的数据，如果失败返回None
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"文件不存在: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"数据已从文件加载: {file_path}")
        return data
        
    except Exception as e:
        logger.error(f"加载JSON文件失败: {file_path} - {str(e)}")
        return None

def setup_logging(log_level: str = "INFO", log_file: str = None) -> None:
    """设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径（可选）
    """
    # 配置日志格式
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # 设置日志级别
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # 配置根日志记录器
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[
            logging.StreamHandler()  # 控制台输出
        ]
    )
    
    # 如果指定了日志文件，添加文件处理器
    if log_file:
        # 确保日志目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        
        logger.info(f"日志将保存到文件: {log_file}")

def validate_file_exists(file_path: str) -> bool:
    """验证文件是否存在
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件是否存在
    """
    exists = os.path.exists(file_path)
    if not exists:
        logger.warning(f"文件不存在: {file_path}")
    return exists

def get_file_size(file_path: str) -> int:
    """获取文件大小（字节）
    
    Args:
        file_path: 文件路径
    
    Returns:
        文件大小，如果文件不存在返回0
    """
    try:
        return os.path.getsize(file_path)
    except OSError:
        logger.warning(f"无法获取文件大小: {file_path}")
        return 0

def format_file_size(size_bytes: int) -> str:
    """格式化文件大小显示
    
    Args:
        size_bytes: 文件大小（字节）
    
    Returns:
        格式化的文件大小字符串
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"

def clean_text(text: str) -> str:
    """清理文本内容
    
    Args:
        text: 原始文本
    
    Returns:
        清理后的文本
    """
    if not text:
        return ""
    
    # 移除多余的空白字符
    text = ' '.join(text.split())
    
    # 移除特殊字符（保留基本标点）
    import re
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()\[\]{}"\'-]', '', text)
    
    return text.strip()

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本到指定长度
    
    Args:
        text: 原始文本
        max_length: 最大长度
        suffix: 截断后缀
    
    Returns:
        截断后的文本
    """
    if not text or len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def merge_dicts(*dicts: Dict[Any, Any]) -> Dict[Any, Any]:
    """合并多个字典
    
    Args:
        *dicts: 要合并的字典
    
    Returns:
        合并后的字典
    """
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result

def safe_get(data: Dict[Any, Any], key: str, default: Any = None) -> Any:
    """安全获取字典值
    
    Args:
        data: 字典数据
        key: 键名
        default: 默认值
    
    Returns:
        字典值或默认值
    """
    try:
        return data.get(key, default)
    except (AttributeError, TypeError):
        return default