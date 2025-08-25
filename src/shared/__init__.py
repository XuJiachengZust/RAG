"""共享组件模块

包含在LangGraph重构中使用的共享组件和工具函数。
"""

from .utils import (
    generate_conversation_id,
    load_documents_from_directory,
    save_json_file,
    load_json_file,
    setup_logging,
    validate_file_exists,
    get_file_size,
    format_file_size,
    clean_text,
    truncate_text,
    merge_dicts,
    safe_get
)

__version__ = "1.0.0"

__all__ = [
    "generate_conversation_id",
    "load_documents_from_directory",
    "save_json_file",
    "load_json_file",
    "setup_logging",
    "validate_file_exists",
    "get_file_size",
    "format_file_size",
    "clean_text",
    "truncate_text",
    "merge_dicts",
    "safe_get"
]