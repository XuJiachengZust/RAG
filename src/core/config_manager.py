import json
import os
import re
from typing import Any, Dict, Optional
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


class ConfigManager:
    """配置管理器，负责加载和管理应用程序配置"""
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化配置管理器
        
        Args:
            config_path: 配置文件路径，默认为项目根目录下的config.json
        """
        if config_path is None:
            # 默认配置文件路径
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.json"
        
        self.config_path = Path(config_path)
        
        # 加载.env文件
        self._load_env_file()
        
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        default_config = self._get_default_config()
        
        if not self.config_path.exists():
            # 如果配置文件不存在，创建默认配置文件
            self._save_config(default_config)
            return default_config
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                file_config = json.load(f)
            
            # 合并默认配置和文件配置
            merged_config = self._merge_configs(default_config, file_config)
            
            # 解析环境变量占位符
            resolved_config = self._resolve_env_variables(merged_config)
            return resolved_config
            
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"配置文件加载失败: {e}，使用默认配置")
            return default_config
    
    def _load_env_file(self):
        """加载.env文件"""
        try:
            # 查找项目根目录下的.env文件
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / ".env"
            
            if env_file.exists():
                if load_dotenv is not None:
                    # 使用python-dotenv加载.env文件
                    load_dotenv(env_file)
                    print(f"已加载环境变量文件: {env_file}")
                else:
                    # 手动解析.env文件
                    self._manual_load_env(env_file)
                    print(f"已手动加载环境变量文件: {env_file}")
            else:
                print(f"未找到.env文件: {env_file}")
                
        except Exception as e:
            print(f"加载.env文件失败: {e}")
    
    def _manual_load_env(self, env_file: Path):
        """手动加载.env文件（当python-dotenv不可用时）"""
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        # 移除引号
                        if value.startswith('"') and value.endswith('"'):
                            value = value[1:-1]
                        elif value.startswith("'") and value.endswith("'"):
                            value = value[1:-1]
                        os.environ[key] = value
        except Exception as e:
            print(f"手动加载.env文件失败: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "api": {
                "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
                "openai_base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                "model_name": "gpt-4o-mini",
                "temperature": 0.7,
                "max_tokens": 4000
            },
            "embedding": {
                "model_name": "text-embedding-ada-002",
                "chunk_size": 1000,
                "chunk_overlap": 200
            },
            "paths": {
                "knowledge_docs_path": "./rules",
                "vector_store_path": "./vector_store",
                "logs_path": "./logs"
            },
            "chunking": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "separator": "\n\n"
            },
            "semantic_chunking": {
                "breakpoint_type": "percentile",
                "breakpoint_threshold": 95,
                "number_of_chunks": None,
                "buffer_size": 1,
                "sentence_split_regex": r"[.!?]+\s+"
            },
            "conversion": {
                "preserve_tables": True,
                "preserve_headers": True,
                "include_metadata": True,
                "max_table_size": 5000
            },
            "retrieval": {
                "top_k": "${RETRIEVAL_K:5}",
                "similarity_threshold": "${SCORE_THRESHOLD:0.7}",
                "search_type": "${SEARCH_TYPE:similarity}",
                "max_retrieval_attempts": "${MAX_RETRIEVAL_ATTEMPTS:3}"
            },
            "generation": {
                "max_generation_attempts": 2,
                "min_answer_length": 50,
                "quality_threshold": 0.6
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file_enabled": True,
                "console_enabled": True
            },
            "rag": {
                "enable_self_correction": True,
                "max_retry_attempts": 3,
                "relevance_threshold": 0.7,
                "enable_query_rewriting": True
            },
            "node_models": {
                "generation_model": os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
                "validation_model": os.getenv("VALIDATION_MODEL", "gpt-4o-mini"),
                "correction_model": os.getenv("CORRECTION_MODEL", "gpt-4o-mini"),
                "rewrite_model": os.getenv("REWRITE_MODEL", "gpt-3.5-turbo"),
                "grading_model": os.getenv("GRADING_MODEL", "gpt-3.5-turbo")
            }
        }
    
    def _merge_configs(self, default: Dict[str, Any], file_config: Dict[str, Any]) -> Dict[str, Any]:
        """递归合并配置字典"""
        result = default.copy()
        
        for key, value in file_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result

    def _resolve_env_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """递归解析配置中的环境变量占位符
        
        支持格式：
        - ${VAR_NAME} - 必需的环境变量
        - ${VAR_NAME:default_value} - 可选的环境变量，带默认值
        """
        if isinstance(config, dict):
            return {key: self._resolve_env_variables(value) for key, value in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_variables(item) for item in config]
        elif isinstance(config, str):
            return self._parse_env_variable(config)
        else:
            return config
    
    def _parse_env_variable(self, value: str) -> Any:
        """解析单个环境变量占位符
        
        Args:
            value: 可能包含环境变量占位符的字符串
            
        Returns:
            解析后的值
        """
        # 匹配环境变量格式 ${VAR_NAME:default_value} 或 ${VAR_NAME}
        pattern = r'\$\{([^:}]+)(?::([^}]+))?\}'
        
        def replace_env_var(match):
            env_var = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else None
            
            env_value = os.getenv(env_var)
            if env_value is not None:
                return env_value
            elif default_val is not None:
                return default_val
            else:
                # 如果没有默认值且环境变量不存在，保持原样
                return match.group(0)
        
        # 如果整个字符串就是一个环境变量占位符，尝试转换类型
        if re.match(r'^\$\{[^}]+\}$', value):
            resolved = re.sub(pattern, replace_env_var, value)
            # 尝试转换为数字类型
            if resolved != value:  # 如果有替换发生
                try:
                    # 尝试转换为整数
                    if resolved.isdigit() or (resolved.startswith('-') and resolved[1:].isdigit()):
                        return int(resolved)
                    # 尝试转换为浮点数
                    return float(resolved)
                except ValueError:
                    # 如果转换失败，返回字符串
                    return resolved
            return resolved
        else:
            # 部分替换，返回字符串
            return re.sub(pattern, replace_env_var, value)

    def _save_config(self, config: Dict[str, Any]) -> None:
        """保存配置到文件"""
        try:
            # 确保目录存在
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"配置文件保存失败: {e}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值
        
        Args:
            key_path: 配置键路径，使用点号分隔，如 'api.openai_api_key'
            default: 默认值
        
        Returns:
            配置值或默认值
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key_path: str, value: Any) -> None:
        """设置配置值
        
        Args:
            key_path: 配置键路径，使用点号分隔
            value: 要设置的值
        """
        keys = key_path.split('.')
        config = self.config
        
        # 导航到目标位置
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        # 设置值
        config[keys[-1]] = value
        
        # 保存配置
        self._save_config(self.config)
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """获取配置段
        
        Args:
            section: 配置段名称
        
        Returns:
            配置段字典
        """
        return self.config.get(section, {})
    
    def get_config(self) -> Dict[str, Any]:
        """获取完整配置
        
        Returns:
            完整配置字典
        """
        return self.config.copy()
    
    def get_semantic_config(self) -> Dict[str, Any]:
        """获取语义分块配置"""
        semantic_config = self.get_section('semantic_chunking')
        return {
            'breakpoint_threshold_type': semantic_config.get('breakpoint_type', 'percentile'),
            'breakpoint_threshold_amount': semantic_config.get('breakpoint_threshold', 95),
            'number_of_chunks': semantic_config.get('number_of_chunks', None),
            'buffer_size': semantic_config.get('buffer_size', 1),
            'sentence_split_regex': semantic_config.get('sentence_split_regex', r'[.!?]+\s+')
        }
    
    def get_conversion_config(self) -> Dict[str, Any]:
        """获取文档转换配置"""
        conversion_config = self.get_section('conversion')
        return {
            'preserve_tables': conversion_config.get('preserve_tables', True),
            'preserve_headers': conversion_config.get('preserve_headers', True),
            'include_metadata': conversion_config.get('include_metadata', True),
            'max_table_size': conversion_config.get('max_table_size', 5000)
        }
    
    def reload(self) -> None:
        """重新加载配置文件"""
        self.config = self._load_config()


# 全局配置管理器实例
_config_manager = None

def get_config_manager() -> ConfigManager:
    """获取全局配置管理器实例"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config(key_path: str, default: Any = None) -> Any:
    """快捷方式：获取配置值"""
    return get_config_manager().get(key_path, default)

def get_config_section(section: str) -> Dict[str, Any]:
    """快捷方式：获取配置段"""
    return get_config_manager().get_section(section)