"""Prompt管理模块

提供集中化的prompt模板管理功能，包括加载、验证、缓存和动态获取prompt模板。
"""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

class PromptManager:
    """Prompt模板管理器
    
    负责加载、管理和提供各个节点的prompt模板。
    支持动态加载、缓存和模板验证。
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """初始化Prompt管理器
        
        Args:
            config_path: prompt配置文件路径，默认为项目根目录下的config/prompts.json
        """
        if config_path is None:
            # 获取项目根目录
            current_dir = Path(__file__).parent
            project_root = current_dir.parent.parent
            config_path = project_root / "config" / "prompts.json"
        
        self.config_path = Path(config_path)
        self._prompts_cache: Optional[Dict[str, Any]] = None
        self._load_prompts()
    
    def _load_prompts(self) -> None:
        """加载prompt配置文件"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Prompt配置文件不存在: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self._prompts_cache = json.load(f)
            
            self._validate_config()
            logger.info(f"成功加载prompt配置: {self.config_path}")
            
        except Exception as e:
            logger.error(f"加载prompt配置失败: {e}")
            raise
    
    def _validate_config(self) -> None:
        """验证配置文件格式"""
        if not self._prompts_cache:
            raise ValueError("Prompt配置为空")
        
        required_sections = ['generation', 'validation', 'correction', 'rewrite', 'grading']
        prompts = self._prompts_cache.get('prompts', {})
        
        for section in required_sections:
            if section not in prompts:
                raise ValueError(f"缺少必需的prompt配置节: {section}")
        
        logger.debug("Prompt配置验证通过")
    
    def reload_prompts(self) -> None:
        """重新加载prompt配置"""
        self._prompts_cache = None
        self._load_prompts()
        logger.info("Prompt配置已重新加载")
    
    def get_prompt(self, node_type: str, prompt_key: str, **kwargs) -> str:
        """获取指定节点的prompt模板
        
        Args:
            node_type: 节点类型 (generation, validation, correction, rewrite, grading)
            prompt_key: prompt键名
            **kwargs: 模板参数，用于格式化prompt
        
        Returns:
            格式化后的prompt字符串
        
        Raises:
            KeyError: 当指定的节点类型或prompt键不存在时
            ValueError: 当模板参数不匹配时
        """
        try:
            prompts = self._prompts_cache['prompts']
            
            if node_type not in prompts:
                raise KeyError(f"未找到节点类型: {node_type}")
            
            node_prompts = prompts[node_type]
            
            # 支持嵌套键访问，如 'intent_prompts.definition'
            if '.' in prompt_key:
                keys = prompt_key.split('.')
                prompt_template = node_prompts
                for key in keys:
                    if key not in prompt_template:
                        raise KeyError(f"未找到prompt键: {prompt_key} (在 {node_type} 中)")
                    prompt_template = prompt_template[key]
            else:
                if prompt_key not in node_prompts:
                    raise KeyError(f"未找到prompt键: {prompt_key} (在 {node_type} 中)")
                prompt_template = node_prompts[prompt_key]
            
            # 格式化模板
            if kwargs:
                try:
                    return prompt_template.format(**kwargs)
                except KeyError as e:
                    raise ValueError(f"模板参数缺失: {e}")
            
            return prompt_template
            
        except Exception as e:
            logger.error(f"获取prompt失败 - 节点: {node_type}, 键: {prompt_key}, 错误: {e}")
            raise
    
    def get_generation_prompt(self, intent: str = 'general', complexity: str = 'medium', **kwargs) -> str:
        """获取答案生成的完整prompt
        
        Args:
            intent: 查询意图
            complexity: 复杂度级别
            **kwargs: 模板参数 (context, question)
        
        Returns:
            完整的生成prompt
        """
        base_prompt = self.get_prompt('generation', 'base_system_prompt')
        intent_prompt = self.get_prompt('generation', f'intent_prompts.{intent}')
        complexity_prompt = self.get_prompt('generation', f'complexity_prompts.{complexity}')
        
        system_prompt = f"{base_prompt}\n\n{intent_prompt}\n{complexity_prompt}"
        
        if kwargs:
            user_prompt = self.get_prompt('generation', 'user_prompt_template', **kwargs)
            return f"System: {system_prompt}\n\nUser: {user_prompt}"
        
        return system_prompt
    
    def get_validation_prompt(self, **kwargs) -> str:
        """获取答案验证的完整prompt
        
        Args:
            **kwargs: 模板参数 (question, context, answer)
        
        Returns:
            完整的验证prompt
        """
        system_prompt = self.get_prompt('validation', 'system_prompt')
        
        if kwargs:
            user_prompt = self.get_prompt('validation', 'user_prompt_template', **kwargs)
            return f"System: {system_prompt}\n\nUser: {user_prompt}"
        
        return system_prompt
    
    def get_correction_prompt(self, **kwargs) -> str:
        """获取答案纠错的完整prompt
        
        Args:
            **kwargs: 模板参数 (question, context, original_answer, validation_feedback)
        
        Returns:
            完整的纠错prompt
        """
        system_prompt = self.get_prompt('correction', 'system_prompt')
        
        if kwargs:
            user_prompt = self.get_prompt('correction', 'user_prompt_template', **kwargs)
            return f"System: {system_prompt}\n\nUser: {user_prompt}"
        
        return system_prompt
    
    def get_rewrite_prompt(self, strategy: str = 'rephrase', intent: str = 'general', **kwargs) -> str:
        """获取查询重写的完整prompt
        
        Args:
            strategy: 重写策略
            intent: 查询意图
            **kwargs: 模板参数 (original_query)
        
        Returns:
            完整的重写prompt
        """
        base_prompt = self.get_prompt('rewrite', 'base_system_prompt')
        strategy_prompt = self.get_prompt('rewrite', f'strategy_prompts.{strategy}')
        intent_prompt = self.get_prompt('rewrite', f'intent_prompts.{intent}')
        output_instruction = self.get_prompt('rewrite', 'output_instruction')
        
        system_prompt = f"{base_prompt}\n\n{strategy_prompt}\n{intent_prompt}\n\n{output_instruction}"
        
        if kwargs:
            user_prompt = self.get_prompt('rewrite', 'user_prompt_template', **kwargs)
            return f"System: {system_prompt}\n\nUser: {user_prompt}"
        
        return system_prompt
    
    def get_grading_prompt(self, **kwargs) -> str:
        """获取文档评分的完整prompt
        
        Args:
            **kwargs: 模板参数 (query, document)
        
        Returns:
            完整的评分prompt
        """
        system_prompt = self.get_prompt('grading', 'system_prompt')
        
        if kwargs:
            user_prompt = self.get_prompt('grading', 'user_prompt_template', **kwargs)
            return f"System: {system_prompt}\n\nUser: {user_prompt}"
        
        return system_prompt
    
    def list_available_prompts(self) -> Dict[str, list]:
        """列出所有可用的prompt模板
        
        Returns:
            按节点类型分组的prompt键列表
        """
        result = {}
        prompts = self._prompts_cache['prompts']
        
        for node_type, node_prompts in prompts.items():
            result[node_type] = list(node_prompts.keys())
        
        return result
    
    def get_config_info(self) -> Dict[str, Any]:
        """获取配置信息
        
        Returns:
            配置元数据信息
        """
        return {
            'version': self._prompts_cache.get('version', 'unknown'),
            'description': self._prompts_cache.get('description', ''),
            'config_path': str(self.config_path),
            'metadata': self._prompts_cache.get('metadata', {})
        }


# 全局单例实例
_prompt_manager_instance: Optional[PromptManager] = None

def get_prompt_manager() -> PromptManager:
    """获取全局PromptManager实例
    
    Returns:
        PromptManager实例
    """
    global _prompt_manager_instance
    
    if _prompt_manager_instance is None:
        _prompt_manager_instance = PromptManager()
    
    return _prompt_manager_instance

def reload_prompt_manager() -> None:
    """重新加载全局PromptManager实例"""
    global _prompt_manager_instance
    
    if _prompt_manager_instance is not None:
        _prompt_manager_instance.reload_prompts()
    else:
        _prompt_manager_instance = PromptManager()


# 创建全局实例
prompt_manager = get_prompt_manager()