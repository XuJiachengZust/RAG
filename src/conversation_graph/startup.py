"""项目启动模块

负责在项目启动时执行必要的初始化操作，包括文档加载。
"""

import os
import sys
import logging
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.config_manager import ConfigManager
from src.core.document_initializer import initialize_documents, get_document_initializer

logger = logging.getLogger(__name__)

class StartupManager:
    """启动管理器
    
    负责协调项目启动时的各种初始化操作。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化启动管理器
        
        Args:
            config: 启动配置参数
        """
        self.config_manager = ConfigManager()
        self.config = config or self._get_default_startup_config()
        
        # 设置日志
        self._setup_logging()
        
        # 启动统计
        self.startup_stats = {
            'start_time': None,
            'end_time': None,
            'duration': 0.0,
            'document_stats': {},
            'errors': [],
            'success': False
        }
    
    def _get_default_startup_config(self) -> Dict[str, Any]:
        """获取默认启动配置"""
        return {
            'initialize_documents': True,  # 是否初始化文档
            'document_config': {
                'knowledge_docs_path': self.config_manager.get('paths.knowledge_docs_path', './rules'),
                'force_reload': False,  # 是否强制重新加载
                'batch_size': 10
            },
            'log_level': self.config_manager.get('logging.level', 'INFO'),
            'async_initialization': False,  # 是否异步初始化
            'timeout': 300,  # 初始化超时时间（秒）
        }
    
    def _setup_logging(self):
        """设置日志配置"""
        log_level = getattr(logging, self.config.get('log_level', 'INFO').upper())
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('startup.log', encoding='utf-8')
            ]
        )
    
    def startup(self) -> Dict[str, Any]:
        """执行启动流程
        
        Returns:
            启动结果统计
        """
        import time
        
        self.startup_stats['start_time'] = time.time()
        
        try:
            logger.info("开始项目启动流程...")
            
            # 1. 验证环境和配置
            self._validate_environment()
            
            # 2. 初始化文档库
            if self.config.get('initialize_documents', True):
                self._initialize_documents()
            
            # 3. 其他初始化操作可以在这里添加
            # self._initialize_other_components()
            
            self.startup_stats['success'] = True
            logger.info("项目启动完成")
            
        except Exception as e:
            error_msg = f"项目启动失败: {str(e)}"
            logger.error(error_msg)
            self.startup_stats['errors'].append(error_msg)
            self.startup_stats['success'] = False
        
        finally:
            self.startup_stats['end_time'] = time.time()
            self.startup_stats['duration'] = (
                self.startup_stats['end_time'] - self.startup_stats['start_time']
            )
            
            # 输出启动摘要
            self._print_startup_summary()
        
        return self.startup_stats
    
    async def async_startup(self) -> Dict[str, Any]:
        """异步执行启动流程
        
        Returns:
            启动结果统计
        """
        import time
        
        self.startup_stats['start_time'] = time.time()
        
        try:
            logger.info("开始异步项目启动流程...")
            
            # 1. 验证环境和配置
            self._validate_environment()
            
            # 2. 异步初始化文档库
            if self.config.get('initialize_documents', True):
                await self._async_initialize_documents()
            
            self.startup_stats['success'] = True
            logger.info("异步项目启动完成")
            
        except Exception as e:
            error_msg = f"异步项目启动失败: {str(e)}"
            logger.error(error_msg)
            self.startup_stats['errors'].append(error_msg)
            self.startup_stats['success'] = False
        
        finally:
            self.startup_stats['end_time'] = time.time()
            self.startup_stats['duration'] = (
                self.startup_stats['end_time'] - self.startup_stats['start_time']
            )
            
            # 输出启动摘要
            self._print_startup_summary()
        
        return self.startup_stats
    
    def _validate_environment(self):
        """验证环境和配置"""
        logger.info("验证环境和配置...")
        
        # 检查必要的环境变量
        required_env_vars = ['OPENAI_API_KEY']
        missing_vars = []
        
        for var in required_env_vars:
            if not os.getenv(var) and not self.config_manager.get(f'api.{var.lower()}'):
                missing_vars.append(var)
        
        if missing_vars:
            raise EnvironmentError(f"缺少必要的环境变量: {', '.join(missing_vars)}")
        
        # 检查必要的目录
        rules_dir = Path(self.config['document_config']['knowledge_docs_path'])
        if not rules_dir.exists():
            logger.warning(f"文档目录不存在: {rules_dir}，将创建该目录")
            rules_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("环境验证完成")
    
    def _initialize_documents(self):
        """初始化文档库"""
        logger.info("开始初始化文档库...")
        
        try:
            # 使用文档初始化器
            document_config = self.config.get('document_config', {})
            stats = initialize_documents(document_config)
            
            self.startup_stats['document_stats'] = stats
            
            if stats.get('errors'):
                logger.warning(f"文档初始化过程中出现 {len(stats['errors'])} 个错误")
                for error in stats['errors']:
                    logger.warning(f"  - {error}")
            
            logger.info(
                f"文档库初始化完成: 处理 {stats.get('processed_files', 0)} 个文件，"
                f"生成 {stats.get('total_chunks', 0)} 个分块"
            )
            
        except Exception as e:
            error_msg = f"文档库初始化失败: {str(e)}"
            logger.error(error_msg)
            self.startup_stats['errors'].append(error_msg)
            raise
    
    async def _async_initialize_documents(self):
        """异步初始化文档库"""
        logger.info("开始异步初始化文档库...")
        
        try:
            # 在线程池中运行文档初始化
            loop = asyncio.get_event_loop()
            document_config = self.config.get('document_config', {})
            
            # 设置超时
            timeout = self.config.get('timeout', 300)
            
            stats = await asyncio.wait_for(
                loop.run_in_executor(None, initialize_documents, document_config),
                timeout=timeout
            )
            
            self.startup_stats['document_stats'] = stats
            
            if stats.get('errors'):
                logger.warning(f"异步文档初始化过程中出现 {len(stats['errors'])} 个错误")
                for error in stats['errors']:
                    logger.warning(f"  - {error}")
            
            logger.info(
                f"异步文档库初始化完成: 处理 {stats.get('processed_files', 0)} 个文件，"
                f"生成 {stats.get('total_chunks', 0)} 个分块"
            )
            
        except asyncio.TimeoutError:
            error_msg = f"文档库初始化超时（{self.config.get('timeout', 300)}秒）"
            logger.error(error_msg)
            self.startup_stats['errors'].append(error_msg)
            raise
        except Exception as e:
            error_msg = f"异步文档库初始化失败: {str(e)}"
            logger.error(error_msg)
            self.startup_stats['errors'].append(error_msg)
            raise
    
    def _print_startup_summary(self):
        """打印启动摘要"""
        print("\n" + "="*60)
        print("项目启动摘要")
        print("="*60)
        
        print(f"启动状态: {'成功' if self.startup_stats['success'] else '失败'}")
        print(f"启动耗时: {self.startup_stats['duration']:.2f} 秒")
        
        # 文档统计
        doc_stats = self.startup_stats.get('document_stats', {})
        if doc_stats:
            print("\n文档库统计:")
            print(f"  - 总文件数: {doc_stats.get('total_files', 0)}")
            print(f"  - 处理成功: {doc_stats.get('processed_files', 0)}")
            print(f"  - 处理失败: {doc_stats.get('failed_files', 0)}")
            print(f"  - 生成分块: {doc_stats.get('total_chunks', 0)}")
            print(f"  - 处理耗时: {doc_stats.get('processing_time', 0):.2f} 秒")
        
        # 错误信息
        if self.startup_stats['errors']:
            print(f"\n错误信息 ({len(self.startup_stats['errors'])} 个):")
            for i, error in enumerate(self.startup_stats['errors'], 1):
                print(f"  {i}. {error}")
        
        print("="*60)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取启动统计信息"""
        return self.startup_stats.copy()


# 全局启动管理器实例
_startup_manager = None

def get_startup_manager(config: Optional[Dict[str, Any]] = None) -> StartupManager:
    """获取全局启动管理器实例"""
    global _startup_manager
    if _startup_manager is None:
        _startup_manager = StartupManager(config)
    return _startup_manager

def startup_application(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """快捷方式：启动应用程序"""
    manager = get_startup_manager(config)
    return manager.startup()

async def async_startup_application(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """快捷方式：异步启动应用程序"""
    manager = get_startup_manager(config)
    return await manager.async_startup()

def main():
    """主函数，用于直接运行启动脚本"""
    import argparse
    
    parser = argparse.ArgumentParser(description='项目启动脚本')
    parser.add_argument('--async', action='store_true', help='使用异步启动')
    parser.add_argument('--force-reload', action='store_true', help='强制重新加载文档')
    parser.add_argument('--rules-dir', type=str, help='文档目录路径')
    parser.add_argument('--log-level', type=str, default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 构建配置
    config = {
        'log_level': args.log_level,
        'async_initialization': getattr(args, 'async'),
        'document_config': {
            'force_reload': args.force_reload,
            'knowledge_docs_path': args.rules_dir if args.rules_dir else './rules'
        }
    }
    
    # 执行启动
    if getattr(args, 'async'):
        stats = asyncio.run(async_startup_application(config))
    else:
        stats = startup_application(config)
    
    # 退出码
    exit_code = 0 if stats['success'] else 1
    sys.exit(exit_code)

if __name__ == '__main__':
    main()