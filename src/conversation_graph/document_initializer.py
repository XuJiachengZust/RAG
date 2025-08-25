"""文档初始化模块

在项目启动时自动扫描rules目录下的文档，
将Word和PDF文档转换为Markdown，然后进行结构化和语义分块。
高内聚的文档处理逻辑，专注于文档初始化。
"""

import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.core.config_manager import ConfigManager
from src.document_processing.document_converter import DocumentConverter
from src.document_processing.semantic_chunker import HybridSemanticChunker

logger = logging.getLogger(__name__)

class DocumentInitializer:
    """文档初始化器
    
    负责在项目启动时加载和处理文档，建立向量库。
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """初始化文档初始化器
        
        Args:
            config: 配置参数，如果为None则使用默认配置
        """
        self.config_manager = ConfigManager()
        
        # 获取默认配置
        default_config = self._get_default_config()
        
        # 如果传入了config，则与默认配置合并
        if config:
            default_config.update(config)
        
        self.config = default_config
        
        # 初始化组件
        self.embeddings = None
        self.vector_store = None
        self.document_converter = None
        self.semantic_chunker = None
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'processed_files': 0,
            'failed_files': 0,
            'total_chunks': 0,
            'processing_time': 0.0,
            'errors': []
        }
        
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        embedding_model = self.config_manager.get('embedding.model_name', 'text-embedding-ada-002')
        
        return {
            'knowledge_docs_path': self.config_manager.get('paths.knowledge_docs_path', './rules'),
            'vector_store_path': './chroma_db_rules',
            'embedding_model': embedding_model,
            'chunk_size': self.config_manager.get('chunking.chunk_size', 1000),
            'chunk_overlap': self.config_manager.get('chunking.chunk_overlap', 200),
            'min_chunk_size': self.config_manager.get('chunking.min_chunk_size', 100),
            'max_chunk_size': self.config_manager.get('chunking.max_chunk_size', 2000),
            'supported_extensions': ['.pdf', '.docx'],  # 只支持PDF和Word文档
            'force_reload': False,  # 是否强制重新加载所有文档
            'batch_size': 5,  # 批处理大小，减少以提高稳定性
            'enable_structured_chunking': True,  # 启用结构化分块
            'semantic_chunk_threshold': 500,  # 语义分块阈值
            'breakpoint_threshold_type': 'percentile',
            'breakpoint_threshold_amount': 95.0,
            # 新增配置项
            'clear_vector_store_on_startup': self.config_manager.get('vector_store.clear_on_startup', True),
            'enable_chunk_sampling': self.config_manager.get('sampling.enable_chunk_sampling', True),
            'chunk_sample_size': self.config_manager.get('sampling.chunk_sample_size', 100),
            'chunk_sample_output_path': self.config_manager.get('sampling.chunk_sample_output_path', './chunk_samples.txt')
        }
    
    def initialize(self) -> Dict[str, Any]:
        """初始化文档库
        
        Returns:
            初始化结果统计
        """
        start_time = time.time()
        
        try:
            logger.info("开始初始化文档库...")
            
            # 1. 初始化组件
            self._initialize_components()
            
            # 2. 清空向量库（如果配置启用）
            if self.config.get('clear_vector_store_on_startup', True):
                self._clear_vector_store()
            
            # 3. 检查是否需要重新加载
            if not self._should_reload():
                logger.info("向量库已存在且不需要重新加载")
                return self._get_existing_stats()
            
            # 4. 扫描文档文件
            files_to_process = self._scan_documents()
            
            if not files_to_process:
                logger.warning(f"在目录 {self.config['knowledge_docs_path']} 中未找到支持的文档文件")
                return self.stats
            
            # 5. 批量处理文档
            self._process_documents_batch(files_to_process)
            
            # 6. 分块采样（如果配置启用）
            if self.config.get('enable_chunk_sampling', True):
                self._sample_chunks_to_file()
            
            # 7. 保存向量库
            if self.vector_store:
                self.vector_store.persist()
                logger.info("向量库已保存")
            
            # 8. 更新统计信息
            self.stats['processing_time'] = time.time() - start_time
            
            logger.info(
                f"文档初始化完成: 处理了 {self.stats['processed_files']}/{self.stats['total_files']} 个文件，"
                f"生成 {self.stats['total_chunks']} 个分块，耗时 {self.stats['processing_time']:.2f} 秒"
            )
            
            return self.stats
            
        except Exception as e:
            error_msg = f"文档初始化失败: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            self.stats['processing_time'] = time.time() - start_time
            return self.stats
    
    def _test_connectivity(self):
        """测试API连通性"""
        try:
            logger.info("开始进行API连通性测试...")
            
            # 获取配置信息
            api_key = self.config_manager.get('api.openai_api_key')
            api_base = self.config_manager.get('api.openai_base_url', 'https://api.openai.com/v1')
            embedding_model = self.config.get('embedding_model')
            
            # 在日志中输出使用的key和URL（脱敏处理）
            logger.info(f"API配置信息:")
            logger.info(f"  - API Key: {api_key[:10] if api_key else 'N/A'}...")
            logger.info(f"  - API Base URL: {api_base}")
            logger.info(f"  - 嵌入模型: {embedding_model}")
            
            if not api_key:
                raise ValueError("未配置OpenAI API密钥")
            
            if not embedding_model:
                raise ValueError("embedding_model配置未找到")
            
            # 创建临时嵌入实例进行连通性测试
            test_embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                model=embedding_model,
                openai_api_base=api_base
            )
            
            # 进行简单的嵌入测试
            logger.info("正在测试嵌入API连通性...")
            test_result = test_embeddings.embed_query("测试连通性")
            
            if test_result and len(test_result) > 0:
                logger.info(f"✓ API连通性测试成功，嵌入维度: {len(test_result)}")
                return True
            else:
                logger.error("✗ API连通性测试失败：返回结果为空")
                return False
                
        except Exception as e:
            logger.error(f"✗ API连通性测试失败: {str(e)}")
            return False
    
    def _initialize_components(self):
        """初始化各个组件"""
        try:
            # 先进行连通性测试
            if not self._test_connectivity():
                raise RuntimeError("API连通性测试失败，无法继续初始化")
            
            # 初始化嵌入模型
            api_key = self.config_manager.get('api.openai_api_key')
            api_base = self.config_manager.get('api.openai_base_url', 'https://api.openai.com/v1')
            embedding_model = self.config.get('embedding_model')
            
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                model=embedding_model,
                openai_api_base=api_base
            )
            
            # 初始化向量存储
            self.vector_store = Chroma(
                persist_directory=self.config['vector_store_path'],
                embedding_function=self.embeddings
            )
            
            # 初始化文档转换器
            self.document_converter = DocumentConverter(config=self.config)
            
            # 初始化混合语义分块器
            chunker_config = {
                'chunk_size': self.config['chunk_size'],
                'chunk_overlap': self.config['chunk_overlap'],
                'min_chunk_size': self.config['min_chunk_size'],
                'max_chunk_size': self.config['max_chunk_size'],
                'enable_structured_chunking': self.config['enable_structured_chunking'],
                'semantic_chunk_threshold': self.config['semantic_chunk_threshold'],
                'breakpoint_threshold_type': self.config['breakpoint_threshold_type'],
                'breakpoint_threshold_amount': self.config['breakpoint_threshold_amount']
            }
            
            self.semantic_chunker = HybridSemanticChunker(
                embeddings=self.embeddings,
                config=chunker_config
            )
            
            logger.info("组件初始化完成")
            
        except Exception as e:
            raise RuntimeError(f"组件初始化失败: {str(e)}")
    
    def _should_reload(self) -> bool:
        """判断是否需要重新加载文档"""
        # 如果配置了强制重新加载
        if self.config.get('force_reload', False):
            return True
        
        # 检查向量库是否存在
        vector_store_path = Path(self.config['vector_store_path'])
        if not vector_store_path.exists():
            return True
        
        # 检查向量库是否为空
        try:
            collection = self.vector_store._collection
            if collection.count() == 0:
                return True
        except Exception:
            return True
        
        return False
    
    def _get_existing_stats(self) -> Dict[str, Any]:
        """获取现有向量库的统计信息"""
        try:
            collection = self.vector_store._collection
            chunk_count = collection.count()
            
            return {
                'total_files': 0,
                'processed_files': 0,
                'failed_files': 0,
                'total_chunks': chunk_count,
                'processing_time': 0.0,
                'errors': [],
                'status': 'existing',
                'message': f'使用现有向量库，包含 {chunk_count} 个文档分块'
            }
        except Exception as e:
            logger.warning(f"获取现有统计信息失败: {str(e)}")
            return self.stats
    
    def _scan_documents(self) -> List[str]:
        """扫描文档目录，获取需要处理的文件列表"""
        rules_dir = Path(self.config['knowledge_docs_path'])
        
        if not rules_dir.exists():
            logger.warning(f"文档目录不存在: {rules_dir}")
            return []
        
        supported_extensions = self.config['supported_extensions']
        files_to_process = []
        
        # 递归扫描目录
        for file_path in rules_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files_to_process.append(str(file_path))
        
        self.stats['total_files'] = len(files_to_process)
        
        logger.info(f"扫描到 {len(files_to_process)} 个支持的文档文件")
        
        return files_to_process
    
    def _process_documents_batch(self, file_paths: List[str]):
        """批量处理文档"""
        batch_size = self.config.get('batch_size', 10)
        
        for i in range(0, len(file_paths), batch_size):
            batch = file_paths[i:i + batch_size]
            
            logger.info(f"处理批次 {i//batch_size + 1}: {len(batch)} 个文件")
            
            for file_path in batch:
                try:
                    self._process_single_file(file_path)
                    self.stats['processed_files'] += 1
                    
                except Exception as e:
                    error_msg = f"处理文件 {file_path} 失败: {str(e)}"
                    logger.error(error_msg)
                    self.stats['errors'].append(error_msg)
                    self.stats['failed_files'] += 1
    
    def _process_single_file(self, file_path: str):
        """处理单个文件"""
        logger.debug(f"处理文件: {file_path}")
        
        try:
            # 将文档转换为Markdown格式
            markdown_content = self.document_converter.convert_to_markdown(file_path)
            
            if not markdown_content or not markdown_content.strip():
                logger.warning(f"文件 {file_path} 转换后内容为空")
                return
            
            logger.debug(f"文件 {file_path} 成功转换为Markdown，内容长度: {len(markdown_content)}")
            
            # 使用混合语义分块器进行分块
            # 对于PDF和Word文档，转换为Markdown后使用markdown分块策略
            chunks = self.semantic_chunker.chunk_documents(
                content=markdown_content,
                source_path=file_path,
                content_type='markdown'
            )
            
            if not chunks:
                logger.warning(f"文件 {file_path} 分块后未生成任何内容")
                return
            
            # 将分块添加到向量存储
            chunk_count = len(chunks)
            self.vector_store.add_documents(chunks)
            
            # 更新统计信息
            self.stats['total_chunks'] += chunk_count
            
            logger.debug(f"文件 {file_path} 处理完成，生成 {chunk_count} 个分块")
            
        except Exception as e:
            logger.error(f"处理文件 {file_path} 时发生错误: {str(e)}")
            raise
    
    def get_retriever(self, k: int = 5):
        """获取配置好的检索器
        
        Args:
            k: 检索返回的文档数量
            
        Returns:
            配置好的检索器
        """
        if not self.vector_store:
            raise RuntimeError("向量库未初始化，请先调用 initialize() 方法")
        
        return self.vector_store.as_retriever(search_kwargs={"k": k})
    
    def add_document(self, file_path: str) -> Dict[str, Any]:
        """添加单个文档到向量库
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            处理结果
        """
        try:
            if not self.semantic_chunker:
                self._initialize_components()
            
            # 将文档转换为Markdown格式
            markdown_content = self.document_converter.convert_to_markdown(file_path)
            
            if not markdown_content or not markdown_content.strip():
                return {
                    'success': False,
                    'error': '文档转换失败，未获取到任何内容',
                    'file_path': file_path
                }
            
            # 使用混合语义分块器进行分块
            chunks = self.semantic_chunker.chunk_text(
                text=markdown_content,
                metadata={'file_path': file_path, 'source': file_path}
            )
            
            if not chunks:
                return {
                    'success': False,
                    'error': '文档分块失败，未生成任何分块',
                    'file_path': file_path
                }
            
            # 将分块添加到向量存储
            chunk_count = len(chunks)
            self.vector_store.add_documents(chunks)
            
            # 保存向量库
            if self.vector_store:
                self.vector_store.persist()
            
            logger.info(f"成功添加文档 {file_path}，生成 {chunk_count} 个分块")
            
            return {
                'success': True,
                'chunks_created': chunk_count,
                'file_path': file_path
            }
            
        except Exception as e:
            error_msg = f"添加文档失败: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'file_path': file_path
            }
    
    def remove_document(self, file_path: str) -> Dict[str, Any]:
        """从向量库中移除文档
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            移除结果
        """
        try:
            if not self.vector_store:
                raise RuntimeError("向量库未初始化")
            
            # 查找与文件路径相关的文档
            collection = self.vector_store._collection
            
            # 根据元数据中的文件路径过滤
            results = collection.get(
                where={"file_path": file_path}
            )
            
            if results and results['ids']:
                # 删除找到的文档
                collection.delete(ids=results['ids'])
                
                # 保存向量库
                self.vector_store.persist()
                
                logger.info(f"成功移除文档 {file_path}，删除 {len(results['ids'])} 个分块")
                
                return {
                    'success': True,
                    'chunks_removed': len(results['ids']),
                    'file_path': file_path
                }
            else:
                return {
                    'success': False,
                    'error': '未找到相关文档',
                    'file_path': file_path
                }
                
        except Exception as e:
            error_msg = f"移除文档失败: {str(e)}"
            logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'file_path': file_path
            }
    
    def get_stats(self) -> Dict[str, Any]:
        """获取当前统计信息"""
        return self.stats.copy()
    
    def clear_vector_store(self) -> bool:
        """清空向量库
        
        Returns:
            是否成功清空
        """
        try:
            if self.vector_store:
                # 删除持久化目录
                import shutil
                vector_store_path = Path(self.config['vector_store_path'])
                if vector_store_path.exists():
                    shutil.rmtree(vector_store_path)
                
                # 重新初始化向量库
                self.vector_store = Chroma(
                    persist_directory=self.config['vector_store_path'],
                    embedding_function=self.embeddings
                )
                
                logger.info("向量库已清空")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"清空向量库失败: {str(e)}")
            return False
    

    
    def _clear_vector_store(self):
        """清空向量库"""
        try:
            if self.vector_store:
                # 获取集合并删除所有文档
                collection = self.vector_store._collection
                # 获取所有文档的ids
                results = collection.get()
                if results and results.get('ids'):
                    # 删除所有文档
                    collection.delete(ids=results['ids'])
                    logger.info(f"向量库已清空，删除了 {len(results['ids'])} 个文档")
                else:
                    logger.info("向量库已经是空的")
            else:
                logger.warning("向量库未初始化，无法清空")
        except Exception as e:
            logger.error(f"清空向量库失败: {str(e)}")
    
    def _sample_chunks_to_file(self):
        """将分块采样保存到文件 - 按文档分组采样"""
        try:
            if not self.vector_store:
                logger.warning("向量库未初始化，无法进行分块采样")
                return
            
            sample_size = self.config.get('chunk_sample_size', 100)
            output_path = self.config.get('chunk_sample_output_path', './chunk_samples.txt')
            
            # 获取所有文档
            collection = self.vector_store._collection
            results = collection.get()
            
            if not results or not results.get('documents'):
                logger.warning("向量库中没有文档可供采样")
                return
            
            import random
            from collections import defaultdict
            
            documents = results['documents']
            metadatas = results.get('metadatas', [])
            
            # 按文档路径分组
            doc_groups = defaultdict(list)
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                source_path = meta.get('source', meta.get('file_path', 'unknown'))
                doc_groups[source_path].append((i, doc, meta))
            
            total_docs = len(documents)
            total_sampled = 0
            
            # 写入文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(f"# 分块采样报告 - 按文档分组\n")
                f.write(f"# 采样时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# 总分块数: {total_docs}\n")
                f.write(f"# 文档数量: {len(doc_groups)}\n")
                f.write(f"# 每个文档采样数量: {sample_size}\n")
                f.write("\n" + "="*80 + "\n\n")
                
                # 为每个文档进行采样
                for doc_idx, (source_path, chunks) in enumerate(doc_groups.items(), 1):
                    f.write(f"# 文档 {doc_idx}: {source_path}\n")
                    f.write(f"# 该文档总分块数: {len(chunks)}\n")
                    
                    # 计算实际采样数量
                    actual_sample_size = min(sample_size, len(chunks))
                    total_sampled += actual_sample_size
                    
                    f.write(f"# 采样分块数: {actual_sample_size}\n")
                    f.write("\n" + "-"*60 + "\n\n")
                    
                    # 随机采样该文档的分块
                    if actual_sample_size < len(chunks):
                        sampled_chunks = random.sample(chunks, actual_sample_size)
                    else:
                        sampled_chunks = chunks
                    
                    # 按原始索引排序以保持一定的顺序
                    sampled_chunks.sort(key=lambda x: x[0])
                    
                    for chunk_idx, (original_idx, doc_content, meta) in enumerate(sampled_chunks, 1):
                        f.write(f"## 分块 {chunk_idx} (原始索引: {original_idx})\n\n")
                        
                        # 写入元数据
                        if meta:
                            f.write("### 元数据\n")
                            for key, value in meta.items():
                                f.write(f"- **{key}**: {value}\n")
                            f.write("\n")
                        
                        # 写入内容
                        f.write("### 内容\n")
                        f.write(f"```\n{doc_content}\n```\n\n")
                        f.write("-" * 40 + "\n\n")
                    
                    f.write("\n" + "="*80 + "\n\n")
            
            logger.info(f"成功为 {len(doc_groups)} 个文档采样共 {total_sampled} 个分块并保存到 {output_path}")
            
        except Exception as e:
            logger.error(f"分块采样失败: {str(e)}")


# 全局文档初始化器实例
_document_initializer = None

def get_document_initializer(config: Optional[Dict[str, Any]] = None) -> DocumentInitializer:
    """获取全局文档初始化器实例"""
    global _document_initializer
    if _document_initializer is None:
        _document_initializer = DocumentInitializer(config)
    return _document_initializer

def initialize_documents(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """快捷方式：初始化文档库"""
    initializer = get_document_initializer(config)
    return initializer.initialize()

def get_rules_retriever(k: int = 5):
    """快捷方式：获取rules文档检索器"""
    initializer = get_document_initializer()
    return initializer.get_retriever(k)