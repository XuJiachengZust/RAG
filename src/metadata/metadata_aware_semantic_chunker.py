from typing import List, Dict, Any, Optional, Iterator, Union
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
try:
    from langchain_experimental.text_splitter import SemanticChunker
except ImportError:
    try:
        from langchain_text_splitters import SemanticChunker
    except ImportError:
        # 如果都没有，创建一个简单的替代实现
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        SemanticChunker = RecursiveCharacterTextSplitter
import logging
from .metadata_extractor import DocumentContent, ChunkMetadata
from .metadata_enhancer import MetadataEnhancer, EnhancedChunkMetadata

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetadataAwareSemanticChunker:
    """
    元数据感知的语义分块器
    集成LangChain的SemanticChunker，并自动添加增强的元数据信息
    """
    
    def __init__(
        self, 
        embeddings: Embeddings,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        初始化元数据感知语义分块器
        
        Args:
            embeddings: 嵌入模型
            config: 配置参数
        """
        self.embeddings = embeddings
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # 初始化LangChain语义分块器
        try:
            # 尝试使用真正的SemanticChunker
            self.semantic_chunker = SemanticChunker(
                embeddings=embeddings,
                breakpoint_threshold_type=self.config.get('breakpoint_threshold_type', 'percentile'),
                breakpoint_threshold_amount=self.config.get('breakpoint_threshold_amount', 95),
                number_of_chunks=self.config.get('number_of_chunks', None),
                buffer_size=self.config.get('buffer_size', 1),
                sentence_split_regex=self.config.get('sentence_split_regex', r'(?<=[.!?])\s+')
            )
        except TypeError:
            # 如果是RecursiveCharacterTextSplitter，使用不同的参数
            chunk_size = self._parse_config_value(self.config.get('chunk_size', 1000))
            chunk_overlap = self._parse_config_value(self.config.get('chunk_overlap', 200))
            self.semantic_chunker = SemanticChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=self.config.get('separators', ["\n\n", "\n", " ", ""])
            )
        
        # 初始化简化的元数据增强器
        self.metadata_enhancer = MetadataEnhancer(config.get('metadata_enhancer', {}))
        
        # 配置参数
        self.enable_metadata_enhancement = self.config.get('enable_metadata_enhancement', True)
        self.preserve_document_structure = self.config.get('preserve_document_structure', True)
        
        # 确保配置值是整数类型
        self.min_chunk_size = self._parse_config_value(self.config.get('min_chunk_size', 100))
        self.max_chunk_size = self._parse_config_value(self.config.get('max_chunk_size', 4000))
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        分割文档列表并添加增强元数据
        
        Args:
            documents: 文档列表
            
        Returns:
            List[Document]: 带有增强元数据的分块文档列表
        """
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self.split_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                self.logger.error(f"分割文档失败: {str(e)}")
                continue
        
        return all_chunks
    
    def split_document(self, document: Document) -> List[Document]:
        """
        分割单个文档并添加增强元数据
        
        Args:
            document: 单个文档
            
        Returns:
            List[Document]: 带有增强元数据的分块文档列表
        """
        try:
            # 获取文档路径和类型
            file_path = document.metadata.get('source', '')
            document_name = document.metadata.get('document_name', '')
            
            if not document_name and file_path:
                document_name = self._extract_document_name(file_path)
            
            # 提取文档内容结构
            document_content = self._extract_document_content(document, file_path)
            
            # 使用语义分块器进行分块
            semantic_chunks = self.semantic_chunker.split_documents([document])
            
            # 过滤和调整分块大小
            filtered_chunks = self._filter_and_adjust_chunks(semantic_chunks)
            
            # 为每个分块添加增强元数据
            enhanced_chunks = []
            total_chunks = len(filtered_chunks)
            
            for i, chunk in enumerate(filtered_chunks):
                try:
                    enhanced_chunk = self._add_enhanced_metadata(
                        chunk, document_content, i, total_chunks
                    )
                    enhanced_chunks.append(enhanced_chunk)
                except Exception as e:
                    self.logger.warning(f"为分块 {i} 添加元数据失败: {str(e)}")
                    # 添加基础元数据
                    chunk.metadata.update({
                        'document_name': document_name,
                        'chunk_index': i,
                        'total_chunks': total_chunks
                    })
                    enhanced_chunks.append(chunk)
            
            self.logger.info(f"成功分割文档 {document_name}，生成 {len(enhanced_chunks)} 个分块")
            return enhanced_chunks
            
        except Exception as e:
            self.logger.error(f"分割文档失败: {str(e)}")
            # 返回原始文档作为单个分块
            return [document]
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """
        分割纯文本并添加基础元数据
        
        Args:
            text: 文本内容
            metadata: 基础元数据
            
        Returns:
            List[Document]: 分块文档列表
        """
        try:
            # 创建临时文档对象
            temp_doc = Document(
                page_content=text,
                metadata=metadata or {}
            )
            
            # 使用语义分块器分割
            chunks = self.semantic_chunker.split_documents([temp_doc])
            
            # 过滤和调整分块
            filtered_chunks = self._filter_and_adjust_chunks(chunks)
            
            # 添加基础元数据
            for i, chunk in enumerate(filtered_chunks):
                chunk.metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(filtered_chunks),
                    'content_type': 'text',
                    'extraction_method': 'semantic_chunker'
                })
            
            return filtered_chunks
            
        except Exception as e:
            self.logger.error(f"分割文本失败: {str(e)}")
            return [Document(page_content=text, metadata=metadata or {})]
    
    def _extract_document_content(self, document: Document, file_path: str) -> Optional[DocumentContent]:
        """
        创建简化的文档内容结构
        
        Args:
            document: 文档对象
            file_path: 文件路径
            
        Returns:
            Optional[DocumentContent]: 简化的文档内容结构
        """
        try:
            if not file_path:
                return None
            
            # 创建基础文档内容结构
            return DocumentContent(
                text=document.page_content,
                page_mapping={},
                title_hierarchy=[],
                document_name=self._extract_document_name(file_path),
                file_path=file_path,
                total_pages=1
            )
                
        except Exception as e:
            self.logger.warning(f"创建文档内容结构失败: {str(e)}")
            return None
    
    def _add_enhanced_metadata(
        self, 
        chunk: Document, 
        document_content: Optional[DocumentContent],
        chunk_index: int,
        total_chunks: int
    ) -> Document:
        """
        为分块添加增强元数据
        
        Args:
            chunk: 分块文档
            document_content: 文档内容结构
            chunk_index: 分块索引
            total_chunks: 总分块数
            
        Returns:
            Document: 带有增强元数据的分块文档
        """
        try:
            if not self.enable_metadata_enhancement or not document_content:
                # 添加基础元数据
                chunk.metadata.update({
                    'document_name': document_content.document_name if document_content else 'unknown',
                    'chunk_index': chunk_index,
                    'total_chunks': total_chunks
                })
                return chunk
            
            # 创建基础元数据
            base_metadata = ChunkMetadata(
                document_name=document_content.document_name
            )
            
            # 使用元数据增强器增强元数据
            enhanced_metadata = self.metadata_enhancer.enhance_chunk_metadata(
                chunk_text=chunk.page_content,
                base_metadata=base_metadata,
                document_content=document_content,
                chunk_index=chunk_index,
                total_chunks=total_chunks
            )
            
            # 验证元数据
            if self.metadata_enhancer.validate_metadata(enhanced_metadata):
                # 将增强元数据转换为字典并更新到分块
                metadata_dict = self.metadata_enhancer.to_dict(enhanced_metadata)
                chunk.metadata.update(metadata_dict)
            else:
                self.logger.warning(f"分块 {chunk_index} 的元数据验证失败，使用基础元数据")
                chunk.metadata.update({
                    'document_name': document_content.document_name,
                    'chunk_index': chunk_index,
                    'total_chunks': total_chunks
                })
            
            return chunk
            
        except Exception as e:
            self.logger.error(f"添加增强元数据失败: {str(e)}")
            # 添加基础元数据作为后备
            chunk.metadata.update({
                'document_name': document_content.document_name if document_content else 'unknown',
                'chunk_index': chunk_index,
                'total_chunks': total_chunks,
                'error': str(e)
            })
            return chunk
    
    def _filter_and_adjust_chunks(self, chunks: List[Document]) -> List[Document]:
        """
        过滤和调整分块大小
        
        Args:
            chunks: 原始分块列表
            
        Returns:
            List[Document]: 调整后的分块列表
        """
        filtered_chunks = []
        
        for chunk in chunks:
            chunk_text = chunk.page_content.strip()
            chunk_length = len(chunk_text)
            
            # 过滤过小的分块
            if chunk_length < self.min_chunk_size:
                continue
            
            # 分割过大的分块
            if chunk_length > self.max_chunk_size:
                sub_chunks = self._split_large_chunk(chunk)
                filtered_chunks.extend(sub_chunks)
            else:
                filtered_chunks.append(chunk)
        
        return filtered_chunks
    
    def _split_large_chunk(self, chunk: Document) -> List[Document]:
        """
        分割过大的分块
        
        Args:
            chunk: 大分块
            
        Returns:
            List[Document]: 子分块列表
        """
        text = chunk.page_content
        sub_chunks = []
        
        # 按段落分割
        paragraphs = text.split('\n\n')
        current_chunk = ""
        
        for paragraph in paragraphs:
            if len(current_chunk + paragraph) <= self.max_chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    sub_chunks.append(Document(
                        page_content=current_chunk.strip(),
                        metadata=chunk.metadata.copy()
                    ))
                current_chunk = paragraph + "\n\n"
        
        # 添加最后一个分块
        if current_chunk.strip():
            sub_chunks.append(Document(
                page_content=current_chunk.strip(),
                metadata=chunk.metadata.copy()
            ))
        
        return sub_chunks
    
    def _extract_document_name(self, file_path: str) -> str:
        """
        从文件路径提取文档名称
        
        Args:
            file_path: 文件路径
            
        Returns:
            str: 文档名称
        """
        try:
            import os
            base_name = os.path.basename(file_path)
            name_without_ext = os.path.splitext(base_name)[0]
            return name_without_ext
        except Exception:
            return 'unknown_document'
    
    def get_chunk_statistics(self, chunks: List[Document]) -> Dict[str, Any]:
        """
        获取分块统计信息
        
        Args:
            chunks: 分块列表
            
        Returns:
            Dict[str, Any]: 统计信息
        """
        if not chunks:
            return {}
        
        chunk_lengths = [len(chunk.page_content) for chunk in chunks]
        content_types = [chunk.metadata.get('content_type', 'unknown') for chunk in chunks]
        
        from collections import Counter
        
        return {
            'total_chunks': len(chunks),
            'avg_chunk_length': sum(chunk_lengths) / len(chunk_lengths),
            'min_chunk_length': min(chunk_lengths),
            'max_chunk_length': max(chunk_lengths),
            'content_type_distribution': dict(Counter(content_types)),
            'documents_processed': len(set(chunk.metadata.get('document_name', 'unknown') for chunk in chunks))
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """
        更新配置参数
        
        Args:
            new_config: 新的配置参数
        """
        self.config.update(new_config)
        
        # 重新初始化语义分块器
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type=self.config.get('breakpoint_threshold_type', 'percentile'),
            breakpoint_threshold_amount=self.config.get('breakpoint_threshold_amount', 95),
            number_of_chunks=self.config.get('number_of_chunks', None),
            buffer_size=self.config.get('buffer_size', 1)
        )
        
        # 更新其他配置
        self.enable_metadata_enhancement = self.config.get('enable_metadata_enhancement', True)
        self.preserve_document_structure = self.config.get('preserve_document_structure', True)
        self.min_chunk_size = self.config.get('min_chunk_size', 100)
        self.max_chunk_size = self.config.get('max_chunk_size', 4000)
        
        self.logger.info("配置已更新")
    
    def _parse_config_value(self, value):
        """
        解析配置值，支持环境变量格式 ${VAR_NAME:default_value}
        
        Args:
            value: 配置值
            
        Returns:
            解析后的整数值
        """
        import os
        import re
        
        if isinstance(value, str):
            # 匹配环境变量格式 ${VAR_NAME:default_value}
            match = re.match(r'\$\{([^:]+):([^}]+)\}', value)
            if match:
                env_var, default_val = match.groups()
                env_value = os.getenv(env_var)
                if env_value is not None:
                    try:
                        return int(env_value)
                    except ValueError:
                        self.logger.warning(f"环境变量 {env_var} 的值 '{env_value}' 不是有效整数，使用默认值")
                        return int(default_val)
                else:
                    return int(default_val)
            else:
                # 尝试直接转换为整数
                try:
                    return int(value)
                except ValueError:
                    self.logger.warning(f"配置值 '{value}' 不是有效整数，使用默认值 1000")
                    return 1000
        elif isinstance(value, (int, float)):
            return int(value)
        else:
            self.logger.warning(f"未知的配置值类型: {type(value)}，使用默认值 1000")
            return 1000