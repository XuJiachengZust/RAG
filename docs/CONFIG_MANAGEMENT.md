# 配置管理系统说明文档

## 概述

本项目采用统一的配置管理系统，将所有配置项集中管理，支持通过配置文件和环境变量进行配置。

## 配置文件结构

### 主配置文件：`config.json`

位于项目根目录，包含所有配置项的默认值：

```json
{
  "api": {
    "openai_api_key": "${OPENAI_API_KEY}",
    "openai_base_url": "${OPENAI_BASE_URL:https://api.openai.com/v1}"
  },
  "model": {
    "name": "${MODEL_NAME:gpt-3.5-turbo}",
    "temperature": "${MODEL_TEMPERATURE:0.3}",
    "max_tokens": "${MODEL_MAX_TOKENS:1000}"
  },
  "chunking": {
    "chunk_size": "${CHUNK_SIZE:1000}",
    "chunk_overlap": "${CHUNK_OVERLAP:200}",
    "max_chunk_size": "${MAX_CHUNK_SIZE:2000}"
  },
  "paths": {
    "knowledge_docs_path": "${KNOWLEDGE_DOCS_PATH:d:/PycharmProjects/agent/rules}",
    "parameter_docs_path": "${PARAMETER_DOCS_PATH:d:/PycharmProjects/agent/data/parameters}",
    "conversations_path": "${CONVERSATIONS_PATH:d:/PycharmProjects/agent/data/conversations}",
    "vector_store_path": "${VECTOR_STORE_PATH:d:/PycharmProjects/agent/data/vector_stores}",
    "cache_dir": "${CACHE_DIR:d:/PycharmProjects/agent/cache}"
  }
}
```

### 环境变量模板：`.env.template`

提供所有可用环境变量的模板和说明。复制为 `.env` 文件并填入实际值。

## 配置管理器使用方法

### 基本用法

```python
from config_manager import get_config_manager, get_config, get_config_section

# 获取配置管理器实例
config_manager = get_config_manager()

# 获取单个配置项
api_key = config_manager.get('api.openai_api_key')
model_name = config_manager.get('model.name', 'gpt-3.5-turbo')  # 带默认值

# 获取配置段
model_config = config_manager.get('model', {})
api_config = config_manager.get('api', {})

# 使用快捷方法
api_key = get_config('api.openai_api_key')
model_config = get_config_section('model')
```

### 高级用法

```python
# 设置配置项
config_manager.set('model.temperature', 0.5)

# 重新加载配置
config_manager.reload()

# 验证必需配置项
required_keys = ['api.openai_api_key', 'model.name']
config_manager.validate_required(required_keys)

# 获取完整配置字典
full_config = config_manager.get_config()

# 保存配置模板
config_manager.save_template('config_template.json')
```

## 环境变量支持

### 环境变量格式

配置文件中支持两种环境变量格式：

1. `${VAR_NAME}` - 必需的环境变量，如果不存在会保持原样
2. `${VAR_NAME:default_value}` - 可选的环境变量，如果不存在使用默认值

### 环境变量映射

系统自动将以下环境变量映射到对应的配置项：

| 环境变量 | 配置路径 | 说明 |
|---------|---------|------|
| `OPENAI_API_KEY` | `api.openai_api_key` | OpenAI API 密钥 |
| `OPENAI_BASE_URL` | `api.openai_base_url` | OpenAI API 基础URL |
| `MODEL_NAME` | `model.name` | 模型名称 |
| `MODEL_TEMPERATURE` | `model.temperature` | 模型温度 |
| `MODEL_MAX_TOKENS` | `model.max_tokens` | 最大令牌数 |
| `CHUNK_SIZE` | `chunking.chunk_size` | 分块大小 |
| `CHUNK_OVERLAP` | `chunking.chunk_overlap` | 分块重叠 |
| `KNOWLEDGE_DOCS_PATH` | `paths.knowledge_docs_path` | 知识文档路径 |
| `CONVERSATIONS_PATH` | `paths.conversations_path` | 对话存储路径 |
| `ENABLE_CACHE` | `features.enable_cache` | 启用缓存 |

更多映射关系请参考 `.env.template` 文件。

## 配置项说明

### API 配置 (`api`)

- `openai_api_key`: OpenAI API 密钥（必需）
- `openai_base_url`: OpenAI API 基础URL

### 模型配置 (`model`)

- `name`: 使用的模型名称
- `temperature`: 模型温度，控制输出的随机性 (0.0-2.0)
- `max_tokens`: 最大生成令牌数

### 分块配置 (`chunking`)

- `chunk_size`: 文档分块大小
- `chunk_overlap`: 分块之间的重叠大小
- `max_chunk_size`: 最大分块大小

### 路径配置 (`paths`)

- `knowledge_docs_path`: 知识文档存储路径
- `parameter_docs_path`: 参数文档存储路径
- `conversations_path`: 对话历史存储路径
- `vector_store_path`: 向量存储路径
- `cache_dir`: 缓存目录

### 嵌入配置 (`embedding`)

- `model`: 嵌入模型名称
- `dimension`: 嵌入向量维度

### 检索配置 (`retrieval`)

- `k`: 检索返回的文档数量
- `similarity_threshold`: 相似度阈值

### Chroma 配置 (`chroma`)

- `persist_directory`: Chroma 持久化目录
- `collection_name`: Chroma 集合名称

### 日志配置 (`logging`)

- `level`: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `file`: 日志文件路径
- `format`: 日志格式

### Web 配置 (`web`)

- `host`: Web 服务器主机
- `port`: Web 服务器端口
- `debug`: 调试模式

### 处理配置 (`processing`)

- `batch_size`: 批处理大小
- `max_retries`: 最大重试次数
- `timeout`: 超时时间（秒）

### 功能开关 (`features`)

- `enable_cache`: 启用缓存
- `enable_logging`: 启用日志
- `enable_vector_store`: 启用向量存储
- `enable_parameter_matching`: 启用参数匹配

## 最佳实践

### 1. 敏感信息管理

- 将 API 密钥等敏感信息放在 `.env` 文件中
- 不要将 `.env` 文件提交到版本控制系统
- 使用 `.env.template` 作为环境变量模板

### 2. 配置文件组织

- 在 `config.json` 中设置合理的默认值
- 使用环境变量覆盖特定环境的配置
- 保持配置结构的层次性和逻辑性

### 3. 代码中的使用

```python
# 推荐：使用配置管理器
config_manager = get_config_manager()
api_key = config_manager.get('api.openai_api_key')

# 不推荐：直接使用环境变量
import os
api_key = os.getenv('OPENAI_API_KEY')
```

### 4. 配置验证

在应用启动时验证必需的配置项：

```python
required_configs = [
    'api.openai_api_key',
    'paths.knowledge_docs_path',
    'model.name'
]

try:
    config_manager.validate_required(required_configs)
except ValueError as e:
    logger.error(f"配置验证失败: {e}")
    sys.exit(1)
```

## 故障排除

### 常见问题

1. **配置项未生效**
   - 检查环境变量名称是否正确
   - 确认 `.env` 文件是否存在且格式正确
   - 重启应用以重新加载配置

2. **找不到配置文件**
   - 确认 `config.json` 文件在项目根目录
   - 检查文件权限

3. **环境变量未解析**
   - 检查环境变量格式是否正确
   - 确认环境变量已设置

### 调试方法

```python
# 查看当前配置
config_manager = get_config_manager()
print(json.dumps(config_manager.get_config(), indent=2))

# 检查特定配置项
print(f"API Key: {config_manager.get('api.openai_api_key')}")
print(f"Model Name: {config_manager.get('model.name')}")
```

## 迁移指南

如果你正在从旧的配置方式迁移到新的统一配置管理系统：

1. **备份现有配置**
   ```bash
   cp .env .env.backup
   ```

2. **更新代码**
   ```python
   # 旧方式
   import os
   api_key = os.getenv('OPENAI_API_KEY')
   
   # 新方式
   from config_manager import get_config
   api_key = get_config('api.openai_api_key')
   ```

3. **验证配置**
   ```python
   config_manager = get_config_manager()
   config_manager.validate_required(['api.openai_api_key'])
   ```

4. **测试应用**
   确保所有功能正常工作后，删除旧的配置代码。