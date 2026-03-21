# 创建 .gitignore 文件计划

## 任务目标
为 GreenDataCenter 项目创建一个完整的 `.gitignore` 文件，忽略运行中生成的文件和目录。

## 需要忽略的文件类型

### 1. Python 相关
- `__pycache__/` - Python 字节码缓存目录
- `*.pyc`, `*.pyo`, `*.pyd` - Python 编译文件
- `.Python` - Python 虚拟环境标记
- `venv/`, `env/`, `.venv/` - 虚拟环境目录
- `*.egg-info/` - Python 包信息
- `dist/`, `build/` - 打包输出目录
- `*.egg` - Python egg 文件

### 2. Node.js/前端相关
- `node_modules/` - npm 依赖目录
- `dist/` - 构建输出目录
- `.vite/` - Vite 缓存目录
- `*.local` - 本地配置文件

### 3. IDE 和编辑器
- `.vscode/` - VS Code 配置
- `.idea/` - PyCharm/IntelliJ 配置
- `*.swp`, `*.swo` - Vim 临时文件
- `*~` - 备份文件

### 4. 项目特定运行时生成文件
- `vector_store/` - FAISS 向量存储（运行时生成）
- `output/` - 输出文件目录
- `.cache.sqlite` - 缓存数据库

### 5. 日志和临时文件
- `*.log` - 日志文件
- `*.tmp` - 临时文件
- `.cache/` - 缓存目录

### 6. 操作系统文件
- `.DS_Store` - macOS 系统文件
- `Thumbs.db` - Windows 缩略图缓存
- `Desktop.ini` - Windows 配置文件

## 实施步骤

1. 在项目根目录创建 `.gitignore` 文件
2. 按类别添加忽略规则
3. 添加必要的注释说明

## 预期输出

创建一个完整的 `.gitignore` 文件，包含所有需要忽略的文件和目录规则。
