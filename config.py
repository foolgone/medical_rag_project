# config.py

# 1. 虚拟机 IP 地址 (请替换为你真实的虚拟机 IP)
VM_IP = "192.168.150.100"

# 2. Ollama 配置
OLLAMA_BASE_URL = f"http://{VM_IP}:11434"
EMBEDDING_MODEL = "bge-m3"

# 3. 数据库连接字符串 (SQLAlchemy 格式)
# 格式: postgresql+psycopg2://用户名:密码@IP:端口/数据库名
PG_CONNECTION = f"postgresql+psycopg2://myuser:mypassword@{VM_IP}:5432/medical_rag"