# config.py

# --- Ollama 在 Windows 本地 ---
# 既然代码和 Ollama 都在 Windows，直接用 localhost
OLLAMA_BASE_URL = "http://127.0.0.1:11434"

# --- 数据库在虚拟机 ---
# 请确保这是虚拟机 ip addr 查到的真实 IP
VM_IP = "192.168.150.100"
PG_CONNECTION = f"postgresql+psycopg2://myuser:mypassword@{VM_IP}:5432/medical_rag"

EMBEDDING_MODEL = "bge-m3"