from core.database import get_vector_store

try:
    vector_db = get_vector_store()
    print("✅ 数据库连接配置成功！")

    # 模拟一条简单的医疗测试数据
    test_text = ["感冒通常表现为流鼻涕和咳嗽。"]
    # vector_db.add_texts(test_text) # 这一步会真正调用 Embedding 并存入数据库
    # print("✅ 数据存入测试成功！")

except Exception as e:
    print(f"❌ 出错了: {e}")
    print("\n💡 排错提示：")
    print("1. 检查虚拟机防火墙是否放行了 5432 端口")
    print("2. 检查 Ollama 服务是否在虚拟机中正常运行且允许远程访问")