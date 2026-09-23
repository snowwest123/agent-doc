"""最小连接测试 + 超时控制。"""
import os
import sys

# 加载 .env
from dotenv import load_dotenv
load_dotenv()

print("===== 配置信息 =====")
for k in ["HOLOGRES_HOST", "HOLOGRES_PORT", "HOLOGRES_DB", "HOLOGRES_USER", "HOLOGRES_PASSWORD"]:
    v = os.getenv(k)
    if v and k == "HOLOGRES_PASSWORD":
        v = v[:4] + "***" + v[-4:]
    print(f"{k}: {v}")

print("\n===== 尝试连接 =====")
try:
    import psycopg
    conn = psycopg.connect(
        host=os.getenv("HOLOGRES_HOST"),
        port=int(os.getenv("HOLOGRES_PORT", "80")),
        dbname=os.getenv("HOLOGRES_DB"),
        user=os.getenv("HOLOGRES_USER"),
        password=os.getenv("HOLOGRES_PASSWORD"),
        connect_timeout=10,  # 10 秒超时
    )
    print("✅ 连接成功")

    with conn.cursor() as cur:
        cur.execute("SELECT version()")
        print(f"版本: {cur.fetchone()[0][:80]}")

        cur.execute("SELECT COUNT(*) FROM customers")
        print(f"customers 行数: {cur.fetchone()[0]}")

        cur.execute("SELECT name, city FROM customers LIMIT 3")
        for row in cur.fetchall():
            print(f"  - {row}")

    conn.close()
    print("\n✅ 测试完成")

except Exception as e:
    print(f"❌ 错误: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()