"""Step 13 客户端测试（适配 Redis 多 session 接口）。

跑法：cd examples/step13_fastapi && python test_client.py

覆盖：
1. /health 检查（带 Redis 连通性）
2. POST /sessions 创建会话
3. POST /upload 上传文档
4. POST /ask 同步问答 + 历史持久化
5. POST /ask_stream SSE 流式问答
6. GET /info 查看 Brain 状态
7. GET /sessions/{sid} 查看会话
8. 多 session 隔离验证
9. DELETE /sessions/{sid} 删除会话
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import requests

BASE = "http://localhost:8000"
API_KEY = ""  # 如果 .env 里设置了 MY_API_KEY，就填进来


def headers() -> dict[str, str]:
    """统一请求头。"""
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    return h


def section(title: str) -> None:
    print(f"\n{'='*60}\n{title}\n{'='*60}")


# 0. 准备测试文件
section("0. PREPARE TEST FILES")
tmp = Path(tempfile.mkdtemp())
cafe_file = tmp / "cafe_info.txt"
cafe_file.write_text(
    """幻海咖啡馆信息：
- 拿铁 ¥32
- 美式 ¥28
- 营业时间：08:00 - 22:00
- 招牌饮品：幻海拿铁""",
    encoding="utf-8",
)
policy_file = tmp / "policy.txt"
policy_file.write_text(
    """员工手册节选：
1. 服务员需着统一制服
3. 请假需提前 1 天申请
5. 顾客投诉必须在 24 小时内处理""",
    encoding="utf-8",
)
print(f"Created: {cafe_file.name}, {policy_file.name}")

# 1. 健康检查
section("1. HEALTH")
r = requests.get(f"{BASE}/health")
print(f"Status: {r.status_code}")
print(f"Body:   {r.json()}")
assert r.status_code == 200
assert r.json()["status"] == "ok"
assert r.json()["redis"] is True, "Redis 未连通！"

# 2. 创建 session
section("2. CREATE SESSION")
r = requests.post(
    f"{BASE}/sessions",
    headers=headers(),
    json={"brain_name": "咖啡馆客服 Brain"},
)
print(f"Status: {r.status_code}")
print(f"Body:   {r.json()}")
assert r.status_code == 200, "创建 session 失败"
session_id = r.json()["session_id"]
print(f"✅ Session ID: {session_id}")

# 3. 上传文档
section("3. UPLOAD FILES")
r = requests.post(
    f"{BASE}/upload",
    headers=headers(),
    params={"session_id": session_id},
    files=[
        ("files", open(cafe_file, "rb")),
        ("files", open(policy_file, "rb")),
    ],
)
print(f"Status: {r.status_code}")
print(f"Body:   {r.json()}")
assert r.status_code == 200, "上传失败"
assert r.json()["total_chunks"] > 0

# 4. 同步问答
section("4. ASK (SYNC)")
r = requests.post(
    f"{BASE}/ask",
    headers=headers(),
    json={"session_id": session_id, "question": "拿铁多少钱？", "use_history": True},
)
print(f"Status: {r.status_code}")
print(f"Body:   {r.json()}")
assert r.status_code == 200
answer1 = r.json()["answer"]
assert "32" in answer1 or "拿铁" in answer1, f"答案不符合预期：{answer1}"

# 5. 第二轮问答（带历史）
section("5. ASK (WITH HISTORY)")
r = requests.post(
    f"{BASE}/ask",
    headers=headers(),
    json={"session_id": session_id, "question": "那美式呢？", "use_history": True},
)
print(f"Status: {r.status_code}")
print(f"Body:   {r.json()}")
assert r.status_code == 200
answer2 = r.json()["answer"]
print(f"💡 验证历史感知：上一轮问拿铁、这一轮问美式，模型应该能联系上下文")

# 6. 流式问答
section("6. ASK STREAM (SSE)")
r = requests.post(
    f"{BASE}/ask_stream",
    headers=headers(),
    json={"session_id": session_id, "question": "营业时间是几点？", "use_history": True},
    stream=True,
)
print(f"Status: {r.status_code}")
print("Stream chunks:")
chunks: list[str] = []
for line in r.iter_lines():
    if line.startswith("data:"):
        payload = line[5:].strip()
        if payload == "[DONE]":
            print("\n✅ Stream finished")
            break
        if payload.startswith("[ERROR]"):
            print(f"\n❌ Stream error: {payload}")
            break
        chunks.append(payload)
        print(payload, end="", flush=True)
full_answer = "".join(chunks)
assert "08" in full_answer or "22" in full_answer, f"流式答案不符合预期：{full_answer}"
print(f"\n📊 Total chunks: {len(chunks)}, length: {len(full_answer)}")

# 7. 查看 Brain 状态
section("7. INFO")
r = requests.get(
    f"{BASE}/info",
    headers=headers(),
    params={"session_id": session_id},
)
print(f"Status: {r.status_code}")
print(f"Body:   {r.json()}")
assert r.status_code == 200
assert r.json()["nb_chunks"] > 0

# 8. 查看会话详情
section("8. GET SESSION")
r = requests.get(f"{BASE}/sessions/{session_id}", headers=headers())
print(f"Status: {r.status_code}")
print(f"Body:   {r.json()}")
assert r.status_code == 200
assert len(r.json()["files"]) == 2

# 9. 多 session 隔离验证
section("9. MULTI-SESSION ISOLATION")
r2 = requests.post(
    f"{BASE}/sessions",
    headers=headers(),
    json={"brain_name": "另一个 Brain"},
)
session_id_2 = r2.json()["session_id"]
print(f"Created second session: {session_id_2}")
# 第二个 session 没上传文件，应该 404
r = requests.post(
    f"{BASE}/ask",
    headers=headers(),
    json={"session_id": session_id_2, "question": "拿铁多少钱？"},
)
print(f"Second session ask status: {r.status_code} (expect 400 因为没文档)")
assert r.status_code == 400

# 10. 清理：删除会话
section("10. CLEANUP")
r = requests.delete(f"{BASE}/sessions/{session_id}", headers=headers())
print(f"Delete session 1: {r.status_code} {r.json()}")
r = requests.delete(f"{BASE}/sessions/{session_id_2}", headers=headers())
print(f"Delete session 2: {r.status_code} {r.json()}")

section("✅ ALL TESTS PASSED")

# 5. 流式问答（核心）
section("5. ASK_STREAM (SSE)")
r = requests.post(
    f"{BASE}/ask_stream",
    json={"question": "营业时间是什么时候？"},
    stream=True,
)
print(f"Status: {r.status_code}, Content-Type: {r.headers.get('content-type')}")
print("Stream chunks:", end=" ")
for line in r.iter_lines():
    if line:
        text = line.decode().replace("data: ", "").strip()
        if text and text != "[DONE]":
            print(f"[{text}]", end=" ", flush=True)
print("\n[DONE]")