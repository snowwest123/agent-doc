"""E2E 联调测试：建 session → 上传 → 同步问答 → 流式问答 → 查 session"""
import json
import sys
import tempfile
from pathlib import Path

import requests

BASE = "http://localhost:8000"


def section(t):
    print(f"\n{'=' * 60}\n{t}\n{'=' * 60}")


section("1. POST /sessions")
r = requests.post(f"{BASE}/sessions", json={"brain_name": "联调测试"})
print(f"Status: {r.status_code}")
print(r.text)
assert r.status_code == 200, "session 创建失败"
sid = r.json()["session_id"]

section("2. POST /upload")
tmp = Path(tempfile.mkdtemp())
f = tmp / "cafe.txt"
f.write_text(
    "幻海咖啡馆：拿铁 32 元，美式 28 元。营业时间 8:00-22:00。招牌拿铁用了意大利品牌 Illy 的咖啡豆。",
    encoding="utf-8",
)
r = requests.post(
    f"{BASE}/upload",
    params={"session_id": sid},
    files={"files": open(f, "rb")},
)
print(f"Status: {r.status_code}")
print(r.text)
assert r.status_code == 200, "上传失败"

section("3. POST /ask (同步)")
r = requests.post(
    f"{BASE}/ask",
    json={"session_id": sid, "question": "拿铁多少钱？", "use_history": True},
)
print(f"Status: {r.status_code}")
print(r.text)
assert r.status_code == 200
answer = r.json()["answer"]
print(f"💡 答案: {answer}")
assert "32" in answer or "拿铁" in answer, "答案不符合预期"

section("4. POST /ask_stream (SSE)")
r = requests.post(
    f"{BASE}/ask_stream",
    json={"session_id": sid, "question": "营业时间是什么时候？", "use_history": True},
    stream=True,
)
print(f"Status: {r.status_code}")
chunks = []
for line in r.iter_lines():
    if line.startswith(b"data:"):
        payload = line[5:].strip().decode()
        if payload == "[DONE]":
            print("\n[Stream DONE]")
            break
        chunks.append(payload)
        print(payload, end="", flush=True)
full = "".join(chunks)
assert "8" in full or "22" in full or "营业" in full, "流式答案不符合预期"
print(f"\n[Stream chunks: {len(chunks)}, length: {len(full)}]")

section("5. GET /sessions/{sid}")
r = requests.get(f"{BASE}/sessions/{sid}")
print(f"Status: {r.status_code}")
print(r.text)
assert r.status_code == 200
info = r.json()
assert info["nb_chunks"] > 0, "chunks 数应为正"
assert len(info["files"]) == 1

section("6. GET /info")
r = requests.get(f"{BASE}/info", params={"session_id": sid})
print(f"Status: {r.status_code}")
print(r.text)
assert r.status_code == 200

section("7. DELETE /sessions/{sid}（清理）")
r = requests.delete(f"{BASE}/sessions/{sid}")
print(f"Status: {r.status_code}")
print(r.text)

print("\n🎉 全部 E2E 测试通过！")