"""FastAPI 暴露 myrag Brain 作为 REST API。

端点：
  POST /sessions          创建会话
  POST /upload           上传文档（绑定到 session）
  POST /ask              同步问答
  POST /ask_stream       SSE 流式问答
  GET  /info             Brain 状态
  GET  /sessions/{sid}   会话信息
  DELETE /sessions/{sid} 删会话
  GET  /health           健康检查

架构：
  - Redis：存会话元信息 / 聊天历史 / 上传文件元信息
  - volume (/app/data)：存原始上传文件 + FAISS 索引
  - 每个 session_id 独立一个 Brain，多租户隔离
"""
from __future__ import annotations

import asyncio
import os
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from myrag.brain import Brain
from myrag.embedding.dashscope_embedder import DashScopeEmbedder
from myrag.llm import LLMEndpoint
from myrag.processor.registry import ProcessorRegistry

# ===== 路径 / 持久化常量 =====
DATA_DIR = Path(os.getenv("MYRAG_DATA_DIR", "/app/data"))
UPLOADS_DIR = DATA_DIR / "uploads"
FAISS_DIR = DATA_DIR / "faiss"

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# session 在 Redis 里的 key 前缀
K_SESSION = "myrag:session:{sid}"   # hash：brain_name / created_at / nb_chunks / files
K_FILES = "myrag:session:{sid}:files"  # set：上传过的文件名
K_HISTORY = "myrag:session:{sid}:history"  # list：聊天历史（JSON 序列化）


# ===== Redis 客户端（单例）=====
class RedisClient:
    def __init__(self) -> None:
        self.r: aioredis.Redis | None = None

    async def connect(self) -> None:
        self.r = aioredis.Redis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            db=REDIS_DB,
            decode_responses=True,
        )
        await self.r.ping()

    async def close(self) -> None:
        if self.r:
            await self.r.close()

    @property
    def client(self) -> aioredis.Redis:
        assert self.r is not None, "Redis 未连接"
        return self.r

redis_client = RedisClient()


# ===== 鉴权 =====
async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """简易 API Key 鉴权，防止公网被无限制调。"""
    expected = os.getenv("MY_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid API Key")


# ===== Lifespan（替换废弃的 @app.on_event）=====
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 强制 UTF-8 stdout，避免 Windows GBK 编码报错
    import sys
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    FAISS_DIR.mkdir(parents=True, exist_ok=True)
    await redis_client.connect()
    print(f"[OK] Redis connected: {REDIS_HOST}:{REDIS_PORT}")
    print(f"[OK] Data dir: {DATA_DIR}")
    yield
    await redis_client.close()
    print("🛑 Redis disconnected")


app = FastAPI(title="MyRAG API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== Pydantic 模型 =====

class SessionInfo(BaseModel):
    session_id: str
    brain_name: str
    nb_chunks: int
    files: list[str]
    created_at: str

class CreateSessionRequest(BaseModel):
    brain_name: str = "MyRAG Brain"

class AskRequest(BaseModel):
    question: str
    session_id: str
    use_history: bool = True  # 默认开启历史

class AskResponse(BaseModel):
    answer: str
    session_id: str

class BrainInfo(BaseModel):
    session_id: str
    name: str
    llm_model: str
    nb_chunks: int
    uploaded_files: list[str]


# ===== Brain 仓库（In-memory cache，避免重复重建）=====
class BrainRepo:
    """Brain 句柄缓存：重启后从 Redis + FAISS 重建。"""

    def __init__(self) -> None:
        self._cache: dict[str, Brain] = {}

    def get(self, session_id: str) -> Brain | None:
        return self._cache.get(session_id)

    def put(self, session_id: str, brain: Brain) -> None:
        self._cache[session_id] = brain

    def drop(self, session_id: str) -> None:
        self._cache.pop(session_id, None)

    async def load_or_create(
        self,
        session_id: str,
        brain_name: str,
        llm: LLMEndpoint,
        embedder: DashScopeEmbedder,
    ) -> Brain:
        """优先从 Redis 重建，否则创建新的。"""
        brain = self.get(session_id)
        if brain:
            return brain

        # 尝试从磁盘 + Redis 元信息重建
        faiss_path = FAISS_DIR / session_id
        rds = redis_client.client
        session_key = K_SESSION.format(sid=session_id)
        raw = await rds.hgetall(session_key)
        if raw and faiss_path.exists():
            # 有持久化索引 + Redis 元信息 → 重建
            from myrag.storage import LocalStorage
            brain = Brain(
                name=brain_name,
                llm=llm,
                storage=LocalStorage(UPLOADS_DIR / session_id),
                brain_id=uuid.UUID(raw.get("brain_id", str(uuid.uuid4()))),
            )
            brain.vector_store = type("FAISSStore", (), {})()  # 占位，下面替换
            from myrag.vectorstore.faiss_store import FAISSStore
            fs = FAISSStore(embedder=embedder)
            fs.load_local(faiss_path)
            brain.vector_store = fs
            self.put(session_id, brain)
            print(f"♻️ 重建 Brain: {session_id}")
            return brain

        # 全新 session
        brain = Brain(
            name=brain_name,
            llm=llm,
            storage=None,
        )
        self.put(session_id, brain)
        return brain


brain_repo = BrainRepo()


# ===== 辅助函数 =====

async def _save_faiss_async(brain: Brain, session_id: str) -> None:
    """后台任务：异步保存 FAISS 索引到磁盘（不阻塞响应）。"""
    if brain.vector_store is None:
        return
    faiss_path = FAISS_DIR / session_id
    faiss_path.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(brain.vector_store.save_local, faiss_path)


async def _get_history(session_id: str) -> list[dict[str, str]]:
    """从 Redis 读取聊天历史。"""
    rds = redis_client.client
    raw_list = await rds.lrange(K_HISTORY.format(sid=session_id), 0, -1)
    import json
    return [json.loads(x) for x in raw_list]


async def _append_history(session_id: str, role: str, content: str, max_turns: int = 10) -> None:
    """追加一条历史，并裁剪到最近 max_turns*2 条。"""
    rds = redis_client.client
    import json
    item = json.dumps({"role": role, "content": content}, ensure_ascii=False)
    key = K_HISTORY.format(sid=session_id)
    async with rds.pipeline(transaction=True) as pipe:
        pipe.rpush(key, item)
        pipe.ltrim(key, -(max_turns * 2), -1)  # 只留最近 N 轮
        await pipe.execute()


# ===== 端点 =====

@app.get("/health")
async def health() -> dict[str, str]:
    """健康检查：包含 Redis 连通性。"""
    redis_status = "ok"
    try:
        await redis_client.client.ping()
    except Exception:
        redis_status = "down"
    return {"status": "ok", "redis": redis_status}


@app.post("/sessions", response_model=SessionInfo, dependencies=[Depends(verify_api_key)])
async def create_session(req: CreateSessionRequest) -> SessionInfo:
    """创建新会话。返回 session_id，后续接口都要传它。"""
    session_id = str(uuid.uuid4())
    rds = redis_client.client
    now = datetime.now(timezone.utc).isoformat()
    async with rds.pipeline(transaction=True) as pipe:
        await pipe.hset(
            K_SESSION.format(sid=session_id),
            mapping={
                "brain_name": req.brain_name,
                "created_at": now,
                "nb_chunks": 0,
                "files": "",
            },
        )
        await pipe.execute()
    return SessionInfo(
        session_id=session_id,
        brain_name=req.brain_name,
        nb_chunks=0,
        files=[],
        created_at=now,
    )


@app.get("/sessions/{session_id}", response_model=SessionInfo, dependencies=[Depends(verify_api_key)])
async def get_session(session_id: str) -> SessionInfo:
    rds = redis_client.client
    raw = await rds.hgetall(K_SESSION.format(sid=session_id))
    if not raw:
        raise HTTPException(404, f"session {session_id} 不存在")
    files = [f for f in raw.get("files", "").split("|") if f]
    return SessionInfo(
        session_id=session_id,
        brain_name=raw.get("brain_name", ""),
        nb_chunks=int(raw.get("nb_chunks", 0)),
        files=files,
        created_at=raw.get("created_at", ""),
    )


@app.delete("/sessions/{session_id}", dependencies=[Depends(verify_api_key)])
async def delete_session(session_id: str) -> dict[str, str]:
    rds = redis_client.client
    async with rds.pipeline(transaction=True) as pipe:
        await pipe.delete(K_SESSION.format(sid=session_id))
        await pipe.delete(K_FILES.format(sid=session_id))
        await pipe.delete(K_HISTORY.format(sid=session_id))
        await pipe.execute()
    brain_repo.drop(session_id)
    # 删磁盘文件
    upload_dir = UPLOADS_DIR / session_id
    if upload_dir.exists():
        shutil.rmtree(upload_dir, ignore_errors=True)
    faiss_path = FAISS_DIR / session_id
    if faiss_path.exists():
        shutil.rmtree(faiss_path, ignore_errors=True)
    return {"deleted": session_id}


@app.post("/upload", dependencies=[Depends(verify_api_key)])
async def upload(
    session_id: str,
    files: list[UploadFile] = File(...),
) -> dict[str, Any]:
    """上传文档到指定 session。第一个文件上传时创建 Brain。"""
    rds = redis_client.client
    session_key = K_SESSION.format(sid=session_id)
    if not await rds.exists(session_key):
        raise HTTPException(404, f"session {session_id} 不存在，请先调用 session 初始化")

    if not files:
        raise HTTPException(400, "至少上传一个文件")

    # 1. 保存到 volume
    session_upload_dir = UPLOADS_DIR / session_id
    session_upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for upload_file in files:
        ext = Path(upload_file.filename).suffix.lower()
        if ext not in {".txt", ".md", ".csv"}:
            raise HTTPException(400, f"不支持的文件类型：{ext}")
        dst = session_upload_dir / f"{uuid.uuid4()}{ext}"
        with dst.open("wb", encoding=None) as f:
            content = await upload_file.read()
            f.write(content)
        saved_paths.append(dst)

    # 2. 重建/获取 Brain
    llm = LLMEndpoint.from_env()
    embedder = DashScopeEmbedder.from_env()
    raw = await rds.hgetall(session_key)
    brain_name = raw.get("brain_name", "MyRAG Brain")
    brain = await brain_repo.load_or_create(session_id, brain_name, llm, embedder)

    # 3. 处理文件
    from myrag.storage import LocalStorage
    if brain.storage is None:
        brain.storage = LocalStorage(session_upload_dir)
    new_docs: list = []
    for path in saved_paths:
        qf = await brain.storage.upload_file(path, exists_ok=True)
        processor = ProcessorRegistry.get(qf.file_extension)
        if processor:
            docs = await processor.process_file(qf)
            new_docs.extend(docs)

    if new_docs:
        brain.knowledge.extend(new_docs)
        if brain.vector_store is None:
            from myrag.vectorstore.faiss_store import FAISSStore
            brain.vector_store = FAISSStore(embedder=embedder)
        await brain.vector_store.add_documents(new_docs)
        await _save_faiss_async(brain, session_id)

    # 4. 写 Redis 元信息
    old_files = await rds.smembers(K_FILES.format(sid=session_id))
    new_filenames = [f.filename for f in files]
    async with rds.pipeline(transaction=True) as pipe:
        if new_filenames:
            await pipe.sadd(K_FILES.format(sid=session_id), *new_filenames)
        await pipe.hset(
            session_key,
            mapping={
                "nb_chunks": str(len(brain.knowledge)),
                "files": "|".join(sorted(old_files | set(new_filenames))),
                "brain_id": str(brain.id),
            },
        )
        await pipe.execute()

    return {
        "uploaded": new_filenames,
        "session_id": session_id,
        "total_chunks": len(brain.knowledge),
    }


@app.post("/ask", response_model=AskResponse, dependencies=[Depends(verify_api_key)])
async def ask(req: AskRequest) -> AskResponse:
    """同步问答（带 session 历史）。"""
    rds = redis_client.client
    if not await rds.exists(K_SESSION.format(sid=req.session_id)):
        raise HTTPException(404, "session 不存在")
    brain = brain_repo.get(req.session_id) or await brain_repo.load_or_create(
        req.session_id,
        (await rds.hget(K_SESSION.format(sid=req.session_id), "brain_name")) or "MyRAG Brain",
        LLMEndpoint.from_env(),
        DashScopeEmbedder.from_env(),
    )
    history = await _get_history(req.session_id) if req.use_history else []
    answer = await brain.ask(req.question, chat_history=history)
    await _append_history(req.session_id, "user", req.question)
    await _append_history(req.session_id, "assistant", answer)
    return AskResponse(answer=answer, session_id=req.session_id)


@app.post("/ask_stream", dependencies=[Depends(verify_api_key)])
async def ask_stream(req: AskRequest):
    """SSE 流式问答。"""
    rds = redis_client.client
    if not await rds.exists(K_SESSION.format(sid=req.session_id)):
        raise HTTPException(404, "session 不存在")
    brain = brain_repo.get(req.session_id) or await brain_repo.load_or_create(
        req.session_id,
        (await rds.hget(K_SESSION.format(sid=req.session_id), "brain_name")) or "MyRAG Brain",
        LLMEndpoint.from_env(),
        DashScopeEmbedder.from_env(),
    )
    history = await _get_history(req.session_id) if req.use_history else []

    full_answer: list[str] = []

    async def event_generator():
        try:
            async for chunk in brain.ask_streaming(req.question):
                full_answer.append(chunk)
                yield f"data: {chunk}\n\n"
            # 完成后才写历史
            await _append_history(req.session_id, "user", req.question)
            await _append_history(req.session_id, "assistant", "".join(full_answer))
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/info", response_model=BrainInfo, dependencies=[Depends(verify_api_key)])
async def info(session_id: str) -> BrainInfo:
    rds = redis_client.client
    raw = await rds.hgetall(K_SESSION.format(sid=session_id))
    if not raw:
        raise HTTPException(404, "session 不存在")
    files = await rds.smembers(K_FILES.format(sid=session_id))
    brain = brain_repo.get(session_id) or await brain_repo.load_or_create(
        session_id,
        raw.get("brain_name", ""),
        LLMEndpoint.from_env(),
        DashScopeEmbedder.from_env(),
    )
    return BrainInfo(
        session_id=session_id,
        name=brain.name,
        llm_model=brain.llm.info()["model"],
        nb_chunks=int(raw.get("nb_chunks", 0)),
        uploaded_files=sorted(files),
    )
