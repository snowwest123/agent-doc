"""FastAPI 暴露 myrag Brain 作为 REST API。

端点：
  POST /sessions                       创建会话（支持 mode=knowledge/database/auto）
  GET  /sessions                       会话列表（持久化于 Hologres / Redis）
  POST /sessions/{sid}/mode            切换会话工作模式
  POST /upload                         上传文档（绑定到 session）
  POST /ask                            同步问答（knowledge 模式）
  POST /ask_sql                        Text-to-SQL 问答（database 模式）
  POST /ask_sql_stream                  Text-to-SQL Agent 过程 SSE 流式问答
  POST /ask_auto                       自动路由文档问答或数据库查询
  POST /ask_stream                     SSE 流式问答（knowledge 模式）
  GET  /info                           Brain 状态
  GET  /sessions/{sid}                 会话信息
  GET  /sessions/{sid}/history         聊天历史
  DELETE /sessions/{sid}               删会话
  GET  /health                         健康检查

架构：
  - Hologres：会话元数据（session_id / brain_name / created_at / nb_chunks /
              files / brain_id / owner_id / mode），永久存储。
  - Redis：运行时缓存（启动时从 Hologres 恢复）+ 聊天历史 + 文件名 set。
  - volume (/app/data)：存原始上传文件；FAISS 模式下也保存本地索引。
  - 每个 session_id 独立一个 Brain，多租户隔离。
  - mode 字段决定执行路径：
        knowledge  → 现有 Brain RAG 链路（/ask、/ask_stream）
        database   → Text-to-SQL Agent（/ask_sql）
        auto       → LLM 意图路由（/ask_auto）
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
import redis.asyncio as aioredis
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from myrag.agents.sql_agent import TextToSQLAgent
from myrag.brain import Brain
from myrag.embedding.dashscope_embedder import DashScopeEmbedder
from myrag.llm import LLMEndpoint
from myrag.processor.registry import ProcessorRegistry
from myrag.tools.base import ToolRegistry
from myrag.tools.sql_tool import create_hologres_tools
from myrag.vectorstore.factory import create_vector_store, is_hologres_vector_store

# ===== 路径 / 持久化常量 =====
DATA_DIR = Path(os.getenv("MYRAG_DATA_DIR", "/app/data"))
UPLOADS_DIR = DATA_DIR / "uploads"
FAISS_DIR = DATA_DIR / "faiss"
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(10 * 1024 * 1024)))

REDIS_HOST = os.getenv("REDIS_HOST", "redis")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))

# session 在 Redis 里的 key 前缀
K_SESSION = "myrag:session:{sid}"   # hash：brain_name / created_at / nb_chunks / files / brain_id / owner_id / mode
K_FILES = "myrag:session:{sid}:files"  # set：上传过的文件名
K_HISTORY = "myrag:session:{sid}:history"  # list：聊天历史（JSON 序列化）

# 当请求没带 API Key 且后端也没配置 MY_API_KEY 时使用此共享 owner（向后兼容单租户场景）。
SHARED_OWNER = "__shared__"

# 会话工作模式：knowledge=文档问答(database=查业务数据 / auto=LLM 自动路由)
# RAG 与 Text-to-SQL 的关键分离点：不同 mode 走不同执行路径，避免一个 Brain 同时承担两种能力导致 prompt 污染。
SESSION_MODES = ("knowledge", "database", "auto")
DEFAULT_MODE = "knowledge"


# ===== Hologres 会话元数据表（永久存储）=====

HOLOGRES_SESSION_TABLE = os.getenv("HOLOGRES_SESSION_TABLE", "myrag_sessions")
if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", HOLOGRES_SESSION_TABLE):
    raise ValueError("HOLOGRES_SESSION_TABLE 只能是字母、数字和下划线，且首字符为字母/下划线")


class HologresSessionStore:
    """将会话元数据（brain_name / created_at / nb_chunks / files / brain_id / owner_id）持久化到 Hologres。

    设计要点：
    - 单表 myrag_sessions，session_id 为主键；
    - 写入采用 UPSERT，重启后可直接 SELECT 出来回填；
    - 仅当 VECTOR_STORE=hologres 时启用持久化（因为用户已配置 Hologres 连接信息）；
    - 支持 owner 隔离：upsert 必须带 owner_id；list/get/delete 都按 owner 校验。
    """

    def __init__(self) -> None:
        self.table = HOLOGRES_SESSION_TABLE

    def _connect(self):
        host = os.getenv("HOLOGRES_HOST")
        if not host:
            raise ValueError("启用 Hologres 会话持久化需要设置 HOLOGRES_HOST")
        return psycopg.connect(
            host=host,
            port=int(os.getenv("HOLOGRES_PORT", "80")),
            dbname=os.getenv("HOLOGRES_DB", ""),
            user=os.getenv("HOLOGRES_USER", ""),
            password=os.getenv("HOLOGRES_PASSWORD", ""),
            connect_timeout=int(os.getenv("HOLOGRES_CONNECT_TIMEOUT", "10")),
        )

    def _ensure_table(self) -> None:
        """幂等地创建/升级 myrag_sessions 表。

        Hologres 限制：
        - ADD COLUMN 不能再带 NOT NULL DEFAULT
        - 不支持 ALTER COLUMN SET NOT NULL
        - 不支持 ALTER COLUMN DROP DEFAULT
        所以 owner_id 在 schema 层保持 nullable，**应用层强制非空**（每次 upsert 都传）。
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.table} (
                    session_id   TEXT NOT NULL,
                    brain_name   TEXT NOT NULL,
                    created_at   TIMESTAMPTZ NOT NULL,
                    nb_chunks    INTEGER NOT NULL DEFAULT 0,
                    files        TEXT NOT NULL DEFAULT '',
                    brain_id     TEXT,
                    PRIMARY KEY (session_id)
                )
                """
            )
            conn.commit()

            # 兼容旧表：单独执行 ALTER，每条包 try/except，单条失败不影响后续。
            self._safe_alter(cur, conn,
                f"ALTER TABLE {self.table} "
                f"ADD COLUMN IF NOT EXISTS owner_id TEXT DEFAULT '__shared__'"
            )
            self._safe_alter(cur, conn,
                f"UPDATE {self.table} SET owner_id = '__shared__' "
                f"WHERE owner_id IS NULL"
            )
            # mode 字段：标识会话使用哪种能力（knowledge/database/auto），老行默认 'knowledge'。
            self._safe_alter(cur, conn,
                f"ALTER TABLE {self.table} "
                f"ADD COLUMN IF NOT EXISTS mode TEXT DEFAULT 'knowledge'"
            )
            self._safe_alter(cur, conn,
                f"UPDATE {self.table} SET mode = 'knowledge' WHERE mode IS NULL"
            )
            self._safe_alter(cur, conn,
                f"CREATE INDEX IF NOT EXISTS {self.table}_owner_idx "
                f"ON {self.table}(owner_id)"
            )

    @staticmethod
    def _safe_alter(cur, conn, sql: str) -> None:
        """执行 DDL，FeatureNotSupported / DuplicateObject / AlreadyExists 等一律吞掉。

        Hologres 不同版本对外暴露的限制不一致，逐条幂等执行比一次性脚本更稳。
        """
        try:
            cur.execute(sql)
            conn.commit()
        except Exception as e:  # noqa: BLE001
            conn.rollback()
            print(f"[WARN] skip DDL ({type(e).__name__}): {sql.splitlines()[0]}")

    async def ensure_table(self) -> None:
        await asyncio.to_thread(self._ensure_table)

    def _upsert(self, data: dict[str, Any]) -> None:
        self._ensure_table()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.table}
                    (session_id, owner_id, brain_name, created_at, nb_chunks, files, brain_id, mode)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (session_id) DO UPDATE SET
                    owner_id   = EXCLUDED.owner_id,
                    brain_name = EXCLUDED.brain_name,
                    nb_chunks  = EXCLUDED.nb_chunks,
                    files      = EXCLUDED.files,
                    brain_id   = COALESCE(EXCLUDED.brain_id, {self.table}.brain_id),
                    mode       = EXCLUDED.mode
                """,
                (
                    data["session_id"],
                    data.get("owner_id") or "__shared__",
                    data["brain_name"],
                    data["created_at"],
                    int(data.get("nb_chunks", 0)),
                    data.get("files", "") or "",
                    data.get("brain_id") or None,
                    data.get("mode") or "knowledge",
                ),
            )
            conn.commit()

    async def upsert(self, data: dict[str, Any]) -> None:
        await asyncio.to_thread(self._upsert, data)

    def _delete(self, session_id: str, owner_id: str | None) -> None:
        with self._connect() as conn, conn.cursor() as cur:
            if owner_id is None:
                cur.execute(f"DELETE FROM {self.table} WHERE session_id = %s", (session_id,))
            else:
                cur.execute(
                    f"DELETE FROM {self.table} WHERE session_id = %s AND owner_id = %s",
                    (session_id, owner_id),
                )
            conn.commit()

    async def delete(self, session_id: str, owner_id: str | None = None) -> None:
        await asyncio.to_thread(self._delete, session_id, owner_id)

    def _list_all(self, owner_id: str | None) -> list[dict[str, Any]]:
        self._ensure_table()
        with self._connect() as conn, conn.cursor() as cur:
            if owner_id is None:
                cur.execute(
                    f"""
                    SELECT session_id, owner_id, brain_name, created_at, nb_chunks, files, brain_id, mode
                    FROM {self.table}
                    ORDER BY created_at DESC
                    """
                )
            else:
                cur.execute(
                    f"""
                    SELECT session_id, owner_id, brain_name, created_at, nb_chunks, files, brain_id, mode
                    FROM {self.table}
                    WHERE owner_id = %s
                    ORDER BY created_at DESC
                    """,
                    (owner_id,),
                )
            rows = cur.fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            session_id, owner, brain_name, created_at, nb_chunks, files, brain_id, mode = row
            result.append(
                {
                    "session_id": session_id,
                    "owner_id": owner or "__shared__",
                    "brain_name": brain_name,
                    "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
                    "nb_chunks": int(nb_chunks or 0),
                    "files": files or "",
                    "brain_id": brain_id or "",
                    "mode": mode or "knowledge",
                }
            )
        return result

    async def list_all(self, owner_id: str | None = None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_all, owner_id)

    def _get_one(self, session_id: str, owner_id: str | None) -> dict[str, Any] | None:
        with self._connect() as conn, conn.cursor() as cur:
            if owner_id is None:
                cur.execute(
                    f"""
                    SELECT session_id, owner_id, brain_name, created_at, nb_chunks, files, brain_id, mode
                    FROM {self.table} WHERE session_id = %s
                    """,
                    (session_id,),
                )
            else:
                cur.execute(
                    f"""
                    SELECT session_id, owner_id, brain_name, created_at, nb_chunks, files, brain_id, mode
                    FROM {self.table}
                    WHERE session_id = %s AND owner_id = %s
                    """,
                    (session_id, owner_id),
                )
            row = cur.fetchone()
        if not row:
            return None
        sid, owner, brain_name, created_at, nb_chunks, files, brain_id, mode = row
        return {
            "session_id": sid,
            "owner_id": owner or "__shared__",
            "brain_name": brain_name,
            "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
            "nb_chunks": int(nb_chunks or 0),
            "files": files or "",
            "brain_id": brain_id or "",
            "mode": mode or "knowledge",
        }

    async def get_one(self, session_id: str, owner_id: str | None = None) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._get_one, session_id, owner_id)


hologres_sessions = HologresSessionStore()


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


# ===== 鉴权 + owner 解析 =====

class OwnerIdentity(BaseModel):
    """当前请求所属 owner：每个 owner 看到的是完全隔离的会话空间。"""
    owner_id: str


async def resolve_owner(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> OwnerIdentity:
    """从 API Key 解析 owner_id：相同 Key 永远解析到同一 owner；无 Key 时退化为共享 owner。"""
    expected = os.getenv("MY_API_KEY")
    if expected:
        if x_api_key != expected:
            raise HTTPException(status_code=401, detail="Invalid API Key")
        # 用 API Key 自身的 SHA1 前缀当 owner：相同 Key 始终命中同一桶，简单且不会泄漏原 Key。
        import hashlib
        owner_id = "key_" + hashlib.sha1(expected.encode("utf-8")).hexdigest()[:12]
        return OwnerIdentity(owner_id=owner_id)
    # 没配 MY_API_KEY：本地开发模式，全部归到共享 owner，方便多人共用一个 Hologres。
    return OwnerIdentity(owner_id=SHARED_OWNER)


async def verify_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """仅做鉴权，不暴露 owner；用于 /info、/health 这类不需要 owner 的端点。"""
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
    if is_hologres_vector_store():
        await hologres_sessions.ensure_table()
        # lifespan 全量恢复：用 owner_id 做 set 成员关系，session 写入 redis 时再 hash 化 owner。
        sessions = await hologres_sessions.list_all()
        rds = redis_client.client
        for s in sessions:
            owner_id = s.get("owner_id") or SHARED_OWNER
            mode = s.get("mode") or DEFAULT_MODE
            await rds.hset(K_SESSION.format(sid=s["session_id"]), mapping={
                "brain_name": s["brain_name"],
                "created_at": s["created_at"],
                "nb_chunks": str(s["nb_chunks"]),
                "files": s["files"],
                "brain_id": s["brain_id"],
                "owner_id": owner_id,
                "mode": mode,
            })
            if s["files"]:
                await rds.delete(K_FILES.format(sid=s["session_id"]))
                await rds.sadd(K_FILES.format(sid=s["session_id"]), *s["files"].split("|"))
        print(f"[OK] 从 Hologres 恢复 {len(sessions)} 个会话")
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
    mode: str = DEFAULT_MODE  # knowledge / database / auto

class CreateSessionRequest(BaseModel):
    brain_name: str = "MyRAG Brain"
    mode: str = DEFAULT_MODE  # 创建时就指定会话工作模式，后续可切换

class SetModeRequest(BaseModel):
    mode: str

class AskRequest(BaseModel):
    question: str
    session_id: str
    use_history: bool = True  # 默认开启历史

class AskResponse(BaseModel):
    answer: str
    session_id: str

class AskSqlResponse(BaseModel):
    answer: str
    session_id: str
    sql_steps: list[dict[str, Any]] = []

class AskAutoResponse(BaseModel):
    answer: str
    session_id: str
    route: str

class SuggestedQuestionsResponse(BaseModel):
    session_id: str
    questions: list[str]

class BrainInfo(BaseModel):
    session_id: str
    name: str
    llm_model: str
    nb_chunks: int
    uploaded_files: list[str]


# ===== Brain 仓库（In-memory cache，避免重复重建）=====
class BrainRepo:
    """Brain 句柄缓存：重启后从 Redis + 向量库重建。"""

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

        rds = redis_client.client
        session_key = K_SESSION.format(sid=session_id)
        raw = await rds.hgetall(session_key)
        if raw and is_hologres_vector_store():
            from myrag.storage import LocalStorage
            brain = Brain(
                name=brain_name,
                llm=llm,
                storage=LocalStorage(UPLOADS_DIR / session_id),
                brain_id=_safe_brain_id(raw.get("brain_id")),
            )
            brain.vector_store = create_vector_store(embedder, session_id)
            self.put(session_id, brain)
            print(f"♻️ 从 Hologres 重建 Brain: {session_id}")
            return brain

        # 尝试从磁盘 + Redis 元信息重建
        faiss_path = FAISS_DIR / session_id
        if raw and faiss_path.exists():
            # 有持久化索引 + Redis 元信息 → 重建
            from myrag.storage import LocalStorage
            brain = Brain(
                name=brain_name,
                llm=llm,
                storage=LocalStorage(UPLOADS_DIR / session_id),
                brain_id=_safe_brain_id(raw.get("brain_id")),
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


class HologresAgent(TextToSQLAgent):
    """面向 Hologres 的 Text-to-SQL Agent。"""

    def _build_system_prompt(self, db_path: str, attempted: list | None = None) -> str:
        attempted_note = ""
        if attempted:
            attempted_note = (
                "\n\n以下尝试已经失败，不要重复：\n"
                + "\n".join(
                    f"- {a['tool']}({a['args']}) -> {a['error']}"
                    for a in attempted[-3:]
                )
            )
        return (
            "你是一个 SQL 数据分析助手，可以调用阿里云 Hologres 工具。\n\n"
            "可用工具：\n"
            "- get_hologres_schema：查看所有表结构\n"
            "- hologres_query：执行 SELECT 查询，参数名为 query\n\n"
            "工作流程：先查看 schema，再执行查询，最后用自然语言回答。\n"
            f"{attempted_note}\n\n"
            "调工具时只输出 JSON："
            '{"action":"tool_call","tool":"工具名","args":{},"reasoning":"..."}\n'
            "最终回答时只输出 JSON："
            '{"action":"final_answer","answer":"...","reasoning":"..."}'
        )


_sql_agent: HologresAgent | None = None
_auto_router_llm: LLMEndpoint | None = None


def _get_sql_agent() -> HologresAgent:
    """懒加载无状态 SQL Agent，避免服务启动时强依赖数据库连接。"""
    global _sql_agent
    if _sql_agent is None:
        registry = ToolRegistry()
        schema_tool, query_tool = create_hologres_tools()
        registry.register("get_hologres_schema", schema_tool)
        registry.register("hologres_query", query_tool)
        _sql_agent = HologresAgent(
            llm=LLMEndpoint.from_env(),
            tool_registry=registry,
            max_iterations=5,
        )
    return _sql_agent


def _get_auto_router_llm() -> LLMEndpoint:
    global _auto_router_llm
    if _auto_router_llm is None:
        _auto_router_llm = LLMEndpoint.from_env()
    return _auto_router_llm


async def _route_auto(question: str, has_documents: bool) -> str:
    """让 LLM 判断问题应该走文档检索还是数据库查询。"""
    messages = [
        SystemMessage(content=(
            "你是 MyRAG 的意图路由器。只能在 knowledge 和 database 中二选一。\n"
            "knowledge：问题询问已上传文档、规则、说明、方案或文本内容。\n"
            "database：问题询问业务表、客户、订单、销售额、统计、排名或数量。\n"
            f"当前会话是否有文档：{'是' if has_documents else '否'}。\n"
            "只输出 JSON，不要解释："
            '{"route":"knowledge"} 或 {"route":"database"}'
        )),
        HumanMessage(content=question),
    ]
    text = await _get_auto_router_llm().ainvoke(messages)
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            route = json.loads(match.group()).get("route")
            if route in {"knowledge", "database"}:
                return route
        except json.JSONDecodeError:
            pass
    return "knowledge" if has_documents else "database"


def _safe_brain_id(raw_value: str | None) -> uuid.UUID | None:
    """把 Redis/Hologres 里存的 brain_id 字符串解析成 UUID，失败时返回 None。

    历史数据可能存的是空串 / 非 UUID 字面量（之前版本的 brain_id 默认值、
    或者 Hologres 迁移时类型变化），直接 `uuid.UUID(...)` 会抛 ValueError 拖垮请求。
    """
    if not raw_value:
        return None
    try:
        return uuid.UUID(raw_value)
    except (ValueError, TypeError):
        return None


# ===== 辅助函数 =====

async def _save_faiss_async(brain: Brain, session_id: str) -> None:
    """后台任务：异步保存 FAISS 索引到磁盘（不阻塞响应）。"""
    if brain.vector_store is None or is_hologres_vector_store():
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


@app.post("/sessions", response_model=SessionInfo)
async def create_session(req: CreateSessionRequest, owner: OwnerIdentity = Depends(resolve_owner)) -> SessionInfo:
    """创建新会话。返回 session_id，后续接口都要传它。"""
    if req.mode not in SESSION_MODES:
        raise HTTPException(400, f"mode 必须是 {SESSION_MODES} 之一")
    session_id = str(uuid.uuid4())
    rds = redis_client.client
    now = datetime.now(timezone.utc).isoformat()
    record = {
        "session_id": session_id,
        "owner_id": owner.owner_id,
        "brain_name": req.brain_name,
        "created_at": now,
        "nb_chunks": 0,
        "files": "",
        "brain_id": "",
        "mode": req.mode,
    }
    if is_hologres_vector_store():
        await hologres_sessions.upsert(record)
    async with rds.pipeline(transaction=True) as pipe:
        await pipe.hset(
            K_SESSION.format(sid=session_id),
            mapping={
                "brain_name": req.brain_name,
                "created_at": now,
                "nb_chunks": 0,
                "files": "",
                "brain_id": "",
                "owner_id": owner.owner_id,
                "mode": req.mode,
            },
        )
        await pipe.execute()
    return SessionInfo(
        session_id=session_id,
        brain_name=req.brain_name,
        nb_chunks=0,
        files=[],
        created_at=now,
        mode=req.mode,
    )


@app.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(owner: OwnerIdentity = Depends(resolve_owner)) -> list[SessionInfo]:
    """列出当前 owner 的会话：Hologres 为权威源，FAISS 模式下退化为扫描 Redis。"""
    if is_hologres_vector_store():
        rows = await hologres_sessions.list_all(owner.owner_id)
        return [
            SessionInfo(
                session_id=row["session_id"],
                brain_name=row["brain_name"],
                nb_chunks=int(row["nb_chunks"]),
                files=[f for f in (row["files"] or "").split("|") if f],
                created_at=row["created_at"],
                mode=row.get("mode") or DEFAULT_MODE,
            )
            for row in rows
        ]

    # FAISS 模式：从 Redis 扫 myrag:session:* 哈希，仅返回 owner_id 匹配的
    rds = redis_client.client
    out: list[SessionInfo] = []
    async for key in rds.scan_iter(match="myrag:session:*"):
        if ":files" in key or ":history" in key:
            continue
        sid = key.split(":", 2)[2] if key.count(":") >= 2 else None
        if not sid:
            continue
        raw = await rds.hgetall(key)
        if not raw:
            continue
        if (raw.get("owner_id") or SHARED_OWNER) != owner.owner_id:
            continue
        out.append(
            SessionInfo(
                session_id=sid,
                brain_name=raw.get("brain_name", ""),
                nb_chunks=int(raw.get("nb_chunks", 0)),
                files=[f for f in raw.get("files", "").split("|") if f],
                created_at=raw.get("created_at", ""),
                mode=raw.get("mode") or DEFAULT_MODE,
            )
        )
    out.sort(key=lambda s: s.created_at, reverse=True)
    return out


@app.get("/sessions/{session_id}", response_model=SessionInfo)
async def get_session_meta(session_id: str, owner: OwnerIdentity = Depends(resolve_owner)) -> SessionInfo:
    rds = redis_client.client
    raw = await rds.hgetall(K_SESSION.format(sid=session_id))
    if not raw:
        raise HTTPException(404, f"session {session_id} 不存在")
    if (raw.get("owner_id") or SHARED_OWNER) != owner.owner_id:
        raise HTTPException(403, "无权访问此会话")
    files = [f for f in raw.get("files", "").split("|") if f]
    return SessionInfo(
        session_id=session_id,
        brain_name=raw.get("brain_name", ""),
        nb_chunks=int(raw.get("nb_chunks", 0)),
        files=files,
        created_at=raw.get("created_at", ""),
        mode=raw.get("mode") or DEFAULT_MODE,
    )


@app.post("/sessions/{session_id}/mode", response_model=SessionInfo)
async def set_session_mode(
    session_id: str,
    req: SetModeRequest,
    owner: OwnerIdentity = Depends(resolve_owner),
) -> SessionInfo:
    """切换会话的工作模式：knowledge / database / auto。"""
    if req.mode not in SESSION_MODES:
        raise HTTPException(400, f"mode 必须是 {SESSION_MODES} 之一")
    rds = redis_client.client
    raw = await rds.hgetall(K_SESSION.format(sid=session_id))
    if not raw:
        raise HTTPException(404, f"session {session_id} 不存在")
    if (raw.get("owner_id") or SHARED_OWNER) != owner.owner_id:
        raise HTTPException(403, "无权访问此会话")

    await rds.hset(K_SESSION.format(sid=session_id), "mode", req.mode)
    if is_hologres_vector_store():
        # 同步写 Hologres：复用 upsert，保证重启后 mode 还在。
        await hologres_sessions.upsert({
            "session_id": session_id,
            "owner_id": owner.owner_id,
            "brain_name": raw.get("brain_name", "MyRAG Brain"),
            "created_at": raw.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "nb_chunks": int(raw.get("nb_chunks", 0)),
            "files": raw.get("files", ""),
            "brain_id": raw.get("brain_id", ""),
            "mode": req.mode,
        })
    files = [f for f in raw.get("files", "").split("|") if f]
    return SessionInfo(
        session_id=session_id,
        brain_name=raw.get("brain_name", ""),
        nb_chunks=int(raw.get("nb_chunks", 0)),
        files=files,
        created_at=raw.get("created_at", ""),
        mode=req.mode,
    )


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str, owner: OwnerIdentity = Depends(resolve_owner)) -> dict[str, str]:
    rds = redis_client.client
    raw = await rds.hgetall(K_SESSION.format(sid=session_id))
    if not raw:
        raise HTTPException(404, f"session {session_id} 不存在")
    if (raw.get("owner_id") or SHARED_OWNER) != owner.owner_id:
        raise HTTPException(403, "无权删除此会话")
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
    if is_hologres_vector_store():
        embedder = DashScopeEmbedder.from_env()
        vector_store = create_vector_store(embedder, session_id)
        if hasattr(vector_store, "delete_tenant"):
            await vector_store.delete_tenant()
        await hologres_sessions.delete(session_id, owner.owner_id)
    return {"deleted": session_id}


@app.get("/sessions/{session_id}/history")
async def get_history(session_id: str, owner: OwnerIdentity = Depends(resolve_owner)) -> dict[str, Any]:
    """返回聊天历史，便于前端刷新后恢复对话气泡。"""
    rds = redis_client.client
    raw = await rds.hgetall(K_SESSION.format(sid=session_id))
    if not raw:
        raise HTTPException(404, "session 不存在")
    if (raw.get("owner_id") or SHARED_OWNER) != owner.owner_id:
        raise HTTPException(403, "无权访问此会话")
    history = await _get_history(session_id)
    return {"session_id": session_id, "history": history}


@app.post("/upload")
async def upload(
    session_id: str,
    files: list[UploadFile] = File(...),
    owner: OwnerIdentity = Depends(resolve_owner),
) -> dict[str, Any]:
    """上传文档到指定 session。第一个文件上传时创建 Brain。"""
    rds = redis_client.client
    session_key = K_SESSION.format(sid=session_id)
    raw = await rds.hgetall(session_key)
    if not raw:
        raise HTTPException(404, f"session {session_id} 不存在，请先调用 session 初始化")
    if (raw.get("owner_id") or SHARED_OWNER) != owner.owner_id:
        raise HTTPException(403, "无权访问此会话")

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
        content = await upload_file.read()
        if len(content) > MAX_UPLOAD_SIZE:
            raise HTTPException(413, f"文件大小不能超过 {MAX_UPLOAD_SIZE // (1024 * 1024)} MB")
        dst.write_bytes(content)
        saved_paths.append(dst)

    # 2. 重建/获取 Brain
    llm = LLMEndpoint.from_env()
    embedder = DashScopeEmbedder.from_env()
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
            brain.vector_store = create_vector_store(embedder, session_id)
        await brain.vector_store.add_documents(new_docs)
        await _save_faiss_async(brain, session_id)

    # 4. 写 Redis 元信息
    old_files = await rds.smembers(K_FILES.format(sid=session_id))
    new_filenames = [f.filename for f in files]
    files_combined = "|".join(sorted(old_files | set(new_filenames)))
    async with rds.pipeline(transaction=True) as pipe:
        if new_filenames:
            await pipe.sadd(K_FILES.format(sid=session_id), *new_filenames)
        await pipe.hset(
            session_key,
            mapping={
                "nb_chunks": str(len(brain.knowledge)),
                "files": files_combined,
                "brain_id": str(brain.id),
            },
        )
        await pipe.execute()
    if is_hologres_vector_store():
        await hologres_sessions.upsert({
            "session_id": session_id,
            "owner_id": owner.owner_id,
            "brain_name": brain_name,
            "created_at": raw.get("created_at") or datetime.now(timezone.utc).isoformat(),
            "nb_chunks": len(brain.knowledge),
            "files": files_combined,
            "brain_id": str(brain.id),
            "mode": raw.get("mode") or DEFAULT_MODE,
        })

    return {
        "uploaded": new_filenames,
        "session_id": session_id,
        "total_chunks": len(brain.knowledge),
    }


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest, owner: OwnerIdentity = Depends(resolve_owner)) -> AskResponse:
    """同步问答（带 session 历史）。只服务于 knowledge 模式。"""
    rds = redis_client.client
    raw = await rds.hgetall(K_SESSION.format(sid=req.session_id))
    if not raw:
        raise HTTPException(404, "session 不存在")
    if (raw.get("owner_id") or SHARED_OWNER) != owner.owner_id:
        raise HTTPException(403, "无权访问此会话")
    session_mode = raw.get("mode") or DEFAULT_MODE
    if session_mode != "knowledge":
        raise HTTPException(
            400,
            f"当前会话 mode={session_mode}，请使用对应端点：/ask_sql 或 /ask_auto"
        )
    brain = brain_repo.get(req.session_id) or await brain_repo.load_or_create(
        req.session_id,
        raw.get("brain_name", "MyRAG Brain"),
        LLMEndpoint.from_env(),
        DashScopeEmbedder.from_env(),
    )
    history = await _get_history(req.session_id) if req.use_history else []
    answer = await brain.ask(req.question, chat_history=history)
    await _append_history(req.session_id, "user", req.question)
    await _append_history(req.session_id, "assistant", answer)
    return AskResponse(answer=answer, session_id=req.session_id)


@app.post("/ask_sql", response_model=AskSqlResponse)
async def ask_sql(req: AskRequest, owner: OwnerIdentity = Depends(resolve_owner)) -> AskSqlResponse:
    """使用 Hologres Text-to-SQL Agent 回答数据库问题。"""
    rds = redis_client.client
    raw = await rds.hgetall(K_SESSION.format(sid=req.session_id))
    if not raw:
        raise HTTPException(404, "session 不存在")
    if (raw.get("owner_id") or SHARED_OWNER) != owner.owner_id:
        raise HTTPException(403, "无权访问此会话")
    session_mode = raw.get("mode") or DEFAULT_MODE
    if session_mode != "database":
        raise HTTPException(400, f"当前会话 mode={session_mode}，请先切换到 database")

    result = await _get_sql_agent().run(req.question, db_path="hologres")
    await _append_history(req.session_id, "user", req.question)
    await _append_history(req.session_id, "assistant", result.answer)
    return AskSqlResponse(
        answer=result.answer,
        session_id=req.session_id,
        sql_steps=[step.__dict__ for step in result.steps],
    )


@app.post("/ask_sql_stream")
async def ask_sql_stream(req: AskRequest, owner: OwnerIdentity = Depends(resolve_owner)):
    """以 SSE 返回 SQL Agent 的完整执行过程。"""
    # #region debug-point E:server-entry
    await asyncio.to_thread(__import__('urllib.request', fromlist=['Request']).urlopen, __import__('urllib.request', fromlist=['Request']).Request('http://127.0.0.1:7777/event', data=json.dumps({'sessionId': 'sql-agent-stream', 'runId': 'pre', 'hypothesisId': 'E', 'location': 'examples/step13_fastapi/main.py:ask_sql_stream', 'msg': '[DEBUG] ask_sql_stream entered', 'data': {'session_id': req.session_id, 'mode': 'database'}}).encode(), headers={'Content-Type': 'application/json'}))
    # #endregion
    rds = redis_client.client
    raw = await rds.hgetall(K_SESSION.format(sid=req.session_id))
    if not raw:
        raise HTTPException(404, "session 不存在")
    if (raw.get("owner_id") or SHARED_OWNER) != owner.owner_id:
        raise HTTPException(403, "无权访问此会话")
    if (raw.get("mode") or DEFAULT_MODE) != "database":
        raise HTTPException(400, "当前会话不是 database 模式")

    async def event_generator():
        try:
            async for event in _get_sql_agent().astream_events(req.question, db_path="hologres"):
                # #region debug-point C:server-event
                await asyncio.to_thread(__import__('urllib.request', fromlist=['Request']).urlopen, __import__('urllib.request', fromlist=['Request']).Request('http://127.0.0.1:7777/event', data=json.dumps({'sessionId': 'sql-agent-stream', 'runId': 'pre', 'hypothesisId': 'C', 'location': 'examples/step13_fastapi/main.py:event_generator', 'msg': '[DEBUG] SSE event yielded', 'data': {'event_type': event.get('type')}}).encode(), headers={'Content-Type': 'application/json'}))
                # #endregion
                if event.get("type") == "done":
                    await _append_history(req.session_id, "user", req.question)
                    await _append_history(req.session_id, "assistant", event["answer"])
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/ask_auto", response_model=AskAutoResponse)
async def ask_auto(req: AskRequest, owner: OwnerIdentity = Depends(resolve_owner)) -> AskAutoResponse:
    """根据问题意图，在文档 RAG 与 Hologres SQL 之间自动路由。"""
    rds = redis_client.client
    raw = await rds.hgetall(K_SESSION.format(sid=req.session_id))
    if not raw:
        raise HTTPException(404, "session 不存在")
    if (raw.get("owner_id") or SHARED_OWNER) != owner.owner_id:
        raise HTTPException(403, "无权访问此会话")
    if (raw.get("mode") or DEFAULT_MODE) != "auto":
        raise HTTPException(400, "当前会话不是 auto 模式")

    route = await _route_auto(req.question, int(raw.get("nb_chunks", 0)) > 0)
    if route == "database":
        result = await _get_sql_agent().run(req.question, db_path="hologres")
        answer = result.answer
    else:
        brain = brain_repo.get(req.session_id) or await brain_repo.load_or_create(
            req.session_id,
            raw.get("brain_name", "MyRAG Brain"),
            LLMEndpoint.from_env(),
            DashScopeEmbedder.from_env(),
        )
        history = await _get_history(req.session_id) if req.use_history else []
        answer = await brain.ask(req.question, chat_history=history)

    await _append_history(req.session_id, "user", req.question)
    await _append_history(req.session_id, "assistant", answer)
    return AskAutoResponse(answer=answer, session_id=req.session_id, route=route)


@app.get("/sessions/{session_id}/suggested-questions", response_model=SuggestedQuestionsResponse)
async def suggested_questions(
    session_id: str,
    count: int = 5,
    owner: OwnerIdentity = Depends(resolve_owner),
) -> SuggestedQuestionsResponse:
    """基于当前 session 的文档生成可点击的问题。"""
    if count < 1 or count > 10:
        raise HTTPException(400, "count 必须在 1 到 10 之间")
    rds = redis_client.client
    session_key = K_SESSION.format(sid=session_id)
    raw = await rds.hgetall(session_key)
    if not raw:
        raise HTTPException(404, "session 不存在")
    if (raw.get("owner_id") or SHARED_OWNER) != owner.owner_id:
        raise HTTPException(403, "无权访问此会话")
    if int(raw.get("nb_chunks", 0)) <= 0:
        return SuggestedQuestionsResponse(session_id=session_id, questions=[])
    brain = brain_repo.get(session_id) or await brain_repo.load_or_create(
        session_id,
        raw.get("brain_name", "MyRAG Brain"),
        LLMEndpoint.from_env(),
        DashScopeEmbedder.from_env(),
    )
    questions = await brain.suggest_questions(count=count)
    return SuggestedQuestionsResponse(session_id=session_id, questions=questions)


@app.post("/ask_stream")
async def ask_stream(req: AskRequest, owner: OwnerIdentity = Depends(resolve_owner)):
    """SSE 流式问答。只服务于 knowledge 模式。"""
    rds = redis_client.client
    raw = await rds.hgetall(K_SESSION.format(sid=req.session_id))
    if not raw:
        raise HTTPException(404, "session 不存在")
    if (raw.get("owner_id") or SHARED_OWNER) != owner.owner_id:
        raise HTTPException(403, "无权访问此会话")
    session_mode = raw.get("mode") or DEFAULT_MODE
    if session_mode != "knowledge":
        raise HTTPException(
            400,
            f"当前会话 mode={session_mode}，请使用对应端点：/ask_sql 或 /ask_auto"
        )
    brain = brain_repo.get(req.session_id) or await brain_repo.load_or_create(
        req.session_id,
        raw.get("brain_name", "MyRAG Brain"),
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
